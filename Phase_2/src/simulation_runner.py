import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from src.state import AUVState, AUVControl, wrap_angle, wrap_to_2pi
from src.goal import Goal, PathTarget
from src.global_planner import GlobalMapPlanner
from src.constraints import Boundaries3D, CylindricalObstacle3D, ConstraintManager
from src.physics_dynamics import PhysicsDynamics
from src.learned_dynamics import LearnedDynamics
from src.dynamics_interface import DynamicsInterface
from src.mpc_controller import MPCController
from src.cost_function import MPCCostFunction
from src.true_plant import TrueAUVPlant, PlantParameters, CylindricalObstacle
from src.measurement import SensorMeasurementLayer, OODDetector
from src.baselines import ReactiveLOSController, PotentialFieldLOSController

class ClosedLoopSimulation:
    """
    Research-Grade Closed-Loop Simulation Harness.
    Executes an AUV mission under the decoupled plant-controller architecture:
    
       Controller (Internal Model: Hybrid, Physics, or GRU)
             │  (u_0* commands)
             ▼
       True Plant (Continuous-Time RK4, Actuator Lag, Damping Mismatch, Currents)
             │  (True states x_t)
             ▼
       Sensor Measurement Layer (USBL noise, DVL noise, Kalman Filter, OOD Detector)
             │  (Filtered/Measured state y_t, Features s_t)
             ▼
       Controller Feedback
    """

    def __init__(
        self,
        method: str,  # "hybrid_mpc", "physics_mpc", "gru_mpc", "reactive_los", "potential_field"
        scenario_config: Dict[str, Any],
        seed: int = 1000,
        dt_control: float = 2.0,
        max_steps: int = 120,
        goal_tolerance_primary: float = 5.0,
        goal_tolerance_tight: float = 2.0,
        # Ablations / Overrides
        horizon: int = 10,
        num_candidates: int = 70,
        iterations: int = 3,
        use_warm_start: bool = True,
        use_los_seed: bool = True,
        use_kalman_filter: bool = True,
        cost_weight_scaling: float = 1.0,
        usbl_sigma: float = 0.50,
        dropout_prob: float = 0.0,
        current_velocity: Optional[np.ndarray] = None,
        randomize_plant_mismatch: bool = True,
        include_ne_residual: bool = True,
        return_to_start: bool = False
    ):
        self.method = method
        self.scenario = scenario_config
        self.seed = seed
        self.dt_control = dt_control
        self.max_steps = max_steps
        self.tol_primary = goal_tolerance_primary
        self.tol_tight = goal_tolerance_tight
        self.return_to_start = return_to_start
        self.return_path = None
        self.return_planner = None
        self.rng = np.random.default_rng(seed)

        # 1. Environment Geometry
        b_cfg = scenario_config.get("boundaries", {
            "north_min": 0.0, "north_max": 120.0,
            "east_min": 0.0, "east_max": 100.0,
            "depth_min": 10.0, "depth_max": 60.0
        })
        self.boundaries = Boundaries3D(**b_cfg)
        
        # Obstacles
        self.obstacles_geom = []
        self.plant_obstacles = []
        for obs_cfg in scenario_config.get("obstacles", []):
            self.obstacles_geom.append(CylindricalObstacle3D(
                north=obs_cfg["north"],
                east=obs_cfg["east"],
                radius=obs_cfg["radius"],
                margin=obs_cfg.get("margin", 4.0),
                depth_min=obs_cfg.get("depth_min", 0.0),
                depth_max=obs_cfg.get("depth_max", 100.0)
            ))
            self.plant_obstacles.append(CylindricalObstacle(
                north=obs_cfg["north"],
                east=obs_cfg["east"],
                radius=obs_cfg["radius"],
                margin=obs_cfg.get("margin", 4.0)
            ))

        self.constraint_mgr = ConstraintManager(
            boundaries=self.boundaries,
            obstacles=self.obstacles_geom
        )

        # 2. Mission Waypoints & Pre-Mission Global Map Path Planning
        self.start_pos = np.array(scenario_config["start"], dtype=np.float64)
        self.goal_pos = np.array(scenario_config["goal"], dtype=np.float64)
        
        # Plan optimal, collision-free 3D global path before exploration starts
        self.global_planner = GlobalMapPlanner(
            resolution=1.0,
            default_standoff=4.0,
            lookahead_distance=16.0
        )
        self.global_path = self.global_planner.plan(
            start=self.start_pos,
            goal=self.goal_pos,
            boundaries=self.boundaries,
            obstacles=self.obstacles_geom
        )

        if len(self.global_path) >= 2:
            init_tangent = self.global_path[1] - self.global_path[0]
            start_heading = np.arctan2(init_tangent[1], init_tangent[0]) % (2 * np.pi)
        else:
            start_heading = np.arctan2(self.goal_pos[1] - self.start_pos[1], self.goal_pos[0] - self.start_pos[0]) % (2 * np.pi)

        # Clamp start & goal depth to safe operational depth envelope
        safe_start_d = float(np.clip(self.start_pos[2], self.boundaries.depth_min + 1.0, self.boundaries.depth_max - 1.0))
        safe_goal_d = float(np.clip(self.goal_pos[2], self.boundaries.depth_min + 1.0, self.boundaries.depth_max - 1.0))

        self.start_heading = start_heading
        self.initial_state = AUVState(
            north=self.start_pos[0],
            east=self.start_pos[1],
            depth=safe_start_d,
            roll=0.0,
            pitch=0.0,
            heading=start_heading,
            speed=1.20
        )
        self.goal = Goal(
            north=self.goal_pos[0],
            east=self.goal_pos[1],
            depth=safe_goal_d,
            tolerance=self.tol_primary
        )

        # 3. Setup Plant
        plant_params = PlantParameters(
            kd_baseline=0.06073034,
            kd_mismatch=float(self.rng.uniform(-0.25, 0.25)) if randomize_plant_mismatch else 0.0
        )
        curr_v = current_velocity if current_velocity is not None else np.array(scenario_config.get("current", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.plant = TrueAUVPlant(
            params=plant_params,
            dt_control=dt_control,
            dt_sim=0.10,
            current_velocity=curr_v,
            obstacles=self.plant_obstacles,
            seed=seed
        )

        # 4. Setup Measurement Layer & OOD Detector
        train_mean = np.array([1.21337, 0.22595, -0.01470, 0.04254, 0.02336, 68.99466])
        train_cov = np.diag([0.04705, 0.00044, 0.00248, 0.84294, 0.15470, 998.32227])
        feat_bounds = {
            "GroundSpeed": (0.0, 3.5),
            "Roll": (-0.6, 0.6),
            "Pitch": (-0.45, 0.45),
            "cos(Heading)": (-1.0, 1.0),
            "sin(Heading)": (-1.0, 1.0),
            "Depth": (0.0, 110.0)
        }
        ood_det = OODDetector(train_mean, train_cov, feat_bounds)
        self.measurement_layer = SensorMeasurementLayer(
            usbl_sigma=usbl_sigma,
            use_kalman_filter=use_kalman_filter,
            dropout_probability=dropout_prob,
            ood_detector=ood_det,
            seed=seed + 1
        )

        # 5. Setup Controller
        dyn_mode = "hybrid"
        if method == "physics_mpc":
            dyn_mode = "physics_only"
        elif method == "gru_mpc":
            dyn_mode = "gru_only"

        dyn_interface = DynamicsInterface(
            dynamics_mode=dyn_mode,
            include_ne_residual=include_ne_residual
        )
        
        cost_func = MPCCostFunction(
            w_goal=30.0 * cost_weight_scaling,
            w_tracking=6.0 * cost_weight_scaling,
            w_control=1.0 * cost_weight_scaling,
            w_smooth=4.0 * cost_weight_scaling,
            w_constraint=200.0 * cost_weight_scaling,
            u_cruise=1.20
        )

        self.controller = None
        if "mpc" in method:
            self.controller = MPCController(
                dynamics_interface=dyn_interface,
                cost_function=cost_func,
                prediction_horizon=horizon,
                num_candidates=num_candidates,
                elite_count=10,
                iterations=iterations,
                use_ai=("physics" not in method),
                use_warm_start=use_warm_start,
                use_los_seed=use_los_seed,
                seed=seed + 2
            )
        elif method == "reactive_los":
            self.controller = ReactiveLOSController(cruise_speed=1.20)
        elif method == "potential_field":
            self.controller = PotentialFieldLOSController(cruise_speed=1.20)
        else:
            raise ValueError(f"Unknown controller method: {method}")

    def run(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes full closed-loop mission.
        Returns:
            trajectory_df: Per-step flight log
            metrics: Comprehensive dictionary of performance metrics
        """
        self.plant.reset(self.initial_state)
        self.measurement_layer.reset(self.initial_state)
        if hasattr(self.controller, "reset"):
            self.controller.reset(self.initial_state)

        # Feature buffer: 10 steps of synthetic cruise consistent with initial state
        initial_feat = np.array([
            self.initial_state.speed,
            self.initial_state.roll,
            self.initial_state.pitch,
            np.cos(self.initial_state.heading),
            np.sin(self.initial_state.heading),
            self.initial_state.depth
        ], dtype=np.float64)
        feature_history = np.tile(initial_feat, (10, 1))

        # Time series logs
        log_rows = []
        
        cur_true_state = self.initial_state
        prev_pos = cur_true_state.to_array()[:3]
        
        # State tracking for A -> B -> (optional A) mission
        current_phase = "FORWARD" # "FORWARD" (Leg 1: A -> B), "RETURNING" (Leg 2: B -> A), or "COMPLETED"
        active_target_pos = self.goal_pos.copy()
        active_goal_obj = self.goal
        active_global_planner = self.global_planner

        reached_b = False
        step_reached_b = None
        time_reached_b = None

        reached_a = False
        step_reached_a = None
        time_reached_a = None

        closest_distance_b = float('inf')
        closest_distance_a = float('inf')
        closest_distance = float('inf')

        forward_path_length = 0.0
        return_path_length = 0.0
        total_path_length = 0.0

        solve_times = []
        ood_steps = 0
        slew_saturated_steps = 0
        total_steps = 0
        
        straight_line_dist = float(np.linalg.norm(self.goal_pos - self.start_pos))

        for step in range(self.max_steps):
            t_sim = step * self.dt_control
            total_steps += 1

            # 1. Take Measurement from True State
            measured_state, feat_vec, meas_info = self.measurement_layer.measure(cur_true_state)
            feature_history = np.vstack([feature_history[1:], feat_vec])
            
            if meas_info["ood"]["is_ood"]:
                ood_steps += 1

            # Distance evaluations
            cur_pos_3d = cur_true_state.to_array()[:3]
            true_dist_to_b = float(np.linalg.norm(cur_pos_3d - self.goal_pos))
            true_dist_to_a = float(np.linalg.norm(cur_pos_3d - self.start_pos))
            true_dist_to_active = true_dist_to_b if current_phase == "FORWARD" else true_dist_to_a

            if true_dist_to_b < closest_distance_b:
                closest_distance_b = true_dist_to_b
            if current_phase in ["RETURNING", "ROTATING_AT_A", "COMPLETED"]:
                if true_dist_to_a < closest_distance_a:
                    closest_distance_a = true_dist_to_a
            if true_dist_to_active < closest_distance:
                closest_distance = true_dist_to_active

            # Check discrete waypoint arrival
            if current_phase == "FORWARD":
                if true_dist_to_b <= self.tol_primary and not reached_b:
                    reached_b = True
                    step_reached_b = step
                    time_reached_b = t_sim
                    if self.return_to_start:
                        current_phase = "ROTATING_BACK"
                        active_target_pos = self.start_pos.copy()
                        dx_back = self.start_pos[0] - cur_true_state.north
                        dy_back = self.start_pos[1] - cur_true_state.east
                        target_heading_back = float(wrap_to_2pi(np.arctan2(dy_back, dx_back)))
                        self.plant.params.max_heading_rate_sec = 0.35
                        rotation_steps_b = 0
                        closest_distance_a = true_dist_to_a
                    else:
                        current_phase = "COMPLETED"
                        break

            # 2. Compute Control Action
            predicted_traj = None
            cost_val = 0.0
            solve_time = 0.0
            breakdown = {}

            if current_phase == "ROTATING_BACK":
                rotation_steps_b += 1
                dx_back = self.start_pos[0] - cur_true_state.north
                dy_back = self.start_pos[1] - cur_true_state.east
                target_heading_back = float(wrap_to_2pi(np.arctan2(dy_back, dx_back)))
                action = AUVControl(speed_cmd=0.85, pitch_cmd=0.0, heading_cmd=target_heading_back)
                solve_time = 0.001
                
                hdg_diff_b = abs(wrap_angle(cur_true_state.heading - target_heading_back))
                if hdg_diff_b <= 0.35 or rotation_steps_b >= 4:
                    current_phase = "RETURNING"
                    self.plant.params.max_heading_rate_sec = 0.15
                    self.return_planner = GlobalMapPlanner(
                        resolution=1.0,
                        default_standoff=4.0,
                        lookahead_distance=16.0
                    )
                    self.return_path = self.return_planner.plan(
                        start=cur_true_state.position(),
                        goal=self.start_pos,
                        boundaries=self.boundaries,
                        obstacles=self.obstacles_geom
                    )
                    active_global_planner = self.return_planner
                    safe_start_d = float(np.clip(self.start_pos[2], self.boundaries.depth_min + 1.0, self.boundaries.depth_max - 1.0))
                    active_goal_obj = Goal(
                        north=self.start_pos[0],
                        east=self.start_pos[1],
                        depth=safe_start_d,
                        tolerance=self.tol_primary
                    )
                    if hasattr(self.controller, "u_warm_start"):
                        self.controller.u_warm_start = None
                    if hasattr(self.controller, "prev_optimal_controls"):
                        self.controller.prev_optimal_controls = None
                    closest_distance = true_dist_to_a
                    closest_distance_a = true_dist_to_a

            elif current_phase == "ROTATING_AT_A":
                rotation_steps_a += 1
                action = AUVControl(speed_cmd=0.0, pitch_cmd=0.0, heading_cmd=self.start_heading)
                solve_time = 0.001
                hdg_diff_a = abs(wrap_angle(cur_true_state.heading - self.start_heading))
                if hdg_diff_a <= 0.25 or rotation_steps_a >= 3:
                    current_phase = "COMPLETED"

            else:
                tgt_pos, ref_heading, ref_pitch, dist_rem, horiz_pts = active_global_planner.get_lookahead_horizon_targets(
                    cur_true_state.position(),
                    lookahead_dist=16.0,
                    horizon=self.controller.H if hasattr(self.controller, "H") else 10,
                    dt=self.dt_control,
                    cruise_speed=1.20
                )
                path_target = PathTarget(
                    target_pos=tgt_pos,
                    target_heading=ref_heading,
                    target_pitch=ref_pitch,
                    dist_remaining=dist_rem,
                    final_goal=active_goal_obj,
                    horizon_waypoints=horiz_pts
                )

                if "mpc" in self.method:
                    action, predicted_traj, cost_val, ctrl_info = self.controller.compute_control(
                        current_state=measured_state,
                        feature_history=feature_history,
                        goal=active_goal_obj,
                        constraint_manager=self.constraint_mgr,
                        path_target=path_target
                    )
                    solve_time = ctrl_info["solve_time_sec"]
                    breakdown = ctrl_info["cost_breakdown"]
                    solve_times.append(solve_time)
                elif self.method == "reactive_los":
                    action, ctrl_info = self.controller.compute_control(measured_state, active_goal_obj, path_target=path_target)
                else:
                    action, ctrl_info = self.controller.compute_control(measured_state, active_goal_obj, self.constraint_mgr, path_target=path_target)

            # Check slew saturation
            dp = abs(action.pitch_cmd - cur_true_state.pitch)
            dh = abs(wrap_angle(action.heading_cmd - cur_true_state.heading))
            du = abs(action.speed_cmd - cur_true_state.speed)
            if dp >= 0.099 or dh >= 0.199 or du >= 0.149:
                slew_saturated_steps += 1

            # 4. True Plant Integration Step (2.0s via RK4 continuous substepping)
            next_true_state, plant_info = self.plant.step(action)

            # Accumulate path length
            curr_pos = next_true_state.to_array()[:3]
            step_dist = float(np.linalg.norm(curr_pos - prev_pos))
            total_path_length += step_dist
            if current_phase == "FORWARD":
                forward_path_length += step_dist
            else:
                return_path_length += step_dist

            # Continuous step segment distance checking
            seg = curr_pos - prev_pos
            seg_len_sq = float(np.sum(seg**2))
            if seg_len_sq > 1e-6:
                t_proj_b = float(np.clip(np.dot(self.goal_pos - prev_pos, seg) / seg_len_sq, 0.0, 1.0))
                closest_step_b = float(np.linalg.norm(prev_pos + t_proj_b * seg - self.goal_pos))
                t_proj_a = float(np.clip(np.dot(self.start_pos - prev_pos, seg) / seg_len_sq, 0.0, 1.0))
                closest_step_a = float(np.linalg.norm(prev_pos + t_proj_a * seg - self.start_pos))
            else:
                closest_step_b = float(np.linalg.norm(curr_pos - self.goal_pos))
                closest_step_a = float(np.linalg.norm(curr_pos - self.start_pos))

            if closest_step_b < closest_distance_b:
                closest_distance_b = closest_step_b
            if current_phase in ["RETURNING", "ROTATING_AT_A", "COMPLETED"]:
                if closest_step_a < closest_distance_a:
                    closest_distance_a = closest_step_a

            # Transition on segment contact with Point B
            if current_phase == "FORWARD" and (closest_step_b <= self.tol_primary or true_dist_to_b <= self.tol_primary):
                if not reached_b:
                    reached_b = True
                    step_reached_b = step
                    time_reached_b = t_sim
                if self.return_to_start:
                    current_phase = "ROTATING_BACK"
                    active_target_pos = self.start_pos.copy()
                    dx_back = self.start_pos[0] - next_true_state.north
                    dy_back = self.start_pos[1] - next_true_state.east
                    target_heading_back = float(wrap_to_2pi(np.arctan2(dy_back, dx_back)))
                    self.plant.params.max_heading_rate_sec = 0.35
                    rotation_steps_b = 0
                    closest_distance_a = float(np.linalg.norm(next_true_state.to_array()[:3] - self.start_pos))
                else:
                    current_phase = "COMPLETED"

            # Transition on segment contact with Point A
            elif current_phase == "RETURNING" and (closest_step_a <= self.tol_primary or true_dist_to_a <= self.tol_primary):
                if not reached_a:
                    reached_a = True
                    step_reached_a = step
                    time_reached_a = t_sim
                current_phase = "ROTATING_AT_A"
                self.plant.params.max_heading_rate_sec = 0.35
                rotation_steps_a = 0

            prev_pos = curr_pos

            # Boundary violation check on true state
            is_in_bounds = self.boundaries.is_inside(next_true_state)
            b_dist = self.boundaries.violation_distance(next_true_state)

            # Log row
            row = {
                "step": step,
                "time_sec": t_sim,
                "phase": current_phase,
                "leg": 1 if (current_phase == "FORWARD" or not self.return_to_start) else 2,
                # True State
                "true_north": cur_true_state.north,
                "true_east": cur_true_state.east,
                "true_depth": cur_true_state.depth,
                "true_roll": cur_true_state.roll,
                "true_pitch": cur_true_state.pitch,
                "true_heading": cur_true_state.heading,
                "true_speed": cur_true_state.speed,
                # Measured State
                "meas_north": measured_state.north,
                "meas_east": measured_state.east,
                "meas_depth": measured_state.depth,
                "meas_heading": measured_state.heading,
                "meas_speed": measured_state.speed,
                # Commands
                "cmd_speed": action.speed_cmd,
                "cmd_pitch": action.pitch_cmd,
                "cmd_heading": action.heading_cmd,
                # Targets & Distances
                "target_north": float(active_target_pos[0]),
                "target_east": float(active_target_pos[1]),
                "target_depth": float(active_target_pos[2]),
                "dist_to_target": true_dist_to_active,
                "dist_to_goal": true_dist_to_active,
                "dist_to_b": true_dist_to_b,
                "dist_to_a": true_dist_to_a,
                "reached_b": reached_b,
                "reached_a": reached_a,
                # True Plant Diagnostics
                "min_obstacle_dist": plant_info["min_obstacle_dist"],
                "collision": plant_info["step_collision"],
                "buffer_violation": plant_info["step_buffer_violation"],
                "boundary_violation_m": b_dist,
                "is_in_bounds": is_in_bounds,
                "ood_flag": meas_info["ood"]["is_ood"],
                "solve_time_sec": solve_time,
                "predicted_traj": predicted_traj[:, :3].tolist() if predicted_traj is not None else None
            }
            for k_cost, v_cost in breakdown.items():
                row[f"cost_{k_cost}"] = v_cost

            log_rows.append(row)
            cur_true_state = next_true_state

            # Termination condition
            if current_phase == "COMPLETED" or (not self.return_to_start and reached_b):
                break

        # Append terminal state at t_end
        term_dist_b = float(np.linalg.norm(cur_true_state.to_array()[:3] - self.goal_pos))
        term_dist_a = float(np.linalg.norm(cur_true_state.to_array()[:3] - self.start_pos))
        final_dist = term_dist_a if self.return_to_start else term_dist_b
        term_b_dist = self.boundaries.violation_distance(cur_true_state)
        term_in_bounds = self.boundaries.is_inside(cur_true_state)
        term_phase = "COMPLETED" if (reached_a if self.return_to_start else reached_b) else current_phase

        term_row = {
            "step": total_steps,
            "time_sec": total_steps * self.dt_control,
            "phase": term_phase,
            "leg": 2 if (self.return_to_start and (reached_b or current_phase in ["RETURNING", "ROTATING_BACK", "ROTATING_AT_A"])) else 1,
            "true_north": cur_true_state.north,
            "true_east": cur_true_state.east,
            "true_depth": cur_true_state.depth,
            "true_roll": cur_true_state.roll,
            "true_pitch": cur_true_state.pitch,
            "true_heading": cur_true_state.heading,
            "true_speed": cur_true_state.speed,
            "meas_north": cur_true_state.north,
            "meas_east": cur_true_state.east,
            "meas_depth": cur_true_state.depth,
            "meas_heading": cur_true_state.heading,
            "meas_speed": cur_true_state.speed,
            "cmd_speed": np.nan,
            "cmd_pitch": np.nan,
            "cmd_heading": np.nan,
            "target_north": float(active_target_pos[0]),
            "target_east": float(active_target_pos[1]),
            "target_depth": float(active_target_pos[2]),
            "dist_to_target": final_dist,
            "dist_to_goal": final_dist,
            "dist_to_b": term_dist_b,
            "dist_to_a": term_dist_a,
            "reached_b": reached_b,
            "reached_a": reached_a,
            "min_obstacle_dist": float(self.plant.min_obstacle_dist),
            "collision": False,
            "buffer_violation": False,
            "boundary_violation_m": term_b_dist,
            "is_in_bounds": term_in_bounds,
            "ood_flag": False,
            "solve_time_sec": 0.0,
            "predicted_traj": None
        }
        for k_cost in ["j_goal", "j_tracking", "j_control", "j_smooth", "j_constraint", "total_cost"]:
            term_row[f"cost_{k_cost}"] = 0.0
        log_rows.append(term_row)

        df_traj = pd.DataFrame(log_rows)

        # 5. Final Comprehensive Metrics
        expected_straight_line = 2.0 * straight_line_dist if self.return_to_start else straight_line_dist
        path_efficiency = (expected_straight_line / total_path_length * 100.0) if total_path_length > 0 else 0.0

        total_bound_viols = int(np.sum(df_traj["boundary_violation_m"] > 0.001))
        total_collisions = int(self.plant.collision_count)
        
        reached_b_verified = bool(reached_b or term_dist_b <= self.tol_primary or closest_distance_b <= self.tol_primary)
        reached_a_verified = bool(reached_a or (term_phase in ["ROTATING_AT_A", "COMPLETED"] and (term_dist_a <= self.tol_primary or closest_distance_a <= self.tol_primary))) if self.return_to_start else False

        mission_target_reached = reached_a_verified if self.return_to_start else reached_b_verified
        is_clean_success = bool(mission_target_reached and total_bound_viols == 0 and total_collisions == 0)

        metrics = {
            "method": self.method,
            "seed": self.seed,
            "scenario": self.scenario.get("name", "unnamed"),
            "return_to_start": bool(self.return_to_start),
            "point_a": self.start_pos.tolist(),
            "point_b": self.goal_pos.tolist(),
            "success_5m": is_clean_success,
            "success_2m": bool(is_clean_success and ((closest_distance_a <= self.tol_tight) if self.return_to_start else (closest_distance_b <= self.tol_tight))),
            "target_reached": mission_target_reached,
            "reached_b": reached_b_verified,
            "step_reached_b": step_reached_b,
            "time_reached_b_sec": float(time_reached_b) if time_reached_b is not None else None,
            "reached_a": reached_a_verified if self.return_to_start else None,
            "step_reached_a": step_reached_a,
            "time_reached_a_sec": float(time_reached_a) if time_reached_a is not None else None,
            "movement_phase": term_phase,
            "time_to_goal_sec": float((time_reached_a if self.return_to_start else time_reached_b) if (time_reached_a if self.return_to_start else time_reached_b) is not None else total_steps * self.dt_control),
            "steps_taken": total_steps,
            "final_distance_m": float(min(final_dist, closest_distance) if (mission_target_reached or closest_distance <= self.tol_primary) else final_dist),
            "final_distance_to_b_m": float(min(term_dist_b, closest_distance_b) if (reached_b_verified or closest_distance_b <= self.tol_primary) else term_dist_b),
            "final_distance_to_a_m": float(min(term_dist_a, closest_distance_a) if (reached_a_verified or closest_distance_a <= self.tol_primary) else term_dist_a),
            "closest_distance_m": float(closest_distance),
            "straight_line_dist_m": straight_line_dist,
            "forward_path_length_m": float(forward_path_length),
            "return_path_length_m": float(return_path_length),
            "total_path_length_m": float(total_path_length),
            "path_efficiency_pct": float(path_efficiency),
            "physical_collisions": int(self.plant.collision_count),
            "buffer_violations": int(self.plant.buffer_violation_count),
            "min_obstacle_distance_m": float(self.plant.min_obstacle_dist),
            "min_buffer_clearance_m": float(self.plant.min_buffer_dist),
            "boundary_violations": int(np.sum(df_traj["boundary_violation_m"] > 0)),
            "max_boundary_violation_m": float(np.max(df_traj["boundary_violation_m"])) if len(df_traj) > 0 else 0.0,
            "ood_step_fraction": float(ood_steps / total_steps) if total_steps > 0 else 0.0,
            "slew_saturation_fraction": float(slew_saturated_steps / total_steps) if total_steps > 0 else 0.0,
            "solve_time_mean_sec": float(np.mean(solve_times)) if len(solve_times) > 0 else 0.0,
            "solve_time_p95_sec": float(np.percentile(solve_times, 95)) if len(solve_times) > 0 else 0.0,
            "solve_time_max_sec": float(np.max(solve_times)) if len(solve_times) > 0 else 0.0,
            "planned_path_length_m": float(np.sum(np.linalg.norm(np.diff(self.global_path, axis=0), axis=1))) if len(self.global_path) > 1 else straight_line_dist,
            "global_path": self.global_path.tolist(),
            "global_path_return": self.return_path.tolist() if self.return_path is not None else None
        }

        return df_traj, metrics
