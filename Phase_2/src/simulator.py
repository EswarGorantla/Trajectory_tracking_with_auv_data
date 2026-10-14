import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.state import AUVState, AUVControl, wrap_to_2pi
from src.goal import Goal
from src.constraints import ConstraintManager
from src.dynamics_interface import DynamicsInterface
from src.mpc_controller import MPCController

class AUVSimulator:
    """
    Closed-Loop Simulation Environment for Goal-Directed AUV Control.
    Integrates Receding-Horizon MPC with the True Plant (Physics + Phase 1 GRU Residuals).
    """

    def __init__(
        self,
        dynamics: DynamicsInterface,
        controller: MPCController,
        start_state: AUVState,
        goal: Goal,
        constraint_manager: ConstraintManager,
        dt: float = 2.0,
        current_disturbance: Optional[np.ndarray] = None, # [v_north, v_east, v_depth] in m/s
        measurement_noise_std: float = 0.0,               # Position Gaussian noise in meters
        global_planner: Optional[Any] = None
    ):
        self.dynamics = dynamics
        self.controller = controller
        self.start_state = start_state
        self.current_state = start_state
        self.goal = goal
        self.constraint_manager = constraint_manager
        self.dt = dt
        self.current_disturbance = current_disturbance if current_disturbance is not None else np.zeros(3)
        self.measurement_noise_std = measurement_noise_std
        self.global_planner = global_planner

        # Auto-initialize global planner if not provided to guarantee path planning support
        if self.global_planner is None:
            try:
                from src.global_planner import GlobalMapPlanner
                self.global_planner = GlobalMapPlanner()
                self.global_planner.plan(
                    start=self.start_state.to_array()[:3],
                    goal=self.goal.to_array(),
                    boundaries=self.constraint_manager.boundaries,
                    obstacles=self.constraint_manager.obstacles
                )
            except Exception:
                self.global_planner = None

        # Initialize 10-step history with starting state features
        init_feat = self.start_state.to_feature_vector()
        self.history_features = np.tile(init_feat, (10, 1))

        # Telemetry logs
        self.records: List[Dict[str, Any]] = []
        self.step_idx = 0
        self.time_elapsed = 0.0

    def run(self, max_steps: int = 120, verbose: bool = True) -> pd.DataFrame:
        """
        Executes the closed-loop MPC simulation until Goal B is reached
        or maximum simulation steps are exhausted.
        """
        if verbose:
            print(f"--- Starting Closed-Loop AUV Simulation ---")
            print(f"Start A: North={self.start_state.north:.1f}m, East={self.start_state.east:.1f}m, Depth={self.start_state.depth:.1f}m")
            print(f"Goal B:  North={self.goal.north:.1f}m, East={self.goal.east:.1f}m, Depth={self.goal.depth:.1f}m (Tol={self.goal.tolerance}m)")
            print(f"Receding Horizon: {self.controller.H} steps ({self.controller.H * self.dt:.1f}s) | Step dt: {self.dt}s\n")

        for step in range(max_steps):
            self.step_idx = step
            self.time_elapsed = step * self.dt

            # 1. State measurement (with optional sensor noise)
            measured_state = self.current_state
            if self.measurement_noise_std > 0:
                noisy_arr = self.current_state.to_array()
                noisy_arr[:3] += np.random.normal(0, self.measurement_noise_std, 3)
                measured_state = AUVState.from_array(noisy_arr)

            # 2. Check Goal Reached
            dist_to_goal = self.goal.distance_to(self.current_state)
            goal_reached = dist_to_goal <= self.goal.tolerance

            # 3. Audit Constraints
            constraint_audit = self.constraint_manager.check_state(self.current_state)

            # 4. Compute MPC Control Action (Receding Horizon Optimization)
            path_target = None
            if self.global_planner is not None:
                try:
                    tgt_pos, ref_heading, ref_pitch, dist_rem = self.global_planner.get_lookahead_target(
                        self.current_state.position(),
                        lookahead_dist=16.0
                    )
                    from src.goal import PathTarget
                    path_target = PathTarget(
                        target_pos=tgt_pos,
                        target_heading=ref_heading,
                        target_pitch=ref_pitch,
                        dist_remaining=dist_rem,
                        final_goal=self.goal
                    )
                except Exception:
                    path_target = None

            ctrl_res = self.controller.compute_control(
                current_state=measured_state,
                feature_history=self.history_features,
                goal=self.goal,
                constraint_manager=self.constraint_manager,
                path_target=path_target
            )
            if len(ctrl_res) == 4:
                action, opt_trajectory, cost, _ = ctrl_res
            else:
                action, opt_trajectory, cost = ctrl_res

            # Record step telemetry
            record = {
                "step": step,
                "timestamp": self.time_elapsed,
                "north": self.current_state.north,
                "east": self.current_state.east,
                "depth": self.current_state.depth,
                "roll": self.current_state.roll,
                "pitch": self.current_state.pitch,
                "heading": self.current_state.heading,
                "speed": self.current_state.speed,
                "speed_cmd": action.speed_cmd,
                "pitch_cmd": action.pitch_cmd,
                "heading_cmd": action.heading_cmd,
                "distance_to_goal": dist_to_goal,
                "goal_reached": goal_reached,
                "in_boundaries": constraint_audit["in_boundaries"],
                "obstacle_collision": constraint_audit["obstacle_collision"],
                "constraint_status": "OK" if constraint_audit["all_satisfied"] else "VIOLATION",
                "mpc_cost": cost
            }
            self.records.append(record)

            if verbose and (step % 5 == 0 or goal_reached):
                print(f"Step {step:02d} ({self.time_elapsed:4.1f}s) | Pos: ({self.current_state.north:5.1f}, {self.current_state.east:5.1f}, {self.current_state.depth:4.1f})m | "
                      f"Dist to Goal: {dist_to_goal:5.2f}m | Heading: {np.degrees(self.current_state.heading):5.1f}° | Speed: {self.current_state.speed:.2f}m/s")

            if goal_reached:
                if verbose:
                    print(f"\nSUCCESS: AUV reached Goal B within tolerance ({dist_to_goal:.2f}m <= {self.goal.tolerance:.1f}m) at t = {self.time_elapsed:.1f}s!\n")
                break

            # 5. Advance True Plant Dynamics (Physics + Phase 1 GRU Residuals)
            next_state, next_history = self.dynamics.predict_next_state(
                current_state=self.current_state,
                control=action,
                history_features=self.history_features,
                use_ai=True
            )

            # Apply Environmental Ocean Current Disturbance (drift over step dt)
            if np.any(self.current_disturbance != 0):
                next_arr = next_state.to_array()
                next_arr[0] += self.current_disturbance[0] * self.dt
                next_arr[1] += self.current_disturbance[1] * self.dt
                next_arr[2] += self.current_disturbance[2] * self.dt
                next_state = AUVState.from_array(next_arr)

            self.current_state = next_state
            self.history_features = next_history

        df_sim = pd.DataFrame(self.records)
        return df_sim
