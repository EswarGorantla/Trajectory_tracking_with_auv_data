import time
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from src.state import AUVState, AUVControl, wrap_angle, wrap_to_2pi
from src.goal import Goal
from src.constraints import ConstraintManager, Boundaries3D
from src.cost_function import MPCCostFunction
from src.dynamics_interface import DynamicsInterface

class MPCController:
    """
    Receding-Horizon Model Predictive Controller (MPC) for 3D AUV Navigation.
    Utilizes the Phase 1 Physics + GRU Residual Dynamics Model for multi-step predictions.
    
    Optimization Strategy:
    Cross-Entropy Method (CEM) with Warm-Starting and Nominal LOS Guidance Seeding.
    Guarantees strict return of the lowest-cost candidate sequence across iterations.
    """

    def __init__(
        self,
        dynamics_interface: DynamicsInterface,
        cost_function: Optional[MPCCostFunction] = None,
        prediction_horizon: int = 10,
        control_horizon: int = 10,
        num_candidates: int = 70,
        elite_count: int = 10,
        iterations: int = 3,
        use_ai: bool = True,
        use_warm_start: bool = True,
        use_los_seed: bool = True,
        seed: Optional[int] = None,
        elite_fraction: Optional[float] = None,
        **kwargs
    ):
        self.dynamics = dynamics_interface
        self.cost_func = cost_function or MPCCostFunction()
        self.H = prediction_horizon
        self.K = num_candidates
        if elite_fraction is not None:
            elite_count = max(2, int(np.round(num_candidates * elite_fraction)))
        self.elite_count = max(2, min(elite_count, num_candidates // 2))
        self.iterations = max(1, iterations)
        self.use_ai = use_ai
        self.use_warm_start = use_warm_start
        self.use_los_seed = use_los_seed
        self.rng = np.random.default_rng(seed)

        # Warm-start memory: previous optimal sequence of shape (H, 3)
        self.prev_optimal_controls: Optional[np.ndarray] = None
        self.prev_applied_control: np.ndarray = np.array([self.cost_func.u_cruise, 0.0, 0.0], dtype=np.float64)

    def reset(self, initial_state: Optional[AUVState] = None):
        self.prev_optimal_controls = None
        if initial_state:
            self.prev_applied_control = np.array([initial_state.speed, initial_state.pitch, initial_state.heading], dtype=np.float64)
        else:
            self.prev_applied_control = np.array([self.cost_func.u_cruise, 0.0, 0.0], dtype=np.float64)

    def _clip_slew_limits(self, candidate_seq: np.ndarray, init_state: AUVState) -> np.ndarray:
        """
        Clips candidate control actions to enforce maximum actuator slew rates.
        Shape: (H, 3) [Speed, Pitch, Heading]
        """
        clipped = candidate_seq.copy()
        max_dp = self.dynamics.max_pitch_rate
        max_dh = self.dynamics.max_heading_rate
        max_du = self.dynamics.max_speed_rate
        
        # Step 0 relative to initial state
        prev_u = init_state.speed
        prev_p = init_state.pitch
        prev_h = init_state.heading
        
        for k in range(self.H):
            # Speed clipping
            clipped[k, 0] = prev_u + np.clip(clipped[k, 0] - prev_u, -max_du, max_du)
            clipped[k, 0] = np.clip(clipped[k, 0], self.dynamics.min_speed, self.dynamics.max_speed)
            prev_u = clipped[k, 0]
            
            # Pitch clipping
            clipped[k, 1] = prev_p + np.clip(clipped[k, 1] - prev_p, -max_dp, max_dp)
            clipped[k, 1] = np.clip(clipped[k, 1], -0.35, 0.35)
            prev_p = clipped[k, 1]
            
            # Heading clipping
            h_err = wrap_angle(clipped[k, 2] - prev_h)
            clipped[k, 2] = wrap_to_2pi(prev_h + np.clip(h_err, -max_dh, max_dh))
            prev_h = clipped[k, 2]
            
        return clipped

    def compute_control(
        self,
        current_state: AUVState,
        feature_history: np.ndarray,
        goal: Goal,
        constraint_manager: ConstraintManager,
        path_target: Optional[Any] = None
    ) -> Tuple[AUVControl, np.ndarray, float, Dict[str, Any]]:
        """
        Executes one receding-horizon MPC cycle with omnidirectional geofence containment,
        predictive obstacle bypass, and pre-mission global path guidance.
        """
        t_start = time.perf_counter()
        H = self.H
        K = self.K

        if path_target is not None:
            los_heading = path_target.target_heading
            los_pitch = path_target.target_pitch
            dist_to_goal = float(path_target.dist_remaining)
        else:
            los_heading = goal.los_heading_to(current_state)
            los_pitch = goal.los_pitch_to(current_state)
            dist_to_goal = float(np.linalg.norm(current_state.to_array()[:3] - goal.to_array()))

        cur_n, cur_e, cur_d = current_state.north, current_state.east, current_state.depth
        b = constraint_manager.boundaries

        d_n_max = b.north_max - cur_n
        d_n_min = cur_n - b.north_min
        d_e_max = b.east_max - cur_e
        d_e_min = cur_e - b.east_min
        d_d_max = b.depth_max - cur_d
        d_d_min = cur_d - b.depth_min
        min_wall_dist = min(d_n_max, d_n_min, d_e_max, d_e_min, d_d_max, d_d_min)

        # 1. Proactive Obstacle Scan
        min_d_obs = 999.0
        nearby_obstacles = []
        for obs in constraint_manager.obstacles:
            dn = obs.north - cur_n
            de = obs.east - cur_e
            d_obs = float(np.hypot(dn, de))
            if d_obs < min_d_obs:
                min_d_obs = d_obs
            depth_overlap = (cur_d >= obs.depth_min - 4.0) and (cur_d <= obs.depth_max + 4.0)
            if d_obs < 45.0 and depth_overlap:
                nearby_obstacles.append((obs, d_obs, dn, de))

        # Dynamic Speed Adaptation: Slow down when turning tightly or approaching walls/hazards
        speed_proximity = min(min_wall_dist / 12.0, min_d_obs / 16.0)
        speed_proximity = float(np.clip(speed_proximity, 0.0, 1.0))
        target_speed = float(np.clip(
            self.dynamics.min_speed + speed_proximity * (self.cost_func.u_cruise - self.dynamics.min_speed),
            self.dynamics.min_speed,
            self.cost_func.u_cruise
        ))
        if dist_to_goal < 15.0:
            target_speed = min(target_speed, float(np.clip(dist_to_goal / 15.0 * self.cost_func.u_cruise, self.dynamics.min_speed, self.cost_func.u_cruise)))

        nominal_speed = target_speed

        def check_heading_boundary_clearance(hdg, ptch, spd, steps=H, margin=2.5):
            for st in range(1, steps + 1):
                dist_t = st * 2.0 * spd
                np_t = cur_n + dist_t * np.cos(ptch) * np.cos(hdg)
                ep_t = cur_e + dist_t * np.cos(ptch) * np.sin(hdg)
                dp_t = cur_d - dist_t * np.sin(ptch)
                if (np_t < b.north_min + margin or np_t > b.north_max - margin or
                    ep_t < b.east_min + margin or ep_t > b.east_max - margin or
                    dp_t < b.depth_min + margin or dp_t > b.depth_max - margin):
                    return False
            return True

        avoidance_seeds = []

        # 2. Obstacle Evasion Seeds with Geofence Predictive Containment
        for obs, d_obs, dn, de in nearby_obstacles:
            obs_bearing = np.arctan2(de, dn) % (2 * np.pi)
            rel_bearing = abs(wrap_angle(obs_bearing - current_state.heading))
            los_obs_rel = abs(wrap_angle(obs_bearing - los_heading))

            if rel_bearing < (np.pi / 2.0) or los_obs_rel < 0.85:
                r_safe = obs.safe_radius + 3.0
                sin_theta = min(0.95, r_safe / max(d_obs, r_safe + 0.1))
                theta_clear = float(np.arcsin(sin_theta)) + 0.15

                hdg_left = wrap_to_2pi(obs_bearing - theta_clear)
                hdg_right = wrap_to_2pi(obs_bearing + theta_clear)

                port_safe = check_heading_boundary_clearance(hdg_left, los_pitch, nominal_speed)
                stbd_safe = check_heading_boundary_clearance(hdg_right, los_pitch, nominal_speed)

                if port_safe:
                    cand_p = np.zeros((H, 3), dtype=np.float64)
                    cand_p[:, 0] = nominal_speed
                    cand_p[:, 1] = los_pitch
                    cand_p[:, 2] = hdg_left
                    avoidance_seeds.append(cand_p)

                if stbd_safe:
                    cand_s = np.zeros((H, 3), dtype=np.float64)
                    cand_s[:, 0] = nominal_speed
                    cand_s[:, 1] = los_pitch
                    cand_s[:, 2] = hdg_right
                    avoidance_seeds.append(cand_s)

                if cur_d > (b.depth_min + 8.0) and obs.depth_min > 5.0:
                    cand_over = np.zeros((H, 3), dtype=np.float64)
                    cand_over[:, 0] = nominal_speed
                    cand_over[:, 1] = 0.28
                    cand_over[:, 2] = los_heading
                    avoidance_seeds.append(cand_over)
                if cur_d < (b.depth_max - 8.0) and obs.depth_max < 75.0:
                    cand_under = np.zeros((H, 3), dtype=np.float64)
                    cand_under[:, 0] = nominal_speed
                    cand_under[:, 1] = -0.28
                    cand_under[:, 2] = los_heading
                    avoidance_seeds.append(cand_under)

        # 3. Omnidirectional Boundary Standoff Repulsion Seeds
        # When on final approach to goal (within 12m), suppress wall repulsion so vehicle can dock at goal
        if dist_to_goal >= 12.0:
            buf_thresh = 14.0
            center_n = 0.5 * (b.north_min + b.north_max)
            center_e = 0.5 * (b.east_min + b.east_max)

            if min_wall_dist < buf_thresh:
                inward_hdg = np.arctan2(center_e - cur_e, center_n - cur_n) % (2 * np.pi)
                cand_in = np.zeros((H, 3), dtype=np.float64)
                cand_in[:, 0] = self.dynamics.min_speed
                cand_in[:, 1] = los_pitch
                cand_in[:, 2] = inward_hdg
                avoidance_seeds.append(cand_in)

            if d_n_max < buf_thresh and abs(wrap_angle(los_heading - 0.0)) < 0.5 * np.pi:
                cand_s = np.zeros((H, 3), dtype=np.float64)
                cand_s[:, 0] = self.dynamics.min_speed
                cand_s[:, 1] = los_pitch
                cand_s[:, 2] = np.pi if cur_e > center_e else 0.75 * np.pi
                avoidance_seeds.append(cand_s)

            if d_n_min < buf_thresh and abs(wrap_angle(los_heading - np.pi)) < 0.5 * np.pi:
                cand_n = np.zeros((H, 3), dtype=np.float64)
                cand_n[:, 0] = self.dynamics.min_speed
                cand_n[:, 1] = los_pitch
                cand_n[:, 2] = 0.0 if cur_e > center_e else 0.25 * np.pi
                avoidance_seeds.append(cand_n)

            if d_e_max < buf_thresh and abs(wrap_angle(los_heading - 0.5 * np.pi)) < 0.5 * np.pi:
                cand_w = np.zeros((H, 3), dtype=np.float64)
                cand_w[:, 0] = self.dynamics.min_speed
                cand_w[:, 1] = los_pitch
                cand_w[:, 2] = 1.5 * np.pi
                avoidance_seeds.append(cand_w)

            if d_e_min < buf_thresh and abs(wrap_angle(los_heading - 1.5 * np.pi)) < 0.5 * np.pi:
                cand_e = np.zeros((H, 3), dtype=np.float64)
                cand_e[:, 0] = self.dynamics.min_speed
                cand_e[:, 1] = los_pitch
                cand_e[:, 2] = 0.5 * np.pi
                avoidance_seeds.append(cand_e)

        # Initialize distribution mean
        if self.use_warm_start and self.prev_optimal_controls is not None and len(self.prev_optimal_controls) == H:
            mean_controls = np.vstack([self.prev_optimal_controls[1:], self.prev_optimal_controls[-1:]])
        else:
            mean_controls = np.zeros((H, 3), dtype=np.float64)
            mean_controls[:, 0] = nominal_speed
            mean_controls[:, 1] = los_pitch
            mean_controls[:, 2] = los_heading

        std_controls = np.array([0.12, 0.05, 0.25], dtype=np.float64)

        best_cost = float('inf')
        best_controls = mean_controls.copy()
        best_trajectory = None
        best_breakdown = {}

        for it in range(self.iterations):
            noise = self.rng.standard_normal((K, H, 3)) * std_controls
            candidates = mean_controls[np.newaxis, :, :] + noise

            if self.use_warm_start:
                candidates[0] = mean_controls.copy()
            if self.use_los_seed and K > 1:
                candidates[1, :, 0] = nominal_speed
                candidates[1, :, 1] = los_pitch
                candidates[1, :, 2] = los_heading

            seed_idx = 2
            for av_seed in avoidance_seeds:
                if seed_idx < K:
                    candidates[seed_idx] = av_seed.copy()
                    seed_idx += 1

            for i in range(K):
                candidates[i] = self._clip_slew_limits(candidates[i], current_state)

            trajectories, _ = self.dynamics.predict_rollout_batch(
                initial_state=current_state,
                initial_history=feature_history,
                candidate_controls=candidates,
                use_ai=self.use_ai
            )

            costs, breakdown = self.cost_func.evaluate_batch(
                trajectories=trajectories,
                controls=candidates,
                prev_control=self.prev_applied_control,
                goal=goal,
                constraint_manager=constraint_manager,
                return_breakdown=True,
                target_speed=target_speed,
                path_target=path_target
            )

            # Strict Boundary Containment Prioritization
            bound_pens = constraint_manager.boundaries.evaluate_batch_penetrations(trajectories[..., :3])
            tot_bound = np.sum(bound_pens, axis=1)
            safe_mask = tot_bound < 1e8

            if np.any(safe_mask):
                safe_indices = np.where(safe_mask)[0]
                min_safe_idx = safe_indices[np.argmin(costs[safe_indices])]
                min_idx = min_safe_idx
                elite_indices = safe_indices[np.argsort(costs[safe_indices])][:self.elite_count]
            else:
                min_idx = int(np.argmin(costs))
                elite_indices = np.argsort(costs)[:self.elite_count]

            if costs[min_idx] < best_cost:
                best_cost = float(costs[min_idx])
                best_controls = candidates[min_idx].copy()
                best_trajectory = trajectories[min_idx].copy()
                best_breakdown = {k: float(v[min_idx]) for k, v in breakdown.items()}

            elite_controls = candidates[elite_indices]
            alpha = 0.70
            new_mean_u = np.mean(elite_controls[:, :, 0], axis=0)
            new_mean_p = np.mean(elite_controls[:, :, 1], axis=0)
            sin_h = np.mean(np.sin(elite_controls[:, :, 2]), axis=0)
            cos_h = np.mean(np.cos(elite_controls[:, :, 2]), axis=0)
            new_mean_h = np.arctan2(sin_h, cos_h) % (2 * np.pi)

            mean_controls[:, 0] = alpha * new_mean_u + (1.0 - alpha) * mean_controls[:, 0]
            mean_controls[:, 1] = alpha * new_mean_p + (1.0 - alpha) * mean_controls[:, 1]
            h_diff = wrap_angle(new_mean_h - mean_controls[:, 2])
            mean_controls[:, 2] = wrap_to_2pi(mean_controls[:, 2] + alpha * h_diff)

            std_u = np.std(elite_controls[:, :, 0], axis=0) * 0.70 + 0.02
            std_p = np.std(elite_controls[:, :, 1], axis=0) * 0.70 + 0.02
            h_err = wrap_angle(elite_controls[:, :, 2] - new_mean_h[np.newaxis, :])
            std_h = np.std(h_err, axis=0) * 0.70 + 0.02
            std_controls = np.stack([std_u, std_p, std_h], axis=-1)

        self.prev_optimal_controls = best_controls.copy()
        u_0 = best_controls[0]
        self.prev_applied_control = u_0.copy()

        action = AUVControl(speed_cmd=float(u_0[0]), pitch_cmd=float(u_0[1]), heading_cmd=float(u_0[2]))
        solve_time = time.perf_counter() - t_start
        info = {
            "solve_time_sec": solve_time,
            "cost_breakdown": best_breakdown,
            "predicted_horizon": best_trajectory.tolist() if best_trajectory is not None else []
        }
        return action, best_trajectory, best_cost, info
