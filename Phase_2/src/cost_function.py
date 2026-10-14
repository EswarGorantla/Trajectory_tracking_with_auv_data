import numpy as np
from typing import Dict, Any, Union, Tuple, Optional
from src.goal import Goal
from src.constraints import ConstraintManager
from src.state import wrap_angle

class MPCCostFunction:
    """
    Multi-Objective Cost Function for Receding-Horizon Goal-Directed AUV MPC:
    J = w_goal * J_goal + w_tracking * J_tracking + w_control * J_control
        + w_smooth * J_smooth + w_constraint * J_constraint
    
    Includes individual sub-cost decomposition and documented control weight matrix R.
    """

    def __init__(
        self,
        w_goal: float = 30.0,
        w_tracking: float = 6.0,
        w_control: float = 1.0,
        w_smooth: float = 4.0,
        w_constraint: float = 200.0,
        u_cruise: float = 1.2,
        r_smooth_weights: Optional[np.ndarray] = None
    ):
        self.w_goal = w_goal
        self.w_tracking = w_tracking
        self.w_control = w_control
        self.w_smooth = w_smooth
        self.w_constraint = w_constraint
        self.u_cruise = u_cruise
        # Diagonal R-weights for control rate penalty: [Speed, Pitch, Heading]
        self.r_smooth = np.array(r_smooth_weights if r_smooth_weights is not None else [1.0, 4.0, 4.0], dtype=np.float64)

    def evaluate_batch(
        self,
        trajectories: np.ndarray,      # (K, H+1, 7) [North, East, Depth, Roll, Pitch, Heading, Speed]
        controls: np.ndarray,          # (K, H, 3) [Speed, Pitch, Heading]
        prev_control: np.ndarray,      # (3,)
        goal: Goal,
        constraint_manager: ConstraintManager,
        return_breakdown: bool = False,
        target_speed: Optional[float] = None,
        path_target: Optional[Any] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Computes the scalar cost for each of the K candidate trajectories in parallel.
        Supports dynamic lookahead target along pre-mission global path.
        Returns:
            total_cost: array of shape (K,)
            breakdown (optional): dictionary containing individual sub-costs (each shape (K,))
        """
        K, H_plus_1, _ = trajectories.shape
        H = controls.shape[1]
        goal_pos = path_target.target_pos if path_target is not None else goal.to_array()
        u_ref = target_speed if target_speed is not None else self.u_cruise

        positions = trajectories[:, :, :3]  # (K, H+1, 3)

        # 1. Goal / Waypoint Distance Cost
        if path_target is not None and getattr(path_target, "horizon_waypoints", None) is not None:
            ref_pts = path_target.horizon_waypoints  # (H, 3)
            if ref_pts.shape[0] == H:
                horizon_dists = np.linalg.norm(positions[:, 1:, :] - ref_pts[np.newaxis, :, :], axis=2)  # (K, H)
                terminal_dist = horizon_dists[:, -1]
                min_horizon_dist = np.min(horizon_dists, axis=1)
                j_goal = (terminal_dist**2) * 1.5 + (min_horizon_dist**2) * 2.0 + np.mean(horizon_dists**2, axis=1) * 2.0
            else:
                all_dists = np.linalg.norm(positions[:, 1:, :] - goal_pos, axis=2)  # (K, H)
                min_horizon_dist = np.min(all_dists, axis=1)  # (K,)
                terminal_dist = all_dists[:, -1]             # (K,)
                j_goal = (terminal_dist**2) * 1.5 + (min_horizon_dist**2) * 3.0 + np.mean(all_dists, axis=1) * 1.5
        else:
            all_dists = np.linalg.norm(positions[:, 1:, :] - goal_pos, axis=2)  # (K, H)
            min_horizon_dist = np.min(all_dists, axis=1)  # (K,)
            terminal_dist = all_dists[:, -1]             # (K,)
            j_goal = (terminal_dist**2) * 1.5 + (min_horizon_dist**2) * 3.0 + np.mean(all_dists, axis=1) * 1.5

        # 2. Path Tracking Alignment (LOS heading & pitch alignment)
        headings = trajectories[:, 1:, 5]  # (K, H)
        pitches = trajectories[:, 1:, 4]   # (K, H)

        if path_target is not None:
            # Align directly with pre-mission planned path corridor tangent
            los_heading_err = wrap_angle(headings - path_target.target_heading)
            los_pitch_err = pitches - path_target.target_pitch
            j_tracking = np.sum((los_heading_err**2) * 1.2 + (los_pitch_err**2), axis=1)
        else:
            # Vector towards goal from each prediction step
            vec_to_goal = goal_pos - positions[:, 1:, :]  # (K, H, 3)
            los_headings = np.arctan2(vec_to_goal[..., 1], vec_to_goal[..., 0]) % (2 * np.pi)
            los_heading_err = wrap_angle(headings - los_headings)
            
            horiz_dist = np.hypot(vec_to_goal[..., 0], vec_to_goal[..., 1])
            los_pitches = np.arctan2(-vec_to_goal[..., 2], np.maximum(horiz_dist, 0.1))
            los_pitch_err = pitches - los_pitches

            tracking_gate = np.clip(horiz_dist / 3.0, 0.0, 1.0)
            j_tracking = np.sum((los_heading_err**2) * tracking_gate + los_pitch_err**2, axis=1)

        # 3. Control Effort Cost (deviations from target reference speed)
        speed_cmds = controls[:, :, 0]
        j_control = np.sum((speed_cmds - u_ref)**2, axis=1)

        # 4. Control Smoothness / Slew Cost with explicit R matrix
        # [Speed, Pitch, Heading] weighted difference
        du_0 = (
            self.r_smooth[0] * (controls[:, 0, 0] - prev_control[0])**2 +
            self.r_smooth[1] * (controls[:, 0, 1] - prev_control[1])**2 +
            self.r_smooth[2] * (wrap_angle(controls[:, 0, 2] - prev_control[2]))**2
        )
        
        if H > 1:
            du_rest = (
                self.r_smooth[0] * (controls[:, 1:, 0] - controls[:, :-1, 0])**2 +
                self.r_smooth[1] * (controls[:, 1:, 1] - controls[:, :-1, 1])**2 +
                self.r_smooth[2] * (wrap_angle(controls[:, 1:, 2] - controls[:, :-1, 2]))**2
            )
            j_smooth = du_0 + np.sum(du_rest, axis=1)
        else:
            j_smooth = du_0

        # 5. Constraint Penalties (Boundaries, Obstacles, Envelope limits)
        j_constraint = constraint_manager.evaluate_batch_penalties(trajectories)

        # Total Composite Cost
        total_cost = (
            self.w_goal * j_goal +
            self.w_tracking * j_tracking +
            self.w_control * j_control +
            self.w_smooth * j_smooth +
            self.w_constraint * j_constraint
        )

        if return_breakdown:
            breakdown = {
                "j_goal": j_goal,
                "j_tracking": j_tracking,
                "j_control": j_control,
                "j_smooth": j_smooth,
                "j_constraint": j_constraint,
                "total_cost": total_cost
            }
            return total_cost, breakdown

        return total_cost
