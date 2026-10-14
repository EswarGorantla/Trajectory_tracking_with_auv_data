import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from src.state import AUVState, AUVControl, wrap_angle, wrap_to_2pi
from src.goal import Goal
from src.constraints import ConstraintManager

class ReactiveLOSController:
    """
    Standard Reactive 3D Line-Of-Sight (LOS) Guidance Controller.
    Computes steering direct to goal without proactive obstacle avoidance.
    """
    def __init__(self, cruise_speed: float = 1.20):
        self.cruise_speed = cruise_speed

    def compute_control(
        self,
        current_state: AUVState,
        goal: Goal,
        constraint_manager: Optional[ConstraintManager] = None,
        path_target: Optional[Any] = None
    ) -> Tuple[AUVControl, Dict[str, Any]]:
        if path_target is not None:
            los_heading = path_target.target_heading
            los_pitch = path_target.target_pitch
        else:
            los_heading = goal.los_heading_to(current_state)
            los_pitch = goal.los_pitch_to(current_state)

        control = AUVControl(
            speed_cmd=self.cruise_speed,
            pitch_cmd=float(np.clip(los_pitch, -0.35, 0.35)),
            heading_cmd=float(los_heading)
        )
        return control, {"method": "Reactive_LOS"}


class PotentialFieldLOSController:
    """
    Artificial Potential Field (APF) + LOS Guidance Controller.
    Combines attractive force toward Goal B with repulsive vector fields
    emanating from cylindrical obstacles and boundaries.
    Serves as a strong reactive benchmark against MPC.
    """
    def __init__(
        self,
        cruise_speed: float = 1.20,
        k_att: float = 1.0,
        k_rep_obs: float = 80.0,
        k_rep_wall: float = 40.0
    ):
        self.cruise_speed = cruise_speed
        self.k_att = k_att
        self.k_rep_obs = k_rep_obs
        self.k_rep_wall = k_rep_wall

    def compute_control(
        self,
        current_state: AUVState,
        goal: Goal,
        constraint_manager: ConstraintManager,
        path_target: Optional[Any] = None
    ) -> Tuple[AUVControl, Dict[str, Any]]:
        # 1. Attractive Force towards Goal / Lookahead target
        target_pos = path_target.target_pos if path_target is not None else goal.to_array()
        d_goal_n = target_pos[0] - current_state.north
        d_goal_e = target_pos[1] - current_state.east
        d_goal_z = target_pos[2] - current_state.depth
        dist_goal = np.sqrt(d_goal_n**2 + d_goal_e**2 + d_goal_z**2) + 1e-4

        f_att = np.array([
            self.k_att * (d_goal_n / dist_goal),
            self.k_att * (d_goal_e / dist_goal),
            self.k_att * (d_goal_z / dist_goal)
        ], dtype=np.float64)

        f_rep = np.zeros(3, dtype=np.float64)

        # 2. Repulsive Force from Cylindrical Obstacles
        for obs in constraint_manager.obstacles:
            dn = current_state.north - obs.north
            de = current_state.east - obs.east
            dist_horiz = np.hypot(dn, de)
            r_safe = obs.safe_radius

            if dist_horiz < r_safe:
                # Repulsion magnitude scales inversely with clearance
                clearance = max(dist_horiz - obs.radius, 0.05)
                rep_mag = self.k_rep_obs * (1.0 / clearance - 1.0 / obs.margin) * (1.0 / (clearance**2))
                rep_mag = np.clip(rep_mag, 0.0, 100.0)
                f_rep[0] += rep_mag * (dn / dist_horiz)
                f_rep[1] += rep_mag * (de / dist_horiz)

        # 3. Repulsive Force from Geographic Boundaries
        b = constraint_manager.boundaries
        margin_wall = 5.0
        if current_state.north - b.north_min < margin_wall:
            f_rep[0] += self.k_rep_wall / max(current_state.north - b.north_min, 0.1)
        elif b.north_max - current_state.north < margin_wall:
            f_rep[0] -= self.k_rep_wall / max(b.north_max - current_state.north, 0.1)

        if current_state.east - b.east_min < margin_wall:
            f_rep[1] += self.k_rep_wall / max(current_state.east - b.east_min, 0.1)
        elif b.east_max - current_state.east < margin_wall:
            f_rep[1] -= self.k_rep_wall / max(b.east_max - current_state.east, 0.1)

        # Composite guidance force
        f_total = f_att + f_rep
        cmd_heading = np.arctan2(f_total[1], f_total[0]) % (2 * np.pi)
        horiz_f = np.hypot(f_total[0], f_total[1])
        cmd_pitch = np.arctan2(-f_total[2], max(horiz_f, 0.1))
        cmd_pitch = float(np.clip(cmd_pitch, -0.35, 0.35))

        control = AUVControl(
            speed_cmd=self.cruise_speed,
            pitch_cmd=cmd_pitch,
            heading_cmd=float(cmd_heading)
        )
        return control, {
            "method": "Potential_Field_LOS",
            "f_att": f_att.tolist(),
            "f_rep": f_rep.tolist()
        }
