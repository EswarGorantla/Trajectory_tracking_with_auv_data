import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple
from src.state import AUVState, wrap_to_2pi

@dataclass
class Waypoint:
    """Configurable 3D Point in Navigation Space."""
    north: float
    east: float
    depth: float

    def to_array(self) -> np.ndarray:
        return np.array([self.north, self.east, self.depth], dtype=np.float64)


@dataclass
class Goal(Waypoint):
    """
    Destination Target Point B with configurable acceptance radius
    and Line-of-Sight (LOS) guidance vectors.
    """
    tolerance: float = 5.0  # Acceptance radius (meters)

    def is_reached(self, current_pos: Union[AUVState, np.ndarray]) -> bool:
        return self.distance_to(current_pos) <= self.tolerance

    def distance_to(self, current_pos: Union[AUVState, np.ndarray]) -> float:
        if isinstance(current_pos, AUVState):
            p = current_pos.position()
        else:
            p = current_pos[:3]
        return float(np.linalg.norm(self.to_array() - p))

    def horizontal_distance_to(self, current_pos: Union[AUVState, np.ndarray]) -> float:
        if isinstance(current_pos, AUVState):
            n, e = current_pos.north, current_pos.east
        else:
            n, e = current_pos[0], current_pos[1]
        return float(np.hypot(self.north - n, self.east - e))

    def los_heading_to(self, current_pos: Union[AUVState, np.ndarray]) -> float:
        """
        Computes 2D Line-Of-Sight (LOS) target azimuth in radians [0, 2pi) clockwise from North:
        psi = atan2(East_B - East_cur, North_B - North_cur)
        """
        if isinstance(current_pos, AUVState):
            n, e = current_pos.north, current_pos.east
        else:
            n, e = current_pos[0], current_pos[1]
        psi = np.arctan2(self.east - e, self.north - n)
        return float(wrap_to_2pi(psi))

    def los_pitch_to(self, current_pos: Union[AUVState, np.ndarray]) -> float:
        """
        Computes vertical Line-Of-Sight (LOS) pitch angle:
        Positive pitch climbs (decreases depth), negative pitch dives (increases depth).
        theta = atan2(Depth_cur - Depth_B, horizontal_distance)
        """
        if isinstance(current_pos, AUVState):
            z = current_pos.depth
        else:
            z = current_pos[2]
        d_h = self.horizontal_distance_to(current_pos)
        if d_h < 0.1:
            return 0.0
        theta = np.arctan2(z - self.depth, d_h)
        return float(np.clip(theta, -0.35, 0.35))


@dataclass
class PathTarget:
    """
    Dynamic lookahead reference target along pre-mission global path.
    Guides receding-horizon MPC along the collision-free bathymetric corridor.
    """
    target_pos: np.ndarray                            # 3D [North, East, Depth]
    target_heading: float                             # Radians [0, 2pi)
    target_pitch: float                               # Radians [-0.35, 0.35]
    dist_remaining: float                             # Distance to terminal Goal B along global path
    final_goal: Goal                                  # Terminal acceptance goal
    horizon_waypoints: Optional[np.ndarray] = None   # (H, 3) sequential lookahead waypoints along corridor

