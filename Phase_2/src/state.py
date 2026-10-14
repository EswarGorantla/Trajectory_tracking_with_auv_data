import numpy as np
from dataclasses import dataclass
from typing import List, Union

def wrap_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wraps angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wrap_to_2pi(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wraps angle to [0, 2pi)."""
    return angle % (2 * np.pi)


@dataclass
class AUVControl:
    """
    Autopilot Steering Control Input Vector:
    u = [U_cmd, theta_cmd, psi_cmd]
    """
    speed_cmd: float       # Commanded forward surge speed in m/s (e.g. 0.5 to 2.0)
    pitch_cmd: float       # Commanded pitch angle in radians (e.g. -0.35 to +0.35)
    heading_cmd: float     # Commanded heading angle in radians [0, 2pi)

    def to_array(self) -> np.ndarray:
        return np.array([self.speed_cmd, self.pitch_cmd, self.heading_cmd], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'AUVControl':
        return cls(speed_cmd=float(arr[0]), pitch_cmd=float(arr[1]), heading_cmd=float(arr[2]))


@dataclass
class AUVState:
    """
    Validated 7D State Vector strictly matching Phase 1 representation:
    x = [North, East, Depth, Roll, Pitch, Heading, Speed]
    """
    north: float     # Local Cartesian North position (meters)
    east: float      # Local Cartesian East position (meters)
    depth: float     # Distance below sea surface (meters)
    roll: float      # Roll angle (radians)
    pitch: float     # Pitch angle (radians) [positive = nose up, climbing]
    heading: float   # Heading angle clockwise from North (radians) [0, 2pi)
    speed: float     # Forward surge velocity over ground (m/s)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.north, self.east, self.depth,
            self.roll, self.pitch, self.heading, self.speed
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'AUVState':
        return cls(
            north=float(arr[0]),
            east=float(arr[1]),
            depth=float(arr[2]),
            roll=float(arr[3]),
            pitch=float(arr[4]),
            heading=float(wrap_to_2pi(arr[5])),
            speed=float(max(arr[6], 0.0))
        )

    def to_feature_vector(self) -> np.ndarray:
        """
        Extracts continuous 6D input feature representation for the Phase 1 GRU model:
        s = [Speed, Roll, Pitch, cos(Heading), sin(Heading), Depth]
        """
        return np.array([
            self.speed,
            self.roll,
            self.pitch,
            np.cos(self.heading),
            np.sin(self.heading),
            self.depth
        ], dtype=np.float64)

    def position(self) -> np.ndarray:
        return np.array([self.north, self.east, self.depth], dtype=np.float64)

    def horizontal_distance_to(self, target_pos: Union['AUVState', np.ndarray]) -> float:
        if isinstance(target_pos, AUVState):
            return float(np.sqrt((self.north - target_pos.north)**2 + (self.east - target_pos.east)**2))
        return float(np.sqrt((self.north - target_pos[0])**2 + (self.east - target_pos[1])**2))

    def distance_3d_to(self, target_pos: Union['AUVState', np.ndarray]) -> float:
        if isinstance(target_pos, AUVState):
            return float(np.sqrt(
                (self.north - target_pos.north)**2 +
                (self.east - target_pos.east)**2 +
                (self.depth - target_pos.depth)**2
            ))
        return float(np.sqrt(
            (self.north - target_pos[0])**2 +
            (self.east - target_pos[1])**2 +
            (self.depth - target_pos[2])**2
        ))
