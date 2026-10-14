import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
from src.state import AUVState, AUVControl, wrap_angle

@dataclass
class Boundaries3D:
    """Rectangular 3D Operational Boundaries (in meters)."""
    north_min: float = 0.0
    north_max: float = 120.0
    east_min: float = 0.0
    east_max: float = 100.0
    depth_min: float = 10.0
    depth_max: float = 60.0
    margin: float = 4.0

    def is_inside(self, pos: Union[AUVState, np.ndarray]) -> bool:
        if isinstance(pos, AUVState):
            n, e, d = pos.north, pos.east, pos.depth
        else:
            n, e, d = pos[0], pos[1], pos[2]
        return (
            self.north_min <= n <= self.north_max and
            self.east_min <= e <= self.east_max and
            self.depth_min <= d <= self.depth_max
        )

    def violation_distance(self, pos: Union[AUVState, np.ndarray]) -> float:
        """Returns distance outside the boundary (0 if inside)."""
        if isinstance(pos, AUVState):
            n, e, d = pos.north, pos.east, pos.depth
        else:
            n, e, d = pos[0], pos[1], pos[2]

        dn = max(self.north_min - n, 0.0) + max(n - self.north_max, 0.0)
        de = max(self.east_min - e, 0.0) + max(e - self.east_max, 0.0)
        dd = max(self.depth_min - d, 0.0) + max(d - self.depth_max, 0.0)
        return float(np.sqrt(dn**2 + de**2 + dd**2))

    def evaluate_batch_penetrations(self, positions: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation on (K, H+1, 3) or (K, N, 3) positions with formal Control Barrier Function (CBF):
        1. Hard boundary breach (min_wall_dist <= 0): Unbounded penalty (1e8 + 1e6 * dist)
           that strictly disqualifies any candidate that breaches the geofence perimeter.
        2. Standoff buffer zone (0 < min_wall_dist < margin): Non-linear barrier function
           B(d) = 100 * (margin / d)^2 + 500 * (margin - d)^2 that provides strong
           inward repulsive potential gradients pushing the vehicle away from boundaries.
        """
        n = positions[..., 0]
        e = positions[..., 1]
        d = positions[..., 2]

        d_n_min = n - self.north_min
        d_n_max = self.north_max - n
        d_e_min = e - self.east_min
        d_e_max = self.east_max - e
        d_d_min = d - self.depth_min
        d_d_max = self.depth_max - d

        min_wall_dist = np.minimum.reduce([d_n_min, d_n_max, d_e_min, d_e_max, d_d_min, d_d_max])

        is_breach = min_wall_dist <= 0.0
        breach_pen = 1e11 + 1e9 * np.maximum(-min_wall_dist, 0.0)

        is_buffer = (min_wall_dist > 0.0) & (min_wall_dist < self.margin)
        clearance = np.maximum(min_wall_dist, 0.04)
        buffer_intrusion = np.maximum(self.margin - min_wall_dist, 0.0)
        barrier_pen = 200.0 * ((self.margin / clearance) ** 2) + 1000.0 * (buffer_intrusion ** 2)

        pen = np.where(is_breach, breach_pen, np.where(is_buffer, barrier_pen, 0.0))
        return pen

    def evaluate_batch_violations(self, positions: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation on (K, H+1, 3) or (N, 3) positions.
        Returns sum of squared boundary penetration distances.
        """
        n = positions[..., 0]
        e = positions[..., 1]
        d = positions[..., 2]

        dn = np.maximum(self.north_min - n, 0.0) + np.maximum(n - self.north_max, 0.0)
        de = np.maximum(self.east_min - e, 0.0) + np.maximum(e - self.east_max, 0.0)
        dd = np.maximum(self.depth_min - d, 0.0) + np.maximum(d - self.depth_max, 0.0)

        pen = dn**2 + de**2 + dd**2
        return pen


@dataclass
class CylindricalObstacle3D:
    """
    3D Cylindrical Obstacle (e.g. seamount, mooring cable, pipeline structure).
    Defined by horizontal center (north, east), radius, safety margin, and vertical span.
    """
    north: float
    east: float
    radius: float
    margin: float = 3.0
    depth_min: float = 0.0
    depth_max: float = 100.0

    @property
    def safe_radius(self) -> float:
        return self.radius + self.margin

    def is_collision(self, pos: Union[AUVState, np.ndarray]) -> bool:
        if isinstance(pos, AUVState):
            n, e, d = pos.north, pos.east, pos.depth
        else:
            n, e, d = pos[0], pos[1], pos[2]

        if not (self.depth_min <= d <= self.depth_max):
            return False
        dist_2d = np.hypot(n - self.north, e - self.east)
        return dist_2d <= self.radius

    def evaluate_batch_penetrations(self, positions: np.ndarray) -> np.ndarray:
        """
        Vectorized obstacle penalty calculation with formal Control Barrier Function (CBF):
        1. Hard physical collision (dist <= radius): Unbounded penalty (1e8) completely rejects 
           any candidate that penetrates the physical obstacle hull.
        2. Safety buffer zone (radius < dist < safe_radius): Steep non-linear barrier function
           B(d) = 50 * (margin / (d - radius))^2 + 200 * (safe_radius - d)^2 that provides
           strong repulsive potential gradients away from the obstacle.
        """
        n = positions[..., 0]
        e = positions[..., 1]
        d = positions[..., 2]

        dist_2d = np.hypot(n - self.north, e - self.east)
        depth_active = (d >= self.depth_min) & (d <= self.depth_max)

        # 1. Hard physical hull collision
        is_collision = (dist_2d <= self.radius) & depth_active
        collision_pen = 1e8 + 1e6 * (self.radius - dist_2d)

        # 2. Buffer zone intrusion with steep repulsive barrier
        is_buffer = (dist_2d > self.radius) & (dist_2d < self.safe_radius) & depth_active
        clearance = np.maximum(dist_2d - self.radius, 0.02)
        buffer_intrusion = np.maximum(self.safe_radius - dist_2d, 0.0)
        barrier_pen = 50.0 * ((self.margin / clearance) ** 2) + 200.0 * (buffer_intrusion ** 2)

        penalty = np.where(is_collision, collision_pen, np.where(is_buffer, barrier_pen, 0.0))
        return penalty


@dataclass
class VehicleConstraints:
    """Physical AUV Actuation and Envelope Limits."""
    speed_min: float = 0.5
    speed_max: float = 2.0
    pitch_max: float = 0.35      # ~20 degrees
    roll_max: float = 0.50       # ~28.6 degrees
    max_pitch_rate: float = 0.10 # rad/step
    max_heading_rate: float = 0.20 # rad/step


class ConstraintManager:
    """
    Central constraint evaluation and violation accounting for Phase 2.
    """

    def __init__(
        self,
        boundaries: Boundaries3D = None,
        obstacles: List[CylindricalObstacle3D] = None,
        vehicle: VehicleConstraints = None
    ):
        self.boundaries = boundaries or Boundaries3D()
        self.obstacles = obstacles or []
        self.vehicle = vehicle or VehicleConstraints()

    def check_state(self, state: AUVState) -> Dict[str, Any]:
        """Audits a single state against all constraints."""
        in_bounds = self.boundaries.is_inside(state)
        bound_dist = self.boundaries.violation_distance(state)

        obs_collisions = [obs.is_collision(state) for obs in self.obstacles]
        has_obs_collision = any(obs_collisions)

        speed_ok = self.vehicle.speed_min <= state.speed <= self.vehicle.speed_max
        pitch_ok = abs(state.pitch) <= self.vehicle.pitch_max
        roll_ok = abs(state.roll) <= self.vehicle.roll_max

        all_ok = in_bounds and (not has_obs_collision) and speed_ok and pitch_ok and roll_ok

        return {
            "all_satisfied": all_ok,
            "in_boundaries": in_bounds,
            "boundary_violation_distance": bound_dist,
            "obstacle_collision": has_obs_collision,
            "speed_ok": speed_ok,
            "pitch_ok": pitch_ok,
            "roll_ok": roll_ok
        }

    def evaluate_batch_penalties(self, trajectories: np.ndarray) -> np.ndarray:
        """
        Evaluates cumulative constraint penalty for K candidate trajectories of shape (K, H+1, 7).
        Includes sub-step midpoint sampling to prevent inter-step discrete tunneling through obstacles.
        Returns penalty vector of shape (K,).
        """
        K, H_plus_1, _ = trajectories.shape
        positions = trajectories[..., :3]  # (K, H+1, 3)

        # Dense sub-step evaluation: concatenate knot points and midpoints to prevent obstacle tunneling
        if H_plus_1 > 1:
            mid_pts = 0.5 * (positions[:, :-1, :] + positions[:, 1:, :])
            dense_positions = np.concatenate([positions, mid_pts], axis=1)
        else:
            dense_positions = positions

        # 1. Boundary violations penalty with hard breach rejection and standoff barrier
        bound_pen = self.boundaries.evaluate_batch_penetrations(dense_positions)
        total_bound_pen = np.sum(bound_pen, axis=1)

        # 2. Obstacle barrier & collision penalties
        total_obs_pen = np.zeros(K, dtype=np.float64)
        for obs in self.obstacles:
            obs_pen = obs.evaluate_batch_penetrations(dense_positions)
            total_obs_pen += np.sum(obs_pen, axis=1)

        # 3. Attitude & Speed limits penalty
        pitches = trajectories[..., 4]
        rolls = trajectories[..., 3]
        speeds = trajectories[..., 6]

        pitch_viol = np.maximum(np.abs(pitches) - self.vehicle.pitch_max, 0.0)**2
        roll_viol = np.maximum(np.abs(rolls) - self.vehicle.roll_max, 0.0)**2
        speed_viol = (
            np.maximum(self.vehicle.speed_min - speeds, 0.0)**2 +
            np.maximum(speeds - self.vehicle.speed_max, 0.0)**2
        )

        env_pen = np.sum(pitch_viol + roll_viol + speed_viol, axis=1)

        return total_bound_pen + total_obs_pen + env_pen * 50.0
