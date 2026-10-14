import numpy as np
from typing import Union
from src.state import AUVState, wrap_to_2pi

class PhysicsDynamics:
    """
    Reduced-order 3D AUV marine kinematics and quadratic surge drag model.
    Strictly calibrated using Phase 1 system identification parameters.
    """

    def __init__(self, k_d: float = 0.06073, dt: float = 2.0, u_cruise: float = 1.2):
        self.k_d = k_d
        self.dt = dt
        self.u_cruise = u_cruise

    def step(self, state: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Step single state: [North, East, Depth, Roll, Pitch, Heading, Speed]
        """
        dt = dt if dt is not None else self.dt
        x, y, z, phi, theta, psi, u = state

        # 3D Euler Kinematic Propagation
        x_next = x + u * np.cos(theta) * np.cos(psi) * dt
        y_next = y + u * np.cos(theta) * np.sin(psi) * dt
        z_next = z - u * np.sin(theta) * dt  # Positive pitch climbs (depth decreases)

        # Hydrodynamic Surge Drag Model
        u_next = max(u + self.k_d * (self.u_cruise**2 - u**2) * dt, 0.0)

        # Baseline assumption for attitude (static over short dt)
        phi_next = phi
        theta_next = theta
        psi_next = wrap_to_2pi(psi)

        return np.array([x_next, y_next, z_next, phi_next, theta_next, psi_next, u_next], dtype=np.float64)

    def step_batch(self, states: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Vectorized batch physics propagation on (N, 7) state matrix for high-speed MPC rollouts.
        """
        dt = dt if dt is not None else self.dt
        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]
        phi = states[:, 3]
        theta = states[:, 4]
        psi = states[:, 5]
        u = states[:, 6]

        x_next = x + u * np.cos(theta) * np.cos(psi) * dt
        y_next = y + u * np.cos(theta) * np.sin(psi) * dt
        z_next = z - u * np.sin(theta) * dt
        u_next = np.maximum(u + self.k_d * (self.u_cruise**2 - u**2) * dt, 0.0)

        return np.column_stack([x_next, y_next, z_next, phi, theta, psi, u_next])
