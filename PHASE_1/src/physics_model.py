import numpy as np

class PhysicsModel:
    """
    Standard marine vehicle 3D kinematics and speed-drag dynamics model.
    """
    def __init__(self, k_d: float = 0.022498, dt: float = 2.0, U_cruise: float = 1.2):
        self.k_d = k_d
        self.dt = dt
        self.U_cruise = U_cruise

    def step(self, state: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Advances the 7D AUV state by one timestep using 3D kinematics and drag dynamics.
        state: [North (x), East (y), Depth (z), Roll (phi), Pitch (theta), Heading (psi), Speed (U)]
        """
        dt = dt if dt is not None else self.dt
        
        x = state[0]
        y = state[1]
        z = state[2]
        phi = state[3]
        theta = state[4]
        psi = state[5]
        u = state[6]

        # 3D Kinematic updates
        x_next = x + u * np.cos(theta) * np.cos(psi) * dt
        y_next = y + u * np.cos(theta) * np.sin(psi) * dt
        z_next = z - u * np.sin(theta) * dt # positive pitch is nose-up (decreases depth)
        
        # Surge speed dynamics with quadratic hydrodynamic drag
        u_next = max(u + self.k_d * (self.U_cruise**2 - u**2) * dt, 0.0)

        # Baseline assumption for angles (constant velocity / zero commanded turn rate)
        phi_next = phi
        theta_next = theta
        psi_next = psi % (2 * np.pi)

        return np.array([x_next, y_next, z_next, phi_next, theta_next, psi_next, u_next], dtype=np.float64)

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Vectorized batch one-step physics prediction.
        states: (N, 7) array
        """
        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]
        phi = states[:, 3]
        theta = states[:, 4]
        psi = states[:, 5]
        u = states[:, 6]

        x_next = x + u * np.cos(theta) * np.cos(psi) * self.dt
        y_next = y + u * np.cos(theta) * np.sin(psi) * self.dt
        z_next = z - u * np.sin(theta) * self.dt
        u_next = np.maximum(u + self.k_d * (self.U_cruise**2 - u**2) * self.dt, 0.0)
        
        return np.column_stack([x_next, y_next, z_next, phi, theta, psi, u_next])
