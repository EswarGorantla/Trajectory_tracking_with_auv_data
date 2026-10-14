import numpy as np

class PhysicsModel:
    """
    Reduced-order 3D AUV kinematics and quadratic hydrodynamic surge drag model.
    M ν_dot + D(ν)ν = τ + d formulation without fabricated control inputs.
    """

    def __init__(self, k_d: float = 0.06073, dt: float = 2.0, u_cruise: float = 1.2):
        self.k_d = k_d
        self.dt = dt
        self.u_cruise = u_cruise

    @classmethod
    def fit_drag_coefficient_on_train(cls, train_speeds: np.ndarray, dt: float = 2.0, u_cruise: float = 1.2) -> float:
        """
        Estimates the hydrodynamic drag coefficient Kd strictly from TRAINING SET speed telemetry.
        (u[t+1] - u[t]) / dt = Kd * (u_cruise^2 - u[t]^2)
        """
        u_t = train_speeds[:-1]
        u_next = train_speeds[1:]
        du_dt = (u_next - u_t) / dt

        regressor = (u_cruise**2 - u_t**2)
        denom = np.sum(regressor**2)
        if denom == 0:
            return 0.02

        k_d = float(np.sum(regressor * du_dt) / denom)
        return max(k_d, 0.001)

    def step(self, state: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Advances the 7D AUV state by one timestep using 3D kinematics and drag dynamics.
        state: [North (x), East (y), Depth (z), Roll (phi), Pitch (theta), Heading (psi), Speed (U)]
        """
        dt = dt if dt is not None else self.dt
        x, y, z, phi, theta, psi, u = state

        # 3D Kinematic propagation
        x_next = x + u * np.cos(theta) * np.cos(psi) * dt
        y_next = y + u * np.cos(theta) * np.sin(psi) * dt
        z_next = z - u * np.sin(theta) * dt  # nose-up pitch decreases depth
        
        # Surge speed drag dynamics
        u_next = max(u + self.k_d * (self.u_cruise**2 - u**2) * dt, 0.0)

        # Baseline assumption for attitude (static Euler angles over step)
        phi_next = phi
        theta_next = theta
        psi_next = psi % (2 * np.pi)

        return np.array([x_next, y_next, z_next, phi_next, theta_next, psi_next, u_next], dtype=np.float64)

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Vectorized batch physics prediction on (N, 7) state matrix.
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
        u_next = np.maximum(u + self.k_d * (self.u_cruise**2 - u**2) * self.dt, 0.0)

        return np.column_stack([x_next, y_next, z_next, phi, theta, psi, u_next])
