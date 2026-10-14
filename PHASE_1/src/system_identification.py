import numpy as np

class SystemIdentifier:
    """Estimates hydrodynamic parameters from training split telemetry using least squares."""

    def __init__(self, dt: float = 2.0):
        self.dt = dt

    def identify_drag_coefficient(self, speed_series: np.ndarray, u_cruise: float = 1.2) -> float:
        """
        Fits drag coefficient Kd using:
        (u[t+1] - u[t]) / dt = Kd * (u_cruise^2 - u[t]^2)
        """
        u_t = speed_series[:-1]
        u_next = speed_series[1:]
        du_dt = (u_next - u_t) / self.dt
        
        regressor = (u_cruise**2 - u_t**2)
        
        # Least squares regression: Kd = (regressor^T * du_dt) / (regressor^T * regressor)
        denominator = np.sum(regressor**2)
        if denominator == 0:
            return 0.02
            
        k_d = float(np.sum(regressor * du_dt) / denominator)
        # Ensure physical non-negativity
        k_d = max(k_d, 0.001)
        return k_d
