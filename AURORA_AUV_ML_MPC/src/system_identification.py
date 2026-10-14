import numpy as np

class SystemIdentifier:
    def __init__(self, dt=2.0):
        self.dt = dt
        
    def identify_drag_coefficient(self, speed_series, u_cruise):
        """
        Fits first-order drag coefficient k_d from speed time-series.
        U_{t+1} - U_t = k_d * (U_cruise^2 - U_t^2) * dt
        """
        speed_series = np.array(speed_series)
        
        # Calculate speed derivatives
        dU = (speed_series[1:] - speed_series[:-1]) / self.dt
        
        # Formulate X variable: (U_cruise^2 - U_t^2)
        U_t = speed_series[:-1]
        X = u_cruise**2 - U_t**2
        
        # Linear regression: dU = k_d * X
        # Solved via least squares: k_d = (X^T * X)^-1 * X^T * dU
        denom = np.dot(X, X)
        if denom < 1e-6:
            k_d = 0.05  # Default fallback
        else:
            k_d = np.dot(X, dU) / denom
            
        # Ensure drag is physical (non-negative and within reasonable bounds)
        k_d = np.clip(k_d, 0.001, 1.0)
        print(f"Identified drag coefficient k_d: {k_d:.6f}")
        return k_d
