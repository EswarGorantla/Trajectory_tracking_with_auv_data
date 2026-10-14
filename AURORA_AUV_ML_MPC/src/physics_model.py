import numpy as np

class PhysicsModel:
    def __init__(self, k_d=0.05, dt=2.0, U_cruise=1.2):
        self.k_d = k_d
        self.dt = dt
        self.U_cruise = U_cruise
        
    def set_drag_coefficient(self, k_d):
        self.k_d = k_d

    def step(self, state, u_cmd, theta_cmd, psi_cmd):
        """
        Performs a forward step of the physics model.
        State: [North, East, Depth, Roll, Pitch, Heading, Speed]
        u_cmd: commanded speed
        theta_cmd: commanded pitch
        psi_cmd: commanded heading
        """
        x, y, z, roll, pitch, heading, speed = state
        
        # Heading wrapping utility
        def wrap_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
            
        # Command rate limit updates (simulates control fin delays)
        pitch_next = pitch + np.clip(theta_cmd - pitch, -0.1, 0.1)
        heading_next = heading + np.clip(wrap_angle(psi_cmd - heading), -0.2, 0.2)
        heading_next = heading_next % (2 * np.pi)
        
        # Kinematic transitions
        x_next = x + speed * np.cos(pitch) * np.cos(heading) * self.dt
        y_next = y + speed * np.cos(pitch) * np.sin(heading) * self.dt
        z_next = z - speed * np.sin(pitch) * self.dt
        roll_next = roll
        
        # Surge speed dynamic model (thrust vs quadratic drag)
        speed_next = speed + self.k_d * (u_cmd**2 - speed**2) * self.dt
        speed_next = np.maximum(speed_next, 0.0)
        
        return np.array([x_next, y_next, z_next, roll_next, pitch_next, heading_next, speed_next])
