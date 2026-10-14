import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from src.state import AUVState, AUVControl, wrap_angle, wrap_to_2pi

@dataclass
class CylindricalObstacle:
    north: float
    east: float
    radius: float           # Physical collision radius (m)
    margin: float = 4.0     # Safety standoff buffer (m)
    
    @property
    def safe_radius(self) -> float:
        return self.radius + self.margin

@dataclass
class PlantParameters:
    kd_baseline: float = 0.06073034
    kd_mismatch: float = 0.0        # Relative mismatch delta in [-0.25, 0.25]
    c2_nonlinear_drag: float = 0.015 # Quadratic surge damping coefficient
    buoyancy_stiffness: float = 0.08 # Restoring moment coefficient (-k_b * sin(theta))
    depth_pitch_coupling: float = 0.02
    
    # Actuator time constants (first-order lag)
    tau_u: float = 1.5      # Speed response time constant (s)
    tau_pitch: float = 1.0  # Pitch response time constant (s)
    tau_yaw: float = 1.2    # Yaw response time constant (s)
    
    # Actuator rate limits (per second)
    max_pitch_rate_sec: float = 0.05    # 0.10 rad / 2.0s
    max_heading_rate_sec: float = 0.10  # 0.20 rad / 2.0s
    max_speed_rate_sec: float = 0.075   # 0.15 m/s / 2.0s
    
    # Disturbances
    ou_theta_std: float = 0.0043
    ou_psi_std: float = 0.0155
    ou_u_std: float = 0.0219
    ou_correlation_time: float = 10.0   # OU mean reversion time (s)

class TrueAUVPlant:
    """
    Research-Grade Continuous-Time True Plant for Autonomous Underwater Vehicle (AUV).
    
    Key Distinctions from Controller Model (Preventing 'Inverse Crime'):
    1. Independent continuous-time dynamics integrated via 4th-order Runge-Kutta (RK4) with sub-stepping (dt_sim = 0.1s).
    2. Real actuator lag (first-order low-pass) and hard continuous rate limiters.
    3. Hydrodynamic parameter mismatch (Kd differs by delta in [-25%, +25%]).
    4. Quadratic speed drag and depth-pitch restoring hydrostatic moments.
    5. Stochastic Ornstein-Uhlenbeck environmental disturbances calibrated from validation telemetry.
    6. 3D ocean currents (constant or Gauss-Markov drift).
    7. Independent true collision and safety-buffer detection against physical geometry.
    """
    
    def __init__(
        self,
        params: Optional[PlantParameters] = None,
        dt_control: float = 2.0,
        dt_sim: float = 0.10,
        current_velocity: Optional[np.ndarray] = None,
        use_gauss_markov_current: bool = False,
        obstacles: Optional[List[CylindricalObstacle]] = None,
        seed: Optional[int] = None
    ):
        self.params = params or PlantParameters()
        self.dt_control = dt_control
        self.dt_sim = dt_sim
        self.substeps = int(round(dt_control / dt_sim))
        self.current_vel = np.array(current_velocity if current_velocity is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.use_gm_current = use_gauss_markov_current
        self.obstacles = obstacles or []
        self.rng = np.random.default_rng(seed)
        
        # State vector: [x (North), y (East), z (Depth), phi (Roll), theta (Pitch), psi (Heading), U (Speed)]
        self.state = np.zeros(7, dtype=np.float64)
        
        # Internal actuator states: [u_act, theta_act, psi_act]
        self.actuator_state = np.zeros(3, dtype=np.float64)
        
        # Ornstein-Uhlenbeck disturbance states: [ou_u, ou_theta, ou_psi]
        self.ou_state = np.zeros(3, dtype=np.float64)
        self.ou_gamma = 1.0 / self.params.ou_correlation_time
        
        # Tracking metrics
        self.collision_count = 0
        self.buffer_violation_count = 0
        self.min_obstacle_dist = float('inf')
        self.min_buffer_dist = float('inf')

    def reset(self, initial_state: AUVState, randomize_mismatch: bool = False):
        self.state = initial_state.to_array().copy()
        self.actuator_state = np.array([self.state[6], self.state[4], self.state[5]], dtype=np.float64)
        self.ou_state = np.zeros(3, dtype=np.float64)
        
        if randomize_mismatch:
            self.params.kd_mismatch = float(self.rng.uniform(-0.25, 0.25))
            
        self.collision_count = 0
        self.buffer_violation_count = 0
        self.min_obstacle_dist = float('inf')
        self.min_buffer_dist = float('inf')

    def _rk4_derivatives(self, s: np.ndarray, act: np.ndarray, u_cmd: AUVControl) -> Tuple[np.ndarray, np.ndarray]:
        """
        Continuous-time derivatives:
        s: [x, y, z, phi, theta, psi, U]
        act: [u_act, theta_act, psi_act]
        """
        p = self.params
        
        # Actuator lag and rate clipping
        du_raw = (u_cmd.speed_cmd - act[0]) / p.tau_u
        du_dt = np.clip(du_raw, -p.max_speed_rate_sec, p.max_speed_rate_sec)
        
        dtheta_raw = (u_cmd.pitch_cmd - act[1]) / p.tau_pitch
        dtheta_dt = np.clip(dtheta_raw, -p.max_pitch_rate_sec, p.max_pitch_rate_sec)
        
        dpsi_err = wrap_angle(u_cmd.heading_cmd - act[2])
        dpsi_raw = dpsi_err / p.tau_yaw
        dpsi_dt = np.clip(dpsi_raw, -p.max_heading_rate_sec, p.max_heading_rate_sec)
        
        dact_dt = np.array([du_dt, dtheta_dt, dpsi_dt], dtype=np.float64)
        
        # Vehicle motion kinematics + ocean current drift
        u_speed = s[6]
        theta = s[4]
        psi = s[5]
        
        v_current = self.current_vel
        
        # Kinematics
        dx_dt = u_speed * np.cos(theta) * np.cos(psi) + v_current[0]
        dy_dt = u_speed * np.cos(theta) * np.sin(psi) + v_current[1]
        dz_dt = -u_speed * np.sin(theta) + v_current[2]  # Depth positive-down
        dphi_dt = 0.0 # PHINS roll damping equilibrium
        
        # Dynamics with hydrodynamic mismatch and restoring moments
        kd_eff = p.kd_baseline * (1.0 + p.kd_mismatch)
        u_cruise = 1.20
        
        # True acceleration with quadratic drag
        du_surge = kd_eff * (u_cruise**2 - u_speed**2) - p.c2_nonlinear_drag * u_speed * abs(u_speed) + self.ou_state[0]
        # Actuator drives speed toward u_act
        du_plant = du_surge + (act[0] - u_speed) * 0.8
        
        # True pitch dynamic with hydrostatic restoring moment
        dtheta_plant = np.clip((act[1] - theta) / p.tau_pitch, -p.max_pitch_rate_sec, p.max_pitch_rate_sec) - p.buoyancy_stiffness * np.sin(theta) + self.ou_state[1]
        
        # True yaw rate
        dpsi_plant = np.clip(wrap_angle(act[2] - psi) / p.tau_yaw, -p.max_heading_rate_sec, p.max_heading_rate_sec) + self.ou_state[2]
        
        ds_dt = np.array([dx_dt, dy_dt, dz_dt, dphi_dt, dtheta_plant, dpsi_plant, du_plant], dtype=np.float64)
        return ds_dt, dact_dt

    def step(self, control: AUVControl) -> Tuple[AUVState, Dict[str, Any]]:
        """
        Integrates the plant over dt_control (2.0s) using substeps of dt_sim (0.1s) with RK4.
        """
        dt = self.dt_sim
        
        # Update Ornstein-Uhlenbeck disturbance once per control step
        ou_noise = self.rng.standard_normal(3)
        self.ou_state[0] += -self.ou_gamma * self.ou_state[0] * self.dt_control + self.params.ou_u_std * np.sqrt(2 * self.ou_gamma * self.dt_control) * ou_noise[0]
        self.ou_state[1] += -self.ou_gamma * self.ou_state[1] * self.dt_control + self.params.ou_theta_std * np.sqrt(2 * self.ou_gamma * self.dt_control) * ou_noise[1]
        self.ou_state[2] += -self.ou_gamma * self.ou_state[2] * self.dt_control + self.params.ou_psi_std * np.sqrt(2 * self.ou_gamma * self.dt_control) * ou_noise[2]
        
        # Update Gauss-Markov current if enabled
        if self.use_gm_current:
            gm_noise = self.rng.standard_normal(3) * 0.015
            self.current_vel += -0.05 * self.current_vel * self.dt_control + gm_noise
            
        step_collision = False
        step_buffer_violation = False
        
        for _ in range(self.substeps):
            # RK4 Integration
            k1_s, k1_act = self._rk4_derivatives(self.state, self.actuator_state, control)
            
            s2 = self.state + 0.5 * dt * k1_s
            act2 = self.actuator_state + 0.5 * dt * k1_act
            k2_s, k2_act = self._rk4_derivatives(s2, act2, control)
            
            s3 = self.state + 0.5 * dt * k2_s
            act3 = self.actuator_state + 0.5 * dt * k2_act
            k3_s, k3_act = self._rk4_derivatives(s3, act3, control)
            
            s4 = self.state + dt * k3_s
            act4 = self.actuator_state + dt * k3_act
            k4_s, k4_act = self._rk4_derivatives(s4, act4, control)
            
            self.state += (dt / 6.0) * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s)
            self.actuator_state += (dt / 6.0) * (k1_act + 2.0 * k2_act + 2.0 * k3_act + k4_act)
            
            # Post-substep sanitization
            self.state[5] = wrap_to_2pi(self.state[5])
            self.state[6] = np.maximum(self.state[6], 0.0)
            self.actuator_state[2] = wrap_to_2pi(self.actuator_state[2])
            
            # Sub-step obstacle collision check
            cur_north = self.state[0]
            cur_east = self.state[1]
            for obs in self.obstacles:
                d = np.hypot(cur_north - obs.north, cur_east - obs.east)
                if d < self.min_obstacle_dist:
                    self.min_obstacle_dist = d
                if (d - obs.radius) < self.min_buffer_dist:
                    self.min_buffer_dist = d - obs.radius
                    
                if d <= obs.radius:
                    step_collision = True
                if d <= obs.safe_radius:
                    step_buffer_violation = True
                    
        if step_collision:
            self.collision_count += 1
        if step_buffer_violation:
            self.buffer_violation_count += 1
            
        realized_state = AUVState(
            north=float(self.state[0]),
            east=float(self.state[1]),
            depth=float(self.state[2]),
            roll=float(self.state[3]),
            pitch=float(self.state[4]),
            heading=float(self.state[5]),
            speed=float(self.state[6])
        )
        
        info = {
            "step_collision": step_collision,
            "step_buffer_violation": step_buffer_violation,
            "min_obstacle_dist": float(self.min_obstacle_dist),
            "min_buffer_dist": float(self.min_buffer_dist),
            "total_collisions": self.collision_count,
            "total_buffer_violations": self.buffer_violation_count,
            "actuator_state": self.actuator_state.tolist(),
            "current_vel": self.current_vel.tolist(),
            "ou_disturbance": self.ou_state.tolist()
        }
        
        return realized_state, info
