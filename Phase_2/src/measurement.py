import numpy as np
from typing import Tuple, Optional, Dict, Any
from src.state import AUVState, wrap_to_2pi

class PositionKalmanFilter:
    """
    2D Constant-Velocity Kinematic Kalman Filter for smoothing noisy USBL acoustic position fixes.
    State: [x, y, vx, vy]^T
    """
    def __init__(self, dt: float = 2.0, q_pos: float = 0.05, q_vel: float = 0.10, r_usbl: float = 0.50):
        self.dt = dt
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)
        
        # Process noise covariance
        q_p = (q_pos**2) * dt
        q_v = (q_vel**2) * dt
        self.Q = np.diag([q_p, q_p, q_v, q_v])
        
        # Measurement noise covariance
        self.R = np.diag([r_usbl**2, r_usbl**2])
        
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 0.5, 0.5])
        self.initialized = False

    def reset(self, init_north: float, init_east: float, init_speed: float, init_heading: float):
        vx = init_speed * np.cos(init_heading)
        vy = init_speed * np.sin(init_heading)
        self.x = np.array([init_north, init_east, vx, vy], dtype=np.float64)
        self.P = np.diag([0.25, 0.25, 0.1, 0.1])
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measured_north: float, measured_east: float):
        z = np.array([measured_north, measured_east], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])


class OODDetector:
    """
    Out-Of-Distribution (OOD) Monitor for 6D GRU Feature Inputs:
    [GroundSpeed, Roll, Pitch, cos(Heading), sin(Heading), Depth]
    Uses train-split bounding boxes and Mahalanobis distance metric.
    """
    def __init__(self, train_mean: np.ndarray, train_cov: np.ndarray, feature_bounds: Dict[str, Tuple[float, float]]):
        self.mean = np.array(train_mean, dtype=np.float64)
        self.inv_cov = np.linalg.pinv(np.array(train_cov, dtype=np.float64))
        self.bounds = feature_bounds
        # Mahalanobis threshold (chi-squared at 99.9% for 6 DOF is ~22.46)
        self.mahalanobis_threshold = 22.46

    def check(self, feature_vec: np.ndarray) -> Dict[str, Any]:
        """
        feature_vec: shape (6,) [Speed, Roll, Pitch, cos(Heading), sin(Heading), Depth]
        """
        diff = feature_vec - self.mean
        d_m = float(np.sqrt(np.dot(diff, np.dot(self.inv_cov, diff))))
        
        is_ood_mahalanobis = d_m > self.mahalanobis_threshold
        
        # Bounding box checks
        box_violations = []
        feat_keys = ["GroundSpeed", "Roll", "Pitch", "cos(Heading)", "sin(Heading)", "Depth"]
        for i, k in enumerate(feat_keys):
            b_min, b_max = self.bounds[k]
            if feature_vec[i] < b_min or feature_vec[i] > b_max:
                box_violations.append(k)
                
        is_ood = is_ood_mahalanobis or len(box_violations) > 0
        return {
            "is_ood": is_ood,
            "mahalanobis_dist": d_m,
            "box_violations": box_violations
        }


class SensorMeasurementLayer:
    """
    Simulates Realistic Multi-Sensor Telemetry Acquisition for AUV:
    1. USBL acoustic coordinates with Gaussian jitter, heavy-tailed outliers, and dropouts.
    2. Digiquartz pressure depth sensor noise.
    3. PHINS inertial navigation attitude noise.
    4. Doppler Velocity Log (DVL) bottom-track speed noise.
    5. Integrated Kalman filtering mode vs raw sensor feedback.
    6. Continuous OOD monitor logging.
    """
    def __init__(
        self,
        usbl_sigma: float = 0.50,
        depth_sigma: float = 0.05,
        attitude_sigma: float = 0.005,
        dvl_sigma: float = 0.02,
        use_kalman_filter: bool = True,
        dropout_probability: float = 0.0,
        ood_detector: Optional[OODDetector] = None,
        seed: Optional[int] = None
    ):
        self.usbl_sigma = usbl_sigma
        self.depth_sigma = depth_sigma
        self.attitude_sigma = attitude_sigma
        self.dvl_sigma = dvl_sigma
        self.use_kf = use_kalman_filter
        self.dropout_prob = dropout_probability
        self.ood_detector = ood_detector
        self.rng = np.random.default_rng(seed)
        
        self.kf = PositionKalmanFilter(dt=2.0, r_usbl=usbl_sigma)
        self.dropout_active = False
        self.last_valid_usbl = (0.0, 0.0)

    def reset(self, initial_state: AUVState):
        self.kf.reset(initial_state.north, initial_state.east, initial_state.speed, initial_state.heading)
        self.last_valid_usbl = (initial_state.north, initial_state.east)
        self.dropout_active = False

    def measure(self, true_state: AUVState) -> Tuple[AUVState, np.ndarray, Dict[str, Any]]:
        """
        Returns:
        1. measured_state (AUVState used by controller)
        2. continuous_features (6D feature vector for GRU history buffer)
        3. diagnostic_info
        """
        # 1. USBL measurement (North, East)
        is_dropout = self.rng.random() < self.dropout_prob
        
        if is_dropout:
            # Acoustic dropout: transponder return lost
            measured_north, measured_east = self.last_valid_usbl
        else:
            noise_ne = self.rng.normal(0.0, self.usbl_sigma, size=2)
            measured_north = true_state.north + noise_ne[0]
            measured_east = true_state.east + noise_ne[1]
            self.last_valid_usbl = (measured_north, measured_east)

        # 2. Kalman filter update or raw
        self.kf.predict()
        if not is_dropout:
            self.kf.update(measured_north, measured_east)
            
        if self.use_kf:
            est_north, est_east = self.kf.get_position()
        else:
            est_north, est_east = measured_north, measured_east

        # 3. Attitude & Speed sensors
        measured_depth = float(true_state.depth + self.rng.normal(0.0, self.depth_sigma))
        measured_roll = float(true_state.roll + self.rng.normal(0.0, self.attitude_sigma))
        measured_pitch = float(true_state.pitch + self.rng.normal(0.0, self.attitude_sigma))
        measured_heading = float(wrap_to_2pi(true_state.heading + self.rng.normal(0.0, self.attitude_sigma)))
        measured_speed = float(np.maximum(0.0, true_state.speed + self.rng.normal(0.0, self.dvl_sigma)))

        measured_state = AUVState(
            north=est_north,
            east=est_east,
            depth=measured_depth,
            roll=measured_roll,
            pitch=measured_pitch,
            heading=measured_heading,
            speed=measured_speed
        )

        # Continuous 6D feature vector for GRU
        features_6d = np.array([
            measured_speed,
            measured_roll,
            measured_pitch,
            np.cos(measured_heading),
            np.sin(measured_heading),
            measured_depth
        ], dtype=np.float64)

        # OOD Check
        ood_info = {"is_ood": False, "mahalanobis_dist": 0.0, "box_violations": []}
        if self.ood_detector:
            ood_info = self.ood_detector.check(features_6d)

        diagnostics = {
            "usbl_raw_north": measured_north,
            "usbl_raw_east": measured_east,
            "is_dropout": is_dropout,
            "used_kf": self.use_kf,
            "ood": ood_info
        }

        return measured_state, features_6d, diagnostics
