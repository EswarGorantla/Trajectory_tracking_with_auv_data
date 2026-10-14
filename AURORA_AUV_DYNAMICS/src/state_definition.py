import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

class StateEstimator:
    """Handles coordinate frame transformations, geographic projections, and state representations."""
    
    R_EARTH = 6371000.0  # Mean Earth radius in meters

    def __init__(self):
        self.lat0 = None
        self.lon0 = None
        self.state_names = ["North", "East", "depth", "Roll", "Pitch", "Heading", "GroundSpeed"]
        self.feature_names = ["GroundSpeed", "Roll", "Pitch", "cos(Heading)", "sin(Heading)", "depth"]

    def fit_origin(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Fits local tangent plane origin from the initial coordinates."""
        self.lat0 = float(df['Latitude'].iloc[0])
        self.lon0 = float(df['Longitude'].iloc[0])
        return self.lat0, self.lon0

    def to_cartesian(self, df: pd.DataFrame) -> pd.DataFrame:
        """Projects Latitude and Longitude to North (x) and East (y) Cartesian meters."""
        if self.lat0 is None or self.lon0 is None:
            self.fit_origin(df)

        df_out = df.copy()
        lat_rad = np.radians(df['Latitude'].values)
        lon_rad = np.radians(df['Longitude'].values)
        lat0_rad = np.radians(self.lat0)
        lon0_rad = np.radians(self.lon0)

        # Equirectangular projection
        north = (lat_rad - lat0_rad) * self.R_EARTH
        east = (lon_rad - lon0_rad) * self.R_EARTH * np.cos(lat0_rad)

        df_out['North'] = north
        df_out['East'] = east
        return df_out

    @staticmethod
    def wrap_angle(angle: np.ndarray) -> np.ndarray:
        """Wraps angular values to the interval [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def extract_state_vector(df: pd.DataFrame) -> np.ndarray:
        """Extracts the 7D state matrix: [x, y, z, roll, pitch, heading, speed]."""
        return np.column_stack([
            df['North'].values,
            df['East'].values,
            df['depth'].values,
            df['Roll'].values,
            df['Pitch'].values,
            df['Heading'].values,
            df['GroundSpeed'].values
        ])

    @staticmethod
    def extract_features(state_matrix: np.ndarray) -> np.ndarray:
        """
        Converts 7D state matrix into 6D neural network features:
        [Speed, Roll, Pitch, cos(Heading), sin(Heading), Depth]
        """
        # state_matrix columns: 0:North, 1:East, 2:depth, 3:Roll, 4:Pitch, 5:Heading, 6:Speed
        speed = state_matrix[:, 6]
        roll = state_matrix[:, 3]
        pitch = state_matrix[:, 4]
        heading = state_matrix[:, 5]
        depth = state_matrix[:, 2]

        return np.column_stack([
            speed,
            roll,
            pitch,
            np.cos(heading),
            np.sin(heading),
            depth
        ])
