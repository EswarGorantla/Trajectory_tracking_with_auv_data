import numpy as np
import pandas as pd

class StateEstimator:
    def __init__(self, lat0=None, lon0=None):
        self.R = 6371000.0  # Earth radius in meters
        self.lat0 = lat0
        self.lon0 = lon0
        
    def fit_origin(self, df):
        """Fits reference latitude/longitude to the first row of training data."""
        self.lat0 = np.radians(df['Latitude'].iloc[0])
        self.lon0 = np.radians(df['Longitude'].iloc[0])
        print(f"Origin fit to Lat0: {np.degrees(self.lat0):.6f}, Lon0: {np.degrees(self.lon0):.6f}")
        
    def to_cartesian(self, df):
        """Converts Latitude/Longitude to North/East coordinates relative to origin."""
        if self.lat0 is None or self.lon0 is None:
            raise ValueError("Origin must be fit using fit_origin() before calling to_cartesian()")
        df_out = df.copy()
        
        lat_rad = np.radians(df_out['Latitude'])
        lon_rad = np.radians(df_out['Longitude'])
        
        df_out['North'] = self.R * (lat_rad - self.lat0)
        df_out['East'] = self.R * (lon_rad - self.lon0) * np.cos(self.lat0)
        return df_out

    @staticmethod
    def wrap_angle(angle):
        """Wraps angle (radians) to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def wrap_to_2pi(angle):
        """Wraps angle (radians) to [0, 2*pi]"""
        return angle % (2 * np.pi)
