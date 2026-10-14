import numpy as np
import pandas as pd
from typing import Tuple

class StateDefinition:
    """Manages state vector extraction, continuous angular features, and temporal sequences."""

    STATE_COLUMNS = ["North", "East", "depth", "Roll", "Pitch", "Heading", "GroundSpeed"]
    FEATURE_COLUMNS = ["GroundSpeed", "Roll", "Pitch", "cos(Heading)", "sin(Heading)", "depth"]

    @staticmethod
    def wrap_angle(angle: np.ndarray) -> np.ndarray:
        """Wraps angular differences to the interval [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @classmethod
    def extract_state_matrix(cls, df: pd.DataFrame) -> np.ndarray:
        """Extracts (N, 7) state matrix: [North, East, Depth, Roll, Pitch, Heading, Speed]."""
        return np.column_stack([
            df["North"].values.astype(np.float64),
            df["East"].values.astype(np.float64),
            df["depth"].values.astype(np.float64),
            df["Roll"].values.astype(np.float64),
            df["Pitch"].values.astype(np.float64),
            df["Heading"].values.astype(np.float64),
            df["GroundSpeed"].values.astype(np.float64)
        ])

    @classmethod
    def extract_feature_matrix(cls, state_mat: np.ndarray) -> np.ndarray:
        """
        Converts (N, 7) state matrix to (N, 6) continuous feature matrix:
        [Speed, Roll, Pitch, cos(Heading), sin(Heading), Depth]
        """
        speed = state_mat[:, 6]
        roll = state_mat[:, 3]
        pitch = state_mat[:, 4]
        heading = state_mat[:, 5]
        depth = state_mat[:, 2]

        return np.column_stack([
            speed,
            roll,
            pitch,
            np.cos(heading),
            np.sin(heading),
            depth
        ])

    @staticmethod
    def build_sequence_windows(features: np.ndarray, targets: np.ndarray, seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds rolling temporal history sequences of length seq_len.
        features: (N, D_in) -> X_seq: (N - seq_len + 1, seq_len, D_in)
        targets:  (N, D_out) -> y_seq: (N - seq_len + 1, D_out)
        """
        num_samples = len(features) - seq_len + 1
        X_seq = np.zeros((num_samples, seq_len, features.shape[1]), dtype=np.float32)
        y_seq = np.zeros((num_samples, targets.shape[1]), dtype=np.float32)

        for i in range(num_samples):
            X_seq[i] = features[i : i + seq_len]
            y_seq[i] = targets[i + seq_len - 1]

        return X_seq, y_seq
