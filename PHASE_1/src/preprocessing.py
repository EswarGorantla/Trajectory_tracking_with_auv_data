import numpy as np
import pandas as pd
from typing import Tuple

class Preprocessor:
    """Handles data cleaning, outlier removal, chronological splitting, and windowing."""

    def __init__(self):
        self.required_columns = [
            'Latitude', 'Longitude', 'depth', 'Roll', 'Pitch', 'Heading', 'GroundSpeed'
        ]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts telemetry columns to numeric and removes missing rows."""
        df_clean = df.copy()
        for col in self.required_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        df_clean = df_clean.dropna(subset=self.required_columns).copy()

        # Physical range checks
        df_clean = df_clean[
            (df_clean['GroundSpeed'] >= 0.0) & (df_clean['GroundSpeed'] <= 4.0) &
            (df_clean['depth'] >= 0.0) & (df_clean['depth'] <= 6000.0)
        ].reset_index(drop=True)

        print(f"Cleaned dataset rows: {len(df_clean)} (from {len(df)} raw rows)")
        return df_clean

    def split_chronologically(self, df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits time-series chronologically to guarantee zero future data leakage.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)

        print(f"Chronological Splits Generated:")
        print(f"  Train:      {len(train_df)} rows ({len(train_df)/n*100:.1f}%)")
        print(f"  Validation: {len(val_df)} rows ({len(val_df)/n*100:.1f}%)")
        print(f"  Test:       {len(test_df)} rows ({len(test_df)/n*100:.1f}%) [UNSEEN]")

        return train_df, val_df, test_df

    @staticmethod
    def create_sequence_windows(features: np.ndarray, targets: np.ndarray, seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds rolling temporal history sequences of length seq_len.
        features: (N, D_in) -> X_seq: (N - seq_len + 1, seq_len, D_in)
        targets: (N, D_out) -> y_seq: (N - seq_len + 1, D_out)
        """
        num_samples = len(features) - seq_len + 1
        X_seq = np.zeros((num_samples, seq_len, features.shape[1]), dtype=np.float32)
        y_seq = np.zeros((num_samples, targets.shape[1]), dtype=np.float32)

        for i in range(num_samples):
            X_seq[i] = features[i : i + seq_len]
            y_seq[i] = targets[i + seq_len - 1]

        return X_seq, y_seq
