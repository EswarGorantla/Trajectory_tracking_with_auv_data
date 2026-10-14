import os
import pandas as pd
from typing import Tuple, List

class DataLoader:
    """Loads processed EDA dataset splits and reports dataset specifications."""

    def __init__(self, processed_dir: str):
        self.processed_dir = processed_dir
        self.input_features = [
            "GroundSpeed", "Roll", "Pitch", "cos(Heading)", "sin(Heading)", "depth"
        ]
        self.target_features = [
            "North", "East", "depth", "Roll", "Pitch", "Heading", "GroundSpeed"
        ]
        self.seq_len = 10
        self.dt = 2.0
        self.sampling_rate_hz = 0.5

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_path = os.path.join(self.processed_dir, "train.csv")
        val_path = os.path.join(self.processed_dir, "val.csv")
        test_path = os.path.join(self.processed_dir, "test.csv")

        for p in [train_path, val_path, test_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing processed data split at: {p}")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # Print required Section 1 verification block
        print("============================================================")
        print("1. MODEL-READY DATASET SPECIFICATIONS (FROM EDA)")
        print("============================================================")
        print("FINAL INPUT FEATURES:")
        print(self.input_features)
        print("\nFINAL TARGET FEATURES:")
        print(self.target_features)
        print(f"\nSEQUENCE LENGTH:\n{self.seq_len} timesteps ({self.seq_len * self.dt:.1f} seconds)")
        print(f"\nSAMPLING RATE:\n{self.sampling_rate_hz} Hz (dt = {self.dt} s)")
        print(f"\nTRAIN SAMPLES:\n{len(train_df)}")
        print(f"\nVALIDATION SAMPLES:\n{len(val_df)}")
        print(f"\nTEST SAMPLES:\n{len(test_df)}")
        print("============================================================\n")

        return train_df, val_df, test_df
