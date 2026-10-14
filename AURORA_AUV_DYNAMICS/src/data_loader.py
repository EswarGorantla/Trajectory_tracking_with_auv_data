import os
import pandas as pd
from typing import Tuple

class DataLoader:
    """Handles raw data ingestion from IEEE DataPort AURORA files and saves processed splits."""

    def __init__(self, raw_csv_path: str, processed_dir: str):
        self.raw_csv_path = raw_csv_path
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_raw_data(self) -> pd.DataFrame:
        """Loads raw CSV data and strips header whitespaces."""
        if not os.path.exists(self.raw_csv_path):
            raise FileNotFoundError(f"Raw data file not found at: {self.raw_csv_path}")
        
        print(f"Loading raw AUV telemetry from {self.raw_csv_path}...")
        df = pd.read_csv(self.raw_csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        return df

    def save_processed_split(self, df: pd.DataFrame, split_name: str) -> str:
        """Saves a processed split CSV (train, val, test)."""
        out_path = os.path.join(self.processed_dir, f"{split_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {split_name} split ({len(df)} rows) to {out_path}")
        return out_path
