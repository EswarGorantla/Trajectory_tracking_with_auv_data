import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, raw_csv_path, processed_dir):
        self.raw_csv_path = raw_csv_path
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
    def load_raw_data(self):
        """Loads and returns the raw CSV data."""
        if not os.path.exists(self.raw_csv_path):
            raise FileNotFoundError(f"Raw CSV data not found at {self.raw_csv_path}")
        print(f"Loading raw AUV telemetry from {self.raw_csv_path}...")
        df = pd.read_csv(self.raw_csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        return df

    def save_processed_split(self, df, split_name):
        """Saves a dataframe split to the processed directory."""
        path = os.path.join(self.processed_dir, f"{split_name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved {split_name} split to {path}")

    def load_processed_split(self, split_name):
        """Loads a processed split dataframe."""
        path = os.path.join(self.processed_dir, f"{split_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed split {split_name} not found at {path}")
        return pd.read_csv(path)
