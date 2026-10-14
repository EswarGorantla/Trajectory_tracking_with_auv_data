import pandas as pd
import numpy as np
import os

class Preprocessor:
    def __init__(self, numeric_cols=None):
        if numeric_cols is None:
            self.numeric_cols = ['Seconds', 'Latitude', 'Longitude', 'depth', 'altitude', 'Roll', 'Pitch', 'Heading', 'WaterSpeed', 'GroundSpeed']
        else:
            self.numeric_cols = numeric_cols

    def clean_data(self, df):
        """Cleans, sorts, and checks telemetry data for duplicates and missing values."""
        df_clean = df.copy()
        
        # Convert numeric columns
        for col in self.numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Drop rows with NaN in key fields
        key_cols = ['Date', 'Time'] + self.numeric_cols
        df_clean = df_clean.dropna(subset=key_cols).copy()
        
        # Combine date/time and parse
        df_clean['DateTime'] = pd.to_datetime(
            df_clean['Date'].astype(str).str.strip() + ' ' + df_clean['Time'].astype(str).str.strip(),
            format='%d-%b-%Y %H:%M:%S',
            errors='coerce'
        )
        df_clean = df_clean.dropna(subset=['DateTime']).copy()
        
        # Sort chronologically and drop duplicates
        df_clean = df_clean.sort_values('DateTime').reset_index(drop=True)
        df_clean = df_clean.drop_duplicates(subset=['DateTime']).reset_index(drop=True)
        
        # Simple outlier detection (clip extreme values)
        if 'GroundSpeed' in df_clean.columns:
            df_clean['GroundSpeed'] = np.clip(df_clean['GroundSpeed'], 0.0, 3.0)
        if 'depth' in df_clean.columns:
            df_clean['depth'] = np.clip(df_clean['depth'], 0.0, 6000.0)
            
        print(f"Cleaned dataset rows: {len(df_clean)}")
        return df_clean

    def split_chronologically(self, df, train_ratio=0.70, val_ratio=0.15):
        """Splits the dataframe into chronological train/val/test splits."""
        N = len(df)
        train_end = int(N * train_ratio)
        val_end = train_end + int(N * val_ratio)
        
        train_df = df.iloc[0:train_end].copy().reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)
        
        print(f"Splits generated: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
