import pandas as pd
import numpy as np
import os

# Set file paths
csv_path = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\original_data\M86\multibeam_echosounder\multi-beam-echosounder-index.csv"
output_dir = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\splits"
os.makedirs(output_dir, exist_ok=True)

# 1. Load and clean data
print("Loading dataset...")
df = pd.read_csv(csv_path, low_memory=False)
df.columns = df.columns.str.strip()

numeric_cols = ['Seconds', 'Latitude', 'Longitude', 'depth', 'altitude', 'Roll', 'Pitch', 'Heading', 'WaterSpeed', 'GroundSpeed']

# Convert numeric columns
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaNs in key navigation columns
df = df.dropna(subset=['Date', 'Time'] + numeric_cols).copy()

df['DateTime'] = pd.to_datetime(df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip(), format='%d-%b-%Y %H:%M:%S')
df = df.sort_values('DateTime').reset_index(drop=True)

total_rows = len(df)
print(f"Total cleaned rows: {total_rows}")

# 2. Define ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Calculate split index cutoffs
train_end = int(total_rows * train_ratio)
val_end = train_end + int(total_rows * val_ratio)

# Split the data chronologically
train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

# Save splits to CSV
train_path = os.path.join(output_dir, "train.csv")
val_path = os.path.join(output_dir, "val.csv")
test_path = os.path.join(output_dir, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

# 3. Print details and check for overlap
print("\n=== Split Summary ===")
print(f"Train Set:      {len(train_df)} rows ({len(train_df)/total_rows*100:.2f}%)")
print(f"  Row Range:    0 to {train_end - 1}")
print(f"  Time Range:   {train_df['DateTime'].min()} to {train_df['DateTime'].max()}")

print(f"Validation Set: {len(val_df)} rows ({len(val_df)/total_rows*100:.2f}%)")
print(f"  Row Range:    {train_end} to {val_end - 1}")
print(f"  Time Range:   {val_df['DateTime'].min()} to {val_df['DateTime'].max()}")

print(f"Test Set:       {len(test_df)} rows ({len(test_df)/total_rows*100:.2f}%)")
print(f"  Row Range:    {val_end} to {total_rows - 1}")
print(f"  Time Range:   {test_df['DateTime'].min()} to {test_df['DateTime'].max()}")

# Confirmations
print("\n=== Verification Checks ===")
# 1. Total count check
sum_rows = len(train_df) + len(val_df) + len(test_df)
count_match = sum_rows == total_rows
print(f"  Row counts sum to total: {count_match} ({sum_rows} / {total_rows})")

# 2. Chronological order & no overlap checks
check1 = train_df['DateTime'].max() < val_df['DateTime'].min()
check2 = val_df['DateTime'].max() < test_df['DateTime'].min()
overlap_free = check1 and check2
print(f"  Train max time < Val min time: {check1} ({train_df['DateTime'].max()} < {val_df['DateTime'].min()})")
print(f"  Val max time < Test min time: {check2} ({val_df['DateTime'].max()} < {test_df['DateTime'].min()})")
print(f"  Temporal Overlap-Free: {overlap_free}")

if count_match and overlap_free:
    print("\nSUCCESS: Dataset split successfully by chronological order with ZERO overlap.")
else:
    print("\nERROR: Verification failed.")
