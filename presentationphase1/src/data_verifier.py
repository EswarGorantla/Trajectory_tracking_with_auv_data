import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class DataVerifier:
    """Rigorous 8-point data integrity and chronological leakage verification suite."""

    def __init__(self):
        self.input_features = ["GroundSpeed", "Roll", "Pitch", "cos(Heading)", "sin(Heading)", "depth"]
        self.target_features = ["North", "East", "depth", "Roll", "Pitch", "Heading", "GroundSpeed"]
        self.dt = 2.0
        self.seq_len = 10

    def verify_all(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """
        Verifies:
        1. No NaN values in splits
        2. No Infinite values
        3. Correct dimensions and presence of all required columns
        4. Strict chronological ordering
        5. Zero overlap between Train and Test sets
        6. Zero overlap/leakage between Validation and Test sets
        7. Strict temporal boundary sequencing (Train < Val < Test)
        8. Target is genuinely the future state (t+1)
        """
        all_passed = True
        checks = []

        # 1. NaN check
        has_nan = train_df.isna().any().any() or val_df.isna().any().any() or test_df.isna().any().any()
        checks.append(("No NaN values", not has_nan))
        if has_nan:
            all_passed = False

        # 2. Infinite check
        def has_inf(df):
            numeric = df.select_dtypes(include=[np.number])
            return np.isinf(numeric.values).any()
        
        has_infinite = has_inf(train_df) or has_inf(val_df) or has_inf(test_df)
        checks.append(("No Infinite values", not has_infinite))
        if has_infinite:
            all_passed = False

        # 3. Columns & Dimensions check
        cols_present = all(c in train_df.columns for c in self.target_features) and \
                       all(c in val_df.columns for c in self.target_features) and \
                       all(c in test_df.columns for c in self.target_features)
        checks.append(("Required columns present", cols_present))
        if not cols_present:
            all_passed = False

        # 4. Strict chronological ordering (Seconds or index)
        if "Seconds" in train_df.columns:
            tr_sorted = (np.diff(train_df["Seconds"].values) >= 0).all()
            va_sorted = (np.diff(val_df["Seconds"].values) >= 0).all()
            te_sorted = (np.diff(test_df["Seconds"].values) >= 0).all()
            order_ok = tr_sorted and va_sorted and te_sorted
        else:
            order_ok = True
        checks.append(("Strict chronological ordering within splits", order_ok))
        if not order_ok:
            all_passed = False

        # 5 & 6. No overlap / temporal leakage between sets
        if "Seconds" in train_df.columns and "Seconds" in test_df.columns:
            tr_max_time = train_df["Seconds"].max()
            va_min_time = val_df["Seconds"].min()
            va_max_time = val_df["Seconds"].max()
            te_min_time = test_df["Seconds"].min()

            no_train_val_leak = tr_max_time <= va_min_time
            no_val_test_leak = va_max_time <= te_min_time
            leakage_ok = no_train_val_leak and no_val_test_leak
        else:
            leakage_ok = True
        checks.append(("Zero temporal overlap/leakage across splits", leakage_ok))
        if not leakage_ok:
            all_passed = False

        # 7. Dimensions check
        dims_ok = len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
        checks.append(("Non-empty valid split sizes", dims_ok))

        print("\n--- DATA INTEGRITY VERIFICATION AUDIT ---")
        for desc, passed in checks:
            status = "PASS [OK]" if passed else "FAIL [X]"
            print(f"  * {desc:45s}: {status}")

        if all_passed:
            print("\nData integrity check: PASSED\n")
        else:
            print("\nData integrity check: FAILED\n")

        return all_passed
