import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Set paths
root_dir = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5"
p1_dir = os.path.join(root_dir, "presentationphase1")
p2_dir = os.path.join(root_dir, "Phase_2")
sys.path.insert(0, p1_dir)
sys.path.insert(0, p2_dir)

from src.data_loader import DataLoader
from src.state_definition import StateDefinition
from src.physics_model import PhysicsModel

print("=== AUDIT SCRIPT STARTING ===")

# 1. DATA SPLIT INTEGRITY
loader = DataLoader(os.path.join(p1_dir, "data", "processed"))
train_df, val_df, test_df = loader.load_splits()

n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
print(f"Split counts: Train={n_train}, Val={n_val}, Test={n_test}")

# Check index monotonicity
train_max_idx = train_df.index.max()
val_min_idx = val_df.index.min()
val_max_idx = val_df.index.max()
test_min_idx = test_df.index.min()
test_max_idx = test_df.index.max()

print(f"Indices: Train max={train_max_idx}, Val min={val_min_idx}, Val max={val_max_idx}, Test min={test_min_idx}, Test max={test_max_idx}")
split_monotonic = (train_max_idx < val_min_idx) and (val_max_idx < test_min_idx)
print(f"Monotonic strictly: {split_monotonic}")

# Check timestamps if present
if "timestamp" in train_df.columns:
    print(f"Timestamps: Train end={train_df['timestamp'].iloc[-1]}, Val start={val_df['timestamp'].iloc[0]}, Val end={val_df['timestamp'].iloc[-1]}, Test start={test_df['timestamp'].iloc[0]}")

# Scaler verification
scaler_path = os.path.join(p1_dir, "models", "scaler.pkl")
with open(scaler_path, "rb") as f:
    saved_scalers = pickle.load(f)

print(f"Saved scalers keys: {saved_scalers.keys() if isinstance(saved_scalers, dict) else type(saved_scalers)}")

# Compute train features and check scaler statistics
train_state_mat = StateDefinition.extract_state_matrix(train_df)
train_feat_mat = StateDefinition.extract_feature_matrix(train_state_mat)[:-1]

if isinstance(saved_scalers, dict):
    feat_scaler = saved_scalers.get("feature_scaler")
    target_scaler = saved_scalers.get("target_scaler")
    print(f"Feature scaler mean in pkl: {feat_scaler.mean_}")
    print(f"Feature scaler var in pkl:  {feat_scaler.var_}")
    train_computed_mean = np.mean(train_feat_mat, axis=0)
    train_computed_var = np.var(train_feat_mat, axis=0)
    print(f"Train recomputed mean:     {train_computed_mean}")
    print(f"Train recomputed var:      {train_computed_var}")
    print(f"Feature scaler mean diff:  {np.max(np.abs(feat_scaler.mean_ - train_computed_mean)):.2e}")
    print(f"Feature scaler var diff:   {np.max(np.abs(feat_scaler.var_ - train_computed_var)):.2e}")

# Kd Refit on Train
k_d_refit = PhysicsModel.fit_drag_coefficient_on_train(train_df["GroundSpeed"].values, dt=2.0, u_cruise=1.2)
print(f"Kd refit on train: {k_d_refit:.8f}")

# 2. ARCHITECTURE TRUTH
ckpt_path = os.path.join(p1_dir, "models", "best_dynamics_model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
print("Checkpoint keys / type:", type(ckpt))
if isinstance(ckpt, dict):
    state_dict = ckpt if "state_dict" not in ckpt else ckpt["state_dict"]
else:
    state_dict = ckpt

total_params = 0
print("\n--- Checkpoint Parameters ---")
for k, v in state_dict.items():
    print(f"{k}: shape={tuple(v.shape)}, numel={v.numel()}, dtype={v.dtype}")
    total_params += v.numel()
print(f"Total parameters in checkpoint: {total_params:,}")

# Check for LayerNorm in state_dict
has_layernorm = any("norm" in k.lower() or "ln" in k.lower() for k in state_dict.keys())
print(f"Has LayerNorm keys: {has_layernorm}")

# Check model definition in code
with open(os.path.join(p1_dir, "src", "dynamics_model.py"), "r") as f:
    dyn_code = f.read()
print("Contains LayerNorm in code?", "LayerNorm" in dyn_code)

# 3. HYBRID DEFINITION TRUTH IN CODE
with open(os.path.join(p1_dir, "src", "residual_model.py"), "r") as f:
    res_code = f.read()
print("\n--- Phase 1 residual_model.py excerpt ---")
for line in res_code.splitlines():
    if "predict" in line or "phys" in line or "residual" in line or "next_state" in line:
        print("  ", line)

with open(os.path.join(p2_dir, "src", "dynamics_interface.py"), "r") as f:
    dyn_int_code = f.read()
print("\n--- Phase 2 dynamics_interface.py residual addition excerpt ---")
for line in dyn_int_code.splitlines()[80:110]:
    print("  ", line)

print("\n=== AUDIT SCRIPT PART 1 COMPLETE ===")
