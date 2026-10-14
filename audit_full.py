import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

root_dir = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5"
p1_dir = os.path.join(root_dir, "presentationphase1")
p2_dir = os.path.join(root_dir, "Phase_2")
sys.path.insert(0, p1_dir)
sys.path.insert(0, p2_dir)

from src.data_loader import DataLoader
from src.state_definition import StateDefinition
from src.physics_model import PhysicsModel
from src.residual_model import HybridDynamicsModel
from src.dynamics_model import GRUModel
from src.predict import Predictor
from src.evaluate import ModelEvaluator

audit_results = {}

# 1. SPLIT INTEGRITY & SOURCE CHECK
raw_splits_dir = os.path.join(root_dir, "splits")
p1_proc_dir = os.path.join(p1_dir, "data", "processed")

train_df = pd.read_csv(os.path.join(p1_proc_dir, "train.csv"), low_memory=False)
val_df = pd.read_csv(os.path.join(p1_proc_dir, "val.csv"), low_memory=False)
test_df = pd.read_csv(os.path.join(p1_proc_dir, "test.csv"), low_memory=False)

audit_results["splits"] = {
    "train_len": len(train_df),
    "val_len": len(val_df),
    "test_len": len(test_df),
    "total_len": len(train_df) + len(val_df) + len(test_df),
}

# Check if original index or timestamp exists
time_col = None
for col in ["timestamp", "time", "Time", "index", "Unnamed: 0"]:
    if col in train_df.columns:
        time_col = col
        break

audit_results["splits"]["time_col"] = time_col
if time_col:
    audit_results["splits"]["train_time_range"] = [str(train_df[time_col].iloc[0]), str(train_df[time_col].iloc[-1])]
    audit_results["splits"]["val_time_range"] = [str(val_df[time_col].iloc[0]), str(val_df[time_col].iloc[-1])]
    audit_results["splits"]["test_time_range"] = [str(test_df[time_col].iloc[0]), str(test_df[time_col].iloc[-1])]
    # Check strict chronological progression between splits
    audit_results["splits"]["train_val_chronological"] = bool(train_df[time_col].iloc[-1] < val_df[time_col].iloc[0])
    audit_results["splits"]["val_test_chronological"] = bool(val_df[time_col].iloc[-1] < test_df[time_col].iloc[0])

# Verify Scaler
scaler_path = os.path.join(p1_dir, "models", "scaler.pkl")
with open(scaler_path, "rb") as f:
    saved_scalers = pickle.load(f)

feat_scaler = saved_scalers["feature_scaler"]
target_scaler = saved_scalers["target_scaler"]

train_state_mat = StateDefinition.extract_state_matrix(train_df)
train_feat_mat = StateDefinition.extract_feature_matrix(train_state_mat)[:-1]

audit_results["scaler"] = {
    "feature_mean_saved": feat_scaler.mean_.tolist(),
    "feature_var_saved": feat_scaler.var_.tolist(),
    "feature_mean_computed": np.mean(train_feat_mat, axis=0).tolist(),
    "feature_var_computed": np.var(train_feat_mat, axis=0).tolist(),
    "max_abs_diff_mean": float(np.max(np.abs(feat_scaler.mean_ - np.mean(train_feat_mat, axis=0)))),
    "max_abs_diff_var": float(np.max(np.abs(feat_scaler.var_ - np.var(train_feat_mat, axis=0))))
}

# Refit Kd on train
kd_refit = float(PhysicsModel.fit_drag_coefficient_on_train(train_df["GroundSpeed"].values, dt=2.0, u_cruise=1.2))
audit_results["kd"] = {
    "refit": kd_refit,
    "claimed": 0.06073,
    "diff": abs(kd_refit - 0.06073)
}

# 2. ARCHITECTURE TRUTH
ckpt_path = os.path.join(p1_dir, "models", "best_dynamics_model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt if not isinstance(ckpt, dict) or "state_dict" not in ckpt else ckpt["state_dict"]

param_details = {}
total_params = 0
for k, v in state_dict.items():
    param_details[k] = list(v.shape)
    total_params += v.numel()

audit_results["architecture"] = {
    "total_params": total_params,
    "param_shapes": param_details,
    "has_layernorm_weights": any("norm" in k.lower() or "ln" in k.lower() for k in state_dict.keys()),
    "fc_layers": [k for k in state_dict.keys() if "fc" in k]
}

# 3. HYBRID DEFINITION IN CODE
with open(os.path.join(p1_dir, "src", "residual_model.py"), "r", encoding="utf-8") as f:
    p1_res_code = f.read()

with open(os.path.join(p2_dir, "src", "dynamics_interface.py"), "r", encoding="utf-8") as f:
    p2_dyn_code = f.read()

audit_results["hybrid_code"] = {
    "p1_north_east_residual_used": "hybrid_next[0] += residuals[0]" in p1_res_code,
    "p2_north_east_residual_used": "x_next = phys_next[0] + residuals[0]" in p2_dyn_code,
    "p2_pitch_damping_scaling": "theta_next = theta_act + residuals[4] * 0.5" in p2_dyn_code,
    "p2_yaw_damping_scaling": "psi_next = wrap_to_2pi(psi_act + residuals[5] * 0.5)" in p2_dyn_code
}

# 4. RE-RUN 1-STEP TEST EVALUATION
physics = PhysicsModel(k_d=kd_refit, dt=2.0, u_cruise=1.2)
device = torch.device("cpu")
model = GRUModel(input_dim=6, hidden_dim=64, output_dim=7, num_layers=2)
model.load_state_dict(state_dict)
model.eval()

hybrid_model = HybridDynamicsModel(physics, model, feat_scaler, target_scaler, device)
predictor = Predictor(dt=2.0, seq_len=10)

# Evaluate on test_df
evaluator = ModelEvaluator(os.path.join(p1_dir, "results"))
df_preds, raw_preds = predictor.evaluate_test_split(
    test_df, hybrid_model, model, feat_scaler, target_scaler, device
)

comp_df = evaluator.compute_comparison_table(raw_preds)
audit_results["one_step_test_metrics"] = comp_df.to_dict(orient="records")

# 5. FEATURE DISTRIBUTION & OOD CHECK (Train set percentiles vs Phase 2 operating range)
feature_names = ["GroundSpeed", "Roll", "Pitch", "cos(Heading)", "sin(Heading)", "Depth"]
feat_stats = {}
for i, name in enumerate(feature_names):
    vals = train_feat_mat[:, i]
    feat_stats[name] = {
        "min": float(np.min(vals)),
        "p01": float(np.percentile(vals, 1)),
        "p05": float(np.percentile(vals, 5)),
        "p50": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals))
    }
audit_results["feature_distributions"] = feat_stats

# Phase 2 scenario ranges check:
# Speed: [0.5, 2.0], Depth: [10.0, 60.0], Pitch: [-0.35, 0.35], Roll: [-0.5, 0.5]
p2_speed_ood = float(np.mean((train_feat_mat[:, 0] < 0.5) | (train_feat_mat[:, 0] > 2.0)))
p2_depth_train_min = float(np.min(train_feat_mat[:, 5]))
p2_depth_train_max = float(np.max(train_feat_mat[:, 5]))
audit_results["ood_analysis"] = {
    "train_depth_range": [p2_depth_train_min, p2_depth_train_max],
    "phase2_specified_depth_range": [10.0, 60.0],
    "train_speed_range": [float(np.min(train_feat_mat[:, 0])), float(np.max(train_feat_mat[:, 0]))],
    "phase2_specified_speed_range": [0.5, 2.0]
}

# 6. SIGN CONVENTIONS
# Depth: z_{t+1} = z_t - U * sin(theta) * dt
# If theta > 0 (climbing), sin(theta) > 0, so z_{t+1} < z_t (depth decreases -> vehicle gets closer to surface).
# This matches positive-down depth convention (z=0 surface, z>0 deep).
sample_state = np.array([0.0, 0.0, 20.0, 0.0, 0.1, 0.0, 1.2]) # theta = +0.1 rad (climb)
next_st = physics.step(sample_state)
audit_results["sign_convention_check"] = {
    "initial_depth": 20.0,
    "pitch_rad": 0.1,
    "next_depth": float(next_st[2]),
    "depth_decreased_on_climb": bool(next_st[2] < 20.0)
}

# Save audit results to JSON
with open(os.path.join(p2_dir, "audit_raw.json"), "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)

print("=== AUDIT SCRIPT COMPLETED SUCCESSFULLY ===")
