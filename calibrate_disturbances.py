import os
import sys
import numpy as np
import pandas as pd

root_dir = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5"
p1_dir = os.path.join(root_dir, "presentationphase1")
sys.path.insert(0, p1_dir)

from src.state_definition import StateDefinition
from src.physics_model import PhysicsModel

train_df = pd.read_csv(os.path.join(p1_dir, "data", "processed", "train.csv"), low_memory=False)
val_df = pd.read_csv(os.path.join(p1_dir, "data", "processed", "val.csv"), low_memory=False)

# Train split maximum observed rate of change over dt=2.0s
u_diff = np.abs(np.diff(train_df["GroundSpeed"].values)) / 2.0
theta_diff = np.abs(np.diff(train_df["Pitch"].values)) / 2.0
psi_raw = train_df["Heading"].values
psi_diff = np.abs(np.arctan2(np.sin(np.diff(psi_raw)), np.cos(np.diff(psi_raw)))) / 2.0

print(f"Max observed rates in Train split (per second):")
print(f"  Surge Speed rate: p99={np.percentile(u_diff, 99):.4f} m/s^2, max={np.max(u_diff):.4f} m/s^2")
print(f"  Pitch rate:       p99={np.percentile(theta_diff, 99):.4f} rad/s ({np.rad2deg(np.percentile(theta_diff, 99)):.2f} deg/s), max={np.max(theta_diff):.4f} rad/s")
print(f"  Heading rate:     p99={np.percentile(psi_diff, 99):.4f} rad/s ({np.rad2deg(np.percentile(psi_diff, 99)):.2f} deg/s), max={np.max(psi_diff):.4f} rad/s")

# Validation split physics residual standard deviation
physics = PhysicsModel(k_d=0.06073034, dt=2.0, u_cruise=1.2)
val_states = StateDefinition.extract_state_matrix(val_df)
cur_val = val_states[:-1]
act_val = val_states[1:]
pred_val = physics.predict_batch(cur_val)

res_val = act_val - pred_val
res_val[:, 5] = StateDefinition.wrap_angle(res_val[:, 5])

std_res_val = np.std(res_val, axis=0)
print("\nValidation split physics residual standard deviations (per 2.0s step):")
state_names = ["North", "East", "Depth", "Roll", "Pitch", "Heading", "Speed"]
for name, s in zip(state_names, std_res_val):
    print(f"  {name:<10}: {s:.6f}")
