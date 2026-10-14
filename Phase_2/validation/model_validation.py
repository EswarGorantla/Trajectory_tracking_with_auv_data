import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p1_dir = os.path.join(root_dir, "presentationphase1")
p2_dir = os.path.join(root_dir, "Phase_2")
sys.path.insert(0, p1_dir)
sys.path.insert(0, p2_dir)

from src.state_definition import StateDefinition
from src.physics_model import PhysicsModel
from src.dynamics_model import GRUModel
from src.residual_model import HybridDynamicsModel

def run_model_validation(seed: int = 1000, num_bootstrap: int = 1000):
    print("============================================================")
    print("PHASE 2 VALIDATION: OPEN-LOOP MULTI-STEP PREDICTION BENCHMARK")
    print("============================================================")
    print("Evaluating held-out TEST split only (zero leakage).")
    
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    # 1. Load Data
    test_path = os.path.join(p1_dir, "data", "processed", "test.csv")
    test_df = pd.read_csv(test_path, low_memory=False)
    print(f"Loaded held-out test split: {len(test_df)} samples")
    
    # 2. Load Checkpoint and Scalers
    ckpt_path = os.path.join(p1_dir, "models", "best_dynamics_model.pt")
    scaler_path = os.path.join(p1_dir, "models", "scaler.pkl")
    
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    feat_scaler = scalers["feature_scaler"]
    target_scaler = scalers["target_scaler"]
    
    device = torch.device("cpu")
    model = GRUModel(input_dim=6, hidden_dim=64, output_dim=7, num_layers=2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    kd = 0.06073034
    physics = PhysicsModel(k_d=kd, dt=2.0, u_cruise=1.2)
    hybrid_model = HybridDynamicsModel(physics, model, feat_scaler, target_scaler, device)
    
    # 3. Extract Matrices
    state_matrix = StateDefinition.extract_state_matrix(test_df)
    feat_matrix = StateDefinition.extract_feature_matrix(state_matrix)
    
    N = len(state_matrix)
    seq_len = 10
    max_k = 10
    
    # We evaluate multi-step rollouts starting from valid initial windows of length seq_len
    # Stride of 5 to balance statistical power and evaluation speed
    start_indices = np.arange(seq_len, N - max_k, 5)
    num_rollouts = len(start_indices)
    print(f"Evaluating {num_rollouts} rollouts of horizon k=1..{max_k} steps...")
    
    # Storage for errors: shape (num_rollouts, max_k, 7)
    # States: [North, East, Depth, Roll, Pitch, Heading, Speed]
    err_physics = np.zeros((num_rollouts, max_k, 7), dtype=np.float64)
    err_hybrid = np.zeros((num_rollouts, max_k, 7), dtype=np.float64)
    err_ml_only = np.zeros((num_rollouts, max_k, 7), dtype=np.float64)
    
    for r_idx, t_start in enumerate(start_indices):
        # Initial ground-truth state and history
        init_state = state_matrix[t_start - 1].copy()
        init_hist = feat_matrix[t_start - seq_len : t_start].copy()
        
        # Current states for recursion
        st_phys = init_state.copy()
        st_hyb = init_state.copy()
        st_ml = init_state.copy()
        
        hist_hyb = init_hist.copy()
        hist_ml = init_hist.copy()
        
        for k in range(max_k):
            t_curr = t_start + k
            actual_next = state_matrix[t_curr]
            
            # ORACLE COMMAND PROTOCOL:
            # Since control commands were not logged in the mission dataset,
            # future attitude & speed from actual telemetry are passed as proxy guidance
            oracle_speed = actual_next[6]
            oracle_pitch = actual_next[4]
            oracle_heading = actual_next[5]
            
            # --- 1. Physics Rollout ---
            phys_input = st_phys.copy()
            phys_input[4] = oracle_pitch
            phys_input[5] = oracle_heading
            phys_input[6] = oracle_speed
            st_phys = physics.step(phys_input)
            
            # --- 2. Hybrid Rollout ---
            hyb_next, _, _ = hybrid_model.predict_step(st_hyb, hist_hyb)
            # Apply oracle guidance to attitude/speed for fair propagation
            hyb_next[4] = oracle_pitch
            hyb_next[5] = oracle_heading
            hyb_next[6] = oracle_speed
            st_hyb = hyb_next.copy()
            
            # Advance hybrid feature history window
            new_feat_hyb = StateDefinition.extract_feature_matrix(st_hyb.reshape(1, -1))[0]
            hist_hyb = np.vstack([hist_hyb[1:], new_feat_hyb])
            
            # --- 3. ML-Only Rollout ---
            hist_ml_scaled = feat_scaler.transform(hist_ml)
            with torch.no_grad():
                res_t = model(torch.tensor(hist_ml_scaled[np.newaxis, :], dtype=torch.float32)).numpy()[0]
            ml_delta = target_scaler.inverse_transform([res_t])[0]
            st_ml = st_ml + ml_delta
            st_ml[5] = (st_ml[5]) % (2 * np.pi)
            st_ml[6] = max(st_ml[6], 0.0)
            st_ml[4] = oracle_pitch
            st_ml[5] = oracle_heading
            st_ml[6] = oracle_speed
            
            new_feat_ml = StateDefinition.extract_feature_matrix(st_ml.reshape(1, -1))[0]
            hist_ml = np.vstack([hist_ml[1:], new_feat_ml])
            
            # Record errors
            diff_phys = actual_next - st_phys
            diff_phys[5] = StateDefinition.wrap_angle(diff_phys[5])
            err_physics[r_idx, k] = diff_phys
            
            diff_hyb = actual_next - st_hyb
            diff_hyb[5] = StateDefinition.wrap_angle(diff_hyb[5])
            err_hybrid[r_idx, k] = diff_hyb
            
            diff_ml = actual_next - st_ml
            diff_ml[5] = StateDefinition.wrap_angle(diff_ml[5])
            err_ml_only[r_idx, k] = diff_ml

    # Compute RMSE per horizon step
    # Shape: (max_k, 7)
    rmse_phys = np.sqrt(np.mean(err_physics**2, axis=0))
    rmse_hyb = np.sqrt(np.mean(err_hybrid**2, axis=0))
    rmse_ml = np.sqrt(np.mean(err_ml_only**2, axis=0))
    
    # 2D horizontal position RMSE
    pos_err_phys = np.sqrt(np.mean(err_physics[:, :, 0]**2 + err_physics[:, :, 1]**2, axis=0))
    pos_err_hyb = np.sqrt(np.mean(err_hybrid[:, :, 0]**2 + err_hybrid[:, :, 1]**2, axis=0))
    pos_err_ml = np.sqrt(np.mean(err_ml_only[:, :, 0]**2 + err_ml_only[:, :, 1]**2, axis=0))
    
    # Block Bootstrap 95% CIs on position RMSE
    block_size = 10  # 10 consecutive rollouts (~50s of flight time)
    n_blocks = num_rollouts // block_size
    bootstrap_pos_phys = []
    bootstrap_pos_hyb = []
    
    for _ in range(num_bootstrap):
        sampled_blocks = rng.integers(0, n_blocks, size=n_blocks)
        sampled_indices = np.concatenate([np.arange(b * block_size, (b + 1) * block_size) for b in sampled_blocks])
        b_phys = np.sqrt(np.mean(err_physics[sampled_indices, :, 0]**2 + err_physics[sampled_indices, :, 1]**2, axis=0))
        b_hyb = np.sqrt(np.mean(err_hybrid[sampled_indices, :, 0]**2 + err_hybrid[sampled_indices, :, 1]**2, axis=0))
        bootstrap_pos_phys.append(b_phys)
        bootstrap_pos_hyb.append(b_hyb)
        
    bootstrap_pos_phys = np.array(bootstrap_pos_phys)
    bootstrap_pos_hyb = np.array(bootstrap_pos_hyb)
    
    ci_phys_lower = np.percentile(bootstrap_pos_phys, 2.5, axis=0)
    ci_phys_upper = np.percentile(bootstrap_pos_phys, 97.5, axis=0)
    ci_hyb_lower = np.percentile(bootstrap_pos_hyb, 2.5, axis=0)
    ci_hyb_upper = np.percentile(bootstrap_pos_hyb, 97.5, axis=0)
    
    # Determine fraction of horizons where hybrid beats physics
    hybrid_wins_per_state = {}
    state_names = ["North (m)", "East (m)", "Depth (m)", "Roll (rad)", "Pitch (rad)", "Heading (rad)", "Speed (m/s)"]
    for s_idx, s_name in enumerate(state_names):
        wins = np.sum(rmse_hyb[:, s_idx] < rmse_phys[:, s_idx])
        hybrid_wins_per_state[s_name] = f"{wins}/{max_k} horizons ({wins/max_k*100:.1f}%)"
    
    print("\n--- Model Validation Results Summary ---")
    print(f"{'Horizon (k)':<12}{'Physics Pos RMSE (m)':<24}{'Hybrid Pos RMSE (m)':<24}{'Hybrid Beat Physics?'}")
    for k in range(max_k):
        beat = "YES" if pos_err_hyb[k] < pos_err_phys[k] else "NO"
        print(f"{k+1:<12}{pos_err_phys[k]:<24.4f}{pos_err_hyb[k]:<24.4f}{beat}")
        
    print("\nState-wise Horizon Win Rate (Hybrid vs Physics):")
    for k, v in hybrid_wins_per_state.items():
        print(f"  {k}: {v}")
        
    # Save validation metrics to JSON
    out_dir = os.path.join(p2_dir, "results_v2", "metrics")
    plots_dir = os.path.join(p2_dir, "results_v2", "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    results_payload = {
        "dataset": "Autosub6000 AURORA M86 Test Split",
        "sample_count": len(test_df),
        "evaluated_rollouts": num_rollouts,
        "oracle_protocol": "Command proxy from actual future attitude & speed",
        "horizons_steps": list(range(1, max_k + 1)),
        "horizons_seconds": [k * 2.0 for k in range(1, max_k + 1)],
        "position_rmse": {
            "physics": pos_err_phys.tolist(),
            "physics_ci_95": [ci_phys_lower.tolist(), ci_phys_upper.tolist()],
            "hybrid": pos_err_hyb.tolist(),
            "hybrid_ci_95": [ci_hyb_lower.tolist(), ci_hyb_upper.tolist()],
            "ml_only": pos_err_ml.tolist()
        },
        "state_rmse_hybrid": {s: rmse_hyb[:, i].tolist() for i, s in enumerate(state_names)},
        "state_rmse_physics": {s: rmse_phys[:, i].tolist() for i, s in enumerate(state_names)},
        "hybrid_wins_per_state": hybrid_wins_per_state
    }
    
    json_path = os.path.join(out_dir, "model_validation_kstep.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved validation metrics JSON to: {json_path}")
    
    # Plotting K-Step Degradation
    plt.figure(figsize=(9, 5))
    h_sec = [k * 2.0 for k in range(1, max_k + 1)]
    plt.plot(h_sec, pos_err_phys, 'o-', color='#dc2626', label='Physics-Only Kinematics', lw=2)
    plt.fill_between(h_sec, ci_phys_lower, ci_phys_upper, color='#dc2626', alpha=0.15)
    
    plt.plot(h_sec, pos_err_hyb, 's-', color='#2563eb', label='Hybrid (Physics + GRU)', lw=2)
    plt.fill_between(h_sec, ci_hyb_lower, ci_hyb_upper, color='#2563eb', alpha=0.15)
    
    plt.plot(h_sec, pos_err_ml, '^--', color='#7c3aed', label='Pure Data-Driven GRU', lw=1.5, alpha=0.7)
    
    plt.title("Multi-Step Open-Loop Position Error Growth (Held-Out Test Split)", fontsize=12, fontweight='bold')
    plt.xlabel("Prediction Horizon (seconds)", fontsize=11)
    plt.ylabel("Horizontal Position RMSE (meters)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    fig_path = os.path.join(plots_dir, "kstep_error_growth.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved validation plot to: {fig_path}")

if __name__ == "__main__":
    run_model_validation()
