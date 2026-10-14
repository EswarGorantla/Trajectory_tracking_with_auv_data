import os
import yaml
import pickle
import torch
import numpy as np
import pandas as pd
from src.ml_models import GRUModel
from src.physics_model import PhysicsModel

def main():
    print("=========================================================")
    print("HYBRID PHYSICS-AI DYNAMICS FLOW DEMONSTRATION")
    print("=========================================================")

    # 1. Load configuration and directories
    config_path = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\AURORA_AUV_ML_MPC\configs\config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    models_dir = config['paths']['models_dir']
    processed_dir = config['paths']['processed_dir']
    
    # Check if models exist (if not, train them by running run_pipeline.py first)
    gru_path = os.path.join(models_dir, "gru.pth")
    if not os.path.exists(gru_path):
        raise FileNotFoundError("Trained GRU model not found. Please run 'python run_pipeline.py' first to train models.")

    # 2. Ingest the dataset (loading the test set)
    test_path = os.path.join(processed_dir, "test.csv")
    df_test = pd.read_csv(test_path)
    
    # 3. Load preprocessing parameters (scalers) and the trained ML model
    with open(os.path.join(models_dir, "feature_scaler.pkl"), "rb") as f:
        feature_scaler = pickle.load(f)
    with open(os.path.join(models_dir, "target_scaler.pkl"), "rb") as f:
        target_scaler = pickle.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUModel(input_dim=6, hidden_dim=config['hyperparameters']['hidden_dim'], output_dim=7).to(device)
    model.load_state_dict(torch.load(gru_path, map_location=device))
    model.eval()

    # Load parameters for the physics model
    dt = config['simulation']['dt']
    u_cruise = config['simulation']['U_cruise']
    k_d = 0.022498  # identified drag coefficient
    physics = PhysicsModel(k_d=k_d, dt=dt, U_cruise=u_cruise)

    # ----------------- FLOW SIMULATION -----------------
    # We will pick a sample current state index from the test set (e.g. index 100)
    sample_idx = 100
    seq_len = config['hyperparameters']['seq_len']
    
    # Step A: Ingest current state from dataset
    current_row = df_test.iloc[sample_idx]
    # Current AUV State: [North, East, Depth, Roll, Pitch, Heading, Speed]
    state_t = np.array([
        current_row['North'], current_row['East'], current_row['depth'],
        current_row['Roll'], current_row['Pitch'], current_row['Heading'],
        current_row['GroundSpeed']
    ])
    
    # Step B: Get command inputs from telemetry (what the vehicle was commanded to do next)
    u_cmd = df_test.iloc[sample_idx]['GroundSpeed']
    theta_cmd = df_test.iloc[sample_idx]['Pitch']
    psi_cmd = df_test.iloc[sample_idx]['Heading']

    # Step C: Run raw Physics dynamics model prediction
    physics_pred_t1 = physics.step(state_t, u_cmd, theta_cmd, psi_cmd)

    # Step D: Pre-populate history window of size 10 (20 seconds) for the time-series ML model
    history = []
    for idx in range(seq_len):
        row = df_test.iloc[sample_idx - seq_len + idx + 1]
        history.append([
            row['GroundSpeed'], row['Roll'], row['Pitch'],
            np.cos(row['Heading']), np.sin(row['Heading']), row['depth']
        ])
    
    # Scale features and run the Residual ML Model to predict the correction
    history_scaled = feature_scaler.transform(np.array(history))
    with torch.no_grad():
        history_tensor = torch.tensor([history_scaled], dtype=torch.float32).to(device)
        res_scaled = model(history_tensor).cpu().numpy()[0]
        # Unscale the predicted residuals back to physical units
        residuals = target_scaler.inverse_transform([res_scaled])[0]

    # Step E: Combine Physics Prediction + ML Correction to get Improved Dynamics
    improved_pred_t1 = physics_pred_t1.copy()
    improved_pred_t1[0] += residuals[0] # North correction
    improved_pred_t1[1] += residuals[1] # East correction
    improved_pred_t1[2] += residuals[2] # Depth correction
    improved_pred_t1[3] += residuals[3] # Roll correction
    improved_pred_t1[5] = (physics_pred_t1[5] + residuals[5]) % (2 * np.pi) # Heading correction
    improved_pred_t1[6] = max(physics_pred_t1[6] + residuals[6], 0.0) # Speed correction

    actual_t1 = np.array([
        df_test.iloc[sample_idx + 1]['North'],
        df_test.iloc[sample_idx + 1]['East'],
        df_test.iloc[sample_idx + 1]['depth']
    ])

    # ----------------- PRINT OUTPUTS -----------------
    print(f"\n[1] CURRENT STATE (at time t):")
    print(f"    North: {state_t[0]:.4f} m | East: {state_t[1]:.4f} m | Depth: {state_t[2]:.4f} m")
    print(f"    Speed: {state_t[6]:.4f} m/s | Heading: {np.degrees(state_t[5]):.2f} deg")

    print(f"\n[2] PHYSICS-ONLY PREDICTION (at time t+1):")
    print(f"    North: {physics_pred_t1[0]:.4f} m | East: {physics_pred_t1[1]:.4f} m | Depth: {physics_pred_t1[2]:.4f} m")

    print(f"\n[3] ML-LEARNED DYNAMICS CORRECTION (Residual):")
    print(f"    dNorth: {residuals[0]:.4f} m | dEast: {residuals[1]:.4f} m | dDepth: {residuals[2]:.4f} m")

    print(f"\n[4] FUTURE POSITION (Improved Dynamics Prediction at time t+1):")
    print(f"    North: {improved_pred_t1[0]:.4f} m | East: {improved_pred_t1[1]:.4f} m | Depth: {improved_pred_t1[2]:.4f} m")

    print(f"\n[5] GROUND TRUTH (Actual AUV telemetry at time t+1):")
    print(f"    North: {actual_t1[0]:.4f} m | East: {actual_t1[1]:.4f} m | Depth: {actual_t1[2]:.4f} m")

    # Compute errors
    phys_err = np.sqrt(np.sum((physics_pred_t1[:3] - actual_t1)**2))
    hyb_err = np.sqrt(np.sum((improved_pred_t1[:3] - actual_t1)**2))
    print(f"\n=== EVALUATION ===")
    print(f"    Physics-only Position Error: {phys_err:.4f} meters")
    print(f"    Hybrid Physics-AI Error:     {hyb_err:.4f} meters")
    print(f"    Prediction Improvement:      {((phys_err - hyb_err) / phys_err * 100):.2f}%")
    print("=========================================================")

if __name__ == "__main__":
    main()
