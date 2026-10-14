# PHASE 1: Learning-Based Dynamic Modeling of an AUV Using AURORA Multi-Sensor Data

This folder contains the complete Phase 1 implementation, trained models, evaluation metrics, prediction data, and visual plots for learning Autonomous Underwater Vehicle (AUV) dynamics from real multi-sensor telemetry.

---

## 1. System Architecture & Information Flow

```
                 REAL AURORA
                     │  (M86 Navigation Index: 31,944 rows)
                     ▼
             ┌──────────────┐
             │ Preprocessing│  (Cleaning, equirectangular projection, chronological split)
             └──────┬───────┘
                    ▼
              AUV State X_t    ([North, East, Depth, Roll, Pitch, Heading, Speed]^T)
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Physics Model          ML Model   (2-Layer GRU on 10-step history)
          │                   │
          ▼                   ▼
   X physics             Residual   (Predicted unmodeled dynamics)
          │                   │
          └─────────┬─────────┘
                    ▼
             X predicted       (X_physics + ML Residual)
                    │
                    ▼
            ┌───────────────┐
            │  TEST SPLIT   │  (Unseen 15% future data: 4,792 rows)
            │ Actual X_t+1  │
            └───────┬───────┘
                    ▼
             ACTUAL vs PREDICTED  (test_predictions.csv)
                    │
                    ▼
              RMSE / MAE / R^2    (model_metrics.csv)
                    │
                    ▼
              GRAPHS + MODEL     (best_dynamics_model.pt + 5 PNG charts)
```

---

## 2. Dataset & Chronological Splits

* **Dataset:** Autosub6000 AUV Cruise JC125 Mission M86 (`multi-beam-echosounder-index.csv`)
* **Total Cleaned Rows:** 31,944 samples
* **Sampling Rate:** $0.50\text{ Hz}$ ($\Delta t = 2.0\text{ seconds}$)
* **Splits:**
  * **Train Set:** 22,360 samples (70.0%, ~14.1 hours)
  * **Validation Set:** 4,792 samples (15.0%, ~3.0 hours)
  * **Test Set:** 4,792 samples (15.0%, ~3.0 hours — **Unseen Future Telemetry**)

---

## 3. Test Set Performance Metrics (Evaluated on Unseen Data)

| State Variable | Physics-Only RMSE | ML-Only RMSE | Hybrid (Physics+ML) RMSE | Error Reduction (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Pitch Angle (rad)** | 0.0478 | 0.0412 | **0.0394** | **+17.53%** |
| **Depth (m)** | 0.1284 | **0.0970** | 0.1083 | **+15.64%** |
| **Heading Angle (rad)** | 0.0603 | 0.0515 | **0.0513** | **+15.01%** |
| **Surge Speed (m/s)** | 0.2416 | **0.2117** | 0.2253 | **+6.73%** |
| **Roll Angle (rad)** | 0.0805 | **0.0756** | 0.0768 | **+4.59%** |
| **North Position (m)** | **0.7310** | 0.7289 | 0.7664 | -4.84% (USBL noise) |
| **East Position (m)** | **0.7382** | 0.7511 | 0.7794 | -5.59% (USBL noise) |

---

## 4. Key Findings

1. **Hydrodynamic Damping & Propulsion Lags:**
   * The baseline rigid-body kinematic equations assume static velocity and zero turning lag.
   * The trained recurrent GRU neural network captures propulsion response delays and damping cross-coupling, achieving significant error reductions in pitch ($+17.53\%$), depth ($+15.64\%$), and heading ($+15.01\%$).
2. **Measurement Noise vs Dynamic State:**
   * Raw geographic coordinates (`Latitude` and `Longitude`) from USBL acoustic positioning contain transceiver jump noise. The physics model provides a smoother baseline, whereas direct ML delta fitting on position coordinates captures sensor jitter.
3. **Multi-Step Horizon Stability:**
   * Across recursive rollouts up to 20 steps (40 seconds), the hybrid dynamics model exhibits stable error growth without compounding divergence.

---

## 5. Folder Structure & Files

```
PHASE 1/
├── data/
│   └── processed/
│       ├── train.csv              # 22,360 rows (70%)
│       ├── val.csv                # 4,792 rows (15%)
│       └── test.csv               # 4,792 rows (15% Unseen)
│
├── models/
│   ├── best_dynamics_model.pt     # Saved best PyTorch model checkpoint
│   ├── scaler.pkl                 # Feature & target scalers
│   ├── model_config.json          # Model metadata & hyperparameters
│   └── training_history.json      # Loss per epoch for train and val
│
├── src/
│   ├── data_loader.py             # Data loading
│   ├── preprocessing.py           # Cleaning & chronological splitting
│   ├── state_definition.py        # Coordinate transformations & state extraction
│   ├── physics_model.py           # 3D marine kinematics & drag dynamics
│   ├── system_identification.py   # Least squares Kd identification (Kd = 0.06073)
│   ├── dynamics_model.py          # PyTorch MLP, LSTM, and GRU models
│   ├── residual_model.py          # Hybrid physics + ML residual wrapper
│   ├── train.py                   # PyTorch training loop with early stopping
│   ├── predict.py                 # Test-set rollout inference
│   └── evaluate.py                # Regression metrics & plot generation
│
├── results/
│   ├── training/training_loss.png
│   ├── predictions/test_predictions.csv
│   ├── predictions/multistep_degradation.png
│   ├── metrics/model_metrics.csv
│   └── plots/trajectory_actual_vs_predicted.png
│   └── plots/state_variables_comparison.png
│   └── plots/metrics_comparison_bar.png
│
├── reports/
│   ├── data_inspection.md         # Sensor & variable audit
│   ├── state_definition.md        # State vector definitions & frames
│   ├── methodology.md             # Kinematic equations & ML architecture
│   └── results.md                 # Full results report with metrics
│
├── requirements.txt
└── run_pipeline.py                # Single-command executable script
```

---

## 6. How to Re-Run the Entire Pipeline

To re-run data cleaning, parameter identification, model training, test predictions, and plot generation from scratch:
```bash
python run_pipeline.py
```
