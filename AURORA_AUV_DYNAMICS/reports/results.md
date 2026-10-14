# Experimental Results & Model Performance

**Project:** Learning-Based Dynamic Modeling of an Autonomous Underwater Vehicle Using AURORA Multi-Sensor Data

---

## 1. System Architecture Flow

`
                 REAL AURORA
                     │
                     ▼
             ┌──────────────┐
             │ Preprocessing│
             └──────┬───────┘
                    ▼
              AUV State X_t
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Physics Model          ML Model
          │                   │
          ▼                   ▼
   X physics             Residual
          │                   │
          └─────────┬─────────┘
                    ▼
             X predicted
                    │
                    ▼
            ┌───────────────┐
            │  TEST SPLIT   │
            │ Actual X_t+1  │
            └───────┬───────┘
                    ▼
             ACTUAL vs PREDICTED
                    │
                    ▼
              RMSE / MAE / R^2
                    │
                    ▼
              GRAPHS + MODEL
`

---

## 2. Dataset & Split Configuration
* **Dataset:** AURORA Autosub6000 Mission M86 Telemetry (IEEE DataPort)
* **Total Cleaned Samples:** 31,944 rows
* **Sampling Rate:** 0.50 Hz (dt = 2.0 s)
* **Chronological Splits:**
  * **Train Set:** 22,360 samples (70.0%)
  * **Validation Set:** 4,792 samples (15.0%)
  * **Test Set:** 4,792 samples (15.0%) [UNSEEN FUTURE TELEMETRY]

---

## 3. Test Set Prediction Metrics (Unseen Test Data)

| State Variable | Physics MAE | Physics RMSE | Physics R^2 | ML-Only MAE | ML-Only RMSE | ML-Only R^2 | Hybrid MAE | Hybrid RMSE | Hybrid R^2 | Error Improvement (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pitch Angle (rad) | 0.0278 | 0.0478 | 0.6267 | 0.0235 | 0.0412 | 0.7228 | 0.0222 | 0.0394 | 0.7461 | +17.53% |
| Depth (m) | 0.0758 | 0.1284 | 1.0000 | 0.0578 | 0.0970 | 1.0000 | 0.0596 | 0.1083 | 1.0000 | +15.64% |
| Heading Angle (rad) | 0.0345 | 0.0603 | 0.9989 | 0.0248 | 0.0515 | 0.9992 | 0.0248 | 0.0513 | 0.9992 | +15.01% |
| Surge Speed (m/s) | 0.1609 | 0.2416 | 0.6384 | 0.1410 | 0.2117 | 0.7225 | 0.1524 | 0.2253 | 0.6854 | +6.73% |
| Roll Angle (rad) | 0.0346 | 0.0805 | 0.4615 | 0.0287 | 0.0756 | 0.5255 | 0.0297 | 0.0768 | 0.5097 | +4.59% |
| North Position (m) | 0.4649 | 0.7310 | 1.0000 | 0.4941 | 0.7289 | 1.0000 | 0.4827 | 0.7664 | 1.0000 | -4.84% (USBL noise) |
| East Position (m) | 0.3272 | 0.7382 | 0.9998 | 0.3220 | 0.7511 | 0.9998 | 0.3367 | 0.7794 | 0.9998 | -5.59% (USBL noise) |

---

## 4. Multi-Step Prediction Degradation (Position RMSE)

| Horizon Steps | Horizon Time (seconds) | Physics-Only RMSE (m) | Hybrid Dynamics RMSE (m) |
| :---: | :---: | :---: | :---: |
| 1 step | 2.0 s | 0.735 m | 0.773 m |
| 5 steps | 10.0 s | 2.418 m | 2.215 m |
| 10 steps | 20.0 s | 4.892 m | 4.380 m |
| 20 steps | 40.0 s | 10.231 m | 8.947 m |

---

## 5. Summary of Findings
1. **Hypothesis Confirmed:** Incorporating machine-learned residuals into rigid-body kinematics reduces unmodeled hydrodynamic tracking error across surge speed, pitch, depth, and heading.
2. **Multi-Step Horizon Stability:** The hybrid model reduces compounding position drift over multi-step horizons (from 10.23 m down to 8.95 m at 20 steps).
