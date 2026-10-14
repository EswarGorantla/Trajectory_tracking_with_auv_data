# Quantitative Claims Reconciliation: Spec vs Empirical Verification

**Module:** Phase 2 (AUV Model Predictive Control)  
**Reference Document:** `COMBINED_METHODOLOGY.md`  
**Ground Rule:** Non-negotiable anti-hallucination policy. If empirical code produces numbers differing from the text, the code's numbers win and the discrepancy is documented honestly.  
**Auditor:** Senior Robotics/Control Research Engineer  
**Date:** September 2026  

---

## 1. Master Claims Reconciliation Table

| # | Claimed Item / Metric in Spec | Claimed Value in `COMBINED_METHODOLOGY.md` | Empirically Reproduced Value | Verification Status | Detailed Engineering Explanation & Proposed Spec Correction |
| :-: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Training Dataset Split Size** | $22,360\text{ samples}$ | $22,360\text{ samples}$ | **CONFIRMED** | Exactly matches raw lines in `train.csv`. |
| **2** | **Validation Dataset Split Size** | $4,791\text{ samples}$ | $4,792\text{ samples}$ | **PARTIALLY CONFIRMED** | Discrepancy of 1 sample. `val.csv` has exactly 4,792 rows. Proposed correction: update text to 4,792. |
| **3** | **Held-Out Test Split Size** | $4,792\text{ samples}$ | $4,792\text{ samples}$ | **CONFIRMED** | Exactly matches raw lines in `test.csv`. |
| **4** | **Model Parameter Count** | $41,095\text{ parameters}$ | $41,095\text{ parameters}$ | **CONFIRMED** | Exact parameter tensor count verified in PyTorch checkpoint. |
| **5** | **Layer Normalization in Model** | "Layer Normalization in intermediate layer" | **No LayerNorm exists** in weights or code | **NOT REPRODUCED** | Checkpoint state_dict has only GRU and Linear weights (`fc.0`, `fc.2`). Text was a documentation error; model is pure GRU + MLP. |
| **6** | **Fitted Drag Coefficient ($K_d$)** | $K_d = 0.06073$ | $K_d = 0.06073034$ | **CONFIRMED** | Refitted strictly on Train split ($|\Delta| = 3.39 \times 10^{-7}$). |
| **7** | **Surge Speed 1-Step RMSE** | Physics: $0.2254$, Hybrid: $0.1759$ (+21.96%) | Physics: $0.2416$, Hybrid: $0.2218$ (+8.20%) | **PARTIALLY CONFIRMED** | Hybrid model significantly outperforms physics on speed (+8.2%), but exact numerical RMSEs on held-out test split differ from text. |
| **8** | **Heading Angle 1-Step RMSE** | Physics: $0.0572$, Hybrid: $0.0472$ (+17.48%) | Physics: $0.0603$, Hybrid: $0.0511$ (+15.35%) | **PARTIALLY CONFIRMED** | Heading improvement is confirmed (+15.35% vs claimed +17.48%). |
| **9** | **Pitch Angle 1-Step RMSE** | Physics: $0.0454$, Hybrid: $0.0375$ (+17.40%) | Physics: $0.0478$, Hybrid: $0.0398$ (+16.71%) | **PARTIALLY CONFIRMED** | Pitch improvement is confirmed (+16.71% vs claimed +17.40%). |
| **10**| **Depth 1-Step RMSE** | Physics: $0.1230$, Hybrid: $0.1049$ (+14.72%) | Physics: $0.1284$, Hybrid: $0.1141$ (+11.14%) | **PARTIALLY CONFIRMED** | Depth prediction improvement confirmed (+11.14% vs claimed +14.72%). |
| **11**| **Position Residual Addition** | "retains deterministic kinematics for planar translation" | **Residuals ARE added to North & East** | **NOT REPRODUCED** | In both Phase 1 and Phase 2 code, `residuals[0]` and `residuals[1]` are actively added to $(x, y)$. Text proposed an intended design not enacted in original code. |
| **12**| **Prediction Horizon Lookahead** | $H=10$ steps ($20.0\text{ s}$ lookahead) | $10 \times 2.0\text{ s} = 20.0\text{ s}$ | **CONFIRMED** | Lookahead distance is $24\text{ m}$ at $1.2\text{ m/s}$ cruise. |
| **13**| **Basic A to B Mission Reaching** | Success: YES, Time: $90.0\text{ s}$, Final Error: $4.94\text{ m}$ | Success: YES, Time: $88.0 - 92.0\text{ s}$, Final Error: $4.2 - 4.9\text{ m}$ | **CONFIRMED** | Verified under continuous-time true plant with actuator lag and drag parameter mismatch. |
| **14**| **Cylindrical Obstacle Avoidance** | Min clearance: $12.30\text{ m}$ ($>12.0\text{ m}$ buffer), Collisions: 0 | Min dist: $12.1 - 12.8\text{ m}$, Collisions: 0 | **CONFIRMED** | Proactive MPC lookahead prevents physical hull contact across all seeds. |
| **15**| **Baseline Comparison vs Reactive LOS** | Reactive: 6 collisions (12s inside obstacle), MPC: 0 collisions | Reactive: 5-7 collision steps, MPC: 0 collisions | **CONFIRMED** | Reactive LOS tracker crashes directly through the cylinder. MPC achieves 100% collision-free transit. |
| **16**| **Current Drift & USBL Noise Robustness**| Reached goal without steady-state offset under $0.20\text{ m/s}$ current | Reached goal in $92.0 - 96.0\text{ s}$, final error $< 5.0\text{ m}$ | **CONFIRMED** | Receding-horizon feedback and Kalman filtering reject continuous current drift. |

---

## 2. Key Synthesis & Recommended Revisions for Specification

1. **Architecture Description:**  
   Update `COMBINED_METHODOLOGY.md` Section 3.2 to remove mention of "Layer Normalization". The model checkpoint strictly contains a 2-layer GRU with 64 units and a Linear(64, 32) -> ReLU -> Linear(32, 7) output head (total 41,095 weights).
2. **Dataset Counts:**  
   Correct the validation split sample count from 4,791 to **4,792**.
3. **Planar Residual Handling:**  
   Acknowledge that `Phase_2/src/dynamics_interface.py` actively includes `residuals[0]` (North) and `residuals[1]` (East). While the methodology proposed using pure kinematics for planar translation to avoid overfitting acoustic noise, the actual codebase incorporates them. Our research ablation demonstrates that while 1-step test metrics show East residual slightly increases error ($-4.04\%$), closed-loop receding-horizon MPC remains stable under both configurations.
4. **Decoupled Plant Distinction:**  
   Emphasize that the original Phase 2 numbers were obtained in an idealized discrete simulator identical to the internal controller model (an "inverse crime"). The new research-grade numbers validated above were obtained against an independent continuous-time plant with unmodeled actuator lags, continuous-time RK4 integration, and hydrodynamic drag mismatch.
