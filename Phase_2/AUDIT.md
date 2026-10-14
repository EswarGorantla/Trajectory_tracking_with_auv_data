# Comprehensive Pre-Build Audit of Phase 1 and Phase 2 Codebase

**Date:** September 2026  
**Auditor:** Senior Robotics/Control Research Engineer  
**Repository:** `RoboticsSEM5`  
**Purpose:** Rigorous empirical audit of all data splits, model architecture, hybrid definitions, metrics, control formulations, distribution shifts, sign conventions, and optimizer mechanics prior to building the research-grade simulation.

---

## 1. Data Split Integrity & Zero-Leakage Verification

### Empirical Findings:
* **Train Split Size:** `22,360` samples
* **Validation Split Size:** `4,792` samples (Note: `COMBINED_METHODOLOGY.md` previously claimed `4,791`; actual line count in `presentationphase1/data/processed/val.csv` is `4,792`).
* **Test Split Size:** `4,792` samples
* **Total Cleaned Telemetry:** `31,944` samples ($17.74\text{ hours}$ at $\Delta t = 2.0\text{ s}$).

### Chronological Progression Evidence:
Column `Time` inspected directly from `train.csv`, `val.csv`, and `test.csv`:
* **Train Interval:** `18:48:14` to `07:24:12` (next morning, 22,360 rows)
* **Val Interval:** `07:24:14` to `10:03:56` (4,792 rows)
* **Test Interval:** `10:03:58` to `12:44:04` (4,792 rows)
* **Assertion:** `max(train_time) < min(val_time) < max(val_time) < min(test_time)` $\to$ **PASS (Strictly Chronological, Zero Overlap)**.

### Scaler Verification:
* Evaluated against `presentationphase1/models/scaler.pkl`:
  * `feature_scaler.mean_`: Recomputed strictly on the Train split ($22,360$ rows).
  * `max_abs_diff_mean`: `0.00e+00`
  * `max_abs_diff_var`: `0.00e+00`
* **Conclusion:** The feature scaler was fitted strictly on the Training split. There is zero contamination from validation or test splits.

### $K_d$ Refit Verification:
* Non-linear least-squares refit on `train_df["GroundSpeed"]` with $U_{\text{cruise}} = 1.20\text{ m/s}$ and $\Delta t = 2.0\text{ s}$:
  $$K_{d, \text{refit}} = 0.06073034 \quad (\text{claimed } 0.06073, \quad |\Delta| = 3.39 \times 10^{-7})$$
* **Conclusion:** $K_d$ refit on train split strictly reproduces $0.06073$.

---

## 2. Model Architecture Truth & Checkpoint Audit

### Checkpoint File:
`presentationphase1/models/best_dynamics_model.pt` (SHA-256: `9BF7305EDD14FB2BA61FD9A293E687D604F1DD9B76592D835A1016E242AA9FF2`)

### Exact Parameter Tensor Shape Breakdown:
| Parameter Name | Shape | Number of Elements | Data Type |
| :--- | :---: | :---: | :---: |
| `gru.weight_ih_l0` | `(192, 6)` | 1,152 | `torch.float32` |
| `gru.weight_hh_l0` | `(192, 64)` | 12,288 | `torch.float32` |
| `gru.bias_ih_l0` | `(192,)` | 192 | `torch.float32` |
| `gru.bias_hh_l0` | `(192,)` | 192 | `torch.float32` |
| `gru.weight_ih_l1` | `(192, 64)` | 12,288 | `torch.float32` |
| `gru.weight_hh_l1` | `(192, 64)` | 12,288 | `torch.float32` |
| `gru.bias_ih_l1` | `(192,)` | 192 | `torch.float32` |
| `gru.bias_hh_l1` | `(192,)` | 192 | `torch.float32` |
| `fc.0.weight` | `(32, 64)` | 2,048 | `torch.float32` |
| `fc.0.bias` | `(32,)` | 32 | `torch.float32` |
| `fc.2.weight` | `(7, 32)` | 224 | `torch.float32` |
| `fc.2.bias` | `(7,)` | 7 | `torch.float32` |
| **TOTAL** | — | **41,095** | — |

### LayerNorm Audit:
* **Checkpoint Keys:** Checked for `norm`, `ln`, `gamma`, `beta`. Result: **Zero matching keys**.
* **Code in `presentationphase1/src/dynamics_model.py`:** Contains `nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 7))`. No `nn.LayerNorm`.
* **Discrepancy Resolution:** `COMBINED_METHODOLOGY.md` claimed "ReLU activation and Layer Normalization". This is **FALSE in code**. The actual architecture is pure 2-layer GRU with a 2-layer MLP projection without normalization layers. Total parameters: exactly **41,095**.

---

## 3. Hybrid Definition Truth: Residual Channels in Code vs Text

### Code Inspection:
1. **In Phase 1 Evaluation (`presentationphase1/src/residual_model.py:46-53`):**
   ```python
   hybrid_next[0] += residuals[0]  # North
   hybrid_next[1] += residuals[1]  # East
   hybrid_next[2] += residuals[2]  # Depth
   hybrid_next[3] += residuals[3]  # Roll
   hybrid_next[4] += residuals[4]  # Pitch
   hybrid_next[5] = (phys_next[5] + residuals[5]) % (2 * np.pi)
   hybrid_next[6] = max(phys_next[6] + residuals[6], 0.0)
   ```
2. **In Phase 2 Controller Rollout (`Phase_2/src/dynamics_interface.py:90-96`):**
   ```python
   x_next = phys_next[0] + residuals[0]
   y_next = phys_next[1] + residuals[1]
   z_next = phys_next[2] + residuals[2]
   phi_next = phys_next[3] + residuals[3]
   theta_next = theta_act + residuals[4] * 0.5  # AI pitch damping adjustment
   psi_next = wrap_to_2pi(psi_act + residuals[5] * 0.5)  # AI yaw drift adjustment
   speed_next = np.clip(phys_next[6] + residuals[6], self.min_speed, self.max_speed)
   ```

### Discrepancy & MPC Reality:
* **The Claim:** `COMBINED_METHODOLOGY.md` states: *"the hybrid system retains deterministic kinematics for planar translation while applying neural residual damping to dynamics"*.
* **The Reality in Code:** In both Phase 1 and Phase 2, `residuals[0]` and `residuals[1]` **are actively added** to the North and East positions!
* Furthermore, in Phase 2 `dynamics_interface.py`, pitch and yaw residuals are multiplied by an ad-hoc damping factor of `0.5`, and added to the post-actuated command (`theta_act`, `psi_act`), not `phys_next`.
* **Action for Research Simulation:** We evaluate both options:
  1. Default MPC mode: Pure kinematics for North/East (eliminating USBL acoustic noise amplification).
  2. Full residual MPC mode: Directly matching `dynamics_interface.py`. Both will be benchmarked side-by-side.

---

## 4. Metric Definition Truth: 1-Step Benchmark on Held-Out Test Split

Evaluated strictly on the $4,792$ held-out test samples (`presentationphase1/results/model_comparison.csv`):

| State Variable | Claimed Physics RMSE | Reproduced Physics RMSE | Claimed Hybrid RMSE | Reproduced Hybrid RMSE | Reproduced Hybrid Improvement (%) | Reproduced ML-Only RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Surge Speed ($m/s$)** | $0.2254$ | $0.2416$ | $0.1759$ | $0.2218$ | **$+8.20\%$** | $0.1992$ |
| **Heading Angle ($\text{rad}$)** | $0.0572$ | $0.0603$ | $0.0472$ | $0.0511$ | **$+15.35\%$** | $0.0511$ |
| **Pitch Angle ($\text{rad}$)** | $0.0454$ | $0.0478$ | $0.0375$ | $0.0398$ | **$+16.71\%$** | $0.0398$ |
| **Depth ($m$)** | $0.1230$ | $0.1284$ | $0.1049$ | $0.1141$ | **$+11.14\%$** | $0.1742$ |
| **Roll Angle ($\text{rad}$)** | $0.0770$ | $0.0805$ | $0.0718$ | $0.0752$ | **$+6.64\%$** | $0.0752$ |
| **North Position ($m$)** | $0.7112$ | $0.7310$ | $0.9518$ | $0.7211$ | **$+1.36\%$** | $2.0301$ |
| **East Position ($m$)** | $0.6927$ | $0.7382$ | $0.7724$ | $0.7679$ | **$-4.04\%$** | $1.1589$ |

### Critical Observations:
* The metrics in the report tables are **Root Mean Squared Error (RMSE)**.
* The reproduced numbers on the test split confirm the qualitative findings: **Hybrid significantly improves speed (+8.2%), heading (+15.4%), pitch (+16.7%), and depth (+11.1%) over kinematics alone**.
* For East position, the hybrid model degrades performance by $-4.04\%$ compared to kinematics because the GRU attempts to fit random acoustic USBL jumps.

---

## 5. Target Scaling, De-Normalization & Angle Wrapping

1. **Feature Normalization:** Applied via `trainer.feature_scaler.transform()` using sample mean and variance.
2. **Residual Targets:** Normalized via `StandardScaler` during training. At inference, outputs are de-normalized via `target_scaler.inverse_transform()`.
3. **Heading Residual Wrapping:** Heading residual during training was wrapped via `StateDefinition.wrap_angle(residuals[:, 5])`. At inference, heading is modular wrapped via `(phys_next[5] + residuals[5]) % (2 * np.pi)`.

---

## 6. Control Injection & Ordering Audit

In `Phase_2/src/dynamics_interface.py:56-82`:
1. **Actuator Slew Limiting:**
   ```python
   theta_act = current_state.pitch + np.clip(control.pitch_cmd - current_state.pitch, -0.10, 0.10)
   heading_err = wrap_angle(control.heading_cmd - current_state.heading)
   psi_act = wrap_to_2pi(current_state.heading + np.clip(heading_err, -0.20, 0.20))
   u_act = current_state.speed + np.clip(control.speed_cmd - current_state.speed, -0.15, 0.15)
   ```
2. **Kinematic Integration:**
   `act_state` with updated $(\theta_{\text{act}}, \psi_{\text{act}}, u_{\text{act}})$ is fed to `physics.step(act_state)`.
   Therefore, **post-step commanded attitude is used immediately for translation over the entire 2.0s timestep** (implicit Euler assumption).
3. **Residual Addition:**
   Residual is computed from past 10-step features (which does NOT include the newly commanded action, because the GRU has no control input channels).
4. **Modeling Implication:** The GRU models autonomous hydrodynamic settling, but does NOT see future commands. Control enters purely through the kinematic baseline and actuator slew model.

---

## 7. Distribution Shift & Out-of-Distribution (OOD) Analysis

Analyzed on $22,360$ training samples:

| Feature | Train Min | Train Median (p50) | Train Max | Phase 2 Operating Range | OOD Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Surge Speed ($m/s$)** | $0.025$ | $1.236$ | $3.249$ | $[0.5, 2.0]\text{ m/s}$ | **In-Distribution** (0% OOD) |
| **Depth ($m$)** | $0.001$ | $58.89$ | $108.38$ | $[10.0, 60.0]\text{ m}$ | **In-Distribution** (0% OOD) |
| **Pitch ($\text{rad}$)** | $-0.409$ | $-0.012$ | $+0.464$ | $[-0.35, +0.35]\text{ rad}$ | **In-Distribution** (0% OOD) |
| **Roll ($\text{rad}$)** | $+0.019$ | $+0.223$ | $+0.478$ | $[-0.50, +0.50]\text{ rad}$ | **Shift Detected:** Telemetry has steady positive roll bias ($+12.8^\circ$). |
| **$\cos\psi, \sin\psi$** | $-1.000$ | $+0.245 / +0.029$ | $+1.000$ | $[-1.0, +1.0]$ | Full circle visited. |

* **OOD Monitor Requirement:** A multivariate Mahalanobis distance monitor on the scaled feature window $\mathbf{s}$ must be logged per step during simulation to detect if unmodeled vehicle states push the GRU into extrapolation.

---

## 8. Sign Conventions & Geometry Verification

1. **Depth ($z$):** Positive-down ($z=0$ at sea surface).
   * Empirical test: Start at $z=20\text{m}$, pitch $\theta = +0.10\text{ rad}$ (climb).
   * Realized $z_{t+1} = 20 - 1.2 \times \sin(0.1) \times 2.0 = 19.76\text{ m} < 20.0\text{ m}$.
   * **Result:** Confirmed positive-down. Positive pitch decreases depth.
2. **Pitch ($\theta$):** Positive nose-up climbing.
3. **Heading ($\psi$):** Clockwise from North, mapped to $[0, 2\pi)$.
4. **LOS Pitch Angle Formulation:**
   $$\theta_{\text{LOS}} = \text{atan2}(-(z_{\text{goal}} - z), \sqrt{(x_{\text{goal}} - x)^2 + (y_{\text{goal}} - y)^2})$$
   When $z_{\text{goal}} > z$ (goal is deeper), $-(z_{\text{goal}} - z) < 0 \implies \theta_{\text{LOS}} < 0$ (nose-down dive). Confirmed correct.

---

## 9. Under-Specified Items in Existing Implementation

| Item | Methodology Text | Actual Code (`Phase_2/src/*`) | Resolution for Research Simulation |
| :--- | :--- | :--- | :--- |
| **CEM Iterations** | Unstated | `iterations = 3` in `mpc_controller.py` | Configurable in `mpc_config.yaml` (`N_iter=3`, ablate 1, 2, 3) |
| **Elite Fraction** | Text says "top-10" | `elite_fraction = 0.20` ($70 \times 0.2 = 14$ elites) | Fixed to strict top-10 ($E=10$) matching spec |
| **Random Generator** | Unseeded `np.random.randn` | Unseeded global random | Refactored to explicit `np.random.Generator(seed)` |
| **Optimal Selection** | Implied $\arg\min$ | `best_controls = elite_controls[0]` | Verified lowest-cost candidate is strictly tracked and returned |
| **$J_{\text{smooth}}$ R-matrix** | Not defined | Implicit identity weights for $[\Delta U, \Delta\theta, \Delta\psi]$ | Explicit diagonal matrix $R = \text{diag}(1.0, 4.0, 4.0)$ documented in `ASSUMPTIONS.md` |
| **True Plant Hydrodynamics** | Identical to controller ("inverse crime") | Identical in `Phase_2/src/simulator.py` | Built independent `true_plant.py` with continuous sub-stepping and parameter mismatch |
