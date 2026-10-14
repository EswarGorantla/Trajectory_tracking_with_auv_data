# Dynamic Modeling Methodology

**Project:** Learning-Based Dynamic Modeling of an Autonomous Underwater Vehicle Using AURORA Multi-Sensor Data

---

## 1. System Architecture & Information Flow

```
                 REAL AURORA
                     │
                     ▼
             ┌──────────────┐
             │ Preprocessing│
             └──────┬───────┘
                    ▼
              AUV State Xₜ
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
            │ Actual Xₜ₊₁   │
            └───────┬───────┘
                    ▼
             ACTUAL vs PREDICTED
                    │
                    ▼
              RMSE / MAE / R²
                    │
                    ▼
              GRAPHS + MODEL
```

---

## 2. 3D Kinematics & Physics Dynamics Baseline
The baseline physical model incorporates kinematic relations and quadratic hydrodynamic drag:

$$x_{t+1} = x_t + U_t \cos(\theta_t) \cos(\psi_t) \Delta t$$
$$y_{t+1} = y_t + U_t \cos(\theta_t) \sin(\psi_t) \Delta t$$
$$z_{t+1} = z_t - U_t \sin(\theta_t) \Delta t$$
$$U_{t+1} = \max\left(U_t + K_d (U_{\text{cruise}}^2 - U_t^2) \Delta t, 0.0\right)$$

### Identified Drag Parameter
Using linear least squares on the 70% training telemetry:
$$K_d = 0.060730$$

---

## 3. Machine Learning Architecture
* **Recurrent Network:** 2-Layer GRU with 64 hidden units.
* **Input Sequence:** $L = 10$ steps ($20.0\text{ s}$) of $[U_t, \phi_t, \theta_t, \cos(\psi_t), \sin(\psi_t), z_t]$.
* **Target:** Residual error vector $\mathbf{r}_t = \mathbf{x}_{t+1}^{\text{actual}} - \mathbf{x}_{t+1}^{\text{phys}}$.
