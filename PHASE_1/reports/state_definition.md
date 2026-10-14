# AUV State Vector Definition

**Project Title:** Learning-Based Dynamic Modeling of an Autonomous Underwater Vehicle Using AURORA Multi-Sensor Data

---

## 1. State Vector Formulation

Based strictly on real variables available in the AURORA dataset, the state vector $\mathbf{x}_t \in \mathbb{R}^7$ is defined as:

$$\mathbf{x}_t = [x_t, y_t, z_t, \phi_t, \theta_t, \psi_t, U_t]^T$$

Where:

| State Variable | Symbol | Physical Meaning | Unit | Coordinate Frame | Source Sensor |
| :--- | :---: | :--- | :---: | :--- | :--- |
| **North Position** | $x$ | Displacement North relative to mission start origin | Meters ($m$) | Local Earth Tangent Plane (NED) | USBL / GPS (Latitude) |
| **East Position** | $y$ | Displacement East relative to mission start origin | Meters ($m$) | Local Earth Tangent Plane (NED) | USBL / GPS (Longitude) |
| **Depth** | $z$ | Vertical distance below water surface | Meters ($m$) | Earth Z (positive-down) | Pressure Sensor (`depth`) |
| **Roll** | $\phi$ | Rotation about longitudinal body axis | Radians ($rad$) | Body Frame | IMU (`Roll`) |
| **Pitch** | $\theta$ | Rotation about lateral body axis (positive nose-up) | Radians ($rad$) | Body Frame | IMU (`Pitch`) |
| **Heading** | $\psi$ | Compass orientation clockwise from North | Radians ($rad$) | Earth Frame | Gyrocompass (`Heading`) |
| **Surge Speed** | $U$ | Forward linear velocity over ground | $\text{m/s}$ | Body X-axis | Doppler Velocity Log (`GroundSpeed`) |

---

## 2. Neural Network Feature Vector

To respect angular continuity and avoid discontinuities at the $0 \leftrightarrow 2\pi$ wrap boundary, heading is decomposed into its continuous orthogonal components:

$$\mathbf{s}_t = [U_t, \phi_t, \theta_t, \cos(\psi_t), \sin(\psi_t), z_t]^T \in \mathbb{R}^6$$

For recurrent sequential models (LSTM / GRU), the input is a rolling sequence of $L = 10$ timesteps ($20.0\text{ seconds}$ of history):

$$\mathbf{S}_t = [\mathbf{s}_{t-9}, \mathbf{s}_{t-8}, \dots, \mathbf{s}_t] \in \mathbb{R}^{10 \times 6}$$

---

## 3. Dynamics Learning Problem Formulation

The objective is to predict the state at the next timestep $t+1$ ($\Delta t = 2.0\text{ s}$):

$$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{S}_t)$$

In our hybrid residual architecture:
1. **Physics Model:** Computes deterministic kinematic projection $\mathbf{x}_{t+1}^{\text{phys}} = f_{\text{phys}}(\mathbf{x}_t)$.
2. **Residual Target:** Defined as the discrepancy $\mathbf{r}_t = \mathbf{x}_{t+1}^{\text{actual}} - \mathbf{x}_{t+1}^{\text{phys}}$.
3. **ML Prediction:** The neural network learns $\hat{\mathbf{r}}_t = g_{\text{ML}}(\mathbf{S}_t)$.
4. **Final Hybrid Prediction:** $\mathbf{x}_{t+1}^{\text{hybrid}} = \mathbf{x}_{t+1}^{\text{phys}} + \hat{\mathbf{r}}_t$.
