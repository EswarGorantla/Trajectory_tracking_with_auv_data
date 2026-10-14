# Learning-Based Dynamic Modeling and Model Predictive Control for Autonomous Underwater Vehicles Using Multi-Sensor AURORA Data

This repository contains the complete codebase and reports for our hybrid physics-machine learning modeling and Model Predictive Control (MPC) framework for Autonomous Underwater Vehicles (AUVs).

---

## 1. Problem Statement
Accurately predicting AUV motion in the ocean is essential for precise steering, collision avoidance, and trajectory tracking. However, underwater environments are highly dynamic and difficult to model.

## 2. Why AUV Dynamics Are Difficult
AUVs experience complex, nonlinear forces that are difficult to write down analytically, including:
* **Hydrodynamic Damping:** Drag forces that scale quadratically with velocity.
* **Added Mass:** The volume of water that the vehicle must accelerate along with itself.
* **Environmental Disturbances:** Unmeasured ocean currents, waves, and buoyancy variations.
* **Thruster and Fin Lags:** Delays in control surface actuation and propeller response.

## 3. What AURORA Provides
The AURORA dataset (from the Autosub6000 cruise JC125) logs high-frequency multi-sensor measurements:
* Geographic position (Latitude, Longitude) from GPS/USBL acoustic transponders.
* Pressures representing AUV depth.
* Accelerations and Euler angles (Roll, Pitch, Heading) from Inertial Measurement Units (IMUs).
* Relative water speed and ground speed from Doppler Velocity Logs (DVL).

## 4. Project Objective
We combine a simple rigid-body physics model with a machine learning model. Instead of replacing physics, the AI model is trained to learn the **residual mismatch** between physics-based predictions and the observed vehicle behavior. This hybrid model is then utilized inside a receding-horizon MPC loop to steer the AUV.

### Information Flow Diagram
```
                 AURORA DATA
                      ↓
              Current AUV State
                      ↓
          ┌─────────────────────┐
          │ Physics Dynamics    │
          │       Model         │
          └──────────┬──────────┘
                     ↓
             Physics Prediction
                     ↓
          ┌─────────────────────┐
          │   LSTM / GRU        │
          │ Residual ML Model   │
          └──────────┬──────────┘
                     ↓
              ML Correction
                     ↓
        ┌────────────────────────┐
        │ Improved Dynamics      │
        │        Model           │
        └───────────┬────────────┘
                    ↓
                   MPC
                    ↓
          Optimal Control Input
                    ↓
             AUV Simulation
                    ↓
             Future Position
```

---

## 5. Input and Output Structures
* **Model Inputs (Features):** $6$ variables representing the vehicle's current state:
  $$\mathbf{s}_t = [U_t, \phi_t, \theta_t, \cos(\psi_t), \sin(\psi_t), z_t]^T$$
* **Model Outputs (Residual Targets):** $7$ variables representing the prediction error of the physics model over a $2.0s$ step:
  $$\mathbf{r}_t = \mathbf{x}_{t+1}^{\text{actual}} - \mathbf{x}_{t+1}^{\text{physics}}$$

---

## 6. Mathematical Framework

### Physics Model
We implement the standard 3D kinematic relationships:
$$x_{t+1} = x_t + U_t \cos(\theta_t) \cos(\psi_t) \Delta t$$
$$y_{t+1} = y_t + U_t \cos(\theta_t) \sin(\psi_t) \Delta t$$
$$z_{t+1} = z_t - U_t \sin(\theta_t) \Delta t$$

We identify the surge velocity drag coefficient $K_d$ using linear least squares regression on the training split:
$$U_{t+1} = U_t + K_d (U_c^2 - U_t^2) \Delta t \implies K_d \approx 0.022498$$

### ML Residual Learning
We train multiple architectures:
1. **Linear Regression:** Baseline estimator.
2. **Random Forest:** Baseline nonlinear tree-based regressor.
3. **MLP:** A 2-layer feedforward neural network.
4. **LSTM / GRU:** Recurrent models using a sequence history of $L = 10$ steps (20 seconds).

The final state prediction combines the model outputs:
$$\mathbf{x}_{t+1}^{\text{final}} = \mathbf{x}_{t+1}^{\text{physics}} + \mathbf{r}_t^{\text{ML\_predicted}}$$

### Receding-Horizon MPC
We implement a randomized Monte Carlo shooting MPC. At each step, the optimizer generates $K = 300$ candidate control trajectories over a horizon $H = 5$ steps (10 seconds), simulates the rollouts using our hybrid model, and selects the command sequence that minimizes the tracking cost function:
$$J = \sum_{h=1}^H \|\mathbf{x}_{t+h} - \mathbf{x}_{t+h}^{\text{ref}}\|_Q^2 + \sum_{h=0}^{H-1} \|\mathbf{u}_{t+h}\|_R^2 + \sum_{h=0}^{H-1} \|\Delta \mathbf{u}_{t+h}\|_S^2$$

---

## 7. Project Directory Structure
* `configs/config.yaml`: Configurations and hyperparameter bounds.
* `src/`: Modular Python classes for preprocessing, training, simulating, and visualizing.
* `notebooks/`: Jupyter Notebook walk-throughs for each pipeline step.
* `results/`: Output directories for generated plots.
* `reports/`: Data inspection, methodology, and final experimental reports.

---

## 8. How to Run the Pipeline
Ensure all requirements are installed:
```bash
pip install -r requirements.txt
```
To run the entire pipeline end-to-end (inspecting, preprocessing, training all models, running MPC simulations with disturbances, generating charts, and compiling reports):
```bash
python run_pipeline.py
```

---

## 9. Expected Results (Summary of Pipeline Execution)
* **One-Step-Ahead Prediction:** The GRU model achieves the lowest test set RMSE for attitude angles and surge speed.
* **Closed-Loop Control Tracking:**
  * **PID Position RMSE:** 104.97 meters
  * **Physics-only MPC Position RMSE:** 73.46 meters
  * **AI-enhanced MPC (LSTM) Position RMSE:** **21.42 meters**
  * **Improvement:** Incorporating the LSTM residual predictions inside the MPC loop reduces trajectory tracking error by **70.84%** compared to the physics MPC and **79.59%** compared to PID.
