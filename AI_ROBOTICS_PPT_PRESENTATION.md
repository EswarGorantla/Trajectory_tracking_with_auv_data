# Presentation Dossier: Trajectory Tracking of an Underwater Robot Using Physics-AI Modeling & Model Predictive Control

**Course:** INTRODUCTION TO AI ROBOTICS (22AIE214)  
**Institution:** Amrita Vishwa Vidyapeetham, Amaravati Campus  
**Date:** 11-10-2026  
**Instructor / Guide:** Dr. Ravishankar Prakash Desai  

---

## Slide Index & Overview

| Slide # | Slide Title | Core Purpose & Deliverables |
| :---: | :--- | :--- |
| **Slide 1** | **Title Slide** | Institution branding, lab project title, presenter details, course code, and faculty guide. |
| **Slide 2** | **Introduction** | Importance of AUV autonomy, ocean environmental challenges, failure of pure physics vs. pure AI, and hybrid motivation. |
| **Slide 3** | **Problem Statement, Objectives & Programming Language** | Mathematical formulation of tracking under ocean disturbance, 4 concrete technical goals, software stack & dataset. |
| **Slide 4** | **Methodology** | Two-phase architecture: 3D kinematics + quadratic drag, GRU/LSTM residual learning, and receding-horizon CEM-MPC. |
| **Slide 5** | **Flow-chart** | Visual architectural block diagram (Mermaid) mapping offline training pipeline to real-time closed-loop control loop. |
| **Slide 6** | **Simulation Result** | One-step predictor test RMSE comparison, closed-loop trajectory tracking benchmarks, and core Python control code. |
| **Slide 7** | **Conclusion** | Core scientific takeaways (69.4% tracking gain over PID, acoustic sensor overfitting insights), and future research directions. |
| **Slide 8** | **References** | Scholarly references spanning marine robotics, optimal control, deep learning for dynamics, and mission datasets. |
| **Slide 9** | **Thank You** | Acknowledgments, repository artifact references, contact info, and open Q&A invitation. |

---

<!-- SLIDE 1 -->
# Slide 1: Title Slide

### **Amrita Vishwa Vidyapeetham, Amaravati Campus**
#### *Department of Artificial Intelligence & Robotics*
---

### **Lab-Based Project Title:**
## **Trajectory Tracking of an Autonomous Underwater Vehicle (AUV) Using Hybrid Physics-AI Dynamic Modeling and Model Predictive Control**

**Course:** INTRODUCTION TO AI ROBOTICS (22AIE214)  
**Date:** 11-10-2026  

---

### **Presented by:**
| Name of the Student | Roll No. | Department |
| :--- | :--- | :--- |
| **Eswar Gorantla** | *AM.EN.U4AIE220XX* *(Student Roll No.)* | Artificial Intelligence & Robotics (Sem 5) |

---

### **Under the Guidance of:**
### **Dr. Ravishankar Prakash Desai**
*Assistant Professor / Project Guide*  
*School of Artificial Intelligence, Amrita Vishwa Vidyapeetham*

> **Slide Footer:** `11-10-2026 | INTRODUCTION TO AI ROBOTICS (22AIE214) | Slide 1`

---
> 🎙️ **Speaker Notes for Slide 1:**
> *"Respected guide Dr. Ravishankar Prakash Desai and fellow peers, good morning. Today, I am presenting our semester 5 lab-based project for Introduction to AI Robotics: 'Trajectory Tracking of an Autonomous Underwater Vehicle Using Hybrid Physics-AI Dynamic Modeling and Model Predictive Control'. This project investigates how combining analytical physics equations with deep recurrent neural networks enables an underwater robotic submarine to accurately follow complex 3D paths in deep-sea environments despite currents, noise, and complex fluid damping."*

---

<!-- SLIDE 2 -->
# Slide 2: Introduction

## **Introduction: Autonomous Underwater Robotics**
### *Highlighting the Crucial Importance of the Project Work*

### 1. The Deep-Ocean Challenge
* **GPS-Denied Operation:** Radio waves attenuate rapidly in seawater; autonomous underwater vehicles (AUVs) cannot rely on GPS and must navigate purely using inertial (IMU), acoustic (DVL/USBL), and depth sensors.
* **Severe Environmental Disturbances:** Subsea currents, buoyancy variations, density stratification, and turbulence induce continuous, non-linear drift.
* **Complex Hydrodynamics:** Water is roughly 800 times denser than air. Forces like added mass, vortex shedding, and quadratic fluid drag are notoriously difficult to model analytically.

### 2. The Dual Failure of Conventional Approaches
```
        [Pure Analytical Physics]                    [Pure Data-Driven AI]
                   ❌                                          ❌
• Requires expensive tow-tank parameter tuning. • Struggles with out-of-distribution drift.
• Ignores unmodeled thruster and fluid lag.       • Overfits acoustic sensor noise (USBL jumps).
• Suffers cumulative dead-reckoning drift.      • Lacks geometric & physical guarantees.
```

### 3. The Proposed Solution: Physics-AI Hybridization + Model Predictive Control
* **Nominal Physics Prior:** A deterministic 3D marine kinematic and hydrodynamic drag model provides foundational stability and geometric conservation.
* **Neural Residual Corrector:** A deep recurrent network (GRU/LSTM) predicts the step-by-step discrepancy (*residuals*) caused by unmodeled hydrodynamic forces and thruster lag.
* **Model Predictive Control (MPC):** Embeds the hybrid forward model into an online receding-horizon optimizer that plans multi-step future maneuvers while strictly adhering to actuator limits and boundary constraints.

### 4. Real-World Applications & Impact
* Subsea oil & gas pipeline and transatlantic telecommunication cable inspections.
* High-resolution bathymetric ocean floor mapping and benthic habitat monitoring.
* Search, rescue, and salvage operations at extreme depths (up to 6,000 meters).

> **Slide Footer:** `11-10-2026 | INTRODUCTION TO AI ROBOTICS (22AIE214) | Slide 2`

---
> 🎙️ **Speaker Notes for Slide 2:**
> *"Autonomous underwater vehicles operate in one of the harshest environments on Earth. In the deep sea, GPS is unavailable, and vehicles experience complex fluid forces like drag and added mass. If we only use a textbook physics model, we miss hydrodynamic nonlinearities and thruster dynamics, causing large tracking errors. On the other hand, if we only train a pure neural network, it can hallucinate or overfit acoustic sensor noise. In this project, we create a hybrid: the physics model handles the known geometry, the AI learns the complex residual errors from real sea-trial data, and an MPC controller uses both to steer the vehicle along target trajectories."*

---

<!-- SLIDE 3 -->
# Slide 3: Problem Statement, Objectives and Programming Language

## **Problem Statement, Objectives & Technical Specifications**

### 1. Problem Statement
> *"To design, implement, and benchmark an autonomous 3D trajectory tracking framework for a deep-sea AUV subjected to low-frequency telemetry ($0.50\text{ Hz}$), acoustic measurement noise, and hydrodynamic disturbances, eliminating the tracking inaccuracies of classical PID controllers and the drift of uncalibrated physics models."*

### 2. Core Project Objectives
1. **Dynamic Modeling:** Formulate a baseline 3D kinematic model with quadratic surge drag calibrated against open-sea mission telemetry.
2. **Residual Neural Learning:** Train and benchmark feedforward (MLP) and recurrent (LSTM, GRU) neural architectures to capture 7D state transition errors under strict chronological splitting.
3. **Optimal Controller Synthesis:** Implement a receding-horizon Model Predictive Controller utilizing the Cross-Entropy Method (CEM) over a 20-second lookahead window ($H=10$ steps).
4. **Benchmarking & Validation:** Quantify one-step prediction error (RMSE), closed-loop path tracking accuracy (3D Position RMSE), depth regulation, and actuator control effort against benchmark PID and Physics-only MPC systems.

### 3. Technical Platform & Programming Stack

| Category | Specifications & Tools Adopted |
| :--- | :--- |
| **Programming Language** | **Python 3.10+** (Selected for rapid numerical prototyping, PyTorch integration, and scientific computing) |
| **Deep Learning Framework** | **PyTorch 2.x** (CUDA accelerated GPU/CPU tensors, custom recurrent residual architectures) |
| **Numerical & Scientific Math** | **NumPy, SciPy** (Vectorized matrix kinematics, numerical integration, statistical analysis) |
| **Data Processing & Analytics** | **Pandas** (Chronological sequence processing, sliding window feature extraction) |
| **Visualization & Reporting** | **Matplotlib, Seaborn** (3D trajectory renders, error distributions, radar plots) |
| **Physical Vehicle Platform** | **Autosub6000 Deep-Sea AUV** ($6.8\text{ m}$ length, $3.5\text{ tonnes}$ displacement, rated to $6,000\text{ m}$ depth) |
| **Dataset Source** | **IEEE DataPort: AURORA Mission M86 Telemetry** ($31,944$ time-series samples logged at $\Delta t = 2.0\text{ s}$) |

> **Slide Footer:** `11-10-2026 | INTRODUCTION TO AI ROBOTICS (22AIE214) | Slide 3`

---
> 🎙️ **Speaker Notes for Slide 3:**
> *"Our problem statement focuses on closing the gap between simplified physical models and real subsea telemetry. We established four clear objectives: formulating the baseline marine kinematics, training neural residual models, embedding them into a receding-horizon predictive controller, and benchmarking against industry-standard PID control. We built the complete codebase in Python 3 using PyTorch for the AI models and NumPy/SciPy for vectorized physics and MPC optimization. All our models were trained on 31,944 telemetry points from the real Autosub6000 submarine during its North Atlantic mission."*

---

<!-- SLIDE 4 -->
# Slide 4: Methodology

## **Methodology: Unified Physics-AI & Receding-Horizon Control**

### 1. 7D Physical State & 6D Continuous Feature Space
The vehicle state adheres to a full 7-degree-of-freedom kinematic representation:
$$\mathbf{x}_t = \big[x_t \ (\text{North, m}),\ y_t \ (\text{East, m}),\ z_t \ (\text{Depth, m}),\ \phi_t \ (\text{Roll, rad}),\ \theta_t \ (\text{Pitch, rad}),\ \psi_t \ (\text{Heading, rad}),\ U_t \ (\text{Surge, m/s})\big]^T \in \mathbb{R}^7$$

To prevent discontinuity singularities at the $[-\pi, \pi)$ angular boundary, the neural model receives a continuous 6D feature vector over a rolling history buffer of $T=10$ steps ($20.0\text{ seconds}$):
$$\mathbf{s}_t = [U_t,\ \phi_t,\ \theta_t,\ \cos\psi_t,\ \sin\psi_t,\ z_t]^T \in \mathbb{R}^6$$

---

### 2. Analytical Marine Physics Baseline
A deterministic 3D kinematic model projects state transitions combined with hydrodynamic quadratic drag ($K_d = 0.06073$):
$$x_{t+1}^{\text{phys}} = x_t + U_t \cos(\theta_t) \cos(\psi_t) \Delta t, \qquad y_{t+1}^{\text{phys}} = y_t + U_t \cos(\theta_t) \sin(\psi_t) \Delta t$$
$$z_{t+1}^{\text{phys}} = z_t - U_t \sin(\theta_t) \Delta t, \qquad U_{t+1}^{\text{phys}} = \max\Big(U_t + K_d (U_{\text{cruise}}^2 - U_t^2)\Delta t,\ 0.0\Big)$$

---

### 3. Neural Residual Learning (Physics + AI)
The neural network models the unmodeled dynamics residual $\mathbf{r}_t = \mathbf{x}_{t+1}^{\text{actual}} - \mathbf{x}_{t+1}^{\text{phys}}$.  
The final one-step forward forecast is given by:
$$\hat{\mathbf{x}}_{t+1} = \mathbf{x}_{t+1}^{\text{phys}} + \hat{\mathbf{r}}_t, \quad \text{where } \hat{\mathbf{r}}_t = f_{\theta_{\text{NN}}}(\mathbf{S}_{t-9:t})$$
* **Chronological Splitting:** 70% Train ($22,360$ steps), 15% Validation ($4,791$ steps), 15% strictly held-out Test ($4,792$ steps) to prevent data leakage.
* **Architectures Evaluated:** 3-layer MLP, 2-layer LSTM, and 2-layer Recurrent GRU ($64$ hidden units, $41,095$ parameters).

---

### 4. Receding-Horizon Model Predictive Control (MPC)
At each control step, the controller optimizes a guidance command sequence $\mathbf{u}_{0:H-1} = [U_{\text{cmd}}, \theta_{\text{cmd}}, \psi_{\text{cmd}}]^T$ over lookahead horizon $H=10$ ($20.0\text{ s}$):
$$\min_{\mathbf{u}_{0:H-1}} \mathcal{J} = \sum_{k=1}^H \Big( w_{\text{pos}} \|\mathbf{p}_k - \mathbf{p}_k^{\text{ref}}\|^2 + w_{\text{depth}} (z_k - z_k^{\text{ref}})^2 + w_{\text{head}} (1 - \cos(\psi_k - \psi_k^{\text{ref}})) + w_{\Delta u} \|\Delta \mathbf{u}_k\|^2 \Big) + \mathcal{J}_{\text{terminal}}$$
* **Optimizer:** Vectorized Cross-Entropy Method (CEM) sampling $K=70$ candidate trajectories over 3 refinement iterations.
* **Actuator Slew Constraints:** Enforces $|\Delta \theta| \le 0.10\text{ rad/step}$ ($2.86^\circ/\text{s}$), $|\Delta \psi| \le 0.20\text{ rad/step}$ ($5.73^\circ/\text{s}$), and $|\Delta U| \le 0.15\text{ m/s/step}$.

> 🖼️ **[NOTE: ADD OPTIONAL IMAGE HERE — Autosub6000 Sea-Trial Survey Path]**
> * **File to insert:** [`presentationphase1/results/eda/eda_survey_path.png`](file:///c:/Users/eswar/OneDrive/Dokumen/RoboticsSEM5/presentationphase1/results/eda/eda_survey_path.png)
> * **Slide Placement:** Bottom-right corner or side banner.
> * **Visual Description / Caption:** 2D East-North lawnmower survey path of the Autosub6000 in the North Atlantic during Mission M86 showing the 31,944 collected telemetry points.

> **Slide Footer:** `11-10-2026 | INTRODUCTION TO AI ROBOTICS (22AIE214) | Slide 4`

---
> 🎙️ **Speaker Notes for Slide 4:**
> *"Our methodology is divided into two distinct components. First, the nominal physics model integrates surge speed, pitch, and heading to estimate where the vehicle will translate over the next 2 seconds. Second, we extract the residual error between what the physics predicted and what the vehicle actually did in the sea trials shown in the survey plot. A 2-layer GRU neural network learns these residuals from a 10-step history window. In Phase 2, our receding-horizon MPC uses this hybrid model to roll out 70 candidate trajectories 20 seconds into the future, selecting the best controls via the Cross-Entropy Method while penalizing actuator jerk and respecting physical turning limits."*

---

<!-- SLIDE 5 -->
# Slide 5: Flow-chart

## **System Architecture & Operational Flow-chart**

```mermaid
flowchart TD
    subgraph S1["PHASE 1: Offline Dynamic Identification & Residual Learning"]
        D1["Raw Autosub6000 Telemetry (M86)<br>31,944 Samples @ 0.50 Hz (dt=2.0s)"] --> D2["Chronological Partition<br>Train: 70% | Val: 15% | Test: 15%"]
        D2 --> D3["3D Marine Kinematic Baseline<br>+ Quadratic Surge Drag (Kd=0.06073)"]
        D2 & D3 --> D4["Compute Residual Targets<br>r(t) = x_actual(t+1) - x_physics(t+1)"]
        D4 --> D5["Feature Normalization & Windowing<br>10 Timesteps x 6D State Matrix"]
        D5 --> D6["Train Neural Residual Models<br>MLP | LSTM | GRU (Adam, lr=1e-3)"]
        D6 --> D7["Best Model Checkpoint Export<br>best_dynamics_model.pt + scaler.pkl"]
    end

    subgraph S2["PHASE 2: Online Receding-Horizon Model Predictive Control"]
        C1["3D Reference Mission Path<br>Sinusoidal Waypoints (x_ref, y_ref, z_ref)"] --> C2["Current Vehicle State Estimate<br>x_t = [x, y, z, phi, theta, psi, U]"]
        C2 --> C3["MPC Receding Horizon (H=10, 20s Lookahead)<br>Initialize Candidate Action Distributions"]
        C3 --> C4["Cross-Entropy Method (CEM)<br>Sample K=70 Candidate Sequences"]
        D7 -.->|"Zero-Duplication Model Import"| C5["Vectorized Hybrid Forward Rollout<br>Nominal Physics + GRU Residual Prediction"]
        C4 --> C5
        C5 --> C6["Multi-Objective Cost Evaluation<br>Position Error + Depth + Heading + Slew Penalty"]
        C6 --> C7["Elite Trajectory Selection & Gaussian Update<br>(Top 20% Candidates, 3 Iterations)"]
        C7 --> C8["Optimal Control Extraction u_0*<br>[Speed_cmd, Pitch_cmd, Heading_cmd]"]
        C8 --> C9["Actuator Slew Rate Limiter<br>Enforce Physical Fin & Thruster Bounds"]
        C9 --> C10["Simulated Plant Execution<br>Under Environmental Disturbances"]
        C10 --> C11["Slide History Buffer Window (T=10)"]
        C11 --> C2
    end
```

### Detailed Execution Stages:
1. **Telemetry Processing:** Telemetry is cleaned and parsed chronologically into 7D states.
2. **Residual Generation:** The analytical model predicts step $t+1$; residuals isolate fluid damping and thruster latency.
3. **Deep Residual Network:** A 2-layer GRU minimizes MSE on validation data with early stopping.
4. **CEM Optimization:** The MPC generates 70 candidate maneuvers over 10 future steps, evaluating each using the hybrid model.
5. **Closed-Loop Execution:** Only the first control setpoint ($u_0^*$) is commanded; the vehicle moves, measurements are updated, and the horizon recedes.

> **Slide Footer:** `11-10-2026 | INTRODUCTION TO AI ROBOTICS (22AIE214) | Slide 5`

---
> 🎙️ **Speaker Notes for Slide 5:**
> *"This flowchart shows how our offline learning connects seamlessly to our online control loop. In the top block, offline training extracts the residual targets and trains our GRU network. In the bottom block, the online control loop runs in real-time: given the current vehicle state and the target trajectory, the MPC samples 70 candidate control sequences. The candidate paths are projected forward 20 seconds using our hybrid physics-AI model, scored against our cost function, and the optimal first control step is applied to the vehicle. Then the time window shifts forward by 2 seconds and repeats."*

---

<!-- SLIDE 6 -->
# Slide 6: Simulation Result

## **Simulation Results & Quantitative Benchmarking**

---

### 1. One-Step-Ahead Predictor Test Set Benchmark (RMSE)
*Evaluated on 4,792 unseen test steps ($\Delta t = 2.0\text{ s}$). Lower is better.*

| Telemetry State Variable | Analytical Physics | Physics + MLP | Physics + LSTM | **Physics + GRU (Best)** | **AI Improvement (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Depth $z$ (m)** | 0.1230 | 0.1060 | 0.1057 | **0.1049** | **+14.7%** |
| **Roll $\phi$ (rad)** | 0.0770 | 0.0938 | 0.0730 | **0.0718** | **+6.8%** |
| **Pitch $\theta$ (rad)** | 0.0454 | 0.0407 | 0.0378 | **0.0375** | **+17.4%** |
| **Heading $\psi$ (rad)** | 0.0572 | 0.0873 | 0.0500 | **0.0472** | **+17.5%** |
| **Surge Speed $U$ (m/s)** | 0.2254 | 0.2426 | 0.1775 | **0.1759** | **+22.0%** |
| **North Position $x$ (m)** | **0.7112** | 1.3140 | 1.0110 | 0.9518 | *Baseline Best (Noise Overfit)* |
| **East Position $y$ (m)** | **0.6927** | 0.7833 | 0.7437 | 0.7724 | *Baseline Best (Noise Overfit)* |

> 🖼️ **[NOTE: ADD IMAGE 1 HERE — Predictor RMSE Benchmark Bar Chart]**
> * **File to insert:** [`outputs/predictor_rmse_comparison.png`](file:///c:/Users/eswar/OneDrive/Dokumen/RoboticsSEM5/outputs/predictor_rmse_comparison.png)  
>   *(Alternative location: `AURORA_AUV_ML_MPC/results/model_comparison/predictor_rmse_comparison.png`)*
> * **Slide Placement:** Place on the **Right Side** or **Directly Below** Table 1.
> * **Visual Description / Caption:** Normalized error bar chart comparing Physics vs. MLP vs. LSTM vs. GRU. Visually demonstrates that GRU/LSTM models reduce Heading error by 17.5% and Speed error by 22.0% relative to the 100% baseline physics.

---

### 2. Closed-Loop 3D Trajectory Tracking Simulation Benchmark
*Simulated over a 3D wave-tracking mission under ocean current disturbances. Lower RMSE indicates superior tracking.*

| Controller Configuration | Position RMSE (m) | Depth RMSE (m) | Heading RMSE (rad) | Control Effort | Performance Summary |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Benchmark PID Controller** | 39.79 m | **2.87 m** | **0.3990 rad** | **0.054** | Reactive, lags behind sharp turns. |
| **Physics-Only MPC** | 116.79 m | 4.34 m | 1.0567 rad | 68.94 | Drifts significantly due to unmodeled drag. |
| **AI-Enhanced MPC (GRU)** | 81.98 m | 6.88 m | 0.4594 rad | 53.52 | Susceptible to multi-step recursive error drift. |
| **AI-Enhanced MPC (LSTM)** | **12.17 m** | **2.91 m** | 0.4037 rad | 61.92 | **BEST:** **69.4% error reduction vs. PID; 89.6% vs. Physics MPC.** |

> 🖼️ **[NOTE: ADD IMAGE 2 HERE — Closed-Loop Controller Comparison Trajectory Plot]**
> * **File to insert:** [`outputs/controller_comparison.png`](file:///c:/Users/eswar/OneDrive/Dokumen/RoboticsSEM5/outputs/controller_comparison.png)  
>   *(Alternative location: `AURORA_AUV_ML_MPC/results/control/controller_comparison.png`)*
> * **Slide Placement:** Place on the **Right Side** or **Directly Below** Table 2.
> * **Visual Description / Caption:** 2D & 3D comparative path tracking showing desired reference path vs. PID vs. Physics-MPC vs. AI-Enhanced MPC. Shows how LSTM-MPC tightly tracks the target while standard controllers experience significant lateral overshoot and drift.

> 💡 *Optional Additional Plot (If creating a 2nd Results Slide / 3D Perspective):*
> * **File to insert:** [`Phase_2/results/plots/A_to_B_3D.png`](file:///c:/Users/eswar/OneDrive/Dokumen/RoboticsSEM5/Phase_2/results/plots/A_to_B_3D.png) *(3D Goal-directed waypoint tracking flight path)* or [`Phase_2/results/plots/baseline_comparison_2D.png`](file:///c:/Users/eswar/OneDrive/Dokumen/RoboticsSEM5/Phase_2/results/plots/baseline_comparison_2D.png)

---

### 3. Core Simulation Execution Code (Python / PyTorch)
```python
# Receding Horizon MPC Step with Physics-AI Hybrid Dynamics
def mpc_step(current_state, history_buffer, ref_trajectory, model, horizon=10, K=70):
    # Sample candidate control sequences using Cross-Entropy Method
    candidate_actions = np.random.normal(mean_u, std_u, size=(K, horizon, 3))
    candidate_actions = enforce_actuator_slew_limits(candidate_actions, current_state)
    
    # Vectorized forward rollout: Nominal Physics + GRU Residual Correction
    trajectories = np.zeros((K, horizon, 7))
    for step in range(horizon):
        phys_next = analytical_marine_step(trajectories[:, step], candidate_actions[:, step])
        ai_residual = model.predict_residual(history_buffer)
        trajectories[:, step + 1] = phys_next + ai_residual
        
    # Evaluate Multi-Objective Cost & pick optimal first action
    costs = compute_trajectory_cost(trajectories, ref_trajectory)
    best_action = candidate_actions[np.argmin(costs), 0]
    return best_action # Command u_0* to vehicle thrusters & rudders
```

> **Slide Footer:** `11-10-2026 | INTRODUCTION TO AI ROBOTICS (22AIE214) | Slide 6`

---
> 🎙️ **Speaker Notes for Slide 6:**
> *"Looking at the numerical results: on the predictor side, as shown in Image 1, our GRU residual network reduced heading error by 17.5% and speed error by 22% compared to pure physics. More importantly, when we look at the closed-loop tracking plot in Image 2, the LSTM-Enhanced MPC achieved an exceptional position RMSE of just 12.17 meters. Compare that to the standard PID controller, which had an error of 39.79 meters, and the Physics-only MPC, which had an error of 116.79 meters. Our hybrid architecture delivers an overall 69.4% error reduction over PID and 89.6% over classical physics MPC."*

---

<!-- SLIDE 7 -->
# Slide 7: Conclusion

## **Conclusion & Key Takeaways**
### *Summary of Final Project Outcomes and Discoveries*

### 1. Key Project Achievements
* **Validated Superiority of Hybrid AI:** Demonstrated that combining deterministic marine physics with recurrent residual learning outperforms both purely classical controllers and uncalibrated models.
* **Significant Tracking Precision Gain:** The AI-Enhanced MPC achieved **$12.17\text{ m}$** 3D tracking error, providing a **$69.4\%$ performance gain over classical PID** and **$89.6\%$ over Physics-only MPC**.
* **Zero Duplication Architecture:** Engineered a modular, production-ready codebase linking Phase 1 dynamic identification artifacts directly into Phase 2 control optimization without redundant code.

### 2. Critical Scientific Discoveries & Insights
* **The Acoustic Sensor Noise Paradox:** All neural networks worsened raw North and East position predictions because raw acoustic USBL measurements contain jumpy measurement noise. The network overfit to high-frequency sensor noise rather than vehicle dynamics.
  * *Lesson:* Kinematic position should remain grounded in analytical physics integration, while AI should focus strictly on fluid damping, attitude rates, and surge drag.
* **GRU vs. LSTM in Multi-Step Rollouts:** While the GRU achieved slightly lower one-step test RMSE ($0.1759\text{ m/s}$ vs. $0.1775\text{ m/s}$), the LSTM proved substantially superior inside the closed-loop MPC ($12.17\text{ m}$ vs. $81.98\text{ m}$). The LSTM’s decoupled cell state memory mitigates cumulative compounding errors across multi-step recursive rollouts.

### 3. Future Work & Enhancements
1. **Sensor Filtering (EKF / UKF):** Integrate an Extended Kalman Filter to fuse DVL, IMU, and USBL data before passing features to the neural network.
2. **Online Parameter Identification:** Implement recursive least-squares (RLS) to adaptively estimate hydrodynamics ($K_d$, added mass) during mission execution.
3. **Hardware-in-the-Loop (HIL):** Deploy the MPC pipeline onto an embedded NVIDIA Jetson platform to validate real-time computational execution speeds under $100\text{ ms}$ control loops.

> **Slide Footer:** `11-10-2026 | INTRODUCTION TO AI ROBOTICS (22AIE214) | Slide 7`

---
> 🎙️ **Speaker Notes for Slide 7:**
> *"In conclusion, our research demonstrates that hybrid physics-AI modeling is a practical and superior paradigm for underwater robotics. We discovered that neural networks excel at modeling fluid drag and turn lag, but should not be allowed to fit noisy acoustic position measurements directly. Furthermore, we showed that while GRUs can look slightly better on single-step offline metrics, LSTMs are far more robust for long-horizon closed-loop control. In future work, we plan to implement an Extended Kalman Filter to clean sensor noise upfront and deploy our code onto an onboard embedded computer."*

---

<!-- SLIDE 8 -->
# Slide 8: References

## **References & Academic Literature**

1. **Fossen, T. I. (2011).** *Handbook of Marine Craft Hydrodynamics and Motion Control.* John Wiley & Sons. (Foundational formulations for 6-DOF marine kinematics, added mass, and damping).
2. **McPhail, S. D., et al. (2010).** "Autosub6000: A Deep Diving Autonomous Underwater Vehicle for Oceanographic Science." *IEEE/OES Autonomous Underwater Vehicles (AUV)*, Monterey, CA, pp. 1–6.
3. **Williams, D. P., & Fakiris, E. (2020).** "AURORA: Multi-Sensor Autonomous Underwater Vehicle Dataset." *IEEE DataPort*, doi: 10.21227/aurora-m86.
4. **Camacho, E. F., & Alba, C. B. (2013).** *Model Predictive Control.* Springer Science & Business Media.
5. **Rubinstein, R. Y., & Kroese, D. P. (2004).** *The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte Carlo Simulation and Machine Learning.* Springer.
6. **Hochreiter, S., & Schmidhuber, J. (1997).** "Long Short-Term Memory." *Neural Computation*, 9(8), pp. 1735–1780.
7. **Cho, K., et al. (2014).** "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." *arXiv preprint arXiv:1406.1078*.
8. **Paszke, A., et al. (2019).** "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

> **Slide Footer:** `11-10-2026 | INTRODUCTION TO AI ROBOTICS (22AIE214) | Slide 8`

---
> 🎙️ **Speaker Notes for Slide 8:**
> *"Our project builds on established literature in marine hydrodynamics, particularly Thor Fossen's textbook, the Autosub6000 engineering documentation, the IEEE DataPort AURORA dataset, standard Model Predictive Control literature, and pioneering papers on recurrent neural architectures like LSTM and GRU."*

---

<!-- SLIDE 9 -->
# Slide 9: Thank You

## **Thank You!**

### **Questions & Discussion**

---

### **Project Summary at a Glance:**
* **Project Title:** Trajectory Tracking of an AUV Using Hybrid Physics-AI Modeling & MPC
* **Platform:** Autosub6000 / AURORA Mission M86 Telemetry ($31,944$ points)
* **Key Achievement:** **$69.4\%$ tracking accuracy improvement** over standard PID controllers
* **Best Architecture:** LSTM-Enhanced Receding-Horizon MPC ($12.17\text{ m}$ 3D RMSE)

---

### **Contact & Project Artifacts:**
* **Student Presenter:** Eswar Gorantla
* **Guide:** Dr. Ravishankar Prakash Desai
* **Course:** Introduction to AI Robotics (22AIE214)
* **Campus:** Amrita Vishwa Vidyapeetham, Amaravati Campus
* **Workspace Repository:** `RoboticsDynamics / RoboticsSEM5`

> **Slide Footer:** `11-10-2026 | INTRODUCTION TO AI ROBOTICS (22AIE214) | Slide 9`

---
> 🎙️ **Speaker Notes for Slide 9:**
> *"Thank you very much, Dr. Ravishankar Prakash Desai and everyone present, for your time and guidance throughout this project. I am now open to any questions, comments, or technical discussion regarding our dynamic models, residual neural networks, or MPC implementation."*
