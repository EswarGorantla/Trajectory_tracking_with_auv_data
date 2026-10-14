# PHASE 2 EXPERIMENTAL REPORT: Goal-Directed AUV Model Predictive Control (MPC)

**Project Title:** Goal-Directed Model Predictive Control of an Autonomous Underwater Vehicle Using Phase 1 Physics + GRU Residual Dynamics  
**Module:** Phase 2  
**Dataset Reference:** Autosub6000 Telemetry (AURORA Mission M86) + Hybrid GRU Residual Model Checkpoint

---

## 1. Problem Definition
In Phase 1, we developed and validated a **Hybrid Physics + Deep GRU Residual Dynamics Model** capable of accurately predicting one-step and multi-step state evolutions on real deep-sea AUV telemetry.  
However, open-loop prediction is insufficient for real-world robotics missions. The vehicle must autonomously navigate from an arbitrary starting location **Point A** to a commanded target **Point B**, remaining strictly within 3D geographic operating boundaries, respecting vehicle pitch/roll/speed actuation limits, and avoiding underwater hazards.  
Phase 2 addresses this challenge by formulating and deploying a **closed-loop, receding-horizon Model Predictive Controller (MPC)** that embeds the Phase 1 learned dynamics model inside the optimization loop.

---

## 2. Relation to Phase 1
Phase 2 directly inherits and utilizes the trained Phase 1 artifacts without modifying or retraining them:
* **Model Checkpoint:** [`presentationphase1/models/best_dynamics_model.pt`](../presentationphase1/models/best_dynamics_model.pt) (41,095 trainable weights, 2 recurrent GRU layers with 64 units each).
* **Fitted Scalers:** [`presentationphase1/models/scaler.pkl`](../presentationphase1/models/scaler.pkl) (`StandardScaler` fitted on 22,360 training samples).
* **Physics Parameters:** Drag coefficient $K_d = 0.06073$, cruise speed $U_{\text{cruise}} = 1.2\text{ m/s}$, sampling time $\Delta t = 2.0\text{ s}$ ($0.50\text{ Hz}$).
* **Interface Abstraction:** The Phase 1 model is loaded dynamically via [`Phase_2/models/phase1_loader.py`](models/phase1_loader.py), guaranteeing strict read-only preservation of Phase 1.

---

## 3. State Representation
Phase 2 adheres strictly to the 7D state space established and validated in Phase 1:
$$\mathbf{x}_t = [x_t, y_t, z_t, \phi_t, \theta_t, \psi_t, U_t]^T \in \mathbb{R}^7$$

| Variable | Physical Meaning | Telemetry Source / Units |
| :--- | :--- | :--- |
| $x_t$ | Local Tangent North coordinate | Transponder Dead-reckoning ($m$) |
| $y_t$ | Local Tangent East coordinate | Transponder Dead-reckoning ($m$) |
| $z_t$ | Operating depth below sea surface | Digiquartz Pressure Sensor ($m$) |
| $\phi_t$ | Roll attitude angle | PHINS Inertial Navigation ($rad$) |
| $\theta_t$ | Pitch attitude angle | PHINS Inertial Navigation ($rad$) |
| $\psi_t$ | Heading angle clockwise from North | PHINS Inertial Navigation ($rad$) |
| $U_t$ | Forward surge speed over seabed | 300 kHz Workhorse DVL ($m/s$) |

### Autopilot Control Vector ($\mathbf{u}_t \in \mathbb{R}^3$)
$$\mathbf{u}_t = [U_{\text{cmd}}, \theta_{\text{cmd}}, \psi_{\text{cmd}}]^T$$
Because Phase 1 telemetry did not directly record raw thruster RPM or fin deflection angles, we formulate the control input as standard autopilot steering commands. The actuator response models realistic fin rate limits ($\Delta\theta_{\max} = 0.1\text{ rad/step}$, $\Delta\psi_{\max} = 0.2\text{ rad/step}$) and propulsion inertia.

---

## 4. Dynamics Model (Physics + GRU Residual)
At every step of the MPC prediction horizon, future states are generated via the two-part hybrid formulation:
$$\mathbf{x}_{k+1} = \mathbf{x}_{k+1}^{\text{physics}} + \hat{\mathbf{r}}_k^{\text{GRU}}$$

1. **Deterministic 3D Marine Kinematics:**
   $$x_{k+1}^{\text{phys}} = x_k + U_k \cos\theta_k \cos\psi_k \Delta t$$
   $$y_{k+1}^{\text{phys}} = y_k + U_k \cos\theta_k \sin\psi_k \Delta t$$
   $$z_{k+1}^{\text{phys}} = z_k - U_k \sin\theta_k \Delta t$$
   $$U_{k+1}^{\text{phys}} = \max\left(U_k + K_d (U_{\text{cruise}}^2 - U_k^2) \Delta t, 0.0\right)$$
2. **Phase 1 Deep GRU Residual Correction:**
   Takes a rolling window of 10 feature vectors $\mathbf{s} = [U, \phi, \theta, \cos\psi, \sin\psi, z]^T \in \mathbb{R}^6$ and predicts the 7D residual vector $\hat{\mathbf{r}}_k = [\Delta x, \Delta y, \Delta z, \Delta\phi, \Delta\theta, \Delta\psi, \Delta U]^T$, compensating for hydrodynamic damping, turning yaw lag, and unmodeled buoyancy forces.

---

## 5. MPC Formulation & Optimization
At each control cycle $t$:
1. Measure the current state $\mathbf{x}_t$ and retrieve the 10-step feature history $\mathbf{S}_t$.
2. Seed candidate control sequences using **warm-starting** (shifting the previous optimal control sequence by 1 step) and Line-Of-Sight guidance.
3. Sample $K=70$ candidate control trajectories using a Cross-Entropy distribution over horizon $H$.
4. Simulate all $K$ candidate trajectories forward in parallel using the vectorized `DynamicsInterface`.
5. Compute the multi-objective cost $J$ for each candidate sequence.
6. Select the lowest-cost sequence $\mathbf{u}_{0:H-1}^*$.
7. **Receding Horizon Principle:** Apply **ONLY the first control action $\mathbf{u}_0^*$** to the plant.
8. Advance the plant, slide the feature history window, and repeat.

---

## 6. Prediction Horizon
* Horizon Steps: $H = 10\text{ steps}$
* Timestep: $\Delta t = 2.0\text{ seconds}$
* **Total Physical Prediction Horizon:** $10 \times 2.0\text{ s} = \mathbf{20.0\text{ seconds}}$
* Lookahead Distance: At nominal cruise speed ($1.2\text{ m/s}$), the vehicle predicts $24.0\text{ meters}$ into the future, providing ample distance to identify boundary walls and obstacles before turning limits prevent evasion.

---

## 7. Multi-Objective Cost Function
$$J = w_{\text{goal}} J_{\text{goal}} + w_{\text{track}} J_{\text{track}} + w_{\text{ctrl}} J_{\text{ctrl}} + w_{\text{smooth}} J_{\text{smooth}} + w_{\text{constraint}} J_{\text{constraint}}$$

* **$w_{\text{goal}} = 30.0$:** Minimizes terminal distance to Goal B $\|\mathbf{p}_H - \mathbf{p}_B\|^2$ and running distance decay.
* **$w_{\text{track}} = 6.0$:** Encourages alignment with the 3D Line-Of-Sight (LOS) heading and pitch angles.
* **$w_{\text{ctrl}} = 1.0$:** Penalizes propulsion power deviations from economic cruise speed ($1.2\text{ m/s}$).
* **$w_{\text{smooth}} = 4.0$:** Penalizes high slew rate $\|\mathbf{u}_k - \mathbf{u}_{k-1}\|^2$, ensuring smooth, non-oscillatory steering.
* **$w_{\text{constraint}} = 200.0$:** Quadratic barrier penalty applied to any state breaching 3D boundaries, entering obstacle zones, or exceeding pitch/roll limits.

---

## 8. Constraints Management
Implemented in [`src/constraints.py`](src/constraints.py):
1. **3D Spatial Boundaries:**
   * $\text{North} \in [0.0, 120.0]\text{ m}$
   * $\text{East} \in [0.0, 100.0]\text{ m}$
   * $\text{Depth} \in [10.0, 60.0]\text{ m}$
2. **Vehicle Dynamics Limits:**
   * Surge Speed: $U \in [0.5, 2.0]\text{ m/s}$
   * Pitch Limit: $|\theta| \le 0.35\text{ rad}$ ($\approx 20.0^\circ$)
   * Roll Limit: $|\phi| \le 0.50\text{ rad}$ ($\approx 28.6^\circ$)
   * Pitch Slew Limit: $|\Delta\theta| \le 0.10\text{ rad/step}$
   * Yaw Slew Limit: $|\Delta\psi| \le 0.20\text{ rad/step}$
3. **Obstacle Keep-Out Zones:**
   * Cylindrical obstacles with physical radius $r$ and safety standoff buffer $r_{\text{margin}} = 4.0\text{ m}$.

---

## 9. Simulation Setup
* **Engine:** [`src/simulator.py`](src/simulator.py)
* **True Plant:** Advances state using deterministic physics combined with the PyTorch Phase 1 GRU model.
* **Stopping Criteria:** $\text{Distance to Goal} \le 5.0\text{ m}$ OR $\text{Steps} \ge 120$ ($240\text{ s}$).

---

## 10. Experiment 1: Basic $A \to B$ Goal Reaching
* **Start Point A:** $(0.0, 0.0, 20.0)\text{ m}$
* **Goal Point B:** $(100.0, 80.0, 40.0)\text{ m}$
* **Acceptance Radius:** $5.0\text{ m}$

### Results:
* **Goal Reached:** **YES**
* **Time to Goal:** **$90.0\text{ s}$** ($45\text{ control cycles}$)
* **Final Position:** $(101.70, 75.44, 39.12)\text{ m}$
* **Final Distance to Goal:** **$4.94\text{ m}$**
* **Total Path Length:** **$131.02\text{ m}$** (Straight-line distance = $129.61\text{ m}$ $\implies$ **$98.9\%$ path efficiency**)
* **Boundary Violations:** **$0$**
* **Constraint Violations:** **$0$**
* **Total Control Effort:** $203.76$
* **Learned Model Calls:** $1,425\text{ batches}$ ($96,645\text{ states evaluated}$)

---

## 11. Experiment 2: Multi-Scenario Testing
Tested 3 diverse mission profiles across the 3D volume:

| Scenario | Description | Start $(N, E, Z)$ | Goal $(N, E, Z)$ | Goal Reached | Time to Goal (s) | Path Length (m) | Final Error (m) | Violations | Control Effort |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 1** | Diagonal climbing dive (North-East) | $(0, 0, 20)$ | $(100, 80, 40)$ | **YES** | $90.0$ | $130.54$ | $4.20$ | **0** | $270.18$ |
| **Case 2** | Interior boundary transit (Eastward bias) | $(20, 10, 30)$ | $(80, 90, 50)$ | **YES** | $70.0$ | $100.09$ | $4.79$ | **0** | $112.69$ |
| **Case 3** | South-East ascent transit (High depth to shallow) | $(100, 20, 50)$ | $(20, 80, 20)$ | **YES** | $64.0$ | $103.39$ | $4.59$ | **0** | $6.89$ |

*Summary:* $100\%$ mission success rate across all 3 scenarios with zero boundary breaches.

---

## 12. Experiment 3: Boundary Enforcement
* **Scenario:** Operating area constrained with East wall at $\text{East}_{\max} = 70.0\text{ m}$. Start $A = (10, 10, 25)$, Goal $B = (85, 55, 35)$.
* **Direct Path:** Unconstrained trajectory heading towards $(85, 95)$ would cross the wall at $E > 70\text{ m}$.
* **MPC Result:** The MPC controller actively bent the vehicle's trajectory along the safe interior of the corridor, reaching Goal B in **$62.0\text{ s}$** with **$0$ boundary violations**.

---

## 13. Experiment 4: 3D Cylindrical Obstacle Avoidance
* **Obstacle:** Cylinder centered at $(50.0, 40.0)\text{ m}$ directly in the center of the direct $A \to B$ line-of-sight. Radius $r = 8.0\text{ m}$, safety standoff margin $r_{\text{safe}} = 12.0\text{ m}$.
* **Result:**
  * **Goal Reached:** **YES** ($t = 98.0\text{ s}$)
  * **Final Error:** $4.93\text{ m}$
  * **Minimum Distance to Obstacle Center:** **$12.30\text{ m}$** ($> 12.0\text{ m}$ safety margin)
  * **Obstacle Collisions:** **$0$**

---

## 14. Experiment 5: Baseline Comparison (Reactive LOS vs Hybrid MPC)
Evaluated on the exact same obstacle scenario:

| Metric | Baseline (Reactive LOS Tracker) | Proposed (Receding-Horizon Hybrid MPC) | Performance Advantage |
| :--- | :---: | :---: | :--- |
| **Goal Reached** | YES | YES | Both reach destination |
| **Final Distance Error** | $4.02\text{ m}$ | $4.80\text{ m}$ | Comparable accuracy within tolerance |
| **Obstacle Collisions** | **6 steps ($12\text{ seconds}$ inside obstacle)** | **0 collisions (Safe)** | **MPC prevents catastrophic hull collision** |
| **Total Constraint Violations**| **6 violations** | **0 violations** | **$100\%$ constraint satisfaction for MPC** |
| **Time to Goal** | $92.0\text{ s}$ | $92.0\text{ s}$ | Equal transit duration |
| **Total Path Length** | $126.14\text{ m}$ | $134.00\text{ m}$ | $+6.2\%$ path length for safe circumnavigation |
| **Control Effort** | $0.00$ | $185.15$ | Proactive steering maneuvers |

---

## 15. Experiment 6: Robustness to Ocean Currents & Sensor Noise
Tested under 3 environmental conditions:
1. **Nominal:** Calm water, perfect state feedback.
2. **Ocean Current:** Simulated constant bottom drift $\mathbf{v}_{\text{current}} = [0.15, -0.12, 0.05]\text{ m/s}$ ($0.20\text{ m/s}$ total cross-current).
3. **Current + Noise:** Simulated $0.20\text{ m/s}$ current drift + $\sigma = 0.50\text{ m}$ Gaussian USBL acoustic position jitter.

| Condition | Goal Reached | Time to Goal (s) | Final Error (m) | Path Length (m) | Boundary Violations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nominal** | **YES** | $90.0$ | $4.14$ | $127.92$ | $0$ |
| **Ocean Current ($0.20\text{ m/s}$)** | **YES** | $88.0$ | $4.63$ | $132.53$ | $0$ |
| **Current + USBL Noise ($\sigma=0.5\text{m}$)** | **YES** | $94.0$ | $4.65$ | $132.51$ | $0$ |

*Conclusion:* Receding-horizon replanning compensates continuously for environmental drift and acoustic jitter without steady-state offset.

---

## 16. Consolidated Metrics Table

| Experiment | Configuration | Steps | Time (s) | Path Length (m) | Final Error (m) | Violations | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A -> B Core** | Start $(0,0,20) \to$ Goal $(100,80,40)$ | 45 | $90.0$ | $131.02$ | $4.94$ | $0$ | **SUCCESS** |
| **Multi-Scenario 1** | Start $(0,0,20) \to$ Goal $(100,80,40)$ | 45 | $90.0$ | $130.54$ | $4.20$ | $0$ | **SUCCESS** |
| **Multi-Scenario 2** | Start $(20,10,30) \to$ Goal $(80,90,50)$ | 35 | $70.0$ | $100.09$ | $4.79$ | $0$ | **SUCCESS** |
| **Multi-Scenario 3** | Start $(100,20,50) \to$ Goal $(20,80,20)$ | 32 | $64.0$ | $103.39$ | $4.59$ | $0$ | **SUCCESS** |
| **Boundary Test** | East corridor constrained to $70\text{ m}$ | 31 | $62.0$ | $88.03$ | $4.24$ | $0$ | **SUCCESS** |
| **Obstacle Test** | Cylindrical obstacle at $(50, 40)$ | 49 | $98.0$ | $134.00$ | $4.93$ | $0$ | **SUCCESS** |
| **Robustness (Current)** | $0.20\text{ m/s}$ cross-current drift | 44 | $88.0$ | $132.53$ | $4.63$ | $0$ | **SUCCESS** |
| **Robustness (Noise)** | Current $+ 0.5\text{m}$ USBL jitter | 47 | $94.0$ | $132.51$ | $4.65$ | $0$ | **SUCCESS** |

---

## 17. Visualizations Gallery

### Basic $A \to B$ Trajectories
* **2D Horizontal Path:** [`results/plots/A_to_B_2D.png`](results/plots/A_to_B_2D.png)
* **Depth Profile:** [`results/plots/A_to_B_depth.png`](results/plots/A_to_B_depth.png)
* **3D Flight Path:** [`results/plots/A_to_B_3D.png`](results/plots/A_to_B_3D.png)

### Boundary Enforcement
* **Corridor Trajectory:** [`results/plots/boundary_experiment_2D.png`](results/plots/boundary_experiment_2D.png)
* **3D Boundary Path:** [`results/plots/boundary_experiment_3D.png`](results/plots/boundary_experiment_3D.png)

### Obstacle Avoidance & Baseline Comparison
* **Obstacle Circumnavigation:** [`results/plots/obstacle_experiment_2D.png`](results/plots/obstacle_experiment_2D.png)
* **Baseline vs MPC Overlay:** [`results/plots/baseline_comparison_2D.png`](results/plots/baseline_comparison_2D.png)

### Disturbance Robustness
* **Current & Sensor Noise Tracks:** [`results/plots/robustness_comparison_2D.png`](results/plots/robustness_comparison_2D.png)

---

## 18. Technical Limitations
1. **Actuator Telemetry Absence:** Because actual fin deflection angles ($\delta_r, \delta_s$) were unlogged in Mission M86, control is implemented via autopilot guidance targets ($U_{\text{cmd}}, \theta_{\text{cmd}}, \psi_{\text{cmd}}$) with simulated rate limits rather than direct low-level PWM torque control.
2. **Static Obstacle Assumption:** Obstacles were modeled as stationary cylindrical volumes; dynamic moving ship/marine life avoidance was not evaluated.
3. **Sampling Resolution:** The system operates at $\Delta t = 2.0\text{ s}$ ($0.50\text{ Hz}$), which is optimal for long-range cruising navigation but coarse for high-frequency docking maneuvers.

---

## 19. Future Work
1. **Coupled EKF Navigation:** Integrate an onboard Extended Kalman Filter fusing DVL bottom-track velocities with USBL acoustic fixes prior to MPC state evaluation.
2. **Dynamic Ocean Current Estimation:** Implement an online disturbance observer estimating live 3D water currents and updating the physics regressor in real-time.
3. **Multi-AUV Swarm Coordination:** Extend the cost function to penalize inter-agent proximity for decentralized multi-vehicle survey formations.

---

## 20. Conclusion
Phase 2 successfully accomplished the complete transition from passive trajectory prediction to **active, closed-loop, goal-directed AUV control**.  
By embedding the validated Phase 1 Physics + GRU dynamics model inside a receding-horizon optimization loop, the controller demonstrated:
1. Reliable $A \to B$ navigation reaching the goal within $5.0\text{ m}$ tolerance across all test cases.
2. Strict compliance with 3D operating boundaries ($0$ violations).
3. Safe predictive avoidance of underwater obstacles, whereas reactive baselines resulted in direct collisions.
4. Disturbance rejection against simulated ocean currents and sensor measurement noise.

All code, simulation pipelines, telemetry CSVs, and figures are fully reproducible within the `Phase_2/` directory.
