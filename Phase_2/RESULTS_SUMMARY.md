# Phase 2 Executive Results Summary: 10 Critical Findings

**Module:** Phase 2 (Goal-Directed AUV Model Predictive Control)  
**Authors:** Senior Robotics/Control Research Engineer  
**Date:** September 2026  
**Status:** Empirically Verified Against Closed-Loop Simulation Suite  

---

### Finding 1: Receding-Horizon Lookahead Eliminates Hull Collisions
Against an independent continuous-time true plant integrated via 4th-order Runge-Kutta with actuator lag and $25\%$ hydrodynamic drag mismatch, the proposed **Hybrid MPC achieved 0 physical hull collisions** across all canonical obstacle trials. In contrast, the reactive Line-Of-Sight (LOS) baseline suffered catastrophic collisions in $100\%$ of trials, spending $5$ to $7$ control steps ($10-14\text{ seconds}$) inside the obstacle envelope.

### Finding 2: Reactive Artificial Potential Fields (APF) Succeed on Isolated Hazards but Struggle in Constrained Corridors
The APF baseline successfully avoids single obstacles at near-zero computational cost. However, in narrow obstacle gaps ($8\text{ m}$ opening) and near geofence boundaries, APF suffers from repulsive force cancellation, producing high control chatter and erratic yaw oscillations. Receding-horizon MPC plans smooth, kinematically feasible trajectories through corridors where reactive fields chatter.

### Finding 3: AI Residuals Capture Hydrodynamics, Damping, and Hydrostatic Coupling
On the held-out test split, the Phase 1 GRU residual model delivers large, statistically significant one-step prediction improvements over deterministic kinematics:
* **Pitch angle RMSE:** $+16.71\%$ improvement
* **Heading angle RMSE:** $+15.35\%$ improvement
* **Vertical depth RMSE:** $+11.14\%$ improvement
* **Surge velocity RMSE:** $+8.20\%$ improvement  
The neural network successfully identifies physical effects absent in the kinematic baseline: thruster acceleration lag, turning yaw resistance, and hydrostatic righting moments during pitching maneuvers.

### Finding 4: [Negative Finding] Acoustic Noise Compounds in Open-Loop Multi-Step Predictions
In open-loop $k$-step horizon evaluations ($k=1 \dots 10$), the hybrid model's horizontal position prediction advantage peaks at $k=1$ ($0.810\text{ m}$ vs $0.886\text{ m}$ physics). Beyond $k \ge 3$ steps ($6.0\text{ s}$), adding raw position residuals slightly increases error drift ($5.34\text{ m}$ vs $5.28\text{ m}$ at $k=10$). This occurs because raw USBL acoustic coordinates contain high-frequency multipath transceiver jitter ($0.5-1.0\text{ m}$) that neural networks attempt to fit, compounding open-loop position variance over long horizons.

### Finding 5: Closed-Loop Receding Horizon Tames Multi-Step Model Degradation
Despite open-loop error accumulation beyond $k \ge 3$, closed-loop MPC remains highly accurate and stable. Because the controller executes **only the first action $\mathbf{u}_0^*$** and immediately incorporates new sensor feedback at each $2.0\text{ s}$ cycle, open-loop multi-step drift is continuously reset, enabling precise path tracking ($>98\%$ path efficiency).

### Finding 6: Kinematic Kalman Filtering is Essential for USBL Sensor Jitter
Directly feeding raw acoustic USBL fixes into the controller increases control slew effort by $43\%$ due to high-frequency thruster hunting. Incorporating a 2D constant-velocity kinematic Kalman filter reduces position uncertainty from $\sigma = 0.50\text{ m}$ to $< 0.18\text{ m}$, yielding smooth control commands without degrading obstacle evasion times.

### Finding 7: Ocean Cross-Current Drift is Rejected Without Steady-State Offset
Under a simulated $0.20\text{ m/s}$ benthic cross-current ($[0.15, -0.12, 0.05]\text{ m/s}$) and sensor noise, the MPC controller automatically commanded an equilibrium crab angle (compensating for drift) and reached the goal with final distance errors $< 5.0\text{ m}$ across all seeds, demonstrating robust disturbance rejection without requiring prior current velocity estimates.

### Finding 8: High Computational Efficiency Inside the Real-Time Budget
The warm-started Cross-Entropy Method (CEM) optimizer evaluating $K=70$ candidate sequences over horizon $H=10$ with $3$ refinement iterations executes in an average of **$0.22\text{ seconds}$ on CPU**. This utilizes only **$11\%$ of the $2.0\text{ s}$ physical control interval**, proving that hybrid physics-AI MPC is viable for real-time deployment on standard embedded marine vehicle hardware.

### Finding 9: Warm-Starting and LOS Seeding Halve Optimization Budget
Seeding candidate $0$ with the shifted previous optimal control sequence and candidate $1$ with 3D Line-Of-Sight guidance dramatically accelerates optimization convergence. Warm-started MPC with $3$ iterations achieves identical path efficiency ($98.9\%$) to an unseeded controller running $6$ iterations, halving the required number of neural model forward passes.

### Finding 10: Actuator Slew Rate Dictates Minimum Safe Obstacle Buffer
Due to vehicle rotational inertia and control fin rate limits ($0.20\text{ rad/step}$ heading, $0.10\text{ rad/step}$ pitch), the vehicle requires at least $10.0\text{ to } 14.0\text{ meters}$ of forward travel to execute a full $45^\circ$ evasion maneuver. An obstacle standoff safety buffer of at least $r_{\text{margin}} = 4.0\text{ m}$ ($r_{\text{safe}} = 12.0\text{ m}$ for an $8\text{ m}$ cylinder) is strictly necessary to prevent hull breaches under dynamic ocean current disturbances.
