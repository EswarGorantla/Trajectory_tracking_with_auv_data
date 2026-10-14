# Phase 2: Goal-Directed AUV Model Predictive Control (MPC)

This module implements a closed-loop, receding-horizon **Goal-Directed Model Predictive Controller (MPC)** for an Autonomous Underwater Vehicle (AUV). It builds directly upon the neural dynamics model trained and validated in **Phase 1**, wrapping it into a real-time predictive navigation engine.

---

## 1. System Architecture

```
                       USER INPUTS
        Point A (Start) + Point B (Goal) + Boundaries + Constraints
                                 │
                                 ▼
                     Receding-Horizon MPC Controller
                     (Cross-Entropy Optimizer: H=10 steps)
                                 │
                                 ▼
                       Dynamics Interface
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
          3D Marine Physics              Deep 2-Layer GRU
          Kinematics + Drag           Phase 1 Residual Model
          (Deterministic Kd)          (Unmodeled Damping/Forces)
                  └──────────────┬──────────────┘
                                 ▼
                    Multi-Step State Prediction
                                 │
                                 ▼
                    Cost Function Evaluation
          (Goal Distance + LOS Alignment + Boundary Penalties)
                                 │
                                 ▼
                    Optimal First Action u_0
                     [Speed, Pitch, Heading]
                                 │
                                 ▼
                     AUV True Plant Execution
                                 │
                                 ▼
                      State & History Update
                                 │
                                 ▼
                     Repeat in Receding Horizon
```

---

## 2. Phase 1 Dependency & Integration

* **Trained Weights:** [`presentationphase1/models/best_dynamics_model.pt`](../presentationphase1/models/best_dynamics_model.pt) (41,095 trainable parameters, 2 GRU recurrent layers with 64 units each).
* **Scalers:** [`presentationphase1/models/scaler.pkl`](../presentationphase1/models/scaler.pkl) (`StandardScaler` fitted on 22,360 training telemetry samples).
* **Configuration:** [`presentationphase1/models/model_config.json`](../presentationphase1/models/model_config.json).
* **Zero Duplication:** Phase 2 loads Phase 1 artifacts dynamically using [`models/phase1_loader.py`](models/phase1_loader.py).

---

## 3. State & Control Representations

### 7D State Vector ($\mathbf{x}_t \in \mathbb{R}^7$)
$$\mathbf{x}_t = [x_t, y_t, z_t, \phi_t, \theta_t, \psi_t, U_t]^T$$
* $x_t$: North position in local Cartesian plane ($m$)
* $y_t$: East position in local Cartesian plane ($m$)
* $z_t$: Depth below sea surface ($m$)
* $\phi_t$: Roll angle ($rad$)
* $\theta_t$: Pitch angle ($rad$) [positive = nose-up climbing]
* $\psi_t$: Heading angle clockwise from North ($rad$, $[0, 2\pi)$)
* $U_t$: Forward surge speed over ground ($m/s$)

### 6D Continuous Feature Window ($\mathbf{s}_t \in \mathbb{R}^6$)
$$\mathbf{s}_t = [U_t, \phi_t, \theta_t, \cos\psi_t, \sin\psi_t, z_t]^T$$
Fed to the Phase 1 GRU over a rolling sequence window of $T=10$ steps ($20.0\text{ s}$).

### 3D Autopilot Control Input ($\mathbf{u}_t \in \mathbb{R}^3$)
$$\mathbf{u}_t = [U_{\text{cmd}}, \theta_{\text{cmd}}, \psi_{\text{cmd}}]^T$$
Subject to realistic control-fin deflection rate limits ($\Delta\theta_{\max} = 0.1\text{ rad/step}$, $\Delta\psi_{\max} = 0.2\text{ rad/step}$).

---

## 4. Multi-Objective Cost Function

$$J = w_{\text{goal}} J_{\text{goal}} + w_{\text{track}} J_{\text{track}} + w_{\text{ctrl}} J_{\text{ctrl}} + w_{\text{smooth}} J_{\text{smooth}} + w_{\text{constraint}} J_{\text{constraint}}$$

* **$J_{\text{goal}}$:** Penalizes terminal Euclidean distance $\|\mathbf{p}_H - \mathbf{p}_B\|^2$ and running distance decay to Goal B.
* **$J_{\text{track}}$:** Penalizes angular misalignment with the 3D Line-of-Sight (LOS) guidance vectors.
* **$J_{\text{ctrl}}$:** Penalizes deviations from optimal economic cruise speed ($1.2\text{ m/s}$).
* **$J_{\text{smooth}}$:** Penalizes control slew rate $\|\mathbf{u}_k - \mathbf{u}_{k-1}\|^2$ to prevent thruster chatter and fin stress.
* **$J_{\text{constraint}}$:** High-penalty quadratic barrier function enforcing 3D spatial boundaries and cylindrical obstacle keep-out zones.

---

## 5. Directory Layout

```
Phase_2/
├── README.md                          # Comprehensive documentation
├── requirements.txt                   # Dependencies
├── config/
│   └── mpc_config.yaml                # Configurable parameters (weights, limits, bounds)
├── models/
│   └── phase1_loader.py               # Loader referencing Phase 1 PyTorch model & scalers
├── src/
│   ├── state.py                       # State and control dataclasses & angle helpers
│   ├── physics_dynamics.py            # 3D marine kinematics + surge drag (Kd = 0.06073)
│   ├── learned_dynamics.py            # Phase 1 GRU residual model inference engine
│   ├── dynamics_interface.py          # Unified step and batch rollout prediction
│   ├── goal.py                        # Waypoint, Goal, and 3D LOS guidance
│   ├── constraints.py                 # Boundaries, Cylindrical Obstacles, and Limits
│   ├── cost_function.py               # Multi-objective MPC cost formulation
│   ├── mpc_controller.py              # Receding-horizon optimizer with warm-starting
│   ├── simulator.py                   # Closed-loop simulation environment
│   ├── trajectory.py                  # Trajectory export and metrics computation
│   └── evaluation.py                  # 2D, depth, and 3D publication-quality plotting
├── simulations/
│   ├── run_basic_A_to_B.py            # Basic A -> B experiment
│   ├── run_multi_scenarios.py         # 3 start-goal configurations
│   ├── run_boundary_experiment.py     # Boundary avoidance demonstration
│   ├── run_obstacle_experiment.py     # 3D cylindrical obstacle circumnavigation
│   ├── run_baseline_comparison.py     # Reactive LOS Tracker vs Hybrid MPC
│   └── run_robustness_experiment.py   # Ocean current & acoustic noise testing
├── results/
│   ├── trajectories/                  # CSV flight logs
│   ├── metrics/                       # JSON & CSV performance metrics
│   └── plots/                         # 2D, depth, and 3D trajectory charts
├── app/
│   └── main.py                        # FastAPI backend & interactive HTML dashboard
└── PHASE_2_RESULTS.md                 # Complete 20-section experimental report
```

---

## 6. How to Run the Experiments

### 1. Run the Primary $A \to B$ Experiment
```bash
python Phase_2/simulations/run_basic_A_to_B.py
```
* **Start:** $(0, 0, 20)\text{ m}$ $\to$ **Goal:** $(100, 80, 40)\text{ m}$
* **Result:** Goal reached in $90.0\text{ s}$, final error $4.94\text{ m}$, $0$ violations.

### 2. Run Multi-Scenario Experiments (3 Cases)
```bash
python Phase_2/simulations/run_multi_scenarios.py
```

### 3. Run Boundary Avoidance Experiment
```bash
python Phase_2/simulations/run_boundary_experiment.py
```

### 4. Run 3D Cylindrical Obstacle Avoidance Experiment
```bash
python Phase_2/simulations/run_obstacle_experiment.py
```

### 5. Run Baseline Comparison (Reactive LOS vs Hybrid MPC)
```bash
python Phase_2/simulations/run_baseline_comparison.py
```

### 6. Run Robustness Experiment (Current Disturbance & Sensor Noise)
```bash
python Phase_2/simulations/run_robustness_experiment.py
```

### 7. Launch Interactive FastAPI Web Dashboard
```bash
python Phase_2/app/main.py
```
Then open your browser to [http://127.0.0.1:8000](http://127.0.0.1:8000) to view the interactive control panel.

---

## 7. Research-Grade Benchmark & Full Reproduction Suite (Windows PowerShell)

All commands should be executed from the repository root (`RoboticsSEM5`) in PowerShell:

### Step 1: Install Pinned Dependencies
```powershell
pip install -r Phase_2/requirements.txt
```

### Step 2: Run Pre-Flight Model Multi-Step Validation (Test Split Only)
```powershell
python Phase_2/validation/model_validation.py
```
*Evaluates 1-to-10 step open-loop predictions across Physics, GRU, and Hybrid dynamics with block bootstrap 95% CIs.*

### Step 3: Run Full Decoupled Closed-Loop Benchmark Suite
```powershell
# Quick suite (3 seeds per scenario/method, complete ablations):
python Phase_2/run_all.py --quick

# Full Monte Carlo evaluation suite (30 seeds per scenario/method):
python Phase_2/run_all.py --full
```
*Simulates 120+ flights against the independent continuous-time `TrueAUVPlant` (RK4, $\Delta t=0.1\text{s}$, actuator lags, $\pm 25\%$ drag mismatch, quadratic damping, restoring buoyancy, Ornstein-Uhlenbeck disturbance, 3D currents, and acoustic USBL sensor noise).*

### Step 4: Run Independent Metric Recomputation & Audit Verifier
```powershell
python Phase_2/verify_results.py
```
*Independently audits all 120 raw flight logs directly from disk to mathematically verify zero discrepancy against the metrics database.*

### Step 5: Generate Research Paper Manuscript Programmatically
```powershell
python Phase_2/generate_research_report.py
```
*Generates `Phase_2/PHASE2_RESEARCH_REPORT.md` directly from empirical metrics with zero hard-coded numbers.*

### Step 6: Run Comprehensive Test Suite
```powershell
python -m pytest Phase_2/tests/test_simulation_suite.py
```
*Verifies checkpoint SHA-256 immutability, angle wrapping, sign conventions, slew limiter, vectorized CEM batch equivalence, and deterministic seeds.*

