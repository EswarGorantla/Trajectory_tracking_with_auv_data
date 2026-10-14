import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# File paths
train_path = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\splits\train.csv"
output_plot_path = r"C:\Users\eswar\.gemini\antigravity\brain\54a8d0c6-8ff3-4ae2-a5f4-65100c13a45d\control_comparison.png"

# Load training data to fit scalers and train the LSTM
print("Loading training data for model fitting...")
train_df = pd.read_csv(train_path)
train_df.columns = train_df.columns.str.strip()

numeric_cols = ['Seconds', 'Latitude', 'Longitude', 'depth', 'altitude', 'Roll', 'Pitch', 'Heading', 'WaterSpeed', 'GroundSpeed']
for col in numeric_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

train_df = train_df.dropna(subset=['Date', 'Time'] + numeric_cols).copy()
train_df['DateTime'] = pd.to_datetime(train_df['Date'].astype(str).str.strip() + ' ' + train_df['Time'].astype(str).str.strip(), format='%d-%b-%Y %H:%M:%S')
train_df = train_df.sort_values('DateTime').reset_index(drop=True)

# Coordinate conversion parameters
R = 6371000.0
lat0 = np.radians(train_df['Latitude'].iloc[0])
lon0 = np.radians(train_df['Longitude'].iloc[0])

train_df['North'] = R * np.radians(train_df['Latitude'] - train_df['Latitude'].iloc[0])
train_df['East'] = R * np.radians(train_df['Longitude'] - train_df['Longitude'].iloc[0]) * np.cos(lat0)

# Helpers
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wrap_to_2pi(angle):
    return angle % (2 * np.pi)

# Compute physics residuals on training data
U_cruise = train_df['GroundSpeed'].mean()
k_d = 0.05
dt = 2.0

u = train_df['GroundSpeed'].values[:-1]
theta = train_df['Pitch'].values[:-1]
psi = train_df['Heading'].values[:-1]
phi = train_df['Roll'].values[:-1]

x_pred = train_df['North'].values[:-1] + u * np.cos(theta) * np.cos(psi) * dt
y_pred = train_df['East'].values[:-1] + u * np.cos(theta) * np.sin(psi) * dt
z_pred = train_df['depth'].values[:-1] - u * np.sin(theta) * dt
speed_pred = u + k_d * (U_cruise**2 - u**2) * dt
speed_pred = np.maximum(speed_pred, 0.0)

res_x = train_df['North'].values[1:] - x_pred
res_y = train_df['East'].values[1:] - y_pred
res_z = train_df['depth'].values[1:] - z_pred
res_roll = train_df['Roll'].values[1:] - phi
res_pitch = train_df['Pitch'].values[1:] - theta
res_heading = wrap_angle(train_df['Heading'].values[1:] - psi)
res_speed = train_df['GroundSpeed'].values[1:] - speed_pred

X_train = np.stack([
    train_df['GroundSpeed'].values[:-1],
    train_df['Roll'].values[:-1],
    train_df['Pitch'].values[:-1],
    np.cos(train_df['Heading'].values[:-1]),
    np.sin(train_df['Heading'].values[:-1]),
    train_df['depth'].values[:-1]
], axis=1)

Y_train = np.stack([
    res_x, res_y, res_z, res_roll, res_pitch, res_heading, res_speed
], axis=1)

# Fit scalers
feature_scaler = StandardScaler()
X_train_s = feature_scaler.fit_transform(X_train)

target_scaler = StandardScaler()
Y_train_s = target_scaler.fit_transform(Y_train)

# LSTM Dataset & Model
lookback = 10
X_seq, Y_target = [], []
for i in range(len(X_train_s) - lookback + 1):
    X_seq.append(X_train_s[i : i + lookback])
    Y_target.append(Y_train_s[i + lookback - 1])
X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
Y_target = torch.tensor(np.array(Y_target), dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=7):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

print("Training LSTM model (30 epochs)...")
lstm.train()
X_seq = X_seq.to(device)
Y_target = Y_target.to(device)
for epoch in range(30):
    optimizer.zero_grad()
    # Batch size is full dataset (fast training on CPU)
    pred = lstm(X_seq)
    loss = criterion(pred, Y_target)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1:02d}/30 | Loss: {loss.item():.5f}")

lstm.eval()

# ----------------- CLOSED-LOOP SIMULATION SETUP -----------------

# Define desired 3D path: a sinusoidal wave in horizontal plane, and cosine in vertical plane
N_sim = 100 # 200 seconds of simulation
t_axis = np.arange(N_sim) * dt

# Generate desired trajectory coordinates
# The trajectory goes primarily North-East
v_des = 1.2 # desired speed
s_des = v_des * t_axis

x_des = 0.8 * s_des
y_des = 15.0 * np.sin(2.0 * np.pi * x_des / 150.0)
z_des = 12.0 - 4.0 * np.cos(2.0 * np.pi * x_des / 150.0)

# True Plant class representing the "Real AUV"
# Updates state using integrated physics + LSTM residual predictions
class AUVPlant:
    def __init__(self, x0=0.0, y0=0.0, z0=12.0, u0=1.2, theta0=0.0, psi0=0.0):
        # State vector: [North, East, Depth, Roll, Pitch, Heading, Speed]
        self.state = np.array([x0, y0, z0, 0.0, theta0, psi0, u0])
        # Track history of states for LSTM (initialized with same initial state)
        self.history = []
        for _ in range(lookback):
            self.history.append(self._get_feature_vector())
            
    def _get_feature_vector(self):
        # Features: [Speed, Roll, Pitch, cos(Heading), sin(Heading), depth]
        return [
            self.state[6], # Speed
            self.state[3], # Roll
            self.state[4], # Pitch
            np.cos(self.state[5]), # cos(Heading)
            np.sin(self.state[5]), # sin(Heading)
            self.state[2]  # depth
        ]
        
    def step(self, u_cmd, theta_cmd, psi_cmd):
        # Actuator rate limits (Simulate delay/limits of propulsion and fins)
        U_act = u_cmd # instantaneous thrust speed target
        theta_act = self.state[4] + np.clip(theta_cmd - self.state[4], -0.1, 0.1)
        psi_act = self.state[5] + np.clip(wrap_angle(psi_cmd - self.state[5]), -0.2, 0.2)
        
        # 1. Physics Model prediction
        x_pred_next = self.state[0] + self.state[6] * np.cos(self.state[4]) * np.cos(self.state[5]) * dt
        y_pred_next = self.state[1] + self.state[6] * np.cos(self.state[4]) * np.sin(self.state[5]) * dt
        z_pred_next = self.state[2] - self.state[6] * np.sin(self.state[4]) * dt
        roll_pred_next = self.state[3]
        pitch_pred_next = self.state[4]
        heading_pred_next = self.state[5]
        speed_pred_next = self.state[6] + k_d * (U_cruise**2 - self.state[6]**2) * dt
        speed_pred_next = max(speed_pred_next, 0.0)
        
        phys_pred = np.array([x_pred_next, y_pred_next, z_pred_next, roll_pred_next, pitch_pred_next, heading_pred_next, speed_pred_next])
        
        # 2. LSTM residual prediction
        # Sequence input for LSTM
        seq_input = np.array(self.history)
        seq_input_s = feature_scaler.transform(seq_input)
        
        with torch.no_grad():
            seq_t = torch.tensor([seq_input_s], dtype=torch.float32)
            res_s = lstm(seq_t).cpu().numpy()[0]
            res = target_scaler.inverse_transform([res_s])[0]
            
        # 3. Hybrid true next state
        self.state[0] = phys_pred[0] + res[0]
        self.state[1] = phys_pred[1] + res[1]
        self.state[2] = phys_pred[2] + res[2]
        self.state[3] = phys_pred[3] + res[3]
        self.state[4] = theta_act # controlled pitch
        self.state[5] = wrap_to_2pi(psi_act) # controlled heading
        self.state[6] = phys_pred[6] + res[6] # speed dynamics
        self.state[6] = max(self.state[6], 0.0)
        
        # Update history
        self.history.pop(0)
        self.history.append(self._get_feature_vector())
        
        return self.state.copy()

# ----------------- CONTROLLER IMPLEMENTATIONS -----------------

# (a) PID Controller (Line-Of-Sight guidance)
def run_pid_simulation():
    plant = AUVPlant(psi0=np.arctan2(y_des[1]-y_des[0], x_des[1]-x_des[0]))
    history_states = [plant.state.copy()]
    control_history = []
    
    # PID state
    prev_heading_err = 0.0
    heading_integral = 0.0
    
    lookahead = 4 # look ahead steps on path
    
    for k in range(N_sim - 1):
        state = plant.state
        
        # Target path point
        target_idx = min(k + lookahead, N_sim - 1)
        x_targ = x_des[target_idx]
        y_targ = y_des[target_idx]
        z_targ = z_des[target_idx]
        
        # Heading Control (LOS Guidance)
        psi_targ = np.arctan2(y_targ - state[1], x_targ - state[0])
        heading_err = wrap_angle(psi_targ - state[5])
        heading_integral += heading_err * dt
        heading_deriv = (heading_err - prev_heading_err) / dt
        prev_heading_err = heading_err
        
        # PID commanded heading
        psi_cmd = psi_targ + 0.5 * heading_err + 0.1 * heading_deriv
        
        # Pitch Control (Vertical LOS Guidance)
        dist_horiz = np.sqrt((x_targ - state[0])**2 + (y_targ - state[1])**2)
        theta_targ = np.arctan2(state[2] - z_targ, dist_horiz) # positive pitch climbs (decreases depth)
        theta_cmd = np.clip(theta_targ, -0.3, 0.3)
        
        # Speed Control
        u_cmd = 1.2
        
        # Step plant
        next_state = plant.step(u_cmd, theta_cmd, psi_cmd)
        history_states.append(next_state)
        control_history.append([u_cmd, theta_cmd, psi_cmd])
        
    return np.array(history_states), np.array(control_history)


# Receding-Horizon MPC Controller
class MonteCarloMPC:
    def __init__(self, use_ai=False, horizon=5, num_candidates=400):
        self.use_ai = use_ai
        self.H = horizon
        self.K = num_candidates
        # Previous best control sequence
        self.prev_best = np.zeros((self.H, 3))
        self.prev_best[:, 0] = 1.2 # default speed
        
    def solve(self, current_state, plant_history, step_idx):
        # 1. Generate Candidate Control Sequences
        # Shape of candidates: [K, H, 3] (speed, pitch, heading)
        candidates = np.zeros((self.K, self.H, 3))
        
        # Candidate 0: previous best shifted
        candidates[0, :-1] = self.prev_best[1:]
        candidates[0, -1] = self.prev_best[-1]
        
        # Candidates 1 to K-1: previous best shifted plus random perturbations
        # Speed: mean 1.2, perturb [-0.1, 0.1]
        # Pitch: perturb [-0.05, 0.05]
        # Heading: perturb [-0.1, 0.1]
        noise_u = np.random.normal(0, 0.15, size=(self.K - 1, self.H))
        noise_pitch = np.random.normal(0, 0.05, size=(self.K - 1, self.H))
        noise_heading = np.random.normal(0, 0.15, size=(self.K - 1, self.H))
        
        for k in range(1, self.K):
            candidates[k, :-1] = self.prev_best[1:]
            candidates[k, -1] = self.prev_best[-1]
            
            candidates[k, :, 0] += noise_u[k-1]
            candidates[k, :, 1] += noise_pitch[k-1]
            candidates[k, :, 2] += noise_heading[k-1]
            
        # Clip inputs to physical limits
        candidates[:, :, 0] = np.clip(candidates[:, :, 0], 0.5, 1.8)
        candidates[:, :, 1] = np.clip(candidates[:, :, 1], -0.3, 0.3)
        # Heading wraps naturally, no clipping needed
        
        # 2. Vectorized Rollout Simulation over Horizon
        # We simulate self.K instances of the vehicle state forward in parallel
        # States shape: [K, 7] (North, East, depth, Roll, Pitch, Heading, Speed)
        states = np.tile(current_state, (self.K, 1))
        
        # Keep track of simulated histories for LSTM evaluation if use_ai is True
        # histories shape: [K, lookback, 6]
        histories = np.tile(np.array(plant_history), (self.K, 1, 1))
        
        costs = np.zeros(self.K)
        
        # Simulate forward step by step
        for h in range(self.H):
            # Target path indices for step t+h+1
            targ_idx = min(step_idx + h + 1, N_sim - 1)
            x_t = x_des[targ_idx]
            y_t = y_des[targ_idx]
            z_t = z_des[targ_idx]
            
            u_cmd = candidates[:, h, 0]
            theta_cmd = candidates[:, h, 1]
            psi_cmd = candidates[:, h, 2]
            
            # Actuator dynamics (rate limits) in simulation
            theta_act = states[:, 4] + np.clip(theta_cmd - states[:, 4], -0.1, 0.1)
            psi_act = states[:, 5] + np.clip(wrap_angle(psi_cmd - states[:, 5]), -0.2, 0.2)
            
            # Physics-only next state
            x_pred_next = states[:, 0] + states[:, 6] * np.cos(states[:, 4]) * np.cos(states[:, 5]) * dt
            y_pred_next = states[:, 1] + states[:, 6] * np.cos(states[:, 4]) * np.sin(states[:, 5]) * dt
            z_pred_next = states[:, 2] - states[:, 6] * np.sin(states[:, 4]) * dt
            roll_pred_next = states[:, 3]
            pitch_pred_next = states[:, 4]
            heading_pred_next = states[:, 5]
            speed_pred_next = states[:, 6] + k_d * (U_cruise**2 - states[:, 6]**2) * dt
            speed_pred_next = np.maximum(speed_pred_next, 0.0)
            
            # Apply LSTM residual if AI-enhanced
            if self.use_ai:
                # Format batch input for LSTM: scale the histories
                # Batch size is self.K
                histories_s = np.zeros_like(histories)
                for idx in range(self.K):
                    histories_s[idx] = feature_scaler.transform(histories[idx])
                
                with torch.no_grad():
                    hist_t = torch.tensor(histories_s, dtype=torch.float32).to(device)
                    res_s = lstm(hist_t).cpu().numpy()
                    res = target_scaler.inverse_transform(res_s)
            else:
                res = np.zeros((self.K, 7))
                
            # Update states
            next_states = np.zeros_like(states)
            next_states[:, 0] = x_pred_next + res[:, 0]
            next_states[:, 1] = y_pred_next + res[:, 1]
            next_states[:, 2] = z_pred_next + res[:, 2]
            next_states[:, 3] = roll_pred_next + res[:, 3]
            next_states[:, 4] = theta_act # controlled pitch
            next_states[:, 5] = wrap_to_2pi(psi_act) # controlled heading
            next_states[:, 6] = speed_pred_next + res[:, 6] # speed dynamics
            next_states[:, 6] = np.maximum(next_states[:, 6], 0.0)
            
            states = next_states
            
            # Update histories if AI is used
            if self.use_ai:
                # Slide the history window: pop oldest state, append new state features
                for idx in range(self.K):
                    # Form feature vector for next state
                    f = [
                        states[idx, 6], # Speed
                        states[idx, 3], # Roll
                        states[idx, 4], # Pitch
                        np.cos(states[idx, 5]),
                        np.sin(states[idx, 5]),
                        states[idx, 2]
                    ]
                    # histories[idx] shape: [lookback, 6]
                    histories[idx] = np.vstack([histories[idx][1:], f])
                    
            # Compute Cost for this horizon step
            # Penalize: position error (horizontal), depth error, speed deviation, control rate/jerk
            dist_err_sq = (states[:, 0] - x_t)**2 + (states[:, 1] - y_t)**2
            depth_err_sq = (states[:, 2] - z_t)**2
            speed_err_sq = (states[:, 6] - 1.2)**2
            
            costs += dist_err_sq + 5.0 * depth_err_sq + 10.0 * speed_err_sq
            
        # Select best sequence
        best_idx = np.argmin(costs)
        self.prev_best = candidates[best_idx]
        return self.prev_best[0] # return first step command


def run_mpc_simulation(use_ai):
    plant = AUVPlant(psi0=np.arctan2(y_des[1]-y_des[0], x_des[1]-x_des[0]))
    history_states = [plant.state.copy()]
    control_history = []
    
    mpc = MonteCarloMPC(use_ai=use_ai)
    
    print(f"Running MPC (AI-enhanced={use_ai}) Simulation...")
    for k in range(N_sim - 1):
        # Solve MPC to find optimal inputs
        u_opt = mpc.solve(plant.state, plant.history, k)
        u_cmd, theta_cmd, psi_cmd = u_opt
        
        # Step plant
        next_state = plant.step(u_cmd, theta_cmd, psi_cmd)
        history_states.append(next_state)
        control_history.append([u_cmd, theta_cmd, psi_cmd])
        
    return np.array(history_states), np.array(control_history)

# ----------------- EXECUTE CLOSED-LOOP SIMULATIONS -----------------

print("\nRunning PID Controller...")
states_pid, controls_pid = run_pid_simulation()

print("\nRunning Physics-only MPC...")
states_phys_mpc, controls_phys_mpc = run_mpc_simulation(use_ai=False)

print("\nRunning AI-enhanced MPC...")
states_ai_mpc, controls_ai_mpc = run_mpc_simulation(use_ai=True)

# ----------------- CALCULATE METRICS -----------------

def evaluate_controller(states, controls, name):
    # Position tracking error (horizontal)
    dist_err = np.sqrt((states[:, 0] - x_des)**2 + (states[:, 1] - y_des)**2)
    rmse_pos = np.sqrt(np.mean(dist_err**2))
    
    # Depth error
    depth_err = np.abs(states[:, 2] - z_des)
    rmse_depth = np.sqrt(np.mean(depth_err**2))
    
    # Heading error (need tangent angle of the path to represent target heading)
    # We'll calculate heading tracking error relative to target heading of the path
    dy = np.diff(y_des)
    dx = np.diff(x_des)
    heading_path = np.arctan2(dy, dx)
    heading_path = np.append(heading_path, heading_path[-1]) # pad last
    heading_err = wrap_angle(states[:, 5] - heading_path)
    rmse_heading = np.sqrt(np.mean(heading_err**2))
    
    # Control effort: sum of squared differences of control signals
    # controls has shape [N_sim-1, 3]
    diff_u = np.diff(controls[:, 0])**2
    diff_theta = np.diff(controls[:, 1])**2
    diff_psi = wrap_angle(np.diff(controls[:, 2]))**2
    total_effort = np.sum(diff_u + diff_theta + diff_psi)
    
    return rmse_pos, rmse_depth, rmse_heading, total_effort

metrics_pid = evaluate_controller(states_pid, controls_pid, "PID")
metrics_phys_mpc = evaluate_controller(states_phys_mpc, controls_phys_mpc, "Physics-only MPC")
metrics_ai_mpc = evaluate_controller(states_ai_mpc, controls_ai_mpc, "AI-enhanced MPC")

# Print Comparison Table
print("\n=== CLOSED-LOOP CONTROL EVALUATION ===")
print(f"{'Controller':20s} | {'Pos RMSE (m)':12s} | {'Depth RMSE (m)':14s} | {'Heading RMSE (rad)':18s} | {'Control Effort':14s}")
print("-" * 88)
print(f"{'PID Controller':20s} | {metrics_pid[0]:12.4f} | {metrics_pid[1]:14.4f} | {metrics_pid[2]:18.4f} | {metrics_pid[3]:14.4f}")
print(f"{'Physics-only MPC':20s} | {metrics_phys_mpc[0]:12.4f} | {metrics_phys_mpc[1]:14.4f} | {metrics_phys_mpc[2]:18.4f} | {metrics_phys_mpc[3]:14.4f}")
print(f"{'AI-enhanced MPC':20s} | {metrics_ai_mpc[0]:12.4f} | {metrics_ai_mpc[1]:14.4f} | {metrics_ai_mpc[2]:18.4f} | {metrics_ai_mpc[3]:14.4f}")
print("-" * 88)

# ----------------- PLOT COMPARISON -----------------
fig = plt.figure(figsize=(15, 11))
grid = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel 1: 2D Horizontal Trajectory
ax1 = fig.add_subplot(grid[0, 0])
ax1.plot(y_des, x_des, 'k--', label='Desired Path', linewidth=2)
ax1.plot(states_pid[:, 1], states_pid[:, 0], 'g-', label='PID Controller', linewidth=1.5)
ax1.plot(states_phys_mpc[:, 1], states_phys_mpc[:, 0], 'b-', label='Physics-only MPC', linewidth=1.5)
ax1.plot(states_ai_mpc[:, 1], states_ai_mpc[:, 0], 'r-', label='AI-enhanced MPC (LSTM)', linewidth=1.5)
ax1.set_xlabel('East Position (meters)')
ax1.set_ylabel('North Position (meters)')
ax1.set_title('2D Trajectory Tracking (Horizontal Plane)')
ax1.grid(True)
ax1.legend()
ax1.axis('equal')

# Panel 2: Depth vs Time
ax2 = fig.add_subplot(grid[0, 1])
ax2.plot(t_axis, z_des, 'k--', label='Desired Depth', linewidth=2)
ax2.plot(t_axis, states_pid[:, 2], 'g-', label='PID Controller', linewidth=1.5)
ax2.plot(t_axis, states_phys_mpc[:, 2], 'b-', label='Physics-only MPC', linewidth=1.5)
ax2.plot(t_axis, states_ai_mpc[:, 2], 'r-', label='AI-enhanced MPC (LSTM)', linewidth=1.5)
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Depth (meters)')
ax2.set_title('Depth Tracking over Time')
ax2.grid(True)
ax2.legend()
ax2.invert_yaxis()

# Panel 3: Heading Tracking Error
ax3 = fig.add_subplot(grid[1, 0])
dy = np.diff(y_des)
dx = np.diff(x_des)
heading_path = np.arctan2(dy, dx)
heading_path = np.append(heading_path, heading_path[-1])

ax3.plot(t_axis, wrap_angle(states_pid[:, 5] - heading_path), 'g-', label='PID', linewidth=1.5)
ax3.plot(t_axis, wrap_angle(states_phys_mpc[:, 5] - heading_path), 'b-', label='Phys MPC', linewidth=1.5)
ax3.plot(t_axis, wrap_angle(states_ai_mpc[:, 5] - heading_path), 'r-', label='AI MPC', linewidth=1.5)
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Heading Error (radians)')
ax3.set_title('Heading Tracking Error over Time')
ax3.grid(True)
ax3.legend()

# Panel 4: 3D Trajectory Visualization
ax4 = fig.add_subplot(grid[1, 1], projection='3d')
ax4.plot3D(y_des, x_des, z_des, 'k--', label='Desired Path', linewidth=2)
ax4.plot3D(states_pid[:, 1], states_pid[:, 0], states_pid[:, 2], 'g-', label='PID', linewidth=1.5)
ax4.plot3D(states_phys_mpc[:, 1], states_phys_mpc[:, 0], states_phys_mpc[:, 2], 'b-', label='Phys MPC', linewidth=1.5)
ax4.plot3D(states_ai_mpc[:, 1], states_ai_mpc[:, 0], states_ai_mpc[:, 2], 'r-', label='AI MPC', linewidth=1.5)
ax4.set_xlabel('East (m)')
ax4.set_ylabel('North (m)')
ax4.set_zlabel('Depth (m)')
ax4.set_title('3D Trajectory Visualization')
ax4.invert_zaxis()
ax4.grid(True)
ax4.legend()

plt.suptitle("Receding-Horizon MPC vs PID Controller Closed-Loop Comparison", fontsize=14, weight='bold')
plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_plot_path}")
plt.close()
