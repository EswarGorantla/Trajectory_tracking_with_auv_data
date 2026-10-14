import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import shutil

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# File paths
train_path = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\splits\train.csv"
val_path = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\splits\val.csv"
test_path = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\splits\test.csv"
csv_path = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\original_data\M86\multibeam_echosounder\multi-beam-echosounder-index.csv"

output_dir = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\outputs"
artifact_dir = r"C:\Users\eswar\.gemini\antigravity\brain\54a8d0c6-8ff3-4ae2-a5f4-65100c13a45d"
os.makedirs(output_dir, exist_ok=True)

# 1. Load splits
print("Loading split files...")
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# Coordinate conversion
R = 6371000.0
lat0 = np.radians(train_df['Latitude'].iloc[0])
lon0 = np.radians(train_df['Longitude'].iloc[0])

def convert_coordinates(df):
    df['North'] = R * np.radians(df['Latitude'] - train_df['Latitude'].iloc[0])
    df['East'] = R * np.radians(df['Longitude'] - train_df['Longitude'].iloc[0]) * np.cos(lat0)
    return df

train_df = convert_coordinates(train_df)
val_df = convert_coordinates(val_df)
test_df = convert_coordinates(test_df)

# Angle helpers
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wrap_to_2pi(angle):
    return angle % (2 * np.pi)

# 2. Physics Predictions and Residuals
U_cruise = train_df['GroundSpeed'].mean()
k_d = 0.05
dt = 2.0

def compute_physics_pred_and_residuals(df):
    N = len(df)
    x_pred = np.zeros(N)
    y_pred = np.zeros(N)
    z_pred = np.zeros(N)
    roll_pred = np.zeros(N)
    pitch_pred = np.zeros(N)
    heading_pred = np.zeros(N)
    speed_pred = np.zeros(N)
    
    u = df['GroundSpeed'].values[:-1]
    theta = df['Pitch'].values[:-1]
    psi = df['Heading'].values[:-1]
    phi = df['Roll'].values[:-1]
    
    x_pred[:-1] = df['North'].values[:-1] + u * np.cos(theta) * np.cos(psi) * dt
    y_pred[:-1] = df['East'].values[:-1] + u * np.cos(theta) * np.sin(psi) * dt
    z_pred[:-1] = df['depth'].values[:-1] - u * np.sin(theta) * dt
    
    roll_pred[:-1] = phi
    pitch_pred[:-1] = theta
    heading_pred[:-1] = psi
    
    speed_pred[:-1] = u + k_d * (U_cruise**2 - u**2) * dt
    speed_pred[:-1] = np.maximum(speed_pred[:-1], 0.0)
    
    res_x = df['North'].values[1:] - x_pred[:-1]
    res_y = df['East'].values[1:] - y_pred[:-1]
    res_z = df['depth'].values[1:] - z_pred[:-1]
    res_roll = df['Roll'].values[1:] - roll_pred[:-1]
    res_pitch = df['Pitch'].values[1:] - pitch_pred[:-1]
    res_heading = wrap_angle(df['Heading'].values[1:] - heading_pred[:-1])
    res_speed = df['GroundSpeed'].values[1:] - speed_pred[:-1]
    
    inputs = np.stack([
        df['GroundSpeed'].values[:-1],
        df['Roll'].values[:-1],
        df['Pitch'].values[:-1],
        np.cos(df['Heading'].values[:-1]),
        np.sin(df['Heading'].values[:-1]),
        df['depth'].values[:-1]
    ], axis=1)
    
    targets = np.stack([
        res_x, res_y, res_z, res_roll, res_pitch, res_heading, res_speed
    ], axis=1)
    
    actual_next = np.stack([
        df['North'].values[1:],
        df['East'].values[1:],
        df['depth'].values[1:],
        df['Roll'].values[1:],
        df['Pitch'].values[1:],
        df['Heading'].values[1:],
        df['GroundSpeed'].values[1:]
    ], axis=1)
    
    pred_next_phys = np.stack([
        x_pred[:-1], y_pred[:-1], z_pred[:-1], roll_pred[:-1], pitch_pred[:-1], heading_pred[:-1], speed_pred[:-1]
    ], axis=1)
    
    return inputs, targets, pred_next_phys, actual_next

print("Computing residuals...")
X_train, Y_train, pred_train_phys, act_train = compute_physics_pred_and_residuals(train_df)
X_val, Y_val, pred_val_phys, act_val = compute_physics_pred_and_residuals(val_df)
X_test, Y_test, pred_test_phys, act_test = compute_physics_pred_and_residuals(test_df)

# Scale
feature_scaler = StandardScaler()
X_train_s = feature_scaler.fit_transform(X_train)
X_val_s = feature_scaler.transform(X_val)
X_test_s = feature_scaler.transform(X_test)

target_scaler = StandardScaler()
Y_train_s = target_scaler.fit_transform(Y_train)
Y_val_s = target_scaler.transform(Y_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Define Dataset sliding window for sequential models
lookback = 10
def create_dataset_sequences(X, Y):
    X_seq, Y_target = [], []
    for i in range(len(X) - lookback + 1):
        X_seq.append(X[i : i + lookback])
        Y_target.append(Y[i + lookback - 1])
    return torch.tensor(np.array(X_seq), dtype=torch.float32).to(device), torch.tensor(np.array(Y_target), dtype=torch.float32).to(device)

X_train_seq, Y_train_seq = create_dataset_sequences(X_train_s, Y_train_s)
X_val_seq, Y_val_seq = create_dataset_sequences(X_val_s, Y_val_s)

# MLP Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )
    def forward(self, x):
        return self.net(x)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(6, 64, batch_first=True)
        self.fc = nn.Linear(64, 7)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# GRU Model
class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(6, 64, batch_first=True)
        self.fc = nn.Linear(64, 7)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Instantiate models
mlp = MLP().to(device)
lstm = LSTMModel().to(device)
gru = GRUModel().to(device)

def train_model(model, X_tr, Y_tr, epochs=30, batch_size=256):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.MSELoss()
    model.train()
    N = len(X_tr)
    for e in range(epochs):
        indices = np.arange(N)
        np.random.shuffle(indices)
        for start_idx in range(0, N, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            x_batch = X_tr[batch_idx]
            y_batch = Y_tr[batch_idx]
            opt.zero_grad()
            pred = model(x_batch)
            loss = crit(pred, y_batch)
            loss.backward()
            opt.step()
    model.eval()

print("Training models...")
train_model(mlp, torch.tensor(X_train_s, dtype=torch.float32).to(device), torch.tensor(Y_train_s, dtype=torch.float32).to(device))
print("  MLP Trained.")
train_model(lstm, X_train_seq, Y_train_seq)
print("  LSTM Trained.")
train_model(gru, X_train_seq, Y_train_seq)
print("  GRU Trained.")

# Predictor Evaluation
eval_start_idx = lookback - 1
eval_steps = len(X_test) - eval_start_idx

act_sub = act_test[eval_start_idx:]
phys_sub = pred_test_phys[eval_start_idx:]

X_test_s_mlp = X_test_s[eval_start_idx:]
X_test_s_seq = []
for i in range(eval_steps):
    X_test_s_seq.append(X_test_s[i : i + lookback])
X_test_s_seq = torch.tensor(np.array(X_test_s_seq), dtype=torch.float32).to(device)

# Predictions
with torch.no_grad():
    # MLP
    mlp_p = mlp(torch.tensor(X_test_s_mlp, dtype=torch.float32).to(device)).cpu().numpy()
    mlp_res = target_scaler.inverse_transform(mlp_p)
    # LSTM
    lstm_p = lstm(X_test_s_seq).cpu().numpy()
    lstm_res = target_scaler.inverse_transform(lstm_p)
    # GRU
    gru_p = gru(X_test_s_seq).cpu().numpy()
    gru_res = target_scaler.inverse_transform(gru_p)

# Hybrid addition helper
def hybrid_pred(phys, res):
    h = phys + res
    h[:, 5] = wrap_to_2pi(phys[:, 5] + res[:, 5]) # heading wrap
    h[:, 6] = np.maximum(h[:, 6], 0.0) # speed clamp
    return h

hyb_mlp = hybrid_pred(phys_sub, mlp_res)
hyb_lstm = hybrid_pred(phys_sub, lstm_res)
hyb_gru = hybrid_pred(phys_sub, gru_res)

def calc_rmse(act, pred):
    err = act - pred
    err[:, 5] = wrap_angle(err[:, 5])
    return np.sqrt(np.mean(err**2, axis=0))

rmse_phys = calc_rmse(act_sub, phys_sub)
rmse_mlp = calc_rmse(act_sub, hyb_mlp)
rmse_lstm = calc_rmse(act_sub, hyb_lstm)
rmse_gru = calc_rmse(act_sub, hyb_gru)

# Save Predictor RMSE Table
state_vars = ['North Position (m)', 'East Position (m)', 'Depth (m)', 'Roll (rad)', 'Pitch (rad)', 'Heading (rad)', 'Speed (m/s)']
pred_df = pd.DataFrame({
    'State Variable': state_vars,
    'Physics-only': rmse_phys,
    'Physics+MLP': rmse_mlp,
    'Physics+LSTM': rmse_lstm,
    'Physics+GRU': rmse_gru
})
pred_df.to_csv(os.path.join(output_dir, "predictor_rmse.csv"), index=False)

# ----------------- CLOSED-LOOP SIMULATION -----------------
N_sim = 100
t_axis = np.arange(N_sim) * dt
v_des = 1.2
s_des = v_des * t_axis
x_des = 0.8 * s_des
y_des = 15.0 * np.sin(2.0 * np.pi * x_des / 150.0)
z_des = 12.0 - 4.0 * np.cos(2.0 * np.pi * x_des / 150.0)

class AUVPlant:
    def __init__(self, use_ai=None):
        self.use_ai = use_ai
        # [North, East, Depth, Roll, Pitch, Heading, Speed]
        psi0 = np.arctan2(y_des[1]-y_des[0], x_des[1]-x_des[0])
        self.state = np.array([0.0, 0.0, 12.0, 0.0, 0.0, psi0, 1.2])
        self.history = []
        for _ in range(lookback):
            self.history.append(self._get_feat())
            
    def _get_feat(self):
        return [self.state[6], self.state[3], self.state[4], np.cos(self.state[5]), np.sin(self.state[5]), self.state[2]]
        
    def step(self, u_cmd, theta_cmd, psi_cmd):
        theta_act = self.state[4] + np.clip(theta_cmd - self.state[4], -0.1, 0.1)
        psi_act = self.state[5] + np.clip(wrap_angle(psi_cmd - self.state[5]), -0.2, 0.2)
        
        # Physics transition
        x_pred_n = self.state[0] + self.state[6] * np.cos(self.state[4]) * np.cos(self.state[5]) * dt
        y_pred_n = self.state[1] + self.state[6] * np.cos(self.state[4]) * np.sin(self.state[5]) * dt
        z_pred_n = self.state[2] - self.state[6] * np.sin(self.state[4]) * dt
        roll_pred_n = self.state[3]
        speed_pred_n = max(self.state[6] + k_d * (U_cruise**2 - self.state[6]**2) * dt, 0.0)
        
        phys_pred = np.array([x_pred_n, y_pred_n, z_pred_n, roll_pred_n, roll_pred_n, roll_pred_n, speed_pred_n]) # placeholder for indexing
        
        if self.use_ai is not None:
            seq_s = feature_scaler.transform(np.array(self.history))
            with torch.no_grad():
                seq_t = torch.tensor([seq_s], dtype=torch.float32).to(device)
                if self.use_ai == 'lstm':
                    res_s = lstm(seq_t).cpu().numpy()[0]
                else:
                    res_s = gru(seq_t).cpu().numpy()[0]
                res = target_scaler.inverse_transform([res_s])[0]
        else:
            res = np.zeros(7)
            
        self.state[0] = x_pred_n + res[0]
        self.state[1] = y_pred_n + res[1]
        self.state[2] = z_pred_n + res[2]
        self.state[3] = roll_pred_n + res[3]
        self.state[4] = theta_act
        self.state[5] = wrap_to_2pi(psi_act)
        self.state[6] = speed_pred_n + res[6]
        self.state[6] = max(self.state[6], 0.0)
        
        self.history.pop(0)
        self.history.append(self._get_feat())
        return self.state.copy()

class MonteCarloMPC:
    def __init__(self, use_ai=None, horizon=5, num_candidates=300):
        self.use_ai = use_ai
        self.H = horizon
        self.K = num_candidates
        self.prev_best = np.zeros((self.H, 3))
        self.prev_best[:, 0] = 1.2
        
    def solve(self, current_state, plant_history, step_idx):
        candidates = np.zeros((self.K, self.H, 3))
        candidates[0, :-1] = self.prev_best[1:]
        candidates[0, -1] = self.prev_best[-1]
        
        noise_u = np.random.normal(0, 0.15, size=(self.K - 1, self.H))
        noise_pitch = np.random.normal(0, 0.05, size=(self.K - 1, self.H))
        noise_heading = np.random.normal(0, 0.15, size=(self.K - 1, self.H))
        
        for k in range(1, self.K):
            candidates[k, :-1] = self.prev_best[1:]
            candidates[k, -1] = self.prev_best[-1]
            candidates[k, :, 0] += noise_u[k-1]
            candidates[k, :, 1] += noise_pitch[k-1]
            candidates[k, :, 2] += noise_heading[k-1]
            
        candidates[:, :, 0] = np.clip(candidates[:, :, 0], 0.5, 1.8)
        candidates[:, :, 1] = np.clip(candidates[:, :, 1], -0.3, 0.3)
        
        states = np.tile(current_state, (self.K, 1))
        histories = np.tile(np.array(plant_history), (self.K, 1, 1))
        costs = np.zeros(self.K)
        
        for h in range(self.H):
            targ_idx = min(step_idx + h + 1, N_sim - 1)
            x_t = x_des[targ_idx]
            y_t = y_des[targ_idx]
            z_t = z_des[targ_idx]
            
            u_cmd = candidates[:, h, 0]
            theta_cmd = candidates[:, h, 1]
            psi_cmd = candidates[:, h, 2]
            
            theta_act = states[:, 4] + np.clip(theta_cmd - states[:, 4], -0.1, 0.1)
            psi_act = states[:, 5] + np.clip(wrap_angle(psi_cmd - states[:, 5]), -0.2, 0.2)
            
            x_pred_n = states[:, 0] + states[:, 6] * np.cos(states[:, 4]) * np.cos(states[:, 5]) * dt
            y_pred_n = states[:, 1] + states[:, 6] * np.cos(states[:, 4]) * np.sin(states[:, 5]) * dt
            z_pred_n = states[:, 2] - states[:, 6] * np.sin(states[:, 4]) * dt
            roll_pred_n = states[:, 3]
            speed_pred_n = np.maximum(states[:, 6] + k_d * (U_cruise**2 - states[:, 6]**2) * dt, 0.0)
            
            if self.use_ai is not None:
                histories_s = np.zeros_like(histories)
                for idx in range(self.K):
                    histories_s[idx] = feature_scaler.transform(histories[idx])
                
                with torch.no_grad():
                    hist_t = torch.tensor(histories_s, dtype=torch.float32).to(device)
                    if self.use_ai == 'lstm':
                        res_s = lstm(hist_t).cpu().numpy()
                    else:
                        res_s = gru(hist_t).cpu().numpy()
                    res = target_scaler.inverse_transform(res_s)
            else:
                res = np.zeros((self.K, 7))
                
            next_states = np.zeros_like(states)
            next_states[:, 0] = x_pred_n + res[:, 0]
            next_states[:, 1] = y_pred_n + res[:, 1]
            next_states[:, 2] = z_pred_n + res[:, 2]
            next_states[:, 3] = roll_pred_n + res[:, 3]
            next_states[:, 4] = theta_act
            next_states[:, 5] = wrap_to_2pi(psi_act)
            next_states[:, 6] = speed_pred_n + res[:, 6]
            next_states[:, 6] = np.maximum(next_states[:, 6], 0.0)
            
            states = next_states
            
            if self.use_ai is not None:
                for idx in range(self.K):
                    f = [states[idx, 6], states[idx, 3], states[idx, 4], np.cos(states[idx, 5]), np.sin(states[idx, 5]), states[idx, 2]]
                    histories[idx] = np.vstack([histories[idx][1:], f])
                    
            dist_err_sq = (states[:, 0] - x_t)**2 + (states[:, 1] - y_t)**2
            depth_err_sq = (states[:, 2] - z_t)**2
            speed_err_sq = (states[:, 6] - 1.2)**2
            costs += dist_err_sq + 5.0 * depth_err_sq + 10.0 * speed_err_sq
            
        best_idx = np.argmin(costs)
        self.prev_best = candidates[best_idx]
        return self.prev_best[0]

def run_pid_simulation():
    plant = AUVPlant(use_ai='lstm') # true plant is simulated by LSTM-corrected dynamics
    history_states = [plant.state.copy()]
    control_history = []
    prev_heading_err = 0.0
    heading_integral = 0.0
    lookahead = 4
    for k in range(N_sim - 1):
        state = plant.state
        target_idx = min(k + lookahead, N_sim - 1)
        x_targ, y_targ, z_targ = x_des[target_idx], y_des[target_idx], z_des[target_idx]
        psi_targ = np.arctan2(y_targ - state[1], x_targ - state[0])
        heading_err = wrap_angle(psi_targ - state[5])
        heading_integral += heading_err * dt
        heading_deriv = (heading_err - prev_heading_err) / dt
        prev_heading_err = heading_err
        psi_cmd = psi_targ + 0.5 * heading_err + 0.1 * heading_deriv
        dist_horiz = np.sqrt((x_targ - state[0])**2 + (y_targ - state[1])**2)
        theta_cmd = np.clip(np.arctan2(state[2] - z_targ, dist_horiz), -0.3, 0.3)
        next_state = plant.step(1.2, theta_cmd, psi_cmd)
        history_states.append(next_state)
        control_history.append([1.2, theta_cmd, psi_cmd])
    return np.array(history_states), np.array(control_history)

def run_mpc_simulation(use_ai):
    plant = AUVPlant(use_ai='lstm') # true plant is simulated by LSTM-corrected dynamics
    history_states = [plant.state.copy()]
    control_history = []
    mpc = MonteCarloMPC(use_ai=use_ai)
    for k in range(N_sim - 1):
        u_opt = mpc.solve(plant.state, plant.history, k)
        next_state = plant.step(u_opt[0], u_opt[1], u_opt[2])
        history_states.append(next_state)
        control_history.append(u_opt)
    return np.array(history_states), np.array(control_history)

print("Running controllers...")
states_pid, controls_pid = run_pid_simulation()
states_phys_mpc, controls_phys_mpc = run_mpc_simulation(use_ai=None)
states_lstm_mpc, controls_lstm_mpc = run_mpc_simulation(use_ai='lstm')
states_gru_mpc, controls_gru_mpc = run_mpc_simulation(use_ai='gru')

def evaluate_controller(states, controls):
    dist_err = np.sqrt((states[:, 0] - x_des)**2 + (states[:, 1] - y_des)**2)
    rmse_pos = np.sqrt(np.mean(dist_err**2))
    depth_err = np.abs(states[:, 2] - z_des)
    rmse_depth = np.sqrt(np.mean(depth_err**2))
    dy, dx = np.diff(y_des), np.diff(x_des)
    heading_path = np.append(np.arctan2(dy, dx), np.arctan2(dy[-1], dx[-1]))
    heading_err = wrap_angle(states[:, 5] - heading_path)
    rmse_heading = np.sqrt(np.mean(heading_err**2))
    total_effort = np.sum(np.diff(controls[:, 0])**2 + np.diff(controls[:, 1])**2 + wrap_angle(np.diff(controls[:, 2]))**2)
    return rmse_pos, rmse_depth, rmse_heading, total_effort

m_pid = evaluate_controller(states_pid, controls_pid)
m_phys = evaluate_controller(states_phys_mpc, controls_phys_mpc)
m_lstm = evaluate_controller(states_lstm_mpc, controls_lstm_mpc)
m_gru = evaluate_controller(states_gru_mpc, controls_gru_mpc)

# Save Controller Table
ctrl_df = pd.DataFrame({
    'Controller': ['PID Controller', 'Physics-only MPC', 'AI-enhanced MPC (LSTM)', 'AI-enhanced MPC (GRU)'],
    'Pos RMSE (m)': [m_pid[0], m_phys[0], m_lstm[0], m_gru[0]],
    'Depth RMSE (m)': [m_pid[1], m_phys[1], m_lstm[1], m_gru[1]],
    'Heading RMSE (rad)': [m_pid[2], m_phys[2], m_lstm[2], m_gru[2]],
    'Control Effort': [m_pid[3], m_phys[3], m_lstm[3], m_gru[3]]
})
ctrl_df.to_csv(os.path.join(output_dir, "controller_performance.csv"), index=False)

# ----------------- SAVE SEPARATE PLOTS -----------------
# 1. Physics vs Actual (turning segment open-loop simulation)
print("Saving Plot 1: physics_vs_actual.png...")
# Load segment data
df = pd.read_csv(csv_path, low_memory=False)
df.columns = df.columns.str.strip()
num_cols = ['Seconds', 'Latitude', 'Longitude', 'depth', 'altitude', 'Roll', 'Pitch', 'Heading', 'WaterSpeed', 'GroundSpeed']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df_clean = df.dropna(subset=['Date', 'Time'] + num_cols).copy()
df_clean['DateTime'] = pd.to_datetime(df_clean['Date'].astype(str).str.strip() + ' ' + df_clean['Time'].astype(str).str.strip(), format='%d-%b-%Y %H:%M:%S')
df_clean = df_clean.sort_values('DateTime').reset_index(drop=True)
sub = df_clean.iloc[6268:6268+500].copy().reset_index(drop=True)
sub['North_act'] = R * np.radians(sub['Latitude'] - sub['Latitude'].iloc[0])
sub['East_act'] = R * np.radians(sub['Longitude'] - sub['Longitude'].iloc[0]) * np.cos(lat0)

x_pred_ol = np.zeros(500)
y_pred_ol = np.zeros(500)
z_pred_ol = np.zeros(500)
x_pred_ol[0], y_pred_ol[0], z_pred_ol[0] = sub['North_act'].iloc[0], sub['East_act'].iloc[0], sub['depth'].iloc[0]
u_ol = sub['GroundSpeed'].values

for t in range(499):
    x_pred_ol[t+1] = x_pred_ol[t] + u_ol[t] * np.cos(sub['Pitch'].iloc[t]) * np.cos(sub['Heading'].iloc[t]) * dt
    y_pred_ol[t+1] = y_pred_ol[t] + u_ol[t] * np.cos(sub['Pitch'].iloc[t]) * np.sin(sub['Heading'].iloc[t]) * dt
    z_pred_ol[t+1] = z_pred_ol[t] + u_ol[t] * np.sin(sub['Pitch'].iloc[t]) * dt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(sub['East_act'], sub['North_act'], 'b-', label='Actual')
ax1.plot(y_pred_ol, x_pred_ol, 'r--', label='Physics Predict')
ax1.set_xlabel('East (m)')
ax1.set_ylabel('North (m)')
ax1.set_title('2D Trajectory (Open-loop)')
ax1.grid(True)
ax1.legend()
ax1.axis('equal')

ax2.plot(sub['depth'], 'b-', label='Actual')
ax2.plot(z_pred_ol, 'r--', label='Physics Predict')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Depth (m)')
ax2.set_title('Depth Profile (Open-loop)')
ax2.invert_yaxis()
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "physics_vs_actual.png"), dpi=150)
plt.savefig(os.path.join(artifact_dir, "physics_vs_actual.png"), dpi=150)
plt.close()

# 2. Predictor RMSE Comparison Bar Chart (relative RMSE %)
print("Saving Plot 2: predictor_rmse_comparison.png...")
rel_mlp = (rmse_mlp / rmse_phys) * 100
rel_lstm = (rmse_lstm / rmse_phys) * 100
rel_gru = (rmse_gru / rmse_phys) * 100

x = np.arange(len(state_vars))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, [100]*len(state_vars), width, label='Physics-only (100%)', color='gray', alpha=0.5)
ax.bar(x, rel_mlp, width, label='Physics+MLP', color='orange')
ax.bar(x + width, rel_lstm, width, label='Physics+LSTM', color='blue')
ax.bar(x + 2*width, rel_gru, width, label='Physics+GRU', color='green')

ax.set_ylabel('Relative RMSE (%)')
ax.set_title('Predictor RMSE Comparison (Relative to Physics-only Baseline)')
ax.set_xticks(x + width/2)
ax.set_xticklabels(state_vars, rotation=15, ha='right')
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "predictor_rmse_comparison.png"), dpi=150)
plt.savefig(os.path.join(artifact_dir, "predictor_rmse_comparison.png"), dpi=150)
plt.close()

# 3. Trajectory Comparison (closed-loop)
print("Saving Plot 3: controller_comparison.png...")
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax1.plot(y_des, x_des, 'k--', label='Desired Path')
ax1.plot(states_pid[:, 1], states_pid[:, 0], 'g-', label='PID')
ax1.plot(states_phys_mpc[:, 1], states_phys_mpc[:, 0], 'b-', label='Physics MPC')
ax1.plot(states_lstm_mpc[:, 1], states_lstm_mpc[:, 0], 'r-', label='LSTM MPC')
ax1.plot(states_gru_mpc[:, 1], states_gru_mpc[:, 0], 'c-', label='GRU MPC')
ax1.set_xlabel('East (m)')
ax1.set_ylabel('North (m)')
ax1.set_title('Horizontal Path Tracking')
ax1.grid(True)
ax1.legend()
ax1.axis('equal')

ax2 = fig.add_subplot(122)
ax2.plot(t_axis, z_des, 'k--', label='Desired Depth')
ax2.plot(t_axis, states_pid[:, 2], 'g-', label='PID')
ax2.plot(t_axis, states_phys_mpc[:, 2], 'b-', label='Physics MPC')
ax2.plot(t_axis, states_lstm_mpc[:, 2], 'r-', label='LSTM MPC')
ax2.plot(t_axis, states_gru_mpc[:, 2], 'c-', label='GRU MPC')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Depth (m)')
ax2.set_title('Depth Tracking')
ax2.grid(True)
ax2.legend()
ax2.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "controller_comparison.png"), dpi=150)
plt.savefig(os.path.join(artifact_dir, "controller_comparison.png"), dpi=150)
plt.close()

print("\n=== All tasks finished successfully. ===")
