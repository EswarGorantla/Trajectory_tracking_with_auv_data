import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# File paths
train_path = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\splits\train.csv"
val_path = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\splits\val.csv"
test_path = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\splits\test.csv"

# Load splits
print("Loading split files...")
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# 1. Coordinate Conversion (North-East relative to train's first row)
R = 6371000.0 # Earth's radius in meters
lat0 = np.radians(train_df['Latitude'].iloc[0])
lon0 = np.radians(train_df['Longitude'].iloc[0])

def convert_coordinates(df):
    df['North'] = R * np.radians(df['Latitude'] - train_df['Latitude'].iloc[0])
    df['East'] = R * np.radians(df['Longitude'] - train_df['Longitude'].iloc[0]) * np.cos(lat0)
    return df

train_df = convert_coordinates(train_df)
val_df = convert_coordinates(val_df)
test_df = convert_coordinates(test_df)

# Angle wrapping functions
def wrap_angle(angle):
    # Wrap to [-pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wrap_to_2pi(angle):
    # Wrap to [0, 2pi]
    return angle % (2 * np.pi)

# 2. Physics Model Parameters and Predictions
# Calculate U_cruise from training set
U_cruise = train_df['GroundSpeed'].mean()
k_d = 0.05
dt = 2.0

def compute_physics_pred_and_residuals(df):
    N = len(df)
    
    # Pre-allocate predicted arrays
    x_pred = np.zeros(N)
    y_pred = np.zeros(N)
    z_pred = np.zeros(N)
    roll_pred = np.zeros(N)
    pitch_pred = np.zeros(N)
    heading_pred = np.zeros(N)
    speed_pred = np.zeros(N)
    
    # One-step predictions from current state: s_{t+1}^pred = f(s_t)
    # We slice to N-1 since we predict next state
    u = df['GroundSpeed'].values[:-1]
    theta = df['Pitch'].values[:-1]
    psi = df['Heading'].values[:-1]
    phi = df['Roll'].values[:-1]
    
    # Positions
    x_pred[:-1] = df['North'].values[:-1] + u * np.cos(theta) * np.cos(psi) * dt
    y_pred[:-1] = df['East'].values[:-1] + u * np.cos(theta) * np.sin(psi) * dt
    z_pred[:-1] = df['depth'].values[:-1] - u * np.sin(theta) * dt
    
    # Attitudes (Constant attitude assumption)
    roll_pred[:-1] = phi
    pitch_pred[:-1] = theta
    heading_pred[:-1] = psi
    
    # Speed (Surge dynamics)
    speed_pred[:-1] = u + k_d * (U_cruise**2 - u**2) * dt
    speed_pred[:-1] = np.maximum(speed_pred[:-1], 0.0) # speed cannot be negative
    
    # Compute residuals: actual_{t+1} - predicted_{t+1}
    # Note: these correspond to residuals at index t (predicting step t+1)
    res_x = df['North'].values[1:] - x_pred[:-1]
    res_y = df['East'].values[1:] - y_pred[:-1]
    res_z = df['depth'].values[1:] - z_pred[:-1]
    res_roll = df['Roll'].values[1:] - roll_pred[:-1]
    res_pitch = df['Pitch'].values[1:] - pitch_pred[:-1]
    res_heading = wrap_angle(df['Heading'].values[1:] - heading_pred[:-1])
    res_speed = df['GroundSpeed'].values[1:] - speed_pred[:-1]
    
    # Inputs at step t
    inputs = np.stack([
        df['GroundSpeed'].values[:-1],
        df['Roll'].values[:-1],
        df['Pitch'].values[:-1],
        np.cos(df['Heading'].values[:-1]),
        np.sin(df['Heading'].values[:-1]),
        df['depth'].values[:-1]
    ], axis=1)
    
    # Targets at step t (to predict t+1 residual)
    targets = np.stack([
        res_x, res_y, res_z, res_roll, res_pitch, res_heading, res_speed
    ], axis=1)
    
    # Store predictions and actuals to simplify evaluation
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
        x_pred[:-1],
        y_pred[:-1],
        z_pred[:-1],
        roll_pred[:-1],
        pitch_pred[:-1],
        heading_pred[:-1],
        speed_pred[:-1]
    ], axis=1)
    
    return inputs, targets, pred_next_phys, actual_next

# Process all splits
print("Computing physics predictions and residuals...")
X_train, Y_train, pred_train_phys, act_train = compute_physics_pred_and_residuals(train_df)
X_val, Y_val, pred_val_phys, act_val = compute_physics_pred_and_residuals(val_df)
X_test, Y_test, pred_test_phys, act_test = compute_physics_pred_and_residuals(test_df)

# Scale features and targets
feature_scaler = StandardScaler()
X_train_s = feature_scaler.fit_transform(X_train)
X_val_s = feature_scaler.transform(X_val)
X_test_s = feature_scaler.transform(X_test)

target_scaler = StandardScaler()
Y_train_s = target_scaler.fit_transform(Y_train)
Y_val_s = target_scaler.transform(Y_val)
# Test target is NOT scaled for final RMSE, but we can scale/unscale predictions

# PyTorch Dataset Definitions
class MLPDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class LSTMDataset(Dataset):
    def __init__(self, X, Y, lookback=10):
        self.X_seq = []
        self.Y_target = []
        # Create sliding window sequences
        for i in range(len(X) - lookback + 1):
            self.X_seq.append(X[i : i + lookback])
            self.Y_target.append(Y[i + lookback - 1])
        self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)
        self.Y_target = torch.tensor(np.array(self.Y_target), dtype=torch.float32)
        
    def __len__(self):
        return len(self.X_seq)
    def __getitem__(self, idx):
        return self.X_seq[idx], self.Y_target[idx]

# Parameters
lookback = 10
batch_size = 256
epochs = 30

train_mlp_loader = DataLoader(MLPDataset(X_train_s, Y_train_s), batch_size=batch_size, shuffle=True)
val_mlp_loader = DataLoader(MLPDataset(X_val_s, Y_val_s), batch_size=batch_size, shuffle=False)

train_lstm_loader = DataLoader(LSTMDataset(X_train_s, Y_train_s, lookback=lookback), batch_size=batch_size, shuffle=True)
val_lstm_loader = DataLoader(LSTMDataset(X_val_s, Y_val_s, lookback=lookback), batch_size=batch_size, shuffle=False)

# Model Definitions
class MLP(nn.Module):
    def __init__(self, input_dim=6, output_dim=7):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=7):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        out, (hn, cn) = self.lstm(x)
        # take the last time step output
        return self.fc(out[:, -1, :])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# Helper training function
def train_model(model, train_loader, val_loader, model_name):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    best_weights = None
    
    print(f"\n--- Training {model_name} ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
    # Load best weights
    model.load_state_dict(best_weights)
    return model

# 3. Train MLP
mlp = MLP()
mlp = train_model(mlp, train_mlp_loader, val_mlp_loader, "MLP")

# 4. Train LSTM
lstm = LSTMModel()
lstm = train_model(lstm, train_lstm_loader, val_lstm_loader, "LSTM")

# 5. Evaluation on Test Set (using same subset for fair comparison)
# Since LSTM requires lookback, we can only evaluate starting from index (lookback - 1) of the test set.
# Total test steps = len(X_test)
eval_start_idx = lookback - 1
eval_steps = len(X_test) - eval_start_idx

print(f"\nEvaluating models on {eval_steps} test steps (indices {eval_start_idx} to {len(X_test)-1})...")

# Subset actuals and physics predictions for comparison
act_sub = act_test[eval_start_idx:]
phys_sub = pred_test_phys[eval_start_idx:]

# Feature inputs for MLP (subsetted)
X_test_s_mlp = X_test_s[eval_start_idx:]
# Feature inputs for LSTM
X_test_s_lstm = []
for i in range(eval_steps):
    X_test_s_lstm.append(X_test_s[i : i + lookback])
X_test_s_lstm = np.array(X_test_s_lstm)

# MLP Predictions
mlp.eval()
with torch.no_grad():
    x_mlp_t = torch.tensor(X_test_s_mlp, dtype=torch.float32).to(device)
    mlp_pred_s = mlp(x_mlp_t).cpu().numpy()
    mlp_res_pred = target_scaler.inverse_transform(mlp_pred_s)

# LSTM Predictions
lstm.eval()
with torch.no_grad():
    x_lstm_t = torch.tensor(X_test_s_lstm, dtype=torch.float32).to(device)
    lstm_pred_s = lstm(x_lstm_t).cpu().numpy()
    lstm_res_pred = target_scaler.inverse_transform(lstm_pred_s)

# Compute Hybrid Predictions
# Hybrid state prediction = Physics Prediction + Residual Prediction
# Order of state variables: [North (x), East (y), Depth (z), Roll (phi), Pitch (theta), Heading (psi), Speed (U)]

def compute_hybrid_predictions(phys_pred, res_pred):
    hybrid = np.zeros_like(phys_pred)
    # North, East, Depth, Roll, Pitch, Speed (simple addition)
    hybrid[:, 0] = phys_pred[:, 0] + res_pred[:, 0]
    hybrid[:, 1] = phys_pred[:, 1] + res_pred[:, 1]
    hybrid[:, 2] = phys_pred[:, 2] + res_pred[:, 2]
    hybrid[:, 3] = phys_pred[:, 3] + res_pred[:, 3]
    hybrid[:, 4] = phys_pred[:, 4] + res_pred[:, 4]
    hybrid[:, 6] = phys_pred[:, 6] + res_pred[:, 6]
    # Ensure speed doesn't go negative
    hybrid[:, 6] = np.maximum(hybrid[:, 6], 0.0)
    
    # Heading (angle addition + wrapping)
    hybrid[:, 5] = wrap_to_2pi(phys_pred[:, 5] + res_pred[:, 5])
    return hybrid

hybrid_mlp = compute_hybrid_predictions(phys_sub, mlp_res_pred)
hybrid_lstm = compute_hybrid_predictions(phys_sub, lstm_res_pred)

# Calculate RMSE per state variable
def calculate_rmse(actual, predicted):
    errors = actual - predicted
    # Wrap heading errors
    errors[:, 5] = wrap_angle(errors[:, 5])
    return np.sqrt(np.mean(errors**2, axis=0))

rmse_phys = calculate_rmse(act_sub, phys_sub)
rmse_mlp = calculate_rmse(act_sub, hybrid_mlp)
rmse_lstm = calculate_rmse(act_sub, hybrid_lstm)

# Prepare Comparison Table
state_vars = ['North Position (m)', 'East Position (m)', 'Depth (m)', 'Roll (rad)', 'Pitch (rad)', 'Heading (rad)', 'Speed (m/s)']

print("\n=== Model RMSE Performance Comparison ===")
print(f"{'State Variable':25s} | {'Physics-only':12s} | {'Physics+MLP':12s} | {'Physics+LSTM':12s}")
print("-" * 72)
for i, var in enumerate(state_vars):
    print(f"{var:25s} | {rmse_phys[i]:12.6f} | {rmse_mlp[i]:12.6f} | {rmse_lstm[i]:12.6f}")

# Average improvement percentage
improve_mlp = np.mean((rmse_phys - rmse_mlp) / rmse_phys) * 100
improve_lstm = np.mean((rmse_phys - rmse_lstm) / rmse_phys) * 100
print("-" * 72)
print(f"Avg Improvement vs Phys   | {'-':12s} | {improve_mlp:11.2f}% | {improve_lstm:11.2f}%")
