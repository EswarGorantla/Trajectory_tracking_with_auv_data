import numpy as np
import torch
import os
import pickle
from src.ml_models import LSTMModel

class AUVPlant:
    def __init__(self, config, use_ai=None, k_d=0.05, dt=2.0, u_cruise=1.2, device=None):
        self.config = config
        self.use_ai = use_ai
        self.k_d = k_d
        self.dt = dt
        self.u_cruise = u_cruise
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load scalers and LSTM/GRU for simulation dynamics
        models_dir = config['paths']['models_dir']
        self.seq_len = config['hyperparameters']['seq_len']
        
        with open(os.path.join(models_dir, "feature_scaler.pkl"), "rb") as f:
            self.feature_scaler = pickle.load(f)
        with open(os.path.join(models_dir, "target_scaler.pkl"), "rb") as f:
            self.target_scaler = pickle.load(f)
            
        self.lstm = LSTMModel(input_dim=6, hidden_dim=config['hyperparameters']['hidden_dim'], output_dim=7).to(self.device)
        self.lstm.load_state_dict(torch.load(os.path.join(models_dir, "lstm.pth"), map_location=self.device))
        self.lstm.eval()
        
        self.reset()
        
    def reset(self, initial_state=None):
        """Resets the plant state and pre-populates sequence history."""
        if initial_state is None:
            # Default state: [North, East, Depth, Roll, Pitch, Heading, Speed]
            self.state = np.array([0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 1.2])
        else:
            self.state = np.array(initial_state).copy()
            
        self.history = []
        # Pre-populate history with initial state characteristics
        for _ in range(self.seq_len):
            self.history.append(self._get_feat())
            
    def _get_feat(self):
        """Generates feature row for network inputs."""
        return [self.state[6], self.state[3], self.state[4], np.cos(self.state[5]), np.sin(self.state[5]), self.state[2]]

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def wrap_to_2pi(self, angle):
        return angle % (2 * np.pi)

    def step(self, u_cmd, theta_cmd, psi_cmd, disturbance=None):
        """
        Executes a physics-AI update step on the true plant.
        disturbance: [current_north, current_east, current_depth]
        """
        # Command rate limits
        theta_act = self.state[4] + np.clip(theta_cmd - self.state[4], -0.1, 0.1)
        psi_act = self.state[5] + np.clip(self.wrap_angle(psi_cmd - self.state[5]), -0.2, 0.2)
        
        # Physics baseline
        x_pred_n = self.state[0] + self.state[6] * np.cos(self.state[4]) * np.cos(self.state[5]) * self.dt
        y_pred_n = self.state[1] + self.state[6] * np.cos(self.state[4]) * np.sin(self.state[5]) * self.dt
        z_pred_n = self.state[2] - self.state[6] * np.sin(self.state[4]) * self.dt
        roll_pred_n = self.state[3]
        speed_pred_n = np.maximum(self.state[6] + self.k_d * (self.u_cruise**2 - self.state[6]**2) * self.dt, 0.0)
        
        # AI residuals
        if self.use_ai:
            seq_s = self.feature_scaler.transform(np.array(self.history))
            with torch.no_grad():
                seq_t = torch.tensor([seq_s], dtype=torch.float32).to(self.device)
                res_s = self.lstm(seq_t).cpu().numpy()[0]
                res = self.target_scaler.inverse_transform([res_s])[0]
        else:
            res = np.zeros(7)
            
        # Update plant state
        self.state[0] = x_pred_n + res[0]
        self.state[1] = y_pred_n + res[1]
        self.state[2] = z_pred_n + res[2]
        self.state[3] = roll_pred_n + res[3]
        self.state[4] = theta_act
        self.state[5] = self.wrap_to_2pi(psi_act)
        self.state[6] = speed_pred_n + res[6]
        self.state[6] = np.maximum(self.state[6], 0.0)
        
        # Inject simulated water current disturbance if provided
        if disturbance is not None:
            self.state[0] += disturbance[0] * self.dt
            self.state[1] += disturbance[1] * self.dt
            self.state[2] += disturbance[2] * self.dt
            
        # Update sequence history
        self.history.pop(0)
        self.history.append(self._get_feat())
        
        return self.state.copy()
