import numpy as np
import torch
import os
import pickle
from src.ml_models import LSTMModel, GRUModel

class MonteCarloMPC:
    def __init__(self, config, use_ai=None, device=None):
        self.config = config
        self.use_ai = use_ai
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # MPC Hyperparameters
        self.H = config['mpc']['horizon']
        self.K = config['mpc']['num_candidates']
        self.dt = config['simulation']['dt']
        self.k_d = config['simulation']['k_d_default']
        self.u_cruise = config['simulation']['U_cruise']
        
        # Pre-initialize candidate memory to warm-start optimization
        self.prev_best = np.zeros((self.H, 3))
        self.prev_best[:, 0] = self.u_cruise
        
        # Load scalers and dynamics models if using AI feedback
        if self.use_ai is not None:
            models_dir = config['paths']['models_dir']
            with open(os.path.join(models_dir, "feature_scaler.pkl"), "rb") as f:
                self.feature_scaler = pickle.load(f)
            with open(os.path.join(models_dir, "target_scaler.pkl"), "rb") as f:
                self.target_scaler = pickle.load(f)
                
            if self.use_ai == 'lstm':
                self.model = LSTMModel(input_dim=6, hidden_dim=config['hyperparameters']['hidden_dim'], output_dim=7).to(self.device)
                self.model.load_state_dict(torch.load(os.path.join(models_dir, "lstm.pth"), map_location=self.device))
            else:
                self.model = GRUModel(input_dim=6, hidden_dim=config['hyperparameters']['hidden_dim'], output_dim=7).to(self.device)
                self.model.load_state_dict(torch.load(os.path.join(models_dir, "gru.pth"), map_location=self.device))
            self.model.eval()

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def wrap_to_2pi(self, angle):
        return angle % (2 * np.pi)

    def solve(self, current_state, plant_history, step_idx, x_des, y_des, z_des):
        """
        Solves the optimal control command sequence over a horizon.
        Returns the command vector for the first step [u_cmd, theta_cmd, psi_cmd]
        """
        # Generate candidates: [u_cmd, theta_cmd, psi_cmd]
        candidates = np.zeros((self.K, self.H, 3))
        
        # Candidate 0 is the previous best shifted
        candidates[0, :-1] = self.prev_best[1:]
        candidates[0, -1] = self.prev_best[-1]
        
        # Perturb with Gaussian noise to explore around previous best
        noise_u = np.random.normal(0, 0.15, size=(self.K - 1, self.H))
        noise_pitch = np.random.normal(0, 0.05, size=(self.K - 1, self.H))
        noise_heading = np.random.normal(0, 0.15, size=(self.K - 1, self.H))
        
        for k in range(1, self.K):
            candidates[k, :-1] = self.prev_best[1:]
            candidates[k, -1] = self.prev_best[-1]
            candidates[k, :, 0] += noise_u[k-1]
            candidates[k, :, 1] += noise_pitch[k-1]
            candidates[k, :, 2] += noise_heading[k-1]
            
        # Apply command limits
        candidates[:, :, 0] = np.clip(candidates[:, :, 0], 0.5, 1.8) # speed commands
        candidates[:, :, 1] = np.clip(candidates[:, :, 1], -0.3, 0.3) # pitch command
        
        # Run rollouts for all candidates in parallel (vectorized)
        states = np.tile(current_state, (self.K, 1))
        histories = np.tile(np.array(plant_history), (self.K, 1, 1))
        costs = np.zeros(self.K)
        
        for h in range(self.H):
            targ_idx = min(step_idx + h + 1, len(x_des) - 1)
            x_t = x_des[targ_idx]
            y_t = y_des[targ_idx]
            z_t = z_des[targ_idx]
            
            u_cmd = candidates[:, h, 0]
            theta_cmd = candidates[:, h, 1]
            psi_cmd = candidates[:, h, 2]
            
            # command delays limits
            theta_act = states[:, 4] + np.clip(theta_cmd - states[:, 4], -0.1, 0.1)
            psi_act = states[:, 5] + np.clip(self.wrap_angle(psi_cmd - states[:, 5]), -0.2, 0.2)
            
            # physics kinematics
            x_pred_n = states[:, 0] + states[:, 6] * np.cos(states[:, 4]) * np.cos(states[:, 5]) * self.dt
            y_pred_n = states[:, 1] + states[:, 6] * np.cos(states[:, 4]) * np.sin(states[:, 5]) * self.dt
            z_pred_n = states[:, 2] - states[:, 6] * np.sin(states[:, 4]) * self.dt
            roll_pred_n = states[:, 3]
            speed_pred_n = np.maximum(states[:, 6] + self.k_d * (self.u_cruise**2 - states[:, 6]**2) * self.dt, 0.0)
            
            # Predict residual if AI feedback is enabled
            if self.use_ai is not None:
                histories_s = np.zeros_like(histories)
                for idx in range(self.K):
                    histories_s[idx] = self.feature_scaler.transform(histories[idx])
                
                with torch.no_grad():
                    hist_t = torch.tensor(histories_s, dtype=torch.float32).to(self.device)
                    res_s = self.model(hist_t).cpu().numpy()
                    res = self.target_scaler.inverse_transform(res_s)
            else:
                res = np.zeros((self.K, 7))
                
            # Transition states
            next_states = np.zeros_like(states)
            next_states[:, 0] = x_pred_n + res[:, 0]
            next_states[:, 1] = y_pred_n + res[:, 1]
            next_states[:, 2] = z_pred_n + res[:, 2]
            next_states[:, 3] = roll_pred_n + res[:, 3]
            next_states[:, 4] = theta_act
            next_states[:, 5] = self.wrap_to_2pi(psi_act)
            next_states[:, 6] = speed_pred_n + res[:, 6]
            next_states[:, 6] = np.maximum(next_states[:, 6], 0.0)
            
            states = next_states
            
            # Update prediction sequence history
            if self.use_ai is not None:
                for idx in range(self.K):
                    f = [states[idx, 6], states[idx, 3], states[idx, 4], np.cos(states[idx, 5]), np.sin(states[idx, 5]), states[idx, 2]]
                    histories[idx] = np.vstack([histories[idx][1:], f])
                    
            # Compute tracking and effort costs
            dist_err_sq = (states[:, 0] - x_t)**2 + (states[:, 1] - y_t)**2
            depth_err_sq = (states[:, 2] - z_t)**2
            speed_err_sq = (states[:, 6] - self.u_cruise)**2
            
            # Control effort penalty
            effort = (u_cmd - self.u_cruise)**2 + theta_cmd**2
            
            costs += dist_err_sq + 5.0 * depth_err_sq + 10.0 * speed_err_sq + 2.0 * effort
            
        best_idx = np.argmin(costs)
        self.prev_best = candidates[best_idx]
        return self.prev_best[0]


class PIDController:
    def __init__(self, config):
        self.dt = config['simulation']['dt']
        self.prev_heading_err = 0.0
        self.heading_integral = 0.0
        self.lookahead = 4  # lookahead steps for LOS guidance
        
    def solve(self, state, step_idx, x_des, y_des, z_des):
        """Standard Line-of-Sight feedback logic."""
        target_idx = min(step_idx + self.lookahead, len(x_des) - 1)
        x_targ, y_targ, z_targ = x_des[target_idx], y_des[target_idx], z_des[target_idx]
        
        # Calculate target heading clockwise from North
        psi_targ = np.arctan2(y_targ - state[1], x_targ - state[0])
        
        # Yaw command error and derivation
        def wrap_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
            
        heading_err = wrap_angle(psi_targ - state[5])
        self.heading_integral += heading_err * self.dt
        heading_deriv = (heading_err - self.prev_heading_err) / self.dt
        self.prev_heading_err = heading_err
        
        psi_cmd = psi_targ + 0.5 * heading_err + 0.1 * heading_deriv
        
        # Pitch command based on depth error
        dist_horiz = np.sqrt((x_targ - state[0])**2 + (y_targ - state[1])**2)
        theta_cmd = np.clip(np.arctan2(state[2] - z_targ, dist_horiz), -0.3, 0.3)
        
        # Speed command (constant Cruise Speed)
        u_cmd = 1.2
        
        return np.array([u_cmd, theta_cmd, psi_cmd])
