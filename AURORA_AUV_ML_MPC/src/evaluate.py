import numpy as np
import pandas as pd
import torch
import os
import pickle
from src.ml_models import MLP, LSTMModel, GRUModel

class ModelEvaluator:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = config['hyperparameters']['seq_len']
        
        # Load scalers
        models_dir = config['paths']['models_dir']
        with open(os.path.join(models_dir, "feature_scaler.pkl"), "rb") as f:
            self.feature_scaler = pickle.load(f)
        with open(os.path.join(models_dir, "target_scaler.pkl"), "rb") as f:
            self.target_scaler = pickle.load(f)
            
        # Load models
        self.linear = pickle.load(open(os.path.join(models_dir, "linear.pkl"), "rb"))
        self.rf = pickle.load(open(os.path.join(models_dir, "random_forest.pkl"), "rb"))
        
        self.mlp = MLP(input_dim=6, hidden_dim=config['hyperparameters']['hidden_dim'], output_dim=7).to(self.device)
        self.mlp.load_state_dict(torch.load(os.path.join(models_dir, "mlp.pth"), map_location=self.device))
        self.mlp.eval()
        
        self.lstm = LSTMModel(input_dim=6, hidden_dim=config['hyperparameters']['hidden_dim'], output_dim=7).to(self.device)
        self.lstm.load_state_dict(torch.load(os.path.join(models_dir, "lstm.pth"), map_location=self.device))
        self.lstm.eval()
        
        self.gru = GRUModel(input_dim=6, hidden_dim=config['hyperparameters']['hidden_dim'], output_dim=7).to(self.device)
        self.gru.load_state_dict(torch.load(os.path.join(models_dir, "gru.pth"), map_location=self.device))
        self.gru.eval()

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def wrap_to_2pi(self, angle):
        return angle % (2 * np.pi)

    def evaluate_one_step(self, X_test, Y_test, pred_test_phys, act_test):
        """Computes test set RMSE for all hybrid configurations."""
        # Scale test features
        X_test_s = self.feature_scaler.transform(X_test)
        
        eval_start_idx = self.seq_len - 1
        eval_steps = len(X_test) - eval_start_idx
        
        act_sub = act_test[eval_start_idx:]
        phys_sub = pred_test_phys[eval_start_idx:]
        
        X_test_s_mlp = X_test_s[eval_start_idx:]
        X_test_s_seq = []
        for i in range(eval_steps):
            X_test_s_seq.append(X_test_s[i : i + self.seq_len])
        X_test_s_seq = torch.tensor(np.array(X_test_s_seq), dtype=torch.float32).to(self.device)
        
        # Predictions
        # Linear & Forest
        lin_p = self.linear.predict(X_test_s_mlp)
        rf_p = self.rf.predict(X_test_s_mlp)
        
        lin_res = self.target_scaler.inverse_transform(lin_p)
        rf_res = self.target_scaler.inverse_transform(rf_p)
        
        with torch.no_grad():
            # MLP
            mlp_p = self.mlp(torch.tensor(X_test_s_mlp, dtype=torch.float32).to(self.device)).cpu().numpy()
            mlp_res = self.target_scaler.inverse_transform(mlp_p)
            
            # LSTM
            lstm_p = self.lstm(X_test_s_seq).cpu().numpy()
            lstm_res = self.target_scaler.inverse_transform(lstm_p)
            
            # GRU
            gru_p = self.gru(X_test_s_seq).cpu().numpy()
            gru_res = self.target_scaler.inverse_transform(gru_p)
            
        # Reconstruct full next-state predictions
        def add_residual(phys, res):
            h = phys.copy()
            h[:, 0] = phys[:, 0] + res[:, 0]
            h[:, 1] = phys[:, 1] + res[:, 1]
            h[:, 2] = phys[:, 2] + res[:, 2]
            h[:, 3] = phys[:, 3] + res[:, 3]
            h[:, 4] = phys[:, 4] + res[:, 4]
            h[:, 5] = self.wrap_to_2pi(phys[:, 5] + res[:, 5])
            h[:, 6] = np.maximum(phys[:, 6] + res[:, 6], 0.0)
            return h
            
        hyb_lin = add_residual(phys_sub, lin_res)
        hyb_rf = add_residual(phys_sub, rf_res)
        hyb_mlp = add_residual(phys_sub, mlp_res)
        hyb_lstm = add_residual(phys_sub, lstm_res)
        hyb_gru = add_residual(phys_sub, gru_res)
        
        # Calculate RMSE per variable
        def rmse(act, pred):
            err = act - pred
            err[:, 5] = self.wrap_angle(err[:, 5]) # heading error wrap
            return np.sqrt(np.mean(err**2, axis=0))
            
        rmse_phys = rmse(act_sub, phys_sub)
        rmse_lin = rmse(act_sub, hyb_lin)
        rmse_rf = rmse(act_sub, hyb_rf)
        rmse_mlp = rmse(act_sub, hyb_mlp)
        rmse_lstm = rmse(act_sub, hyb_lstm)
        rmse_gru = rmse(act_sub, hyb_gru)
        
        state_vars = ['North Position (m)', 'East Position (m)', 'Depth (m)', 'Roll (rad)', 'Pitch (rad)', 'Heading (rad)', 'Speed (m/s)']
        df_metrics = pd.DataFrame({
            'State Variable': state_vars,
            'Physics-only': rmse_phys,
            'Physics+Linear': rmse_lin,
            'Physics+Forest': rmse_rf,
            'Physics+MLP': rmse_mlp,
            'Physics+LSTM': rmse_lstm,
            'Physics+GRU': rmse_gru
        })
        
        results_dir = os.path.join(self.config['paths']['results_dir'], "model_comparison")
        os.makedirs(results_dir, exist_ok=True)
        df_metrics.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
        return df_metrics

    def evaluate_multi_step(self, df_test, steps_list=[1, 5, 10, 20], k_d=0.05, dt=2.0, u_cruise=1.2):
        """Tests prediction drift over a multi-step forward horizon."""
        results = {}
        
        # Find a segment inside the test set where we have contiguous steps
        start_idx = 100 # arbitrary point in test set
        for steps in steps_list:
            results[steps] = {}
            actual_trajectory = []
            phys_trajectory = []
            lstm_trajectory = []
            
            # Sub-segment of steps
            sub_df = df_test.iloc[start_idx : start_idx + steps + 1].reset_index(drop=True)
            if len(sub_df) < steps + 1:
                continue
                
            actual_trajectory = np.stack([
                sub_df['North'].values,
                sub_df['East'].values,
                sub_df['depth'].values
            ], axis=1)
            
            # Physics-only rollout
            phys_state = np.array([
                sub_df['North'].iloc[0],
                sub_df['East'].iloc[0],
                sub_df['depth'].iloc[0],
                sub_df['Roll'].iloc[0],
                sub_df['Pitch'].iloc[0],
                sub_df['Heading'].iloc[0],
                sub_df['GroundSpeed'].iloc[0]
            ])
            phys_trajectory.append(phys_state[:3].copy())
            
            # LSTM rollout
            lstm_state = phys_state.copy()
            lstm_trajectory.append(lstm_state[:3].copy())
            
            # Pre-populate history for sequence models
            history = []
            for h in range(self.seq_len):
                row = df_test.iloc[start_idx - self.seq_len + h]
                history.append([row['GroundSpeed'], row['Roll'], row['Pitch'], np.cos(row['Heading']), np.sin(row['Heading']), row['depth']])
                
            for k in range(steps):
                u_cmd = sub_df['GroundSpeed'].iloc[k]
                theta_cmd = sub_df['Pitch'].iloc[k]
                psi_cmd = sub_df['Heading'].iloc[k]
                
                # Update Physics State
                theta_act = phys_state[4] + np.clip(theta_cmd - phys_state[4], -0.1, 0.1)
                psi_act = phys_state[5] + np.clip(self.wrap_angle(psi_cmd - phys_state[5]), -0.2, 0.2)
                
                x_next_p = phys_state[0] + phys_state[6] * np.cos(phys_state[4]) * np.cos(phys_state[5]) * dt
                y_next_p = phys_state[1] + phys_state[6] * np.cos(phys_state[4]) * np.sin(phys_state[5]) * dt
                z_next_p = phys_state[2] - phys_state[6] * np.sin(phys_state[4]) * dt
                speed_next_p = max(phys_state[6] + k_d * (u_cruise**2 - phys_state[6]**2) * dt, 0.0)
                
                phys_state = np.array([x_next_p, y_next_p, z_next_p, phys_state[3], theta_act, self.wrap_to_2pi(psi_act), speed_next_p])
                phys_trajectory.append(phys_state[:3].copy())
                
                # Update LSTM State
                theta_act_l = lstm_state[4] + np.clip(theta_cmd - lstm_state[4], -0.1, 0.1)
                psi_act_l = lstm_state[5] + np.clip(self.wrap_angle(psi_cmd - lstm_state[5]), -0.2, 0.2)
                
                x_next_l = lstm_state[0] + lstm_state[6] * np.cos(lstm_state[4]) * np.cos(lstm_state[5]) * dt
                y_next_l = lstm_state[1] + lstm_state[6] * np.cos(lstm_state[4]) * np.sin(lstm_state[5]) * dt
                z_next_l = lstm_state[2] - lstm_state[6] * np.sin(lstm_state[4]) * dt
                speed_next_l = max(lstm_state[6] + k_d * (u_cruise**2 - lstm_state[6]**2) * dt, 0.0)
                
                # Prediction scaling
                seq_s = self.feature_scaler.transform(np.array(history))
                with torch.no_grad():
                    seq_t = torch.tensor([seq_s], dtype=torch.float32).to(self.device)
                    lstm_p = self.lstm(seq_t).cpu().numpy()[0]
                    res = self.target_scaler.inverse_transform([lstm_p])[0]
                    
                lstm_state[0] = x_next_l + res[0]
                lstm_state[1] = y_next_l + res[1]
                lstm_state[2] = z_next_l + res[2]
                lstm_state[3] = lstm_state[3] + res[3]
                lstm_state[4] = theta_act_l
                lstm_state[5] = self.wrap_to_2pi(psi_act_l)
                lstm_state[6] = speed_next_l + res[6]
                lstm_state[6] = max(lstm_state[6], 0.0)
                
                lstm_trajectory.append(lstm_state[:3].copy())
                
                # Update history with new prediction
                history.pop(0)
                history.append([lstm_state[6], lstm_state[3], lstm_state[4], np.cos(lstm_state[5]), np.sin(lstm_state[5]), lstm_state[2]])
                
            results[steps]['actual'] = np.array(actual_trajectory)
            results[steps]['physics'] = np.array(phys_trajectory)
            results[steps]['lstm'] = np.array(lstm_trajectory)
            
        return results
