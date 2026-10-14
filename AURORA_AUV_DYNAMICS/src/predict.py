import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Tuple, List

from src.physics_model import PhysicsModel
from src.residual_model import HybridDynamicsModel
from src.state_definition import StateEstimator

class Predictor:
    """Evaluates trained dynamics models on unseen test telemetry."""

    def __init__(self, dt: float = 2.0, seq_len: int = 10):
        self.dt = dt
        self.seq_len = seq_len

    def evaluate_test_set(
        self,
        test_df: pd.DataFrame,
        hybrid_model: HybridDynamicsModel,
        ml_only_model: torch.nn.Module,
        feature_scaler,
        target_scaler,
        device: torch.device
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Executes single-step prediction on the entire UNSEEN test split for:
        1. Physics-Only Model
        2. ML-Only Model
        3. Physics + ML Hybrid Model
        """
        n_samples = len(test_df)
        state_matrix = StateEstimator.extract_state_vector(test_df)
        features_all = StateEstimator.extract_features(state_matrix)

        # Storage arrays
        act_next = state_matrix[1:] # t=1 to N
        phys_next = np.zeros_like(act_next)
        hybrid_next = np.zeros_like(act_next)
        ml_only_next = np.zeros_like(act_next)
        residuals_pred = np.zeros_like(act_next)

        ml_only_model.eval()

        print(f"\nRunning 1-Step Dynamics Predictions on {len(act_next)} Unseen Test Samples...")
        for t in range(len(act_next)):
            cur_state = state_matrix[t] # state at t
            
            # Physics prediction
            p_next = hybrid_model.physics.step(cur_state)
            phys_next[t] = p_next

            # Build history window
            if t < self.seq_len - 1:
                # Pad earlier entries
                hist = np.vstack([np.tile(features_all[0], (self.seq_len - 1 - t, 1)), features_all[: t + 1]])
            else:
                hist = features_all[t - self.seq_len + 1 : t + 1]

            # 1. Hybrid Model Step
            feat_scaled = feature_scaler.transform(hist)
            with torch.no_grad():
                tensor_in = torch.tensor(np.array([feat_scaled]), dtype=torch.float32).to(device)
                res_scaled = hybrid_model.ml_model(tensor_in).cpu().numpy()[0]
                res = target_scaler.inverse_transform([res_scaled])[0]
                residuals_pred[t] = res

                # Direct ML-Only Model Step
                ml_res_scaled = ml_only_model(tensor_in).cpu().numpy()[0]
                ml_delta = target_scaler.inverse_transform([ml_res_scaled])[0]

            # Hybrid Combination
            h_next = p_next.copy()
            h_next[0] += res[0]
            h_next[1] += res[1]
            h_next[2] += res[2]
            h_next[3] += res[3]
            h_next[4] += res[4]
            h_next[5] = (p_next[5] + res[5]) % (2 * np.pi)
            h_next[6] = max(p_next[6] + res[6], 0.0)
            hybrid_next[t] = h_next

            # ML-Only Prediction (Delta mapping from current state)
            m_next = cur_state.copy()
            m_next[0] += ml_delta[0]
            m_next[1] += ml_delta[1]
            m_next[2] += ml_delta[2]
            m_next[3] += ml_delta[3]
            m_next[4] += ml_delta[4]
            m_next[5] = (cur_state[5] + ml_delta[5]) % (2 * np.pi)
            m_next[6] = max(cur_state[6] + ml_delta[6], 0.0)
            ml_only_next[t] = m_next

        # Build comprehensive test predictions DataFrame
        df_preds = pd.DataFrame({
            "Time_s": np.arange(len(act_next)) * self.dt,
            "Actual_North": act_next[:, 0],
            "Physics_North": phys_next[:, 0],
            "ML_Only_North": ml_only_next[:, 0],
            "Hybrid_North": hybrid_next[:, 0],

            "Actual_East": act_next[:, 1],
            "Physics_East": phys_next[:, 1],
            "ML_Only_East": ml_only_next[:, 1],
            "Hybrid_East": hybrid_next[:, 1],

            "Actual_Depth": act_next[:, 2],
            "Physics_Depth": phys_next[:, 2],
            "ML_Only_Depth": ml_only_next[:, 2],
            "Hybrid_Depth": hybrid_next[:, 2],

            "Actual_Speed": act_next[:, 6],
            "Physics_Speed": phys_next[:, 6],
            "ML_Only_Speed": ml_only_next[:, 6],
            "Hybrid_Speed": hybrid_next[:, 6],

            "Actual_Pitch": act_next[:, 4],
            "Physics_Pitch": phys_next[:, 4],
            "ML_Only_Pitch": ml_only_next[:, 4],
            "Hybrid_Pitch": hybrid_next[:, 4],

            "Actual_Heading": act_next[:, 5],
            "Physics_Heading": phys_next[:, 5],
            "ML_Only_Heading": ml_only_next[:, 5],
            "Hybrid_Heading": hybrid_next[:, 5]
        })

        raw_predictions = {
            "actual": act_next,
            "physics": phys_next,
            "ml_only": ml_only_next,
            "hybrid": hybrid_next,
            "residuals": residuals_pred
        }

        return df_preds, raw_predictions

    def evaluate_multistep(
        self,
        test_df: pd.DataFrame,
        hybrid_model: HybridDynamicsModel,
        feature_scaler,
        target_scaler,
        device: torch.device,
        horizons: List[int] = [1, 5, 10, 20],
        num_test_sequences: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluates recursive multi-step forecasting accuracy over compounding prediction horizons.
        """
        state_matrix = StateEstimator.extract_state_vector(test_df)
        features_all = StateEstimator.extract_features(state_matrix)
        max_h = max(horizons)

        phy_rmse_by_h = {h: [] for h in horizons}
        hyb_rmse_by_h = {h: [] for h in horizons}

        start_indices = np.linspace(self.seq_len, len(state_matrix) - max_h - 1, num_test_sequences, dtype=int)

        for s_idx in start_indices:
            # True trajectory over horizon
            true_states = state_matrix[s_idx : s_idx + max_h]
            
            # 1. Physics Recursive Rollout
            p_state = state_matrix[s_idx - 1].copy()
            p_rollout = []
            for h in range(max_h):
                p_state = hybrid_model.physics.step(p_state)
                p_rollout.append(p_state.copy())
            p_rollout = np.array(p_rollout)

            # 2. Hybrid Recursive Rollout
            h_state = state_matrix[s_idx - 1].copy()
            hist = features_all[s_idx - self.seq_len : s_idx].copy()
            h_rollout = []
            for h in range(max_h):
                feat_scaled = feature_scaler.transform(hist)
                with torch.no_grad():
                    tensor_in = torch.tensor(np.array([feat_scaled]), dtype=torch.float32).to(device)
                    res_scaled = hybrid_model.ml_model(tensor_in).cpu().numpy()[0]
                    res = target_scaler.inverse_transform([res_scaled])[0]

                p_step = hybrid_model.physics.step(h_state)
                h_step = p_step.copy()
                h_step[0] += res[0]
                h_step[1] += res[1]
                h_step[2] += res[2]
                h_step[3] += res[3]
                h_step[4] += res[4]
                h_step[5] = (p_step[5] + res[5]) % (2 * np.pi)
                h_step[6] = max(p_step[6] + res[6], 0.0)

                h_rollout.append(h_step.copy())
                h_state = h_step.copy()

                # Update rolling history with new prediction
                next_feat = [h_step[6], h_step[3], h_step[4], np.cos(h_step[5]), np.sin(h_step[5]), h_step[2]]
                hist = np.vstack([hist[1:], next_feat])
                
            h_rollout = np.array(h_rollout)

            # Record errors at specific horizons
            for h in horizons:
                err_phy = np.sqrt(np.mean((p_rollout[:h, :3] - true_states[:h, :3])**2))
                err_hyb = np.sqrt(np.mean((h_rollout[:h, :3] - true_states[:h, :3])**2))
                phy_rmse_by_h[h].append(err_phy)
                hyb_rmse_by_h[h].append(err_hyb)

        summary = {
            "horizons": horizons,
            "horizon_seconds": [h * self.dt for h in horizons],
            "physics_position_rmse": [float(np.mean(phy_rmse_by_h[h])) for h in horizons],
            "hybrid_position_rmse": [float(np.mean(hyb_rmse_by_h[h])) for h in horizons]
        }
        return summary
