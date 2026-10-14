import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from src.state_definition import StateDefinition
from src.residual_model import HybridDynamicsModel

class Predictor:
    """Handles independent checkpoint loading and evaluation across unseen test telemetry."""

    def __init__(self, dt: float = 2.0, seq_len: int = 10):
        self.dt = dt
        self.seq_len = seq_len

    def evaluate_test_split(
        self,
        test_df: pd.DataFrame,
        hybrid_model: HybridDynamicsModel,
        ml_only_model: torch.nn.Module,
        feature_scaler,
        target_scaler,
        device: torch.device
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Runs single-step prediction on the entire UNSEEN test split for:
        1. Model A: Physics-Only
        2. Model B: ML-Only
        3. Model C: Physics + ML Hybrid
        """
        state_matrix = StateDefinition.extract_state_matrix(test_df)
        features_all = StateDefinition.extract_feature_matrix(state_matrix)

        act_next = state_matrix[1:]  # actual next state (N-1, 7)
        num_steps = len(act_next)

        phys_next = np.zeros_like(act_next)
        ml_only_next = np.zeros_like(act_next)
        hybrid_next = np.zeros_like(act_next)
        residuals_pred = np.zeros_like(act_next)

        ml_only_model.eval()

        print(f"\n============================================================")
        print(f"12. GENERATING PREDICTIONS ON {num_steps} UNSEEN TEST SAMPLES")
        print(f"============================================================")

        for t in range(num_steps):
            cur_state = state_matrix[t]
            
            # Model A: Physics-Only Step
            p_step = hybrid_model.physics.step(cur_state)
            phys_next[t] = p_step

            # Temporal history window
            if t < self.seq_len - 1:
                hist = np.vstack([np.tile(features_all[0], (self.seq_len - 1 - t, 1)), features_all[: t + 1]])
            else:
                hist = features_all[t - self.seq_len + 1 : t + 1]

            # Model C: Hybrid Residual Step
            feat_scaled = feature_scaler.transform(hist)
            with torch.no_grad():
                tensor_in = torch.tensor(np.array([feat_scaled]), dtype=torch.float32).to(device)
                
                # Hybrid Residual
                res_scaled = hybrid_model.ml_model(tensor_in).cpu().numpy()[0]
                res = target_scaler.inverse_transform([res_scaled])[0]
                residuals_pred[t] = res

                # Model B: Direct ML-Only Delta
                ml_delta_scaled = ml_only_model(tensor_in).cpu().numpy()[0]
                ml_delta = target_scaler.inverse_transform([ml_delta_scaled])[0]

            # Model C Synthesis
            h_step = p_step.copy()
            h_step[0] += res[0]
            h_step[1] += res[1]
            h_step[2] += res[2]
            h_step[3] += res[3]
            h_step[4] += res[4]
            h_step[5] = (p_step[5] + res[5]) % (2 * np.pi)
            h_step[6] = max(p_step[6] + res[6], 0.0)
            hybrid_next[t] = h_step

            # Model B Synthesis (Direct State + Learned Delta)
            m_step = cur_state.copy()
            m_step[0] += ml_delta[0]
            m_step[1] += ml_delta[1]
            m_step[2] += ml_delta[2]
            m_step[3] += ml_delta[3]
            m_step[4] += ml_delta[4]
            m_step[5] = (cur_state[5] + ml_delta[5]) % (2 * np.pi)
            m_step[6] = max(cur_state[6] + ml_delta[6], 0.0)
            ml_only_next[t] = m_step

        timestamps = np.arange(num_steps) * self.dt
        if "Seconds" in test_df.columns:
            timestamps = test_df["Seconds"].iloc[1:].values

        df_preds = pd.DataFrame({
            "timestamp": timestamps,
            "actual_north": act_next[:, 0],
            "predicted_north_physics": phys_next[:, 0],
            "predicted_north_ml": ml_only_next[:, 0],
            "predicted_north_hybrid": hybrid_next[:, 0],

            "actual_east": act_next[:, 1],
            "predicted_east_physics": phys_next[:, 1],
            "predicted_east_ml": ml_only_next[:, 1],
            "predicted_east_hybrid": hybrid_next[:, 1],

            "actual_depth": act_next[:, 2],
            "predicted_depth_physics": phys_next[:, 2],
            "predicted_depth_ml": ml_only_next[:, 2],
            "predicted_depth_hybrid": hybrid_next[:, 2],

            "actual_surge_speed": act_next[:, 6],
            "predicted_surge_speed_physics": phys_next[:, 6],
            "predicted_surge_speed_ml": ml_only_next[:, 6],
            "predicted_surge_speed_hybrid": hybrid_next[:, 6],

            "actual_pitch": act_next[:, 4],
            "predicted_pitch_physics": phys_next[:, 4],
            "predicted_pitch_ml": ml_only_next[:, 4],
            "predicted_pitch_hybrid": hybrid_next[:, 4],

            "actual_heading": act_next[:, 5],
            "predicted_heading_physics": phys_next[:, 5],
            "predicted_heading_ml": ml_only_next[:, 5],
            "predicted_heading_hybrid": hybrid_next[:, 5],

            "actual_roll": act_next[:, 3],
            "predicted_roll_physics": phys_next[:, 3],
            "predicted_roll_ml": ml_only_next[:, 3],
            "predicted_roll_hybrid": hybrid_next[:, 3]
        })

        raw_preds = {
            "actual": act_next,
            "physics": phys_next,
            "ml_only": ml_only_next,
            "hybrid": hybrid_next,
            "residuals": residuals_pred
        }
        return df_preds, raw_preds

    def evaluate_multistep_rollout(
        self,
        test_df: pd.DataFrame,
        hybrid_model: HybridDynamicsModel,
        feature_scaler,
        target_scaler,
        device: torch.device,
        horizons: List[int] = [1, 5, 10],
        num_test_sequences: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluates multi-step recursive rollouts (1, 5, 10 steps) comparing compound position error.
        """
        state_matrix = StateDefinition.extract_state_matrix(test_df)
        features_all = StateDefinition.extract_feature_matrix(state_matrix)
        max_h = max(horizons)

        phy_rmse = {h: [] for h in horizons}
        hyb_rmse = {h: [] for h in horizons}

        start_indices = np.linspace(self.seq_len, len(state_matrix) - max_h - 1, num_test_sequences, dtype=int)

        for s_idx in start_indices:
            true_states = state_matrix[s_idx : s_idx + max_h]

            # Physics rollout
            p_state = state_matrix[s_idx - 1].copy()
            p_roll = []
            for h in range(max_h):
                p_state = hybrid_model.physics.step(p_state)
                p_roll.append(p_state.copy())
            p_roll = np.array(p_roll)

            # Hybrid rollout
            h_state = state_matrix[s_idx - 1].copy()
            hist = features_all[s_idx - self.seq_len : s_idx].copy()
            h_roll = []
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

                h_roll.append(h_step.copy())
                h_state = h_step.copy()

                next_feat = [h_step[6], h_step[3], h_step[4], np.cos(h_step[5]), np.sin(h_step[5]), h_step[2]]
                hist = np.vstack([hist[1:], next_feat])

            h_roll = np.array(h_roll)

            for h in horizons:
                e_p = np.sqrt(np.mean((p_roll[:h, :3] - true_states[:h, :3])**2))
                e_h = np.sqrt(np.mean((h_roll[:h, :3] - true_states[:h, :3])**2))
                phy_rmse[h].append(e_p)
                hyb_rmse[h].append(e_h)

        res = {
            "horizons": horizons,
            "horizon_seconds": [h * self.dt for h in horizons],
            "physics_position_rmse": [float(np.mean(phy_rmse[h])) for h in horizons],
            "hybrid_position_rmse": [float(np.mean(hyb_rmse[h])) for h in horizons]
        }
        return res
