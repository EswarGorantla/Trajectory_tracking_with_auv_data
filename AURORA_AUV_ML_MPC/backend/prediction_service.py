import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from backend.schemas import AUVState, ControlInput, PredictRequest, PredictResponse, PredictSequenceRequest, PredictSequenceResponse
from backend.model_service import model_service

class PredictionService:
    def __init__(self):
        self.ms = model_service

    def _ensure_loaded(self):
        if not self.ms.loaded or self.ms.physics_model is None:
            self.ms.load_all()

    def _state_to_array(self, state: AUVState) -> np.ndarray:
        return np.array([
            state.north,
            state.east,
            state.depth,
            state.roll,
            state.pitch,
            state.heading,
            state.speed
        ], dtype=np.float64)

    def _array_to_state(self, arr: np.ndarray) -> AUVState:
        return AUVState(
            north=float(arr[0]),
            east=float(arr[1]),
            depth=float(arr[2]),
            roll=float(arr[3]),
            pitch=float(arr[4]),
            heading=float(arr[5] % (2 * np.pi)),
            speed=float(max(arr[6], 0.0))
        )

    def _state_to_features(self, state: AUVState) -> List[float]:
        # Exact feature ordering used during training:
        # [GroundSpeed, Roll, Pitch, cos(Heading), sin(Heading), depth]
        return [
            state.speed,
            state.roll,
            state.pitch,
            float(np.cos(state.heading)),
            float(np.sin(state.heading)),
            state.depth
        ]

    def predict(self, req: PredictRequest) -> PredictResponse:
        self._ensure_loaded()
        t0 = time.time()
        
        # 1. Physics baseline prediction
        cur_arr = self._state_to_array(req.current_state)
        phys_arr = self.ms.physics_model.step(
            cur_arr,
            req.commands.u_cmd,
            req.commands.theta_cmd,
            req.commands.psi_cmd
        )
        phys_state = self._array_to_state(phys_arr)

        # 2. Build sequence features (length 10)
        seq_len = self.ms.config.get("sequence_length", 10)
        if req.sequence_history and len(req.sequence_history) >= seq_len:
            hist_states = req.sequence_history[-seq_len:]
        else:
            # Fallback: synthesize static sequence from current state
            hist_states = [req.current_state for _ in range(seq_len)]
            
        feature_matrix = np.array([self._state_to_features(s) for s in hist_states])
        feature_scaled = self.ms.feature_scaler.transform(feature_matrix)

        # 3. Model inference
        model_name = req.model_name.upper() if req.model_name else "LSTM"
        if model_name in ["RANDOM FOREST", "RF"]:
            model_key = "Random Forest"
        elif model_name in ["LINEAR", "LIN"]:
            model_key = "Linear"
        elif model_name in ["GRU"]:
            model_key = "GRU"
        elif model_name in ["MLP"]:
            model_key = "MLP"
        else:
            model_key = "LSTM"

        model = self.ms.models.get(model_key, self.ms.models.get("LSTM"))
        if model is None:
            raise RuntimeError(f"Requested model '{model_key}' is not loaded.")

        if model_key in ["LSTM", "GRU"]:
            # PyTorch recurrent model (expects [1, seq_len, 6])
            with torch.no_grad():
                tensor_in = torch.tensor(np.array([feature_scaled]), dtype=torch.float32).to(self.ms.device)
                res_scaled = model(tensor_in).cpu().numpy()[0]
        elif model_key == "MLP":
            # PyTorch feedforward model (expects [1, 6] of latest step)
            with torch.no_grad():
                tensor_in = torch.tensor(np.array([feature_scaled[-1]]), dtype=torch.float32).to(self.ms.device)
                res_scaled = model(tensor_in).cpu().numpy()[0]
        else:
            # Scikit-learn model (expects [1, 6] of latest step)
            res_scaled = model.predict(np.array([feature_scaled[-1]]))[0]

        # 4. Unscale residuals
        residuals = self.ms.target_scaler.inverse_transform([res_scaled])[0]
        
        # 5. Add residual corrections to physics prediction
        final_arr = phys_arr.copy()
        final_arr[0] += residuals[0] # North
        final_arr[1] += residuals[1] # East
        final_arr[2] += residuals[2] # Depth
        final_arr[3] += residuals[3] # Roll
        final_arr[4] += residuals[4] # Pitch
        final_arr[5] = (phys_arr[5] + residuals[5]) % (2 * np.pi) # Heading wrap
        final_arr[6] = max(phys_arr[6] + residuals[6], 0.0) # Speed clamp

        final_state = self._array_to_state(final_arr)

        res_dict = {
            "d_north_m": float(residuals[0]),
            "d_east_m": float(residuals[1]),
            "d_depth_m": float(residuals[2]),
            "d_roll_rad": float(residuals[3]),
            "d_pitch_rad": float(residuals[4]),
            "d_heading_rad": float(residuals[5]),
            "d_speed_m_s": float(residuals[6])
        }

        dt_ms = (time.time() - t0) * 1000.0
        return PredictResponse(
            success=True,
            physics_prediction=phys_state,
            ai_residual_correction=res_dict,
            final_prediction=final_state,
            model_used=model_key,
            prediction_horizon=1,
            execution_time_ms=round(dt_ms, 2)
        )

    def predict_dataframe(self, df: pd.DataFrame, model_name: str = "LSTM") -> Dict[str, Any]:
        """Processes arbitrary uploaded CSV dataframe and runs step-by-step predictions."""
        self._ensure_loaded()
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        
        def find_col(aliases, default=0.0):
            for a in aliases:
                if a in cols_lower:
                    return df[cols_lower[a]].astype(float).values
            return np.full(len(df), default, dtype=np.float64)

        north_arr = find_col(['north', 'x', 'northing'], 0.0)
        east_arr = find_col(['east', 'y', 'easting'], 0.0)
        depth_arr = find_col(['depth', 'z'], 20.0)
        speed_arr = find_col(['groundspeed', 'speed', 'u', 'surge'], 1.2)
        pitch_arr = find_col(['pitch', 'theta'], 0.0)
        roll_arr = find_col(['roll', 'phi'], 0.0)
        heading_arr = find_col(['heading', 'psi', 'yaw'], 0.0)

        # Detect degree angles and convert to radians
        if np.max(np.abs(heading_arr)) > 6.3:
            heading_arr = np.radians(heading_arr)
        if np.max(np.abs(pitch_arr)) > 3.2:
            pitch_arr = np.radians(pitch_arr)
        if np.max(np.abs(roll_arr)) > 3.2:
            roll_arr = np.radians(roll_arr)

        n_rows = len(df)
        dt = self.ms.config.get("sampling_time_dt", 2.0)
        seq_len = self.ms.config.get("sequence_length", 10)
        
        model_key = model_name.upper() if model_name.upper() in ["LSTM", "GRU", "MLP"] else "LSTM"
        model = self.ms.models.get(model_key, self.ms.models.get("LSTM"))

        results = []
        history_features = []

        for i in range(n_rows):
            cur_state = np.array([
                north_arr[i], east_arr[i], depth_arr[i],
                roll_arr[i], pitch_arr[i], heading_arr[i], speed_arr[i]
            ])
            
            feat = [
                cur_state[6], cur_state[3], cur_state[4],
                float(np.cos(cur_state[5])), float(np.sin(cur_state[5])), cur_state[2]
            ]
            history_features.append(feat)
            if len(history_features) > seq_len:
                history_features.pop(0)

            # Commands: next row command if available, otherwise current
            u_cmd = speed_arr[min(i + 1, n_rows - 1)]
            theta_cmd = pitch_arr[min(i + 1, n_rows - 1)]
            psi_cmd = heading_arr[min(i + 1, n_rows - 1)]

            # 1. Physics Step
            phys_pred = self.ms.physics_model.step(cur_state, u_cmd, theta_cmd, psi_cmd)

            # 2. AI Residual Correction
            if len(history_features) < seq_len:
                padded_hist = [history_features[0]] * (seq_len - len(history_features)) + history_features
            else:
                padded_hist = history_features

            feat_scaled = self.ms.feature_scaler.transform(np.array(padded_hist))
            with torch.no_grad():
                tensor_in = torch.tensor(np.array([feat_scaled]), dtype=torch.float32).to(self.ms.device)
                res_scaled = model(tensor_in).cpu().numpy()[0]
                residuals = self.ms.target_scaler.inverse_transform([res_scaled])[0]

            # 3. Final Predicted State
            final_pred = phys_pred.copy()
            final_pred[0] += residuals[0]
            final_pred[1] += residuals[1]
            final_pred[2] += residuals[2]
            final_pred[3] += residuals[3]
            final_pred[4] += residuals[4]
            final_pred[5] = (phys_pred[5] + residuals[5]) % (2 * np.pi)
            final_pred[6] = max(phys_pred[6] + residuals[6], 0.0)

            results.append({
                "step": i,
                "time_s": round(i * dt, 1),
                "input_state": {
                    "north": round(float(cur_state[0]), 3),
                    "east": round(float(cur_state[1]), 3),
                    "depth": round(float(cur_state[2]), 3),
                    "speed": round(float(cur_state[6]), 3),
                    "heading_deg": round(float(np.degrees(cur_state[5])), 2),
                    "pitch_deg": round(float(np.degrees(cur_state[4])), 2)
                },
                "physics_prediction": {
                    "north": round(float(phys_pred[0]), 3),
                    "east": round(float(phys_pred[1]), 3),
                    "depth": round(float(phys_pred[2]), 3),
                    "speed": round(float(phys_pred[6]), 3),
                    "heading_deg": round(float(np.degrees(phys_pred[5])), 2)
                },
                "ai_correction": {
                    "d_north_m": round(float(residuals[0]), 3),
                    "d_east_m": round(float(residuals[1]), 3),
                    "d_depth_m": round(float(residuals[2]), 3),
                    "d_speed_m_s": round(float(residuals[6]), 3),
                    "d_heading_deg": round(float(np.degrees(residuals[5])), 2)
                },
                "final_predicted_state": {
                    "north": round(float(final_pred[0]), 3),
                    "east": round(float(final_pred[1]), 3),
                    "depth": round(float(final_pred[2]), 3),
                    "speed": round(float(final_pred[6]), 3),
                    "heading_deg": round(float(np.degrees(final_pred[5])), 2)
                }
            })

        return {
            "success": True,
            "total_rows": n_rows,
            "model_used": model_key,
            "predictions": results
        }

    def predict_sequence(self, req: PredictSequenceRequest) -> PredictSequenceResponse:
        """Executes multi-step rolling forecast on real test trajectory segment."""
        df_test = self.ms.df_test
        if df_test is None:
            raise RuntimeError("Test dataset not available.")

        seq_len = self.ms.config.get("sequence_length", 10)
        start_idx = max(seq_len, min(req.start_index, len(df_test) - req.steps - 2))
        steps = min(req.steps, 50)
        dt = self.ms.config.get("sampling_time_dt", 2.0)

        sub_df = df_test.iloc[start_idx : start_idx + steps + 1].reset_index(drop=True)
        
        actual_traj = []
        physics_traj = []
        ai_traj = []
        timesteps = []

        for i, row in sub_df.iterrows():
            actual_traj.append({
                "step": int(i),
                "time_s": float(i * dt),
                "north": float(row["North"]),
                "east": float(row["East"]),
                "depth": float(row["depth"]),
                "speed": float(row["GroundSpeed"]),
                "heading_deg": float(np.degrees(row["Heading"]))
            })

        init_row = sub_df.iloc[0]
        phys_state = np.array([
            init_row["North"], init_row["East"], init_row["depth"],
            init_row["Roll"], init_row["Pitch"], init_row["Heading"],
            init_row["GroundSpeed"]
        ])
        ai_state = phys_state.copy()

        physics_traj.append({
            "step": 0, "time_s": 0.0,
            "north": float(phys_state[0]), "east": float(phys_state[1]), "depth": float(phys_state[2]),
            "speed": float(phys_state[6]), "heading_deg": float(np.degrees(phys_state[5]))
        })
        ai_traj.append({
            "step": 0, "time_s": 0.0,
            "north": float(ai_state[0]), "east": float(ai_state[1]), "depth": float(ai_state[2]),
            "speed": float(ai_state[6]), "heading_deg": float(np.degrees(ai_state[5]))
        })
        timesteps.append(0.0)

        history = []
        for h in range(seq_len):
            r = df_test.iloc[start_idx - seq_len + h + 1]
            history.append([r["GroundSpeed"], r["Roll"], r["Pitch"], np.cos(r["Heading"]), np.sin(r["Heading"]), r["depth"]])

        model = self.ms.models.get("LSTM" if req.model_name.upper() == "LSTM" else "GRU")

        for k in range(steps):
            t_curr = (k + 1) * dt
            timesteps.append(float(t_curr))
            
            u_cmd = sub_df.iloc[k]["GroundSpeed"]
            theta_cmd = sub_df.iloc[k]["Pitch"]
            psi_cmd = sub_df.iloc[k]["Heading"]

            # Physics rollout
            phys_state = self.ms.physics_model.step(phys_state, u_cmd, theta_cmd, psi_cmd)
            physics_traj.append({
                "step": k + 1, "time_s": float(t_curr),
                "north": float(phys_state[0]), "east": float(phys_state[1]), "depth": float(phys_state[2]),
                "speed": float(phys_state[6]), "heading_deg": float(np.degrees(phys_state[5]))
            })

            # AI rollout
            ai_phys = self.ms.physics_model.step(ai_state, u_cmd, theta_cmd, psi_cmd)
            feat_scaled = self.ms.feature_scaler.transform(np.array(history))
            with torch.no_grad():
                tensor_in = torch.tensor(np.array([feat_scaled]), dtype=torch.float32).to(self.ms.device)
                res_scaled = model(tensor_in).cpu().numpy()[0]
                residuals = self.ms.target_scaler.inverse_transform([res_scaled])[0]

            ai_state = ai_phys.copy()
            ai_state[0] += residuals[0]
            ai_state[1] += residuals[1]
            ai_state[2] += residuals[2]
            ai_state[3] += residuals[3]
            ai_state[4] = ai_phys[4]
            ai_state[5] = (ai_phys[5] + residuals[5]) % (2 * np.pi)
            ai_state[6] = max(ai_phys[6] + residuals[6], 0.0)

            ai_traj.append({
                "step": k + 1, "time_s": float(t_curr),
                "north": float(ai_state[0]), "east": float(ai_state[1]), "depth": float(ai_state[2]),
                "speed": float(ai_state[6]), "heading_deg": float(np.degrees(ai_state[5]))
            })

            history.pop(0)
            history.append([ai_state[6], ai_state[3], ai_state[4], np.cos(ai_state[5]), np.sin(ai_state[5]), ai_state[2]])

        act_arr = np.array([[p["north"], p["east"], p["depth"]] for p in actual_traj])
        phy_arr = np.array([[p["north"], p["east"], p["depth"]] for p in physics_traj])
        ai_arr = np.array([[p["north"], p["east"], p["depth"]] for p in ai_traj])

        rmse_phy_pos = float(np.sqrt(np.mean((phy_arr[:, :2] - act_arr[:, :2])**2)))
        rmse_phy_depth = float(np.sqrt(np.mean((phy_arr[:, 2] - act_arr[:, 2])**2)))
        
        rmse_ai_pos = float(np.sqrt(np.mean((ai_arr[:, :2] - act_arr[:, :2])**2)))
        rmse_ai_depth = float(np.sqrt(np.mean((ai_arr[:, 2] - act_arr[:, 2])**2)))

        return PredictSequenceResponse(
            success=True,
            timesteps=timesteps,
            actual_trajectory=actual_traj,
            physics_trajectory=physics_traj,
            ai_trajectory=ai_traj,
            rmse_physics={"position_rmse_m": round(rmse_phy_pos, 4), "depth_rmse_m": round(rmse_phy_depth, 4)},
            rmse_ai={"position_rmse_m": round(rmse_ai_pos, 4), "depth_rmse_m": round(rmse_ai_depth, 4)}
        )

prediction_service = PredictionService()
