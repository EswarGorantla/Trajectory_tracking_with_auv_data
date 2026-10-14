import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.ml_models import MLP, LSTMModel, GRUModel
from src.physics_model import PhysicsModel

class ModelService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_dir: Optional[str] = None):
        if self._initialized:
            return
        
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.processed_dir = os.path.join(self.base_dir, "data", "processed")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config: Dict[str, Any] = {}
        self.feature_scaler = None
        self.target_scaler = None
        self.models: Dict[str, Any] = {}
        self.physics_model: Optional[PhysicsModel] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.loaded = False
        self._initialized = True

    def load_all(self):
        """Loads configuration, scalers, models, and test dataset into memory."""
        print(f"Loading ModelService on device: {self.device}")
        
        # 1. Load config
        config_path = os.path.join(self.models_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "model_type": "LSTM",
                "available_models": ["LSTM", "GRU", "MLP", "Linear", "Random Forest"],
                "sequence_length": 10,
                "sampling_time_dt": 2.0,
                "drag_coefficient_kd": 0.022498,
                "cruise_speed_u": 1.2,
                "dataset": "AURORA Autosub6000 JC125 M86"
            }

        # 2. Load scalers
        feat_scaler_path = os.path.join(self.models_dir, "feature_scaler.pkl")
        targ_scaler_path = os.path.join(self.models_dir, "target_scaler.pkl")
        
        if os.path.exists(feat_scaler_path) and os.path.exists(targ_scaler_path):
            with open(feat_scaler_path, "rb") as f:
                self.feature_scaler = pickle.load(f)
            with open(targ_scaler_path, "rb") as f:
                self.target_scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"Scalers missing in {self.models_dir}. Run run_pipeline.py first.")

        # 3. Load PyTorch models
        hidden_dim = 64
        # LSTM
        lstm_path = os.path.join(self.models_dir, "lstm.pth")
        if os.path.exists(lstm_path):
            lstm = LSTMModel(input_dim=6, hidden_dim=hidden_dim, output_dim=7).to(self.device)
            lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))
            lstm.eval()
            self.models["LSTM"] = lstm
        
        # GRU
        gru_path = os.path.join(self.models_dir, "gru.pth")
        if os.path.exists(gru_path):
            gru = GRUModel(input_dim=6, hidden_dim=hidden_dim, output_dim=7).to(self.device)
            gru.load_state_dict(torch.load(gru_path, map_location=self.device))
            gru.eval()
            self.models["GRU"] = gru

        # MLP
        mlp_path = os.path.join(self.models_dir, "mlp.pth")
        if os.path.exists(mlp_path):
            mlp = MLP(input_dim=6, hidden_dim=hidden_dim, output_dim=7).to(self.device)
            mlp.load_state_dict(torch.load(mlp_path, map_location=self.device))
            mlp.eval()
            self.models["MLP"] = mlp

        # 4. Load Scikit-Learn Baselines
        lin_path = os.path.join(self.models_dir, "linear.pkl")
        if os.path.exists(lin_path):
            with open(lin_path, "rb") as f:
                self.models["Linear"] = pickle.load(f)

        rf_path = os.path.join(self.models_dir, "random_forest.pkl")
        if os.path.exists(rf_path):
            with open(rf_path, "rb") as f:
                self.models["Random Forest"] = pickle.load(f)

        # 5. Physics Model
        kd = self.config.get("drag_coefficient_kd", 0.022498)
        dt = self.config.get("sampling_time_dt", 2.0)
        u_cruise = self.config.get("cruise_speed_u", 1.2)
        self.physics_model = PhysicsModel(k_d=kd, dt=dt, U_cruise=u_cruise)

        # 6. Load Test Dataset for real demo samples
        test_csv_path = os.path.join(self.processed_dir, "test.csv")
        if os.path.exists(test_csv_path):
            self.df_test = pd.read_csv(test_csv_path)
            self.test_records = self.df_test.to_dict('records')
            print(f"Loaded and pre-cached test dataset with {len(self.test_records)} samples.")
        else:
            self.test_records = []
        
        self.loaded = True
        print(f"ModelService successfully initialized. Available models: {list(self.models.keys())}")

    def get_sample(self, index: Optional[int] = None):
        """Fetches a real sample sequence and current state in <1ms."""
        if not self.test_records or len(self.test_records) < 20:
            raise ValueError("Test dataset not loaded.")
        
        seq_len = self.config.get("sequence_length", 10)
        max_idx = len(self.test_records) - 5
        if index is None:
            index = int(np.random.randint(seq_len, max_idx))
        else:
            index = max(seq_len, min(index, max_idx))

        history_rows = self.test_records[index - seq_len + 1 : index + 1]
        current_row = self.test_records[index]
        next_row = self.test_records[index + 1]

        history_list = [
            {
                "north": float(r["North"]),
                "east": float(r["East"]),
                "depth": float(r["depth"]),
                "roll": float(r["Roll"]),
                "pitch": float(r["Pitch"]),
                "heading": float(r["Heading"]),
                "speed": float(r["GroundSpeed"])
            }
            for r in history_rows
        ]

        current_state = {
            "north": float(current_row["North"]),
            "east": float(current_row["East"]),
            "depth": float(current_row["depth"]),
            "roll": float(current_row["Roll"]),
            "pitch": float(current_row["Pitch"]),
            "heading": float(current_row["Heading"]),
            "speed": float(current_row["GroundSpeed"])
        }

        commands = {
            "u_cmd": float(current_row["GroundSpeed"]),
            "theta_cmd": float(current_row["Pitch"]),
            "psi_cmd": float(current_row["Heading"])
        }

        actual_next_state = {
            "north": float(next_row["North"]),
            "east": float(next_row["East"]),
            "depth": float(next_row["depth"]),
            "roll": float(next_row["Roll"]),
            "pitch": float(next_row["Pitch"]),
            "heading": float(next_row["Heading"]),
            "speed": float(next_row["GroundSpeed"])
        }

        return {
            "sample_index": index,
            "current_state": current_state,
            "commands": commands,
            "sequence_history": history_list,
            "actual_next_state": actual_next_state
        }

# Global singleton
model_service = ModelService()
