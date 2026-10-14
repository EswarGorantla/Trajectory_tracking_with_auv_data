import os
import sys
import json
import pickle
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

# Reconstruct Phase 1 GRU Architecture strictly matching saved state_dict
class Phase1GRUModel(nn.Module):
    """
    Exact PyTorch architecture of Phase 1 best_dynamics_model.pt:
    2-Layer Recurrent Gated Recurrent Unit (GRU) with Linear-ReLU decoder.
    Input: [Batch, Sequence=10, Features=6]
    Output: [Batch, Residuals=7]
    Trainable parameters: 41,095
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 7, num_layers: int = 2):
        super(Phase1GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


class Phase1Loader:
    """
    Seamlessly loads and references Phase 1 trained checkpoints, scalers, and configs
    without duplicating datasets or modifying original files.
    """

    @staticmethod
    def find_phase1_paths() -> Dict[str, str]:
        # Search candidate locations for Phase 1
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [
            os.path.join(base_dir, "presentationphase1", "models"),
            os.path.join(base_dir, "PHASE_1", "models"),
            os.path.join(base_dir, "AURORA_AUV_ML_MPC", "models"),
        ]

        for c in candidates:
            ckpt = os.path.join(c, "best_dynamics_model.pt")
            scaler = os.path.join(c, "scaler.pkl")
            cfg = os.path.join(c, "model_config.json")
            if os.path.exists(ckpt) and os.path.exists(scaler) and os.path.exists(cfg):
                return {
                    "checkpoint_path": ckpt,
                    "scaler_path": scaler,
                    "config_path": cfg
                }

        raise FileNotFoundError("Could not locate trained Phase 1 model files in presentationphase1/models!")

    @classmethod
    def load_trained_model(cls, device: torch.device = None) -> Tuple[Phase1GRUModel, Any, Any, Dict[str, Any]]:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        paths = cls.find_phase1_paths()
        
        # 1. Load config
        with open(paths["config_path"], "r", encoding="utf-8") as f:
            config = json.load(f)

        # 2. Instantiate and load model
        model = Phase1GRUModel(
            input_dim=len(config["input_features"]),
            hidden_dim=config.get("hidden_size", 64),
            output_dim=len(config["target_features"]),
            num_layers=config.get("number_of_layers", 2)
        )
        state_dict = torch.load(paths["checkpoint_path"], map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # 3. Load scalers
        with open(paths["scaler_path"], "rb") as f:
            scalers = pickle.load(f)

        feature_scaler = scalers["feature_scaler"]
        target_scaler = scalers["target_scaler"]

        return model, feature_scaler, target_scaler, config
