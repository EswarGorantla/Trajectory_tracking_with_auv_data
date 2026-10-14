import torch
import numpy as np
from typing import Dict, Any, Tuple

from src.physics_model import PhysicsModel

class HybridDynamicsModel:
    """
    Combines deterministic 3D AUV kinematics with a trained PyTorch ML residual model.
    """
    def __init__(self, physics_model: PhysicsModel, ml_model: torch.nn.Module, feature_scaler, target_scaler, device: torch.device):
        self.physics = physics_model
        self.ml_model = ml_model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.device = device
        self.ml_model.eval()

    def predict_next_state(self, current_state: np.ndarray, history_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs one-step hybrid prediction:
        Returns: (hybrid_next_state, physics_next_state, ml_residual)
        """
        # 1. Physics Step
        phys_next = self.physics.step(current_state)

        # 2. ML Residual Step
        feat_scaled = self.feature_scaler.transform(history_features)
        with torch.no_grad():
            if feat_scaled.ndim == 2:
                tensor_in = torch.tensor(np.array([feat_scaled]), dtype=torch.float32).to(self.device)
            else:
                tensor_in = torch.tensor(feat_scaled, dtype=torch.float32).to(self.device)
                
            res_scaled = self.ml_model(tensor_in).cpu().numpy()[0]
            residuals = self.target_scaler.inverse_transform([res_scaled])[0]

        # 3. Hybrid Combination
        hybrid_next = phys_next.copy()
        hybrid_next[0] += residuals[0] # North
        hybrid_next[1] += residuals[1] # East
        hybrid_next[2] += residuals[2] # Depth
        hybrid_next[3] += residuals[3] # Roll
        hybrid_next[4] += residuals[4] # Pitch
        hybrid_next[5] = (phys_next[5] + residuals[5]) % (2 * np.pi) # Heading wrap
        hybrid_next[6] = max(phys_next[6] + residuals[6], 0.0) # Speed non-negative clamp

        return hybrid_next, phys_next, residuals
