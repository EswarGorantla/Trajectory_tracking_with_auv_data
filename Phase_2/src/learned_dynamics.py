import torch
import numpy as np
from typing import Tuple, Dict, Any
from models.phase1_loader import Phase1Loader, Phase1GRUModel

class LearnedDynamics:
    """
    Inference engine for the Phase 1 Deep GRU Residual Model.
    Predicts unmodeled 7D hydrodynamic forces and attitude corrections:
    r_hat = [dr_x, dr_y, dr_z, dr_phi, dr_theta, dr_psi, dr_speed]
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.feature_scaler, self.target_scaler, self.config = Phase1Loader.load_trained_model(self.device)
        self.call_count = 0
        self.total_samples_inferred = 0

    def predict_residual(self, feature_history: np.ndarray) -> np.ndarray:
        """
        Infers 7D residual vector for a single history window of shape (10, 6).
        """
        self.call_count += 1
        self.total_samples_inferred += 1

        # Scale history window features
        feat_scaled = self.feature_scaler.transform(feature_history)
        tensor_in = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            res_scaled = self.model(tensor_in).cpu().numpy()[0]
            residuals = self.target_scaler.inverse_transform([res_scaled])[0]

        return residuals

    def predict_residual_batch(self, feature_histories: np.ndarray) -> np.ndarray:
        """
        Vectorized batch inference on (N, 10, 6) history array for parallel MPC rollouts.
        """
        n_samples = feature_histories.shape[0]
        self.call_count += 1
        self.total_samples_inferred += n_samples

        # Reshape to (N*10, 6) for rapid vectorized scaling
        n_steps = feature_histories.shape[1]
        n_feats = feature_histories.shape[2]
        flat_feats = feature_histories.reshape(-1, n_feats)
        flat_scaled = self.feature_scaler.transform(flat_feats)
        hist_scaled = flat_scaled.reshape(n_samples, n_steps, n_feats)

        tensor_in = torch.tensor(hist_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            res_scaled = self.model(tensor_in).cpu().numpy()
            residuals = self.target_scaler.inverse_transform(res_scaled)

        return residuals

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_architecture": self.config.get("architecture", "2-Layer Recurrent GRU"),
            "call_count": self.call_count,
            "total_samples_inferred": self.total_samples_inferred,
            "device": str(self.device)
        }
