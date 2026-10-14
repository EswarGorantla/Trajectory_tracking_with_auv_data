import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple

from src.dynamics_model import MLP, LSTMModel, GRUModel
from src.preprocessing import Preprocessor

class ModelTrainer:
    """Trains PyTorch dynamics models and manages checkpointing, loss logs, and scalers."""

    def __init__(self, models_dir: str, results_dir: str, device: torch.device = None):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "training"), exist_ok=True)

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def train_epoch(self, model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def eval_epoch(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def train_model(
        self,
        model: nn.Module,
        model_name: str,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        X_val_seq: np.ndarray,
        y_val_seq: np.ndarray,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 0.001,
        patience: int = 7
    ) -> Dict[str, Any]:
        """Runs the PyTorch training loop with early stopping."""
        train_ds = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss = float('inf')
        best_weights = None
        best_epoch = 0
        patience_counter = 0

        train_losses = []
        val_losses = []

        print(f"\n--- Training {model_name} on {self.device} ---")
        for epoch in range(1, epochs + 1):
            t_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            v_loss = self.eval_epoch(model, val_loader, criterion)
            scheduler.step(v_loss)

            train_losses.append(t_loss)
            val_losses.append(v_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:02d}/{epochs} | Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f}")

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_weights = model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping triggered at Epoch {epoch} (Best Epoch: {best_epoch})")
                    break

        # Load best weights
        model.load_state_dict(best_weights)
        print(f"  {model_name} Training Completed. Best Validation Loss: {best_val_loss:.5f} (Epoch {best_epoch})")

        return {
            "model_name": model_name,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "train_losses": train_losses,
            "val_losses": val_losses
        }

    def save_artifacts(
        self,
        best_model: nn.Module,
        best_info: Dict[str, Any],
        model_config: Dict[str, Any]
    ):
        """Saves weights, config, scalers, and training history."""
        # 1. Save Best Model
        model_path = os.path.join(self.models_dir, "best_dynamics_model.pt")
        torch.save(best_model.state_dict(), model_path)
        print(f"Saved best model weights to {model_path}")

        # 2. Save Scalers
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump({
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler
            }, f)
        print(f"Saved scalers to {scaler_path}")

        # 3. Save Config
        config_path = os.path.join(self.models_dir, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        print(f"Saved model config to {config_path}")

        # 4. Save Training History JSON
        history_path = os.path.join(self.models_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(best_info, f, indent=2)
        print(f"Saved training history to {history_path}")

        # 5. Plot and save training loss curve
        self.plot_training_loss(best_info)

    def plot_training_loss(self, history: Dict[str, Any]):
        """Generates results/training/training_loss.png."""
        plt.figure(figsize=(8, 5))
        plt.plot(history['train_losses'], label='Training Loss (MSE)', color='#0077b6', lw=2)
        plt.plot(history['val_losses'], label='Validation Loss (MSE)', color='#e63946', lw=2, linestyle='--')
        plt.axvline(history['best_epoch'] - 1, color='green', linestyle=':', label=f"Best Epoch ({history['best_epoch']})")
        plt.title(f"Training & Validation Loss Convergence ({history['model_name']})", fontsize=12, fontweight='bold')
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Normalized MSE Loss", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(self.results_dir, "training", "training_loss.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved training loss curve to {plot_path}")
