import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import pickle
from src.ml_models import MLP, LSTMModel, GRUModel, LinearBaseline, RandomForestBaseline

class ModelTrainer:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract hyperparameters
        self.seq_len = config['hyperparameters']['seq_len']
        self.batch_size = config['hyperparameters']['batch_size']
        self.lr = config['hyperparameters']['lr']
        self.epochs = config['hyperparameters']['epochs']
        self.hidden_dim = config['hyperparameters']['hidden_dim']
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def prepare_data(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        """Standardizes input features and residual targets."""
        # Fit on train ONLY
        X_train_s = self.feature_scaler.fit_transform(X_train)
        Y_train_s = self.target_scaler.fit_transform(Y_train)
        
        # Transform val/test
        X_val_s = self.feature_scaler.transform(X_val)
        Y_val_s = self.target_scaler.transform(Y_val)
        
        X_test_s = self.feature_scaler.transform(X_test)
        Y_test_s = self.target_scaler.transform(Y_test)
        
        # Save scalers
        models_dir = self.config['paths']['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "feature_scaler.pkl"), "wb") as f:
            pickle.dump(self.feature_scaler, f)
        with open(os.path.join(models_dir, "target_scaler.pkl"), "wb") as f:
            pickle.dump(self.target_scaler, f)
            
        return X_train_s, Y_train_s, X_val_s, Y_val_s, X_test_s, Y_test_s

    def create_sequences(self, X, Y):
        """Creates sliding window input sequences for recurrent models."""
        X_seq, Y_target = [], []
        for i in range(len(X) - self.seq_len + 1):
            X_seq.append(X[i : i + self.seq_len])
            Y_target.append(Y[i + self.seq_len - 1])
        return np.array(X_seq), np.array(Y_target)

    def train_pytorch_model(self, model, X_tr, Y_tr, X_va, Y_va, model_name):
        """Trains a PyTorch neural network model using mini-batches."""
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        crit = nn.MSELoss()
        
        # Convert to tensors and move to device
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
        Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32).to(self.device)
        X_va_t = torch.tensor(X_va, dtype=torch.float32).to(self.device)
        Y_va_t = torch.tensor(Y_va, dtype=torch.float32).to(self.device)
        
        N = len(X_tr_t)
        best_val_loss = float('inf')
        best_weights = None
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            model.train()
            # Shuffle indices
            indices = np.arange(N)
            np.random.shuffle(indices)
            
            epoch_loss = 0.0
            steps = 0
            for start_idx in range(0, N, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                x_batch = X_tr_t[batch_idx]
                y_batch = Y_tr_t[batch_idx]
                
                opt.zero_grad()
                pred = model(x_batch)
                loss = crit(pred, y_batch)
                loss.backward()
                opt.step()
                
                epoch_loss += loss.item()
                steps += 1
                
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_va_t)
                val_loss = crit(val_pred, Y_va_t).item()
                
            avg_train_loss = epoch_loss / steps
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.state_dict().copy()
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  {model_name} | Epoch {epoch+1:02d}/{self.epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f}")
                
        # Load best weights
        if best_weights is not None:
            model.load_state_dict(best_weights)
            
        # Save model
        models_dir = self.config['paths']['models_dir']
        torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}.pth"))
        return train_losses, val_losses

    def run_training_pipeline(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        """Runs the entire training suite for all baseline and neural models."""
        print("\n=== Preprocessing & Scaling Features ===")
        X_tr_s, Y_tr_s, X_va_s, Y_va_s, X_te_s, Y_te_s = self.prepare_data(
            X_train, Y_train, X_val, Y_val, X_test, Y_test
        )
        
        # 1. Classical Baselines
        print("\n=== Training Baselines ===")
        linear = LinearBaseline()
        linear.fit(X_tr_s, Y_tr_s)
        print("  Linear Baseline fitted.")
        
        rf = RandomForestBaseline()
        rf.fit(X_tr_s, Y_tr_s)
        print("  Random Forest Baseline fitted.")
        
        # Save baselines
        models_dir = self.config['paths']['models_dir']
        with open(os.path.join(models_dir, "linear.pkl"), "wb") as f:
            pickle.dump(linear, f)
        with open(os.path.join(models_dir, "random_forest.pkl"), "wb") as f:
            pickle.dump(rf, f)
            
        # 2. PyTorch MLP (one-step)
        print("\n=== Training PyTorch MLP ===")
        mlp = MLP(input_dim=6, hidden_dim=self.hidden_dim, output_dim=7).to(self.device)
        mlp_train_loss, mlp_val_loss = self.train_pytorch_model(
            mlp, X_tr_s, Y_tr_s, X_va_s, Y_va_s, "mlp"
        )
        
        # 3. Create Sequences for LSTM / GRU
        print("\n=== Generating Sequence Windows ===")
        X_tr_seq, Y_tr_seq = self.create_sequences(X_tr_s, Y_tr_s)
        X_va_seq, Y_va_seq = self.create_sequences(X_va_s, Y_va_s)
        
        # 4. PyTorch LSTM
        print("\n=== Training PyTorch LSTM ===")
        lstm = LSTMModel(input_dim=6, hidden_dim=self.hidden_dim, output_dim=7).to(self.device)
        lstm_train_loss, lstm_val_loss = self.train_pytorch_model(
            lstm, X_tr_seq, Y_tr_seq, X_va_seq, Y_va_seq, "lstm"
        )
        
        # 5. PyTorch GRU
        print("\n=== Training PyTorch GRU ===")
        gru = GRUModel(input_dim=6, hidden_dim=self.hidden_dim, output_dim=7).to(self.device)
        gru_train_loss, gru_val_loss = self.train_pytorch_model(
            gru, X_tr_seq, Y_tr_seq, X_va_seq, Y_va_seq, "gru"
        )
        
        # Package history for curves plotting
        history = {
            'mlp': (mlp_train_loss, mlp_val_loss),
            'lstm': (lstm_train_loss, lstm_val_loss),
            'gru': (gru_train_loss, gru_val_loss)
        }
        return history
