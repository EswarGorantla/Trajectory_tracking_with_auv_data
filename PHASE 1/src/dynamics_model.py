import torch
import torch.nn as nn

class MLP(nn.Module):
    """Feedforward Multi-Layer Perceptron for single-step dynamic mapping."""
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 7):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input has sequence dimension [B, L, D], take the latest timestep [B, D]
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x)

class LSTMModel(nn.Module):
    """2-Layer Recurrent LSTM for temporal dynamic sequence modeling."""
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 7, num_layers: int = 2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        out, _ = self.lstm(x)
        # Decode the last hidden state
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

class GRUModel(nn.Module):
    """2-Layer Gated Recurrent Unit for temporal dynamic sequence modeling."""
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 7, num_layers: int = 2):
        super(GRUModel, self).__init__()
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
