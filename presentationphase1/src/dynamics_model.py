import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    2-Layer Gated Recurrent Unit (GRU) for temporal sequence dynamics prediction.
    Input: [Batch, Sequence=10, Features=6]
    Output: [Batch, Residuals=7]
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 7, num_layers: int = 2):
        super(GRUModel, self).__init__()
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

class LSTMModel(nn.Module):
    """
    2-Layer Long Short-Term Memory (LSTM) network for comparative dynamic modeling.
    Input: [Batch, Sequence=10, Features=6]
    Output: [Batch, Residuals=7]
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 7, num_layers: int = 2):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)
