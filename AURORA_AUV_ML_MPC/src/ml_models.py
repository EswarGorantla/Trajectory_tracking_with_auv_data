import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor

class MLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=7):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=7):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output of the last sequence step
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=7):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        # Take the output of the last sequence step
        return self.fc(out[:, -1, :])

# Scikit-Learn Wrapper classes for convenience
class LinearBaseline:
    def __init__(self):
        self.model = SkLinearRegression()
    def fit(self, X, Y):
        self.model.fit(X, Y)
    def predict(self, X):
        return self.model.predict(X)

class RandomForestBaseline:
    def __init__(self, n_estimators=50, max_depth=10, random_state=42):
        self.model = SkRandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
    def fit(self, X, Y):
        self.model.fit(X, Y)
    def predict(self, X):
        return self.model.predict(X)
