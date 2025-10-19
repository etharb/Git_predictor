import torch
import torch.nn as nn

class SmallLSTM(nn.Module):
    def __init__(self, in_features: int, hidden: int = 48, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (B, H)
        p = self.fc(last)     # (B, 1) in (0,1)
        return p.squeeze(-1)
