import torch
import torch.nn as nn

class GazeLSTMNet(nn.Module):
    def __init__(self, input_channels=8, width=60, hidden_dim=128, lstm_layers=2):
        super(GazeLSTMNet, self).__init__()
        self.frame_feature = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=64 * width,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: [B, C, T, W] → [B, T, C, W]
        x = x.permute(0, 2, 1, 3)
        B, T, C, W = x.shape
        x = x.reshape(B * T, C, W)         # → [B*T, C, W]
        f = self.frame_feature(x)         # → [B*T, 64, W]
        f = f.reshape(B, T, -1)           # → [B, T, 64*W]
        _, (h_n, _) = self.lstm(f)        # → [num_layers, B, hidden_dim]
        return self.fc(h_n[-1])           # Use the output of the last LSTM layer
