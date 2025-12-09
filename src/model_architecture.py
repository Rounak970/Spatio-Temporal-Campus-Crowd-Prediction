"""
model_architecture.py
------------------------------------
Defines the Spatio-Temporal Crowd Prediction model used in the project.
Model = 3D CNN + ConvLSTM + WiFi Embedding + 3 Prediction Heads:
    1) Count Regression
    2) Density Map (32×32)
    3) Motion Map (2 × 16 × 16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# 3D CNN Feature Extractor
# --------------------------------------------------------
class CNN3DEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool3d((1, 2, 2)),  # → (B, 32, T, 32, 32)

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool3d((1, 2, 2)),  # → (B, 64, T, 16, 16)
        )

    def forward(self, x):
        return self.encoder(x)


# --------------------------------------------------------
# ConvLSTM Cell
# --------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# --------------------------------------------------------
# ConvLSTM Wrapper
# --------------------------------------------------------
class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels)

    def forward(self, x):  # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=x.device)
        c = torch.zeros(B, self.cell.hidden_channels, H, W, device=x.device)

        for t in range(T):
            h, c = self.cell(x[:, :, t], h, c)

        return h  # final hidden state


# --------------------------------------------------------
# Full Crowd Prediction Model
# --------------------------------------------------------
class CrowdPredictor(nn.Module):
    def __init__(self, wifi_dim=50):
        super().__init__()

        self.cnn3d = CNN3DEncoder()

        # ConvLSTM on top of 3D-CNN features
        self.convlstm = ConvLSTM(64, 64)

        # WiFi embedding branch
        self.wifi_fc = nn.Sequential(
            nn.Linear(wifi_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # Prediction heads
        self.count_head = nn.Linear(64 + 64, 1)
        self.density_head = nn.Linear(64 + 64, 32 * 32)
        self.motion_head = nn.Linear(64 + 64, 2 * 16 * 16)

    def forward(self, video, wifi):
        """
        video: (B, 1, 10, 64, 64)
        wifi:  (B, 10)
        """
        x = self.cnn3d(video)      # → (B, 64, T, 16, 16)
        x = self.convlstm(x)       # → (B, 64, 16, 16)
        x = torch.mean(x, dim=[2, 3])  # GAP → (B, 64)

        wifi_feat = self.wifi_fc(wifi)  # (B, 64)

        fused = torch.cat([x, wifi_feat], dim=1)

        count_out = self.count_head(fused)
        density_out = self.density_head(fused).view(-1, 32, 32)
        motion_out = self.motion_head(fused).view(-1, 2, 16, 16)

        return count_out, density_out, motion_out
