import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.out_scale = 30.0
        self.out_bias = -50.0
        self.net = nn.Sequential(
            nn.Linear(n_channels, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = out.squeeze(-1)
        out = out * self.out_scale + self.out_bias
        return out
