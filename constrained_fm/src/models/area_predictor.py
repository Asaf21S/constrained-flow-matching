import torch
import torch.nn as nn


class AreaPredictor(nn.Module):
    """Neural network that predicts the mass (probability area) of a bounding box.

    The model expects input ``bounds`` of shape ``(B, 4)`` with entries
    ``[x_min, y_min, x_max, y_max]`` and outputs a normalized mass in ``[0, 1]``.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, bounds: torch.Tensor) -> torch.Tensor:
        return self.net(bounds)
