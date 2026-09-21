import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class FourierFeatures(nn.Module):
    """Geometric-frequency sin/cos lift of a few conditioning scalars.

    Two raw numbers concatenated onto a 1024-wide trunk are swamped by the state, and the
    network would have to resolve neighbouring constraints from a single unit of input each.
    The lift spreads every scalar over ``2 * num_frequencies`` channels, so inputs differing
    in the third decimal already separate at the top frequency.
    """

    def __init__(self, num_inputs: int, num_frequencies: int = 16,
                 max_frequency: float = 128.0):
        super().__init__()
        self.out_dim = num_inputs * num_frequencies * 2
        self.register_buffer(
            "freqs", torch.logspace(0.0, math.log10(max_frequency), num_frequencies))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x.unsqueeze(-1) * self.freqs
        return torch.cat((scaled.sin(), scaled.cos()), dim=-1).flatten(start_dim=1)
