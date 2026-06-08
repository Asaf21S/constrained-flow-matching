import torch
import torch.nn as nn

from constrained_fm.src.models.layers import SinusoidalPosEmb, ResBlock


class BboxConstrainedFM(nn.Module):
    def __init__(self, input_dim=2, time_dim=64, cond_dim=4, hidden_dim=1024):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        self.time_emb = SinusoidalPosEmb(time_dim)

        total_in_dim = input_dim + cond_dim + time_dim

        self.input_proj = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.SiLU()
        )

        self.res_blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim)
        )

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, bounds):
        sz = x.size()

        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, 1).float()
        t_expanded = t.expand(x.shape[0], 1)
        t_emb = self.time_emb(t_expanded)

        bounds = bounds.reshape(-1, self.cond_dim)

        d_left = (x[:, 0] - bounds[:, 0]).unsqueeze(1)
        d_bottom = (x[:, 1] - bounds[:, 1]).unsqueeze(1)
        d_right = (bounds[:, 2] - x[:, 0]).unsqueeze(1)
        d_top = (bounds[:, 3] - x[:, 1]).unsqueeze(1)

        h = torch.cat([x, t_emb, d_left, d_bottom, d_right, d_top], dim=1)
        h = self.input_proj(h)
        h = self.res_blocks(h)
        output = self.output_proj(h)

        return output.reshape(*sz)
