import torch
import torch.nn as nn
import torch.nn.functional as F

from constrained_fm.src.consts import BOX_SOFTPLUS_SCALE
from constrained_fm.src.models.layers import ResBlock, SinusoidalPosEmb


class BoxConstrainedFM6D(nn.Module):
    """Vector field on ``R^6`` conditioned on a box over ``p1 = x[:, :3]``, normalized frame.

    ``box`` is ``(centre, log half-width)``. Face distances, box-relative coordinates and softplus
    barriers are derived from ``x`` inside ``forward`` so the exact divergence includes them.
    """

    def __init__(self, input_dim: int = 6, box_dim: int = 3, time_dim: int = 64,
                 hidden_dim: int = 1024, num_blocks: int = 3,
                 softplus_scale: float = BOX_SOFTPLUS_SCALE):
        super().__init__()
        self.input_dim = input_dim
        self.box_dim = box_dim
        self.softplus_scale = softplus_scale

        self.time_emb = SinusoidalPosEmb(time_dim)
        # conditioning (2k) + face distances (2k) + relative coordinates (k) + barriers (2k)
        feature_dim = 7 * box_dim
        self.input_proj = nn.Sequential(nn.Linear(input_dim + time_dim + feature_dim, hidden_dim),
                                        nn.SiLU())
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t_emb = self.time_emb(t.reshape(-1, 1).to(x.dtype).expand(x.shape[0], 1))
        box = box.reshape(-1, 2 * self.box_dim).expand(x.shape[0], -1)

        centre, half_width = box[:, :self.box_dim], box[:, self.box_dim:].exp()
        p1 = x[:, :self.box_dim]
        d_lo = p1 - (centre - half_width)
        d_hi = (centre + half_width) - p1
        relative = (p1 - centre) / half_width
        barriers = F.softplus(-self.softplus_scale * torch.cat([d_lo, d_hi], dim=1))

        h = torch.cat([x, t_emb, box, d_lo, d_hi, relative, barriers], dim=1)
        h = self.res_blocks(self.input_proj(h))
        return self.output_proj(h).reshape(*sz)
