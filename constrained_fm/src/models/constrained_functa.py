import math
from typing import Optional

import torch
import torch.nn as nn

from constrained_fm.src.consts import PLANE_SCALE
from constrained_fm.src.models.base_fm import BaseFM


def _safe_num_groups(num_channels: int, max_groups: int = 32) -> int:
    """Largest group count <= max_groups that evenly divides num_channels."""
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Embeds the continuous ODE time step t into a high-dimensional vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B,) or (B, 1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t * embeddings.unsqueeze(0)

        # Concat sin and cos
        emb = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return emb


class AdaGNBlock(nn.Module):
    """A ResNet block modulated by the master condition vector c via AdaGN."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        num_groups = _safe_num_groups(hidden_dim)
        self.norm1 = nn.GroupNorm(num_groups, hidden_dim)
        self.act1 = nn.SiLU()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.GroupNorm(num_groups, hidden_dim)
        self.act2 = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # The modulation projection
        self.film_proj = nn.Linear(cond_dim, hidden_dim * 2)

        # Initialize the projection so it starts as the identity mapping
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        # Initialize the final layer to output zero, matching standard ResNet initialization
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # First linear block
        h = self.act1(self.norm1(x))
        h = self.linear1(h)

        # Modulation phase
        h = self.norm2(h)
        gamma, beta = self.film_proj(c).chunk(2, dim=-1)

        # (1 + gamma) ensures identity scaling at initialization
        h = h * (1 + gamma) + beta
        h = self.act2(h)

        h = self.linear2(h)

        # Residual connection
        return x + h


class ConstrainedFlowMatcher(BaseFM):
    """
    The full vector field predictor: v_t(x) = f(x_t, t, z)

    latent_dim must match the upstream ModulatedSIREN's latent_dim (512).
    If a frozen `siren` is provided, its pointwise prediction SIREN(x, z) is fed in
    as an extra conditioning feature, giving the network direct boundary-aware
    information per query point instead of relying solely on the global z.
    """

    def __init__(self, siren: Optional[nn.Module] = None, spatial_dim: int = 2, latent_dim: int = 512,
                 time_emb_dim: int = 128, hidden_dim: int = 512, num_blocks: int = 4,
                 plane_scale: float = PLANE_SCALE):
        super().__init__()

        self.plane_scale = plane_scale
        self.use_siren_feature = siren is not None

        if self.use_siren_feature:
            self.siren = siren
            for p in self.siren.parameters():
                p.requires_grad_(False)
            self.siren.eval()

        # Time Embedding
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)

        # Master Condition Combiner: projects (t_emb + z) -> cond_dim
        cond_dim = hidden_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_emb_dim + latent_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # Spatial input projection; +1 extra channel for the SIREN's pointwise feature
        input_dim = spatial_dim + 1 if self.use_siren_feature else spatial_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # ResNet AdaGN Blocks
        self.blocks = nn.ModuleList([
            AdaGNBlock(hidden_dim, cond_dim) for _ in range(num_blocks)
        ])

        # Final output projection back to spatial dimensions (v_t)
        num_groups = _safe_num_groups(hidden_dim)
        self.final_norm = nn.GroupNorm(num_groups, hidden_dim)
        self.final_act = nn.SiLU()
        self.final_proj = nn.Linear(hidden_dim, spatial_dim)

    def train(self, mode: bool = True) -> "ConstrainedFlowMatcher":
        super().train(mode)
        if self.use_siren_feature:
            self.siren.eval()  # frozen dependency always stays in eval mode
        return self

    def trainable_parameters(self):
        """Yields parameters excluding the frozen SIREN, for constructing the optimizer."""
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2)          - 2D point cloud
        t: (B,) or scalar   - ODE time steps (the ODE solver passes a 0-dim tensor)
        z: (B, latent_dim) - Functa latent constraints
        Returns predicted vector field (B, 2)
        """
        # 1. Broadcast t to match x's batch size, then embed and combine with z
        t = t.reshape(-1, 1).float().expand(x.shape[0], 1)
        t_emb = self.time_embed(t)
        c = self.cond_mlp(torch.cat([t_emb, z], dim=-1))

        # 2. Optionally query the frozen SIREN at each point for a direct boundary feature
        if self.use_siren_feature:
            with torch.no_grad():
                x_normalized = (x / self.plane_scale).unsqueeze(1)  # (B, 1, 2) for per-example z
                siren_val = self.siren(x_normalized, z).squeeze(1)  # (B, 1)
            x = torch.cat([x, siren_val], dim=-1)

        # 3. Lift spatial coordinates into hidden dimension
        h = self.input_proj(x)

        # 4. Apply AdaGN ResNet blocks
        for block in self.blocks:
            h = block(h, c)

        # 5. Final projection to output vector field
        h = self.final_act(self.final_norm(h))
        v_pred = self.final_proj(h)

        return v_pred
