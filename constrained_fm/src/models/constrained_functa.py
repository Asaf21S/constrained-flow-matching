import math
import torch
import torch.nn as nn


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
        # GroupNorm requires the number of groups to divide the hidden_dim
        # 32 is standard, but if hidden_dim is small (e.g. 128), 8 or 16 is safer
        self.norm1 = nn.GroupNorm(min(32, hidden_dim // 4), hidden_dim)
        self.act1 = nn.SiLU()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.GroupNorm(min(32, hidden_dim // 4), hidden_dim)
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


class ConstrainedFlowMatcher(nn.Module):
    """
    The full vector field predictor: v_t(x) = f(x_t, t, z)
    """

    def __init__(self, spatial_dim: int = 2, latent_dim: int = 256,
                 time_emb_dim: int = 128, hidden_dim: int = 256,
                 num_blocks: int = 4):
        super().__init__()

        # Time Embedding
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)

        # Master Condition Combiner: projects (t_emb + z) -> cond_dim
        cond_dim = hidden_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_emb_dim + latent_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # Spatial input projection
        self.input_proj = nn.Linear(spatial_dim, hidden_dim)

        # ResNet AdaGN Blocks
        self.blocks = nn.ModuleList([
            AdaGNBlock(hidden_dim, cond_dim) for _ in range(num_blocks)
        ])

        # Final output projection back to spatial dimensions (v_t)
        self.final_norm = nn.GroupNorm(min(32, hidden_dim // 4), hidden_dim)
        self.final_act = nn.SiLU()
        self.final_proj = nn.Linear(hidden_dim, spatial_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2)   - 2D point cloud
        t: (B,)     - ODE time steps
        z: (B, 256) - Functa latent constraints
        Returns predicted vector field (B, 2)
        """
        # 1. Embed time and combine with Functa constraint
        t_emb = self.time_embed(t)
        c = self.cond_mlp(torch.cat([t_emb, z], dim=-1))

        # 2. Lift spatial coordinates into hidden dimension
        h = self.input_proj(x)

        # 3. Apply AdaGN ResNet blocks
        for block in self.blocks:
            h = block(h, c)

        # 4. Final projection to output vector field
        h = self.final_act(self.final_norm(h))
        v_pred = self.final_proj(h)

        return v_pred
