import torch
import torch.nn as nn

from constrained_fm.src.models.base_fm import BaseFM
from constrained_fm.src.models.layers import SinusoidalPosEmb, ResBlock
from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly


class PolynomialConstrainedFM(BaseFM):
    def __init__(self, input_dim=2, time_dim=64, degree=3, hidden_dim=1024, scale_factor=4.0):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.degree = degree
        self.hidden_dim = hidden_dim
        self.scale_factor = scale_factor

        self.time_emb = SinusoidalPosEmb(time_dim)

        self.coeff_dim = (degree + 1) ** 2
        self.cond_dim = self.coeff_dim + 1  # adding P(x)
        total_in_dim = input_dim + time_dim + self.cond_dim

        self.input_proj = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.SiLU()
        )

        self.res_blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim)
        )

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, coeffs):
        sz = x.size()

        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, 1).float()
        t_expanded = t.expand(x.shape[0], 1)
        t_emb = self.time_emb(t_expanded)

        coeffs_flat = coeffs
        coeffs_matrix = coeffs.view(x.shape[0], self.degree + 1, self.degree + 1)
        xt_pow, yt_pow = compute_poly_features(x, degree=self.degree, scale=self.scale_factor)
        p_val = evaluate_poly(xt_pow, yt_pow, coeffs_matrix)

        h = torch.cat([x, t_emb, coeffs_flat, p_val], dim=1)

        h = self.input_proj(h)
        h = self.res_blocks(h)
        output = self.output_proj(h)

        return output.reshape(*sz)
