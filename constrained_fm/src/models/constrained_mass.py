# -*- coding: utf-8 -*-
"""Explicit constraint amortization for the kinematics6d mass shells.

The conditioning here is only two numbers, ``(M_target / s, log(epsilon / s))``, against a
six-dimensional state and a 1024-wide trunk. Concatenating them raw -- the recipe that works
for the 16 polynomial coefficients -- would let the state dominate, so both are lifted into
Fourier features, together with the pointwise constraint value that tells the network where
the current point sits relative to the shell it is being asked to hit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from constrained_fm.src.consts import KIN_MASS_FLOOR
from constrained_fm.src.models.base_fm import BaseFM
from constrained_fm.src.models.layers import FourierFeatures, ResBlock, SinusoidalPosEmb

PARAM_DIM = 2


class MassWindowConstrainedFM(BaseFM):
    r"""Velocity field conditioned on a mass window ``|M(x) - M_\mathrm{target}| \le \epsilon``.

    The frame and the mass scale are carried as buffers so the pointwise constraint feature is
    computed inside :meth:`forward`. It cannot be passed in from outside: the ODE solver only
    forwards the per-shape conditioning, so a feature supplied by the training loop would be
    present during training and silently absent during sampling.

    Args:
        input_dim: dimension of the state.
        time_dim: width of the sinusoidal time embedding.
        hidden_dim: trunk width.
        num_blocks: number of residual blocks.
        num_frequencies: frequencies per conditioning scalar in the Fourier lift.
        frame_mean: per-axis mean of the normalised frame the model is trained in.
        frame_std: per-axis standard deviation of that frame.
        mass_scale: ``s = \sqrt{E[M^2]}``, the unit the window and ``C`` are reported in.
    """

    def __init__(self, input_dim: int = 6, time_dim: int = 128, hidden_dim: int = 1024,
                 num_blocks: int = 4, num_frequencies: int = 16,
                 frame_mean: torch.Tensor | None = None,
                 frame_std: torch.Tensor | None = None, mass_scale: float = 1.0):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.register_buffer("frame_mean", torch.zeros(input_dim)
                             if frame_mean is None else frame_mean.detach().clone())
        self.register_buffer("frame_std", torch.ones(input_dim)
                             if frame_std is None else frame_std.detach().clone())
        self.register_buffer("mass_scale", torch.tensor(float(mass_scale)))

        self.time_emb = SinusoidalPosEmb(time_dim)
        self.cond_emb = FourierFeatures(PARAM_DIM + 1, num_frequencies=num_frequencies)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + time_dim + self.cond_emb.out_dim, hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    @staticmethod
    def invariant_mass(x: torch.Tensor) -> torch.Tensor:
        r"""``M = \sqrt{2 (E_1 E_2 - \vec p_1 \cdot \vec p_2)}`` for two massless particles."""
        pairs = x.view(-1, 2, 3)
        energy = pairs.norm(dim=-1)
        dot = (pairs[:, 0] * pairs[:, 1]).sum(dim=-1)
        return (2.0 * (energy[:, 0] * energy[:, 1] - dot)).clamp_min(KIN_MASS_FLOOR).sqrt()

    def constraint_feature(self, x_physical: torch.Tensor,
                           params: torch.Tensor) -> torch.Tensor:
        r"""``\tanh C(x)`` for the window named by ``params``.

        Squashed because ``C`` is unbounded above: early in the trajectory the state is still
        close to prior noise and ``C`` runs to tens, which would alias through the Fourier
        lift and drown the region near the boundary that actually decides the velocity.
        """
        centre = params[:, 0] * self.mass_scale
        epsilon = torch.exp(params[:, 1]) * self.mass_scale
        residual = (self.invariant_mass(x_physical) - centre).abs() - epsilon
        return torch.tanh(residual / self.mass_scale).unsqueeze(-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        size = x.size()
        x = x.reshape(-1, self.input_dim)
        params = params.reshape(x.shape[0], PARAM_DIM)

        t_emb = self.time_emb(t.reshape(-1, 1).float().expand(x.shape[0], 1))
        feature = self.constraint_feature(x * self.frame_std + self.frame_mean, params)
        cond = self.cond_emb(torch.cat([params, feature], dim=1))

        h = self.input_proj(torch.cat([x, t_emb, cond], dim=1))
        h = self.res_blocks(h)
        return self.output_proj(h).reshape(*size)
