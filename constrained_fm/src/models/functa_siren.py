# -*- coding: utf-8 -*-
"""Modulated SIREN model for CAVIA / Functa Meta-Learning."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class FiLMModulation(nn.Module):
    """Linear mapping from context vector z to layer-wise FiLM parameters.

    Maps z of shape (..., 256) to gamma and beta of shape (..., n_layers, hidden_dim).
    Initialized to zero so z = 0 corresponds to identity modulation.
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Single linear layer for smooth context projection
        self.proj = nn.Linear(latent_dim, n_layers * hidden_dim * 2)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # out shape: (..., n_layers * hidden_dim * 2)
        out = self.proj(z)
        new_shape = out.shape[:-1] + (self.n_layers, self.hidden_dim * 2)
        out = out.view(*new_shape)
        gamma, beta = out.chunk(2, dim=-1)  # each (..., n_layers, hidden_dim)
        return gamma, beta


class ModulatedSIREN(nn.Module):
    """SIREN with FiLM modulation supporting native 2D and 3D batched inputs.

    Input shapes:
        - x: (M, 2) or (B, M, 2)
        - z: (256,) or (B, 256)
    Output shape:
        - p: (M, 1) or (B, M, 1)
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256, n_layers: int = 4, w0: float = 30.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.w0 = w0

        self.input_linear = nn.Linear(2, hidden_dim)
        self.hidden_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_layers)
        ])
        self.output_linear = nn.Linear(hidden_dim, 1)
        self.film = FiLMModulation(latent_dim, hidden_dim, n_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Sitzmann SIREN initialization."""
        bound_in = 1.0 / 2.0
        nn.init.uniform_(self.input_linear.weight, -bound_in, bound_in)
        nn.init.constant_(self.input_linear.bias, 0.0)

        bound_hidden = math.sqrt(6.0 / self.hidden_dim) / self.w0
        for lin in self.hidden_linears:
            nn.init.uniform_(lin.weight, -bound_hidden, bound_hidden)

        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.constant_(self.output_linear.bias, 0.0)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass supporting both unbatched (M, 2) and batched (B, M, 2) inputs.

        Args:
            x: (M, 2) or (B, M, 2) spatial coordinates.
            z: (256,) or (B, 256) context vectors.

        Returns:
            (M, 1) or (B, M, 1) predictions in [0, 1].
        """
        is_batched = x.dim() == 3  # True if (B, M, 2)

        gamma, beta = self.film(z)  # (n_layers, hidden) or (B, n_layers, hidden)

        h = self.input_linear(x)    # (M, hidden) or (B, M, hidden)
        h = torch.sin(self.w0 * h)

        for i, lin in enumerate(self.hidden_linears):
            h = lin(h)
            if is_batched:
                # gamma[:, i]: (B, hidden) -> unsqueeze to (B, 1, hidden) for broadcasting over M points
                g = gamma[:, i].unsqueeze(1)
                b = beta[:, i].unsqueeze(1)
            else:
                g = gamma[i]
                b = beta[i]

            h = torch.sin(self.w0 * ((1.0 + g) * h + b))

        out = self.output_linear(h)
        out = torch.sigmoid(out)
        return out


def build_modulated_siren(latent_dim: int = 256, hidden_dim: int = 256, n_layers: int = 4, w0: float = 30.0) -> ModulatedSIREN:
    return ModulatedSIREN(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, w0=w0)
