# -*- coding: utf-8 -*-
"""Modulated SIREN model for Functa.

The model implements a SIREN (Sinusoidal Representation Network) whose
layers are modulated by a 256‑dimensional latent vector ``z`` via FiLM
(scale ``γ`` and shift ``β``). The modulation network is a small MLP that
maps ``z`` to the per‑layer ``γ``/``β`` parameters.

*Input*
    - ``x``: tensor of shape ``(..., 2)`` – 2‑D coordinates.
    - ``z``: tensor of shape ``(..., 256)`` – Functa latent vector.
*Output*
    - ``p``: probability inside the shape, ``(..., 1)`` after ``sigmoid``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FiLMModulation(nn.Module):
    """Maps a latent vector ``z`` to FiLM parameters for each SIREN layer.

    Returns ``γ`` and ``β`` tensors of shape ``(n_layers, hidden)`` for a single
    latent vector (vmap‑compatible). At initialization the modulation is the
    identity (``γ = 0``, ``β = 0``) so the base SIREN functions unchanged.
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # Two‑layer MLP to produce FiLM parameters.
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_layers * hidden_dim * 2),
        )
        # Initialise final linear layer to zero so modulation starts as identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute FiLM parameters robustly.

        Supports both a single latent vector ``z`` of shape ``(latent_dim,)`` and a
        batched tensor ``z`` of shape ``(B, latent_dim)``. The final dimensions are
        reshaped to ``(..., n_layers, hidden_dim * 2)`` while preserving any leading
        batch dimensions.
        """
        out = self.net(z)  # (..., n_layers * hidden_dim * 2)
        # Append ``n_layers`` and ``hidden_dim*2`` to the existing shape.
        new_shape = out.shape[:-1] + (self.n_layers, self.hidden_dim * 2)
        out = out.view(*new_shape)
        gamma, beta = out.chunk(2, dim=-1)  # each (..., n_layers, hidden_dim)
        return gamma, beta


class ModulatedSIREN(nn.Module):
    """SIREN with FiLM modulation.

    The architecture is as follows:
    * Input dimension: 2 (x, y)
    * Latent dimension: 256 (z)
    * Hidden dimension: configurable (default 256)
    * Number of hidden sinusoidal layers: configurable (default 4)
    * Final output: scalar passed through ``sigmoid``.
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256, n_layers: int = 4, w0: float = 30.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.w0 = w0

        # First linear layer maps 2‑D coordinates to hidden.
        self.input_linear = nn.Linear(2, hidden_dim)
        # Hidden sinusoidal layers (no bias – bias is handled by FiLM).
        self.hidden_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_layers)
        ])
        # Final linear to scalar.
        self.output_linear = nn.Linear(hidden_dim, 1)
        # Modulation network.
        self.film = FiLMModulation(latent_dim, hidden_dim, n_layers)
        # Initialise weights as suggested for SIREN.
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Sitzmann SIREN initialization.

        * Input layer: Uniform[-1/in_features, 1/in_features] (in_features=2).
        * Hidden layers: Uniform[-sqrt(6/hidden_dim)/w0, sqrt(6/hidden_dim)/w0].
        The output layer uses a standard Xavier init.
        """
        # Input layer bound = 1 / in_features (2).
        bound_in = 1.0 / 2.0
        nn.init.uniform_(self.input_linear.weight, -bound_in, bound_in)
        nn.init.constant_(self.input_linear.bias, 0.0)

        # Hidden layers bound = sqrt(6 / hidden_dim) / w0.
        bound_hidden = math.sqrt(6.0 / self.hidden_dim) / self.w0
        for lin in self.hidden_linears:
            nn.init.uniform_(lin.weight, -bound_hidden, bound_hidden)

        # Output layer uses Xavier uniform.
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.constant_(self.output_linear.bias, 0.0)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass (vmap‑compatible).

        Args:
            x: ``(M, 2)`` coordinate tensor for a single shape.
            z: ``(latent_dim,)`` Functa latent vector.
        Returns:
            ``(M, 1)`` probability after ``sigmoid``.
        """
        # Compute FiLM parameters for the single latent vector.
        gamma, beta = self.film(z)  # (n_layers, hidden)

        # Input linear + SIREN activation.
        h = self.input_linear(x)  # (M, hidden)
        h = torch.sin(self.w0 * h)

        # Apply hidden layers with FiLM (identity at init).
        for i, lin in enumerate(self.hidden_linears):
            g = gamma[i]  # (hidden,)
            b = beta[i]   # (hidden,)
            h = lin(h)
            # (1 + g) ensures identity scaling initially.
            h = torch.sin(self.w0 * ((1 + g) * h + b))

        # Output linear → sigmoid.
        out = self.output_linear(h)
        out = torch.sigmoid(out)
        return out


# Convenience factory used by training scripts.
def build_modulated_siren(latent_dim: int = 256, hidden_dim: int = 256, n_layers: int = 4, w0: float = 30.0) -> ModulatedSIREN:
    """Create a ``ModulatedSIREN`` with the default hyper‑parameters.
    The function mirrors the description in the plan and makes model
    creation explicit for downstream code.
    """
    return ModulatedSIREN(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, w0=w0)
