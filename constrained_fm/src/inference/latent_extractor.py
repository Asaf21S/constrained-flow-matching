# -*- coding: utf-8 -*-
"""Phase 3 – Functa extraction via CAVIA / Latent-Modulated Meta-Learning.

The CAVIA meta-learning architecture (Phase 2) learns a universal modulated SIREN.
When a *new* unseen constraint (e.g., a novel polynomial shape) is presented,
we must *extract* its Functa vector without retraining the SIREN's base weights.

This module provides helpers to perform fast adaptation (inference) on novel
shapes. Because the SIREN is meta-learned to adapt quickly, we initialize a
zeroed context vector (which corresponds to an identity modulation) and run a
fast optimization loop using Adam and Cosine Annealing.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def extract_latents_batched(
        siren: nn.Module,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        latent_dim: int = 256,
        lr: float = 0.01,
        steps: int = 300,
        lambda_z: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimise a batch of context vectors (Functas) for unseen shapes.

    Args:
        siren: Trained ``ModulatedSIREN`` (weights frozen).
        X_batch: Coordinate tensor of shape ``(B, M, 2)``.
        Y_batch: Binary label tensor of shape ``(B, M)`` (0/1).
        latent_dim: Dimensionality of the context vector.
        lr: Initial learning rate for Adam.
        steps: Number of optimisation iterations.
        lambda_z: L2 regularisation penalty on the context vector.

    Returns:
        A tuple ``(z_opt, final_losses)`` where ``z_opt`` has shape ``(B, latent_dim)``
        and ``final_losses`` is a tensor of shape ``(B,)`` containing the final loss.
    """
    device = X_batch.device
    B = X_batch.shape[0]

    # Freeze SIREN parameters - we only optimize the context vector `z`
    siren.eval()
    for p in siren.parameters():
        p.requires_grad = False

    # Initialize latent vectors to zero.
    # In our FiLM setup, z=0 corresponds to an identity pass through the base network.
    z_batch = torch.zeros(B, latent_dim, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([z_batch], lr=lr)

    # Cosine Annealing perfectly settles the vector into the local minimum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    loss_fn = nn.BCELoss(reduction='none')

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        # Native 3D batched forward pass (no vmap required)
        preds = siren(X_batch, z_batch).squeeze(-1)

        bce_losses = loss_fn(preds, Y_batch).mean(dim=1)
        l2_penalties = lambda_z * (z_batch ** 2).mean(dim=1)

        total_losses = bce_losses + l2_penalties
        loss = total_losses.mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

    # Final evaluation for logging/returns
    with torch.no_grad():
        preds = siren(X_batch, z_batch).squeeze(-1)
        bce_losses = loss_fn(preds, Y_batch).mean(dim=1)
        l2_penalties = lambda_z * (z_batch ** 2).mean(dim=1)
        final_losses = bce_losses + l2_penalties

    return z_batch.detach(), final_losses.detach()


def extract_latent(
        siren: nn.Module,
        X: torch.Tensor,
        Y: torch.Tensor,
        latent_dim: int = 256,
        lr: float = 0.01,
        steps: int = 300,
        lambda_z: float = 1e-4,
) -> Tuple[torch.Tensor, float]:
    """Convenience wrapper for extracting a single Functa.

    Args:
        siren: Trained ``ModulatedSIREN`` (weights frozen).
        X: Coordinate tensor of shape ``(M, 2)``.
        Y: Binary label tensor of shape ``(M,)``.
        latent_dim: Dimensionality of the context vector.
        lr: Initial learning rate for Adam.
        steps: Number of optimisation iterations.
        lambda_z: L2 regularisation penalty on the context vector.

    Returns:
        ``(z_opt, final_loss)`` with ``z_opt`` shape ``(1, latent_dim)`` and a float loss.
    """
    # Temporarily add a batch dimension for the native 3D forward pass
    X_b = X.unsqueeze(0)
    Y_b = Y.unsqueeze(0)

    z_batch, losses = extract_latents_batched(
        siren=siren,
        X_batch=X_b,
        Y_batch=Y_b,
        latent_dim=latent_dim,
        lr=lr,
        steps=steps,
        lambda_z=lambda_z
    )

    return z_batch, losses[0].item()


__all__ = ["extract_latent", "extract_latents_batched"]