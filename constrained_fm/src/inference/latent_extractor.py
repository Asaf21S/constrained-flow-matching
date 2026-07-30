# -*- coding: utf-8 -*-
"""Functa extraction via CAVIA fast adaptation for unseen polynomial constraints."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_latents_batched(
        siren: nn.Module,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        latent_dim: int | None = None,
        lr: float = 1e-2,
        steps: int = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapts a batch of context vectors to unseen shapes via pure-SGD CAVIA adaptation.

    Replicates the validation-time inner loop of scripts/train_functa.py: mean-reduced
    MSE regression against tanh(P(x, y)) targets, plain SGD, no L2 penalty.

    Args:
        siren: trained ModulatedSIREN, evaluated in inference mode.
        X_batch: (B, M, 2) coordinates normalized to the SIREN's canonical [-1, 1] domain.
        Y_batch: (B, M) regression targets tanh(P(x, y)) in (-1, 1).
        latent_dim: context vector size; defaults to siren.latent_dim.
        lr: SGD step size for the inner loop.
        steps: number of SGD adaptation steps.

    Returns:
        z_opt: (B, latent_dim) adapted context vectors.
        per_shape_mse: (B,) final MSE loss per shape.
    """
    device = X_batch.device
    batch_size = X_batch.shape[0]
    latent_dim = latent_dim or siren.latent_dim

    siren.eval()
    for p in siren.parameters():
        p.requires_grad = False

    z = torch.zeros(batch_size, latent_dim, device=device, requires_grad=True)

    for _ in range(steps):
        preds = siren(X_batch, z).squeeze(-1)
        loss = F.mse_loss(preds, Y_batch)
        grad_z = torch.autograd.grad(loss, z)[0]
        z = z - lr * grad_z

    with torch.no_grad():
        preds = siren(X_batch, z).squeeze(-1)
        per_shape_mse = ((preds - Y_batch) ** 2).mean(dim=1)

    return z.detach(), per_shape_mse.detach()


def extract_latent(
        siren: nn.Module,
        X: torch.Tensor,
        Y: torch.Tensor,
        latent_dim: int | None = None,
        lr: float = 1e-2,
        steps: int = 15,
) -> tuple[torch.Tensor, float]:
    """Single-shape convenience wrapper around extract_latents_batched.

    Args:
        siren: trained ModulatedSIREN, evaluated in inference mode.
        X: (M, 2) coordinates normalized to the SIREN's canonical [-1, 1] domain.
        Y: (M,) regression targets tanh(P(x, y)) in (-1, 1).
        latent_dim: context vector size; defaults to siren.latent_dim.
        lr: SGD step size for the inner loop.
        steps: number of SGD adaptation steps.

    Returns:
        z_opt: (1, latent_dim) adapted context vector.
        final_mse: scalar final MSE loss.
    """
    z_opt, per_shape_mse = extract_latents_batched(
        siren=siren, X_batch=X.unsqueeze(0), Y_batch=Y.unsqueeze(0),
        latent_dim=latent_dim, lr=lr, steps=steps,
    )
    return z_opt, per_shape_mse[0].item()


__all__ = ["extract_latent", "extract_latents_batched"]
