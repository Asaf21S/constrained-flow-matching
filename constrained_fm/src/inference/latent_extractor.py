# -*- coding: utf-8 -*-
"""Functa extraction via CAVIA fast adaptation for unseen polynomial constraints."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def extract_latents_batched(
        siren: nn.Module,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        latent_dim: int | None = None,
        lr: float = 6.25e-4,
        steps: int = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapts a batch of context vectors to unseen shapes via pure-SGD CAVIA adaptation.

    Replicates the inner loop of scripts/train_functa.py: MSE regression against
    tanh(P(x, y)) targets, plain SGD from a zero init, no L2 penalty.

    The loss is averaged over points and summed over shapes, so each z_i receives the
    gradient of its own mean MSE and the step size does not depend on how many shapes
    happen to share a call. Meta-training reduced with a mean over (batch, points) at
    batch 16 and lr 1e-2, so the equivalent per-shape step - and the default here - is
    1e-2 / 16. CAVIA only meta-learns an initialization that is optimal after exactly
    `steps` at the step size it trained with, so this scale must match.

    Args:
        siren: trained ModulatedSIREN, evaluated in inference mode.
        X_batch: (B, M, 2) coordinates normalized to the SIREN's canonical [-1, 1] domain.
        Y_batch: (B, M) regression targets tanh(P(x, y)) in (-1, 1).
        latent_dim: context vector size; defaults to siren.latent_dim.
        lr: per-shape SGD step size for the inner loop.
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
        loss = ((preds - Y_batch) ** 2).mean(dim=1).sum()
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
        lr: float = 6.25e-4,
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


def refine_latents(
        siren: nn.Module,
        z_init: torch.Tensor,
        query_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        steps: int,
        lr: float,
        anchor_weight: float = 0.0,
) -> tuple[torch.Tensor, list[float]]:
    """Test-time Adam on the latents alone, the SIREN weights stay frozen.

    Minimises ``sum_b [ mean_m (f(x_bm, z_b) - y_bm)^2 + lambda ||z_b - z_b^0||^2 ]`` with a
    fresh query batch every step; the anchor keeps ``z`` near the CAVIA solution the flow
    matcher was trained on.

    Args:
        siren: frozen ModulatedSIREN.
        z_init: (B, latent_dim) starting latents, also the anchor ``z^0``.
        query_fn: returns ``(X, Y)`` of shapes (B, M, 2) normalised and (B, M) targets.
        steps, lr: Adam budget.
        anchor_weight: ``lambda``; 0 disables the anchor.

    Returns:
        z: (B, latent_dim) refined latents.
        history: per-step mean-over-shapes MSE.
    """
    anchor = z_init.detach()
    z = anchor.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    history = []
    for _ in range(steps):
        X, Y = query_fn()
        mse = ((siren(X, z).squeeze(-1) - Y) ** 2).mean(dim=1)
        loss = mse.sum() + anchor_weight * ((z - anchor) ** 2).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(mse.mean()))
    return z.detach(), history


__all__ = ["extract_latent", "extract_latents_batched", "refine_latents"]
