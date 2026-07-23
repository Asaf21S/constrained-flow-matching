# -*- coding: utf-8 -*-
"""Phase 3 – Functa extraction via latent optimisation.

The auto‑decoder (Phase 2) learns a universal modulated SIREN and an embedding
table ``embed`` that maps each training shape index to a latent vector ``z``.
When a *new* unseen constraint (e.g. a novel shape) is presented we must
*extract* its Functa without retraining the SIREN.  The standard DeepSDF
procedure is to keep the SIREN weights frozen and optimise a fresh latent
vector ``z`` so that the network’s predictions match the binary mask of the
shape.

This module provides a single public helper ``extract_latent`` that performs
the optimisation for one shape (or a batch of shapes via ``torch.func.vmap``
if desired).  The implementation is as follows:
1. Initialise ``z`` – either randomly or from the mean of the learned
   embedding table (the latter usually speeds up convergence).
2. Freeze the SIREN parameters.
3. Run gradient descent (Adam) on ``z`` using binary cross‑entropy loss
   against the sampled points ``X`` and their inside/outside labels ``Y``.
4. Return the optimized ``z`` (as a tensor on the same device) and the
   final loss value.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

# Import the modulated SIREN definition.
from constrained_fm.src.models.functa_siren import ModulatedSIREN
# Import vmap for batched operations
from torch.func import vmap


def _prepare_initial_z(
    embed: Optional[nn.Embedding],
    device: torch.device,
    latent_dim: int = 256,
    batch_size: int = 1,
) -> torch.Tensor:
    """Return an initial latent vector (or batch of vectors).

    If an ``embed`` table is provided we use its mean as a sensible prior and
    repeat it for the requested ``batch_size``; otherwise we initialise from a
    small isotropic normal distribution.
    """
    if embed is not None:
        # Embedding weight shape: (N, latent_dim)
        mean_z = embed.weight.mean(dim=0, keepdim=True)  # (1, latent_dim)
        return mean_z.repeat(batch_size, 1).clone().detach().to(device)
    # Random normal init – small std to keep early SIREN activations stable.
    return torch.randn(batch_size, latent_dim, device=device) * 0.01


def extract_latent(
    siren: ModulatedSIREN,
    X: torch.Tensor,
    Y: torch.Tensor,
    embed: Optional[nn.Embedding] = None,
    latent_dim: int = 256,
    lr: float = 1e-2,
    steps: int = 500,
    lambda_z: float = 1e-4,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, float]:
    """Optimise a latent vector ``z`` for an unseen shape.

    Args:
        siren: Trained ``ModulatedSIREN`` (weights frozen).
        X: Coordinate tensor of shape ``(M, 2)`` – points sampled from the
           constraint domain.
        Y: Binary label tensor of shape ``(M,)`` (0/1) matching ``X``.
        embed: Optional embedding table from the auto‑decoder.  If supplied the
               mean embedding is used for initialisation, which speeds up
               convergence.
        latent_dim: Dimensionality of the latent space (must match the siren).
        lr: Learning rate for the Adam optimiser.
        steps: Number of optimisation iterations.
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).  If ``None``
                the device of ``X`` is used.

    Returns:
        A tuple ``(z_opt, final_loss)`` where ``z_opt`` has shape ``(1, latent_dim)``
        and ``final_loss`` is the BCE loss after the last optimisation step.
    """
    # Ensure inputs are on the correct device.
    if device is None:
        device = X.device
    else:
        device = torch.device(device)
    X = X.to(device)
    Y = Y.to(device, torch.float32)

    # Freeze SIREN parameters – we only optimise ``z``.
    siren.eval()
    for p in siren.parameters():
        p.requires_grad = False

    # Initialise latent vector.
    z = _prepare_initial_z(embed, device, latent_dim)
    # ``z`` must require gradient for optimisation.
    z.requires_grad_(True)

    optimizer = torch.optim.Adam([z], lr=lr)
    loss_fn = nn.BCELoss()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # Forward pass – ``siren`` expects ``x`` of shape (M,2) and ``z`` of shape (latent_dim,)
        preds = siren(X, z.squeeze(0))  # (M, 1)
        preds = preds.squeeze(-1)  # (M,)
        # Include L2 regularisation on the latent vector to match training loss
        loss = loss_fn(preds, Y) + lambda_z * (z ** 2).mean()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    return z.detach(), final_loss


def extract_latents_batched(
        siren: nn.Module,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        embed: Optional[nn.Module] = None,
        latent_dim: int = 256,
        lr: float = 0.01,
        steps: int = 1500,
        lambda_z: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = X_batch.device
    B = X_batch.shape[0]

    # Freeze SIREN
    siren.eval()
    for p in siren.parameters():
        p.requires_grad = False

    # Initialise latent vectors for the batch
    z_batch = _prepare_initial_z(embed, device, latent_dim, batch_size=B)
    z_batch.requires_grad_(True)

    optimizer = torch.optim.Adam([z_batch], lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    loss_fn = nn.BCELoss(reduction='none')

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        preds = vmap(siren, in_dims=(0, 0))(X_batch, z_batch).squeeze(-1)
        bce_losses = loss_fn(preds, Y_batch).mean(dim=1)
        l2_penalties = lambda_z * (z_batch ** 2).mean(dim=1)

        total_losses = bce_losses + l2_penalties
        loss = total_losses.mean()

        loss.backward()
        optimizer.step()

        scheduler.step()

    with torch.no_grad():
        preds = vmap(siren, in_dims=(0, 0))(X_batch, z_batch).squeeze(-1)
        bce_losses = loss_fn(preds, Y_batch).mean(dim=1)
        l2_penalties = lambda_z * (z_batch ** 2).mean(dim=1)
        final_losses = bce_losses + l2_penalties

    return z_batch.detach(), final_losses.detach()

__all__ = ["extract_latent", "extract_latents_batched"]
