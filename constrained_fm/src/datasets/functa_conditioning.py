# -*- coding: utf-8 -*-
"""Batched Functa-conditioned training data generation for the constrained flow matcher."""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE
from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.geometry.polynomials import compute_poly_features_batched, evaluate_poly_batched
from constrained_fm.src.inference.latent_extractor import extract_latents_batched


def generate_functa_conditioned_batch(
        siren: nn.Module,
        x_1: torch.Tensor,
        proxy_x_pow: torch.Tensor,
        proxy_y_pow: torch.Tensor,
        degree: int = POLYNOMIAL_DEGREE,
        scale: float = PLANE_SCALE,
        points_per_shape: int = 1000,
        extraction_steps: int = 15,
        extraction_lr: float = 1e-2,
        min_area: float = 0.05,
        max_area: float = 0.95,
        extraction_chunk_size: int = 128,
        device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a batch of Functa latents, each guaranteed to represent a valid
    region containing its paired x_1 target sample.

    Mirrors the "flip trick" used to train PolynomialConstrainedFM: samples a random
    balanced-area polynomial per example, then flips its sign wherever x_1 falls on
    the invalid side, guaranteeing P(x_1) <= 0 by construction with no rejection
    sampling. The oriented polynomial is then converted into a Functa context
    vector via frozen-SIREN CAVIA extraction.

    The extraction itself is chunked along the batch dimension so peak SIREN-forward
    memory is bounded by extraction_chunk_size * points_per_shape, independent of the
    caller's (potentially much larger) batch_size -- this is the main memory driver
    of the whole pipeline (a full-batch SIREN forward over batch_size * points_per_shape
    points, run extraction_steps times) and is what OOMs on smaller GPUs otherwise.

    Args:
        siren: trained, frozen ModulatedSIREN.
        x_1: (B, 2) raw-scale target samples (e.g. from the GMM) to condition on.
        proxy_x_pow, proxy_y_pow: precomputed proxy features for rejection sampling,
            computed once outside the training loop (see sample_valid_polynomials).
        degree, scale: polynomial degree and domain half-width; must match training.
        points_per_shape: number of query points used for the CAVIA extraction.
        extraction_steps, extraction_lr: CAVIA inner-loop SGD budget.
        min_area, max_area: accepted area-ratio range for the sampled polynomials.
        extraction_chunk_size: number of shapes processed per SIREN forward pass;
            lower this first if you hit CUDA OOM.
        device: torch device; defaults to x_1's device.

    Returns:
        C: (B, degree + 1, degree + 1) oriented coefficients, satisfying P(x_1) <= 0.
        z: (B, latent_dim) extracted Functa latents.
    """
    if device is None:
        device = x_1.device
    batch_size = x_1.shape[0]

    C = sample_valid_polynomials(
        batch_size=batch_size, degree=degree, scale=scale,
        proxy_x_pow=proxy_x_pow, proxy_y_pow=proxy_y_pow,
        min_area=min_area, max_area=max_area, device=device,
    )

    # Flip trick: orient each polynomial so its own x_1 target satisfies P(x_1) <= 0.
    x1_pow, y1_pow = compute_poly_features_batched(x_1.unsqueeze(1), degree=degree, scale=scale)
    P_x1 = evaluate_poly_batched(x1_pow, y1_pow, C).squeeze(-1)
    flip_mask = (P_x1 > 0).float().view(batch_size, 1, 1)
    C = C * (1.0 - 2.0 * flip_mask)

    # Chunked CAVIA extraction: fresh random query points per chunk, evaluated on
    # the oriented C, converted to a Functa latent via the frozen SIREN.
    z_chunks = []
    for start in range(0, batch_size, extraction_chunk_size):
        end = min(start + extraction_chunk_size, batch_size)
        C_chunk = C[start:end]

        X_raw = (torch.rand(end - start, points_per_shape, 2, device=device) * (scale * 2)) - scale
        x_pow, y_pow = compute_poly_features_batched(X_raw, degree=degree, scale=scale)
        P_vals = evaluate_poly_batched(x_pow, y_pow, C_chunk)
        Y = torch.tanh(P_vals)
        X_scaled = X_raw / scale

        z_chunk, _ = extract_latents_batched(siren, X_scaled, Y, lr=extraction_lr, steps=extraction_steps)
        z_chunks.append(z_chunk)

    z = torch.cat(z_chunks, dim=0)

    return C, z


def build_functa_pool(
        siren: nn.Module,
        proxy_x_pow: torch.Tensor,
        proxy_y_pow: torch.Tensor,
        pool_size: int = 20000,
        degree: int = POLYNOMIAL_DEGREE,
        scale: float = PLANE_SCALE,
        points_per_shape: int = 1000,
        extraction_steps: int = 15,
        extraction_lr: float = 1e-2,
        min_area: float = 0.05,
        max_area: float = 0.95,
        chunk_size: int = 128,
        device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Precomputes a large, reusable pool of Functa latents for both orientations
    of each sampled polynomial, so training-time conditioning needs no further
    SIREN extraction at all (see sample_from_functa_pool).

    Exploits tanh(-P) = -tanh(P): the flipped orientation's regression target is
    simply the negation of the original, so a single extraction pass over the
    pool yields both z_pos (for C) and z_neg (for -C) at roughly double the cost
    of a one-orientation pool -- still a one-time cost, versus paying full
    extraction on every training iteration.

    Polynomial sampling and extraction are both chunked by chunk_size: sample_valid_polynomials'
    internal proxy-grid einsum scales as O(batch_size * num_proxy_points), so generating the
    whole pool_size in one call can itself OOM at large pool sizes, independent of extraction.

    Returns a dict with keys "C" (pool_size, degree+1, degree+1), "z_pos", "z_neg"
    (each (pool_size, latent_dim)), all on CPU, ready to torch.save to disk.
    """
    if device is None:
        device = next(siren.parameters()).device

    C_chunks, z_pos_chunks, z_neg_chunks = [], [], []
    for start in tqdm(range(0, pool_size, chunk_size), desc="Building Functa pool"):
        end = min(start + chunk_size, pool_size)
        current_chunk_size = end - start

        C_chunk = sample_valid_polynomials(
            batch_size=current_chunk_size, degree=degree, scale=scale,
            proxy_x_pow=proxy_x_pow, proxy_y_pow=proxy_y_pow,
            min_area=min_area, max_area=max_area, device=device,
        )

        X_raw = (torch.rand(current_chunk_size, points_per_shape, 2, device=device) * (scale * 2)) - scale
        x_pow, y_pow = compute_poly_features_batched(X_raw, degree=degree, scale=scale)
        P_vals = evaluate_poly_batched(x_pow, y_pow, C_chunk)
        Y_pos = torch.tanh(P_vals)
        X_scaled = X_raw / scale

        z_pos_chunk, _ = extract_latents_batched(siren, X_scaled, Y_pos, lr=extraction_lr, steps=extraction_steps)
        z_neg_chunk, _ = extract_latents_batched(siren, X_scaled, -Y_pos, lr=extraction_lr, steps=extraction_steps)

        C_chunks.append(C_chunk.cpu())
        z_pos_chunks.append(z_pos_chunk.cpu())
        z_neg_chunks.append(z_neg_chunk.cpu())

    return {
        "C": torch.cat(C_chunks, dim=0),
        "z_pos": torch.cat(z_pos_chunks, dim=0),
        "z_neg": torch.cat(z_neg_chunks, dim=0),
    }


def sample_from_functa_pool(
        x_1: torch.Tensor,
        pool: dict[str, torch.Tensor],
        degree: int = POLYNOMIAL_DEGREE,
        scale: float = PLANE_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples a Functa-conditioned batch from a precomputed pool with no SIREN calls.

    For each x_1, picks a random pool entry and selects whichever precomputed
    orientation (z_pos for C, z_neg for -C) actually contains x_1, exactly
    preserving the "flip trick" invariant P(x_1) <= 0 at effectively zero cost.

    Args:
        x_1: (B, 2) raw-scale target samples to condition on.
        pool: dict as returned by build_functa_pool (moved to x_1's device beforehand).
        degree, scale: must match the values used to build the pool.

    Returns:
        C: (B, degree + 1, degree + 1) oriented coefficients, satisfying P(x_1) <= 0.
        z: (B, latent_dim) Functa latents.
    """
    device = x_1.device
    batch_size = x_1.shape[0]
    pool_size = pool["C"].shape[0]

    idx = torch.randint(0, pool_size, (batch_size,), device=device)
    C_batch = pool["C"][idx]
    z_pos_batch = pool["z_pos"][idx]
    z_neg_batch = pool["z_neg"][idx]

    x1_pow, y1_pow = compute_poly_features_batched(x_1.unsqueeze(1), degree=degree, scale=scale)
    P_x1 = evaluate_poly_batched(x1_pow, y1_pow, C_batch).squeeze(-1)
    use_flipped = P_x1 > 0  # x_1 violates the pool's original orientation -> use the flipped one

    z = torch.where(use_flipped.unsqueeze(-1), z_neg_batch, z_pos_batch)
    C = torch.where(use_flipped.view(-1, 1, 1), -C_batch, C_batch)

    return C, z


__all__ = ["generate_functa_conditioned_batch", "build_functa_pool", "sample_from_functa_pool"]
