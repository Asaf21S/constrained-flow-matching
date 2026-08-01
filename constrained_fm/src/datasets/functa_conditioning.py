# -*- coding: utf-8 -*-
"""Batched Functa-conditioned training data generation for the constrained flow matcher."""

from __future__ import annotations

import torch
import torch.nn as nn

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
        device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a batch of Functa latents, each guaranteed to represent a valid
    region containing its paired x_1 target sample.

    Mirrors the "flip trick" used to train PolynomialConstrainedFM: samples a random
    balanced-area polynomial per example, then flips its sign wherever x_1 falls on
    the invalid side, guaranteeing P(x_1) <= 0 by construction with no rejection
    sampling. The oriented polynomial is then converted into a Functa context
    vector via frozen-SIREN CAVIA extraction.

    Args:
        siren: trained, frozen ModulatedSIREN.
        x_1: (B, 2) raw-scale target samples (e.g. from the GMM) to condition on.
        proxy_x_pow, proxy_y_pow: precomputed proxy features for rejection sampling,
            computed once outside the training loop (see sample_valid_polynomials).
        degree, scale: polynomial degree and domain half-width; must match training.
        points_per_shape: number of query points used for the CAVIA extraction.
        extraction_steps, extraction_lr: CAVIA inner-loop SGD budget.
        min_area, max_area: accepted area-ratio range for the sampled polynomials.
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

    # Fresh random query points for CAVIA extraction, evaluated on the oriented C.
    X_raw = (torch.rand(batch_size, points_per_shape, 2, device=device) * (scale * 2)) - scale
    x_pow, y_pow = compute_poly_features_batched(X_raw, degree=degree, scale=scale)
    P_vals = evaluate_poly_batched(x_pow, y_pow, C)
    Y = torch.tanh(P_vals)
    X_scaled = X_raw / scale

    z, _ = extract_latents_batched(siren, X_scaled, Y, lr=extraction_lr, steps=extraction_steps)

    return C, z


__all__ = ["generate_functa_conditioned_batch"]
