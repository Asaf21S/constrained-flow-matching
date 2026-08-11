# -*- coding: utf-8 -*-
"""Fidelity of the Functa conditioning itself, independent of the flow matcher.

The flow matcher only ever sees z, so it can at best fill the region the SIREN decodes
from z. These measures separate "the flow matcher disobeys its conditioning" from
"the conditioning describes the wrong region".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly


def uniform_grid_points(grid_size: int = 200, scale: float = PLANE_SCALE,
                        device: torch.device | str | None = None) -> torch.Tensor:
    """Uniform lattice over the plane, flattened to (grid_size**2, 2)."""
    axis = torch.linspace(-scale, scale, grid_size, device=device)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([gx.flatten(), gy.flatten()], dim=1)


def decode_region(siren: nn.Module, z: torch.Tensor, points: torch.Tensor,
                  scale: float = PLANE_SCALE) -> torch.Tensor:
    """SIREN(x, z) at each point, using the unbatched single-shape path. Returns (M,)."""
    with torch.no_grad():
        return siren(points / scale, z.view(-1)).squeeze(-1)


def true_region_mask(C: torch.Tensor, points: torch.Tensor, degree: int = POLYNOMIAL_DEGREE,
                     scale: float = PLANE_SCALE) -> torch.Tensor:
    """Boolean mask of {P(x) <= 0} for a single polynomial. Returns (M,)."""
    x_pow, y_pow = compute_poly_features(points, degree=degree, scale=scale)
    C_expanded = C.unsqueeze(0).expand(points.shape[0], -1, -1)
    return evaluate_poly(x_pow, y_pow, C_expanded).squeeze(-1) <= 0


def region_iou(siren: nn.Module, z: torch.Tensor, C: torch.Tensor, points: torch.Tensor,
               degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE) -> float:
    """IoU between {SIREN(x, z) <= 0} and {P(x) <= 0}, measured over the given points.

    Pass GMM-distributed points for a mass-weighted IoU: level-set disagreement away
    from the data cannot move the reported distributional metrics, so it should not
    move this diagnostic either.
    """
    true_in = true_region_mask(C, points, degree=degree, scale=scale)
    pred_in = decode_region(siren, z, points, scale=scale) <= 0

    intersection = (true_in & pred_in).sum().float()
    union = (true_in | pred_in).sum().float()
    return float(intersection / union.clamp(min=1.0))


def region_iou_batched(siren: nn.Module, z_batch: torch.Tensor, C_batch: torch.Tensor,
                       points: torch.Tensor, degree: int = POLYNOMIAL_DEGREE,
                       scale: float = PLANE_SCALE) -> torch.Tensor:
    """Per-shape region IoU over a shared point set. Returns (B,) on the CPU."""
    return torch.tensor([
        region_iou(siren, z_batch[i], C_batch[i], points, degree=degree, scale=scale)
        for i in range(C_batch.shape[0])
    ])


def constraint_masses(C_batch: torch.Tensor, x_pool: torch.Tensor,
                      degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                      chunk_size: int = 512) -> torch.Tensor:
    """Fraction of x_pool inside each constraint.

    With a GMM-sampled pool this is the constraint's probability mass, which is also its
    relative training exposure under mass-proportional pairing.
    """
    x_pow, y_pow = compute_poly_features(x_pool, degree=degree, scale=scale)

    masses = []
    for start in range(0, C_batch.shape[0], chunk_size):
        C_chunk = C_batch[start:start + chunk_size]
        # B = batch, N = pool points, I/J = polynomial degrees
        P_vals = torch.einsum("ni, bij, nj -> bn", x_pow, C_chunk, y_pow)
        masses.append((P_vals <= 0).float().mean(dim=1))

    return torch.cat(masses, dim=0)


__all__ = ["uniform_grid_points", "decode_region", "true_region_mask", "region_iou",
           "region_iou_batched", "constraint_masses"]
