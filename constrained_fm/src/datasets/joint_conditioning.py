# -*- coding: utf-8 -*-
"""Joint polynomial / convex-polygon constraints on the 2D GMM plane for one shared SIREN.

Both families are regressed as ``tanh(s * v(x))`` over ``[-S, S]^2`` in SIREN coordinates
``x / S``: polynomials use ``v = P`` and polygons ``v = C / tau`` with the bump2d half-plane
form ``C = max_i (a_i . x - b_i)``. The orientation ``s = -1`` encodes the exact complement
``{-v(x) <= 0}``, and because ``tanh`` is odd its target is the exact negation of ``s = +1``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from constrained_fm.src.consts import (BUMP_DOMAIN, BUMP_POLY_MAX_VERTICES,
                                       BUMP_POLY_MIN_VERTICES, BUMP_POLY_RADIUS_RANGE,
                                       FUNCTA_QUERY_GMM_FRACTION, PLANE_SCALE,
                                       POLY_MAX_AREA_RATIO, POLY_MIN_AREA_RATIO,
                                       POLYNOMIAL_DEGREE)
from constrained_fm.src.datasets.bump_conditioning import polygon_values
from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.geometry.polynomials import (compute_poly_features,
                                                     compute_poly_features_batched,
                                                     evaluate_poly_batched)
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.models.functa_siren import ModulatedSIREN, build_modulated_siren
from constrained_fm.src.problems.bump2d import polygon_mass, propose_polygons

FAMILY_POLYNOMIAL = 0
FAMILY_POLYGON = 1
FAMILY_NAMES = ("polynomial", "polygon")
SHAPE_KEYS = ("family", "sign", "C", "normals", "offsets", "active")
# propose_polygons pads every polygon to max_vertices half-planes plus the four box faces.
NUM_FACES = BUMP_POLY_MAX_VERTICES + 4

Shapes = dict[str, torch.Tensor]


def take(shapes: Shapes, index: torch.Tensor | slice) -> Shapes:
    return {key: value[index] for key, value in shapes.items()}


def proxy_set(num_points: int = 10000, degree: int = POLYNOMIAL_DEGREE,
              scale: float = PLANE_SCALE,
              device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    """GMM draws backing the mass filter of both families, plus their polynomial features."""
    points, _ = get_points(batch_size=num_points, device=device)
    points = points.to(device)
    x_pow, y_pow = compute_poly_features(points, degree=degree, scale=scale)
    return {"points": points, "x_pow": x_pow, "y_pow": y_pow}


def polynomial_boundary_slope(degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                              min_area: float = POLY_MIN_AREA_RATIO,
                              max_area: float = POLY_MAX_AREA_RATIO,
                              num_polys: int = 512, num_points: int = 4000,
                              quantile: float = 0.02,
                              device: torch.device | str | None = None) -> float:
    """Median ``||grad_x P||`` in plane units on the zero sets of training polynomials.

    The lowest ``quantile`` of ``|P|`` per polynomial stands in for its zero set. Polygon
    faces have unit normals, so ``tau = 1 / slope`` gives ``tanh(C / tau)`` the same
    boundary slope as ``tanh(P)``.
    """
    polys = sample_valid_polynomials(num_polys, degree=degree, scale=scale, min_area=min_area,
                                     max_area=max_area, device=device)
    x = ((torch.rand(num_polys, num_points, 2, device=device) * 2 - 1) * scale).requires_grad_(True)
    x_pow, y_pow = compute_poly_features_batched(x, degree=degree, scale=scale)
    values = evaluate_poly_batched(x_pow, y_pow, polys)
    grad_norm = torch.autograd.grad(values.sum(), x)[0].norm(dim=-1)
    magnitude = values.detach().abs()
    threshold = magnitude.quantile(quantile, dim=1, keepdim=True)
    return float(grad_norm[magnitude <= threshold].median())


def plane_polygons(count: int, proxy_points: torch.Tensor, scale: float = PLANE_SCALE,
                   min_mass: float = POLY_MIN_AREA_RATIO,
                   max_mass: float = POLY_MAX_AREA_RATIO,
                   min_vertices: int = BUMP_POLY_MIN_VERTICES,
                   max_vertices: int = BUMP_POLY_MAX_VERTICES, oversample: int = 4,
                   max_rounds: int = 64,
                   device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    """bump2d half-plane polygons mapped from ``[0, 2S]^2`` onto ``[-S, S]^2``, mass filtered.

    The translation ``x -> x - S 1`` keeps each normal and shifts its offset by
    ``-S (a_x + a_y)``, so the four box faces become the plane edges. Radii scale with the
    box side, and the GMM mass window is the polynomial one.
    """
    domain = 2.0 * scale
    radius_range = tuple(r * domain / BUMP_DOMAIN for r in BUMP_POLY_RADIUS_RANGE)
    kept: list[dict[str, torch.Tensor]] = []
    found = 0

    for _ in range(max_rounds):
        if found >= count:
            break
        normals, offsets, active = propose_polygons(count * oversample, domain, min_vertices,
                                                    max_vertices, radius_range, device)
        offsets = offsets - scale * normals.sum(dim=-1)
        mass = polygon_mass(normals, offsets, active, proxy_points)
        keep = ((mass >= min_mass) & (mass <= max_mass)).nonzero(as_tuple=True)[0]
        keep = keep[:count - found]
        kept.append({"normals": normals[keep], "offsets": offsets[keep],
                     "active": active[keep], "mass": mass[keep]})
        found += keep.numel()

    if found < count:
        raise RuntimeError(f"only {found}/{count} polygons fell in [{min_mass}, {max_mass}] "
                           f"after {max_rounds} rounds")
    return {key: torch.cat([chunk[key] for chunk in kept]) for key in kept[0]}


def sample_joint_shapes(count: int, proxy: dict[str, torch.Tensor],
                        polygon_fraction: float = 0.5, random_sign: bool = True,
                        degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                        min_mass: float = POLY_MIN_AREA_RATIO,
                        max_mass: float = POLY_MAX_AREA_RATIO,
                        device: torch.device | str | None = None) -> Shapes:
    """``count`` constraints: polynomials first, then ``round(count * polygon_fraction)`` polygons.

    ``sign`` is +1, or with ``random_sign`` -1 with probability 1/2 to select the complement.
    Fields of the other family are zero, with every polygon face inactive for polynomials.
    """
    num_polygons = int(round(count * polygon_fraction))
    num_polys = count - num_polygons
    C = torch.zeros(count, degree + 1, degree + 1, device=device)
    normals = torch.zeros(count, NUM_FACES, 2, device=device)
    offsets = torch.zeros(count, NUM_FACES, device=device)
    active = torch.zeros(count, NUM_FACES, dtype=torch.bool, device=device)

    if num_polys:
        C[:num_polys] = sample_valid_polynomials(num_polys, degree=degree, scale=scale,
                                                 proxy_x_pow=proxy["x_pow"],
                                                 proxy_y_pow=proxy["y_pow"],
                                                 min_area=min_mass, max_area=max_mass,
                                                 device=device)
    if num_polygons:
        polygons = plane_polygons(num_polygons, proxy["points"], scale, min_mass, max_mass,
                                  device=device)
        normals[num_polys:] = polygons["normals"]
        offsets[num_polys:] = polygons["offsets"]
        active[num_polys:] = polygons["active"]

    family = torch.cat([torch.full((num_polys,), FAMILY_POLYNOMIAL, dtype=torch.long),
                        torch.full((num_polygons,), FAMILY_POLYGON, dtype=torch.long)]).to(device)
    sign = torch.ones(count, device=device)
    if random_sign:
        sign = torch.where(torch.rand(count, device=device) < 0.5, -sign, sign)
    return {"family": family, "sign": sign, "C": C, "normals": normals, "offsets": offsets,
            "active": active}


def constraint_values(shapes: Shapes, x: torch.Tensor, tau: float,
                      degree: int = POLYNOMIAL_DEGREE,
                      scale: float = PLANE_SCALE) -> torch.Tensor:
    """Oriented field ``s * v(x)`` whose ``tanh`` is the regression target; (B, N, 2) -> (B, N).

    ``{s v <= 0}`` is the feasible region: ``v = P`` for polynomials, ``v = C / tau`` for
    polygons, and ``s = -1`` its exact complement.
    """
    values = x.new_empty(x.shape[:2])
    poly = shapes["family"] == FAMILY_POLYNOMIAL
    if bool(poly.any()):
        x_pow, y_pow = compute_poly_features_batched(x[poly], degree=degree, scale=scale)
        values[poly] = evaluate_poly_batched(x_pow, y_pow, shapes["C"][poly])
    gon = ~poly
    if bool(gon.any()):
        values[gon] = polygon_values(x[gon], shapes["normals"][gon], shapes["offsets"][gon],
                                     shapes["active"][gon]) / tau
    return values * shapes["sign"].unsqueeze(-1)


def regression_targets(shapes: Shapes, x_raw: torch.Tensor, tau: float,
                       degree: int = POLYNOMIAL_DEGREE,
                       scale: float = PLANE_SCALE) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x / S, tanh(s v(x)))`` for raw-plane query points (B, N, 2)."""
    return x_raw / scale, torch.tanh(constraint_values(shapes, x_raw, tau, degree, scale))


def draw_joint_batch(count: int, proxy: dict[str, torch.Tensor], points_per_shape: int,
                     tau: float, polygon_fraction: float = 0.5, random_sign: bool = True,
                     query_gmm_fraction: float = FUNCTA_QUERY_GMM_FRACTION,
                     degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                     min_mass: float = POLY_MIN_AREA_RATIO,
                     max_mass: float = POLY_MAX_AREA_RATIO,
                     device: torch.device | str | None = None
                     ) -> tuple[Shapes, torch.Tensor, torch.Tensor]:
    """Mixed constraints with their CAVIA query points ``x / S`` and targets."""
    shapes = sample_joint_shapes(count, proxy, polygon_fraction, random_sign, degree, scale,
                                 min_mass, max_mass, device)
    x_raw = sample_query_points(count, points_per_shape, scale=scale,
                                gmm_fraction=query_gmm_fraction, device=device)
    x, y = regression_targets(shapes, x_raw, tau, degree, scale)
    return shapes, x, y


def constraint_mass(shapes: Shapes, points: torch.Tensor, tau: float,
                    degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                    chunk_size: int = 256) -> torch.Tensor:
    """Fraction of shared ``points`` (N, 2) inside each oriented constraint; (B,)."""
    masses = []
    for start in range(0, shapes["family"].shape[0], chunk_size):
        sub = take(shapes, slice(start, start + chunk_size))
        x = points.unsqueeze(0).expand(sub["family"].shape[0], -1, -1)
        masses.append((constraint_values(sub, x, tau, degree, scale) <= 0).float().mean(dim=1))
    return torch.cat(masses)


def mass_iou(siren: nn.Module, z: torch.Tensor, shapes: Shapes, points: torch.Tensor,
             tau: float, degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
             chunk_size: int = 16) -> torch.Tensor:
    """IoU of ``{SIREN(x, z) <= 0}`` against the oriented constraint over GMM points; (B,) on CPU."""
    scores = []
    with torch.no_grad():
        for start in range(0, z.shape[0], chunk_size):
            sub = take(shapes, slice(start, start + chunk_size))
            x = points.unsqueeze(0).expand(sub["family"].shape[0], -1, -1)
            true_in = constraint_values(sub, x, tau, degree, scale) <= 0
            pred_in = siren(x / scale, z[start:start + chunk_size]).squeeze(-1) <= 0
            union = (true_in | pred_in).sum(dim=1).clamp(min=1)
            scores.append(((true_in & pred_in).sum(dim=1) / union).cpu())
    return torch.cat(scores)


def build_joint_pool(siren: nn.Module, proxy: dict[str, torch.Tensor], tau: float,
                     pool_size: int = 20000, polygon_fraction: float = 0.5,
                     points_per_shape: int = 1000, extraction_steps: int = 15,
                     extraction_lr: float = 6.25e-4, chunk_size: int = 128,
                     query_gmm_fraction: float = FUNCTA_QUERY_GMM_FRACTION,
                     degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                     min_mass: float = POLY_MIN_AREA_RATIO,
                     max_mass: float = POLY_MAX_AREA_RATIO,
                     device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    """Both orientations of every pool constraint, on CPU.

    ``z_pos`` encodes ``{v <= 0}`` and ``z_neg`` its exact complement ``{-v <= 0}``, extracted
    from the negated target on the same query points. ``mass_pos`` is the GMM mass of the
    stored orientation; the complement's is ``1 - mass_pos``.
    """
    if device is None:
        device = next(siren.parameters()).device

    chunks: list[dict[str, torch.Tensor]] = []
    for start in tqdm(range(0, pool_size, chunk_size), desc="Building joint pool"):
        count = min(chunk_size, pool_size - start)
        shapes, x, y = draw_joint_batch(count, proxy, points_per_shape, tau, polygon_fraction,
                                        False, query_gmm_fraction, degree, scale, min_mass,
                                        max_mass, device)
        z_pos, mse_pos = extract_latents_batched(siren, x, y, lr=extraction_lr,
                                                 steps=extraction_steps)
        z_neg, mse_neg = extract_latents_batched(siren, x, -y, lr=extraction_lr,
                                                 steps=extraction_steps)
        mass = constraint_mass(shapes, proxy["points"], tau, degree, scale)
        chunks.append({key: value.cpu() for key, value in {
            **shapes, "mass_pos": mass, "z_pos": z_pos, "z_neg": z_neg,
            "mse_pos": mse_pos, "mse_neg": mse_neg}.items()})

    return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}


def load_joint_siren(siren_dir: Path, checkpoint: str,
                     device: torch.device) -> tuple[ModulatedSIREN, dict]:
    """Frozen joint SIREN plus the training metadata (tau, CAVIA settings) in ``metrics.json``."""
    meta = json.loads((siren_dir / "metrics.json").read_text())
    siren = build_modulated_siren(latent_dim=meta["latent_dim"], hidden_dim=meta["hidden_dim"],
                                  n_layers=meta["n_layers"], w0=meta["w0"]).to(device)
    siren.load_state_dict(torch.load(siren_dir / checkpoint, map_location=device,
                                     weights_only=True))
    siren.eval()
    for p in siren.parameters():
        p.requires_grad_(False)
    return siren, meta


def summarize_by_group(values: torch.Tensor, shapes: Shapes) -> dict[str, float]:
    """Mean of ``values`` per family and per (family, orientation)."""
    values = values.detach().cpu()
    family = shapes["family"].cpu()
    sign = shapes["sign"].cpu()
    summary: dict[str, float] = {}
    for index, name in enumerate(FAMILY_NAMES):
        in_family = family == index
        if bool(in_family.any()):
            summary[f"{name}_mean"] = float(values[in_family].mean())
            summary[f"{name}_p5"] = float(np.percentile(values[in_family].numpy(), 5.0))
        for orientation, label in ((1.0, "inside"), (-1.0, "complement")):
            group = in_family & (sign == orientation)
            if bool(group.any()):
                summary[f"{name}_{label}_mean"] = float(values[group].mean())
    return summary


__all__ = ["FAMILY_POLYNOMIAL", "FAMILY_POLYGON", "FAMILY_NAMES", "SHAPE_KEYS", "NUM_FACES",
           "take", "proxy_set", "polynomial_boundary_slope", "plane_polygons",
           "sample_joint_shapes", "constraint_values", "regression_targets",
           "draw_joint_batch", "constraint_mass", "mass_iou", "build_joint_pool",
           "load_joint_siren", "summarize_by_group"]
