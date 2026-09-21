# -*- coding: utf-8 -*-
"""Functa conditioning for bump2d: convex polygons encoded into SIREN latents.

The polynomial pipeline in :mod:`functa_conditioning` leans on one structural fact that does
not survive the move to polygons. There, the complement of a feasible region is the region of
``-P``, so every sampled shape yields two valid constraints and ``tanh(-P) = -tanh(P)`` makes
the second latent almost free. The complement of a convex polygon is not convex, so there is
no flip trick here: a pool entry has a single orientation, and pairing a target sample with a
constraint that contains it becomes a rejection draw rather than a sign choice.

Everything else carries over. The regression target is ``tanh(C / tau)`` with ``C`` the
max-of-half-planes value, extraction is the same frozen-SIREN CAVIA inner loop, and pool
entries are still keyed by probability mass so training exposure can be reweighted.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from constrained_fm.src.consts import (BUMP_DOMAIN, BUMP_POLY_MAX_MASS, BUMP_POLY_MAX_VERTICES,
                                       BUMP_POLY_MIN_MASS, BUMP_POLY_MIN_VERTICES,
                                       BUMP_POLY_RADIUS_RANGE, BUMP_QUERY_TARGET_FRACTION,
                                       BUMP_SIREN_TAU)
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.problems.bump2d import BumpTarget, polygon_mass, propose_polygons

POOL_KEYS = ("normals", "offsets", "active", "mass")


def to_siren_coords(x: torch.Tensor, domain: float = BUMP_DOMAIN) -> torch.Tensor:
    """Maps the domain box ``[0, L]^d`` onto the SIREN's canonical ``[-1, 1]^d``."""
    return 2.0 * x / domain - 1.0


def polygon_values(x: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor,
                   active: torch.Tensor) -> torch.Tensor:
    """``max_i (a_i . x - b_i)`` per shape; (B, N, 2), (B, K, 2), (B, K), (B, K) -> (B, N).

    Padded half-planes are masked to ``-inf`` rather than dropped, so a batch can mix shapes
    with different face counts. Every polygon keeps the four box faces, so at least one entry
    per row is always finite.
    """
    residual = torch.einsum("bnd,bkd->bnk", x, normals) - offsets.unsqueeze(1)
    return residual.masked_fill(~active.unsqueeze(1), float("-inf")).amax(dim=-1)


def sample_query_points(target: BumpTarget, batch_size: int, num_points: int,
                        domain: float = BUMP_DOMAIN,
                        target_fraction: float = BUMP_QUERY_TARGET_FRACTION,
                        device: torch.device | str | None = None) -> torch.Tensor:
    """Query coordinates for CAVIA extraction, mixing target-distributed and uniform draws.

    The split is the same trade the polynomial pipeline documents, but the balance differs.
    Only the target-distributed half constrains the boundary where the reported mass-IoU can
    see it; the uniform half stops the level set from wandering in the tail, where the latent
    would otherwise be unconstrained and extraction could drift between similar shapes.
    """
    num_target = int(round(num_points * target_fraction))
    parts = []
    if num_target > 0:
        parts.append(target.sample(batch_size * num_target, device=device)
                     .view(batch_size, num_target, 2))
    if num_points - num_target > 0:
        parts.append(torch.rand(batch_size, num_points - num_target, 2, device=device) * domain)
    return torch.cat(parts, dim=1)


def polygon_batch(target: BumpTarget, mass_pool: torch.Tensor, batch_size: int,
                  domain: float = BUMP_DOMAIN,
                  min_vertices: int = BUMP_POLY_MIN_VERTICES,
                  max_vertices: int = BUMP_POLY_MAX_VERTICES,
                  radius_range: tuple[float, float] = BUMP_POLY_RADIUS_RANGE,
                  min_mass: float = BUMP_POLY_MIN_MASS, max_mass: float = BUMP_POLY_MAX_MASS,
                  oversample: int = 4, max_rounds: int = 64,
                  device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    """Exactly ``batch_size`` mass-filtered polygons, kept padded as ``(B, K, ...)``."""
    kept: list[dict[str, torch.Tensor]] = []
    found = 0

    for _ in range(max_rounds):
        if found >= batch_size:
            break
        normals, offsets, active = propose_polygons(batch_size * oversample, domain,
                                                    min_vertices, max_vertices, radius_range,
                                                    device)
        mass = polygon_mass(normals, offsets, active, mass_pool)
        keep = ((mass >= min_mass) & (mass <= max_mass)).nonzero(as_tuple=True)[0]
        keep = keep[:batch_size - found]
        kept.append({"normals": normals[keep], "offsets": offsets[keep],
                     "active": active[keep], "mass": mass[keep]})
        found += keep.numel()

    if found < batch_size:
        raise RuntimeError(f"only {found}/{batch_size} polygons fell in "
                           f"[{min_mass}, {max_mass}] after {max_rounds} rounds")
    return {key: torch.cat([chunk[key] for chunk in kept], dim=0) for key in POOL_KEYS}


def regression_targets(shapes: dict[str, torch.Tensor], points: torch.Tensor,
                       tau: float = BUMP_SIREN_TAU,
                       domain: float = BUMP_DOMAIN) -> tuple[torch.Tensor, torch.Tensor]:
    """``(X in [-1, 1]^2, tanh(C / tau))`` for a padded batch of polygons and query points."""
    values = polygon_values(points, shapes["normals"], shapes["offsets"], shapes["active"])
    return to_siren_coords(points, domain), torch.tanh(values / tau)


def build_polygon_pool(siren: nn.Module, target: BumpTarget, mass_pool: torch.Tensor,
                       pool_size: int = 20000, points_per_shape: int = 1000,
                       extraction_steps: int = 15, extraction_lr: float = 6.25e-4,
                       chunk_size: int = 128, tau: float = BUMP_SIREN_TAU,
                       domain: float = BUMP_DOMAIN,
                       target_fraction: float = BUMP_QUERY_TARGET_FRACTION,
                       min_mass: float = BUMP_POLY_MIN_MASS,
                       max_mass: float = BUMP_POLY_MAX_MASS,
                       device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    """Precomputes latents for a pool of polygons so training needs no SIREN extraction.

    Half the size of the polynomial pool for the same shape count, since only one orientation
    per shape exists. Returned on CPU, ready to save.
    """
    if device is None:
        device = next(siren.parameters()).device

    chunks: list[dict[str, torch.Tensor]] = []
    for start in tqdm(range(0, pool_size, chunk_size), desc="Building polygon pool"):
        count = min(chunk_size, pool_size - start)
        shapes = polygon_batch(target, mass_pool, count, domain=domain, min_mass=min_mass,
                               max_mass=max_mass, device=device)
        points = sample_query_points(target, count, points_per_shape, domain, target_fraction,
                                     device)
        x_scaled, y = regression_targets(shapes, points, tau, domain)
        z, _ = extract_latents_batched(siren, x_scaled, y, lr=extraction_lr,
                                       steps=extraction_steps)
        chunks.append({**{k: shapes[k].cpu() for k in POOL_KEYS}, "z": z.cpu()})

    return {key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in (*POOL_KEYS, "z")}


def sample_from_polygon_pool(x_1: torch.Tensor, pool: dict[str, torch.Tensor],
                             rounds: int = 32, weight_power: float = 0.0,
                             max_weight: float = 20.0
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairs each target sample with a pool polygon that contains it.

    Repeated uniform draws stand in for the polynomial pipeline's sign flip. A shape is drawn
    with probability proportional to its mass, exactly as there, so the same ``mass^-p``
    reweighting equalises exposure. A point in a thin tail may miss every draw; those rows are
    reported rather than silently paired with an infeasible constraint.

    Returns:
        z: (B, latent_dim) latents, undefined where ``hit`` is False.
        w: (B,) loss weights normalised to mean 1 over the rows that hit.
        hit: (B,) whether a containing polygon was found.
    """
    device = x_1.device
    batch_size = x_1.shape[0]
    pool_size = pool["mass"].shape[0]
    query = x_1.unsqueeze(1)

    chosen = torch.zeros(batch_size, dtype=torch.long, device=device)
    hit = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(rounds):
        idx = torch.randint(0, pool_size, (batch_size,), device=device)
        inside = polygon_values(query, pool["normals"][idx], pool["offsets"][idx],
                                pool["active"][idx]).squeeze(-1) <= 0
        take = inside & ~hit
        chosen = torch.where(take, idx, chosen)
        hit |= take
        if bool(hit.all()):
            break

    if weight_power == 0.0:
        w = torch.ones(batch_size, device=device)
    else:
        w = pool["mass"][chosen].clamp(min=1.0 / max_weight) ** (-weight_power)
        w = w.clamp(max=max_weight)
        w = w / w[hit].mean().clamp_min(1e-8)

    return pool["z"][chosen], w, hit


def mass_iou(siren: nn.Module, z: torch.Tensor, shapes: dict[str, torch.Tensor],
             points: torch.Tensor, domain: float = BUMP_DOMAIN,
             chunk_size: int = 16) -> torch.Tensor:
    """Per-shape IoU of ``{SIREN(x, z) <= 0}`` against the true polygon; (B,) on CPU.

    Pass target-distributed points so the intersection and union are counted in probability
    mass. A thin boundary error over an empty region is irrelevant to the sampler; the same
    error where the target concentrates is not, and only a mass-weighted count sees that.

    Chunked over shapes: the SIREN forward is dense over every (shape, point) pair, so a few
    hundred shapes against a few tens of thousands of points would otherwise allocate more
    activation memory than the whole meta-training loop.
    """
    scores = []
    for start in range(0, z.shape[0], chunk_size):
        block = slice(start, start + chunk_size)
        count = z[block].shape[0]
        query = points.unsqueeze(0).expand(count, -1, -1)
        true_in = polygon_values(query, shapes["normals"][block], shapes["offsets"][block],
                                 shapes["active"][block]) <= 0

        with torch.no_grad():
            pred_in = siren(to_siren_coords(query, domain), z[block]).squeeze(-1) <= 0

        intersection = (true_in & pred_in).sum(dim=-1).float()
        union = (true_in | pred_in).sum(dim=-1).float()
        scores.append((intersection / union.clamp_min(1.0)).cpu())

    return torch.cat(scores, dim=0)


__all__ = ["to_siren_coords", "polygon_values", "sample_query_points", "polygon_batch",
           "regression_targets", "build_polygon_pool", "sample_from_polygon_pool", "mass_iou"]
