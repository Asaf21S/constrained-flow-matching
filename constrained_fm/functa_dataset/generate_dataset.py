# -*- coding: utf-8 -*-
"""
Dataset generation for Phase 1 – the spatial oracle.

The script creates a synthetic collection of binary masks representing geometric
constraints (boxes, circles, and convex polygons). For each shape we sample a
set of 2‑D points ``X`` and corresponding inside/outside labels ``Y``. The
sampling strategy follows the plan:

* 50 % uniform random points in the domain ``[-4.5, 4.5]``.
* 50 % points densely sampled near the shape boundary (a thin band of width
  ``epsilon = 0.05`` around the contour).

The generated tensors have shape ``(N, M, 2)`` for ``X`` and ``(N, M)`` for
``Y`` where ``N`` is the number of shapes (default 10 000) and ``M`` is the
total number of points per shape (default 5 000). The tensors are saved as a
single ``.pt`` file for fast loading during training.

Usage
-----
Run the module directly::

    python -m constrained_fm.functa_dataset.generate_dataset \
        --num_shapes 10000 \
        --points_per_shape 5000 \
        --output_path data/functa_spatial_oracle.pt

The script only depends on the standard library, ``numpy`` and ``torch`` –
both are already part of this repository's environment.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Tuple
from tqdm.auto import tqdm

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Geometry helpers ----------------------------------------------------------
# ---------------------------------------------------------------------------

DOMAIN_MIN = -4.5
DOMAIN_MAX = 4.5
DOMAIN_SIZE = DOMAIN_MAX - DOMAIN_MIN

# Width of the thin band used for boundary sampling.
EPSILON = 0.05


def _uniform_points(num: int) -> np.ndarray:
    """Sample ``num`` points uniformly in the square domain.

    Returns
    -------
    np.ndarray
        Shape ``(num, 2)`` with ``float32`` values.
    """
    pts = np.random.uniform(DOMAIN_MIN, DOMAIN_MAX, size=(num, 2)).astype(np.float32)
    return pts

# ---------------------------------------------------------------------------
# Shape generators ----------------------------------------------------------
# ---------------------------------------------------------------------------

def _random_box() -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[int], np.ndarray], dict]:
    # 1. Sample the half-widths first
    hw = random.uniform(0.25, 3.5)
    hh = random.uniform(0.25, 3.5)

    # 2. Constrain the center so the box + jitter fits perfectly inside the domain
    min_cx = DOMAIN_MIN + hw + EPSILON
    max_cx = DOMAIN_MAX - hw - EPSILON
    cx = random.uniform(min_cx, max_cx)

    min_cy = DOMAIN_MIN + hh + EPSILON
    max_cy = DOMAIN_MAX - hh - EPSILON
    cy = random.uniform(min_cy, max_cy)

    x_min, x_max = cx - hw, cx + hw
    y_min, y_max = cy - hh, cy + hh

    def inside(pts: np.ndarray) -> np.ndarray:
        return (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) & (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)

    def sample_boundary(num: int) -> np.ndarray:
        edges = np.random.randint(0, 4, size=(num,))
        pts = np.empty((num, 2), dtype=np.float32)

        # Top edge
        top_mask = edges == 0
        pts[top_mask, 0] = np.random.uniform(x_min, x_max, size=(top_mask.sum(),))
        pts[top_mask, 1] = y_max
        # Bottom edge
        bot_mask = edges == 1
        pts[bot_mask, 0] = np.random.uniform(x_min, x_max, size=(bot_mask.sum(),))
        pts[bot_mask, 1] = y_min
        # Left edge
        left_mask = edges == 2
        pts[left_mask, 0] = x_min
        pts[left_mask, 1] = np.random.uniform(y_min, y_max, size=(left_mask.sum(),))
        # Right edge
        right_mask = edges == 3
        pts[right_mask, 0] = x_max
        pts[right_mask, 1] = np.random.uniform(y_min, y_max, size=(right_mask.sum(),))

        # Apply jitter
        jitter = np.random.uniform(-EPSILON, EPSILON, size=(num, 2))
        pts += jitter

        return np.clip(pts, DOMAIN_MIN, DOMAIN_MAX)  # leaving the clip for safety net against floating point errors.

    params = {"type": "box", "center": [cx, cy], "half_width": hw, "half_height": hh}
    return inside, sample_boundary, params


def _random_circle() -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[int], np.ndarray], dict]:
    """Create a random circle within the domain, with a fast boundary sampler.

    Returns a tuple ``(inside_fn, boundary_sampler, params)`` where ``boundary_sampler``
    generates ``num`` points on (or near) the circle perimeter with a small jitter of
    ``±EPSILON``.
    """
    radius = random.uniform(0.5, 3)
    min_c = DOMAIN_MIN + radius + EPSILON
    max_c = DOMAIN_MAX - radius - EPSILON
    cx = random.uniform(min_c, max_c)
    cy = random.uniform(min_c, max_c)

    def inside(pts: np.ndarray) -> np.ndarray:
        return ((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) <= radius ** 2

    def sample_boundary(num: int) -> np.ndarray:
        # Sample random angles uniformly around the circle.
        theta = np.random.uniform(0, 2 * np.pi, size=(num,))

        # Apply the jitter directly to the radius so it is truly perpendicular to the boundary
        jittered_radius = radius + np.random.uniform(-EPSILON, EPSILON, size=(num,))

        x = cx + jittered_radius * np.cos(theta)
        y = cy + jittered_radius * np.sin(theta)
        pts = np.stack([x, y], axis=-1)

        return np.clip(pts, DOMAIN_MIN, DOMAIN_MAX).astype(np.float32)  # floating-point safety net

    params = {"type": "circle", "center": [cx, cy], "radius": radius}
    return inside, sample_boundary, params


def _random_convex_polygon(num_vertices: int = 5) -> Tuple[
    Callable[[np.ndarray], np.ndarray], Callable[[int], np.ndarray], dict]:
    """Generate a simple convex polygon and a fast boundary sampler.

    Returns a tuple ``(inside_fn, boundary_sampler, params)``. The ``boundary_sampler``
    draws points uniformly along the polygon edges (weighted by edge length) and
    adds a small ``±EPSILON`` jitter strictly perpendicular to the edges.
    """
    num_vertices = max(3, min(num_vertices, 8))

    # Determine a random local scale (similar to circle radius or box half-width)
    scale = random.uniform(0.5, 3.5)

    # Enforce Safe Bounds based on this local scale
    safe_min = DOMAIN_MIN + scale + EPSILON
    safe_max = DOMAIN_MAX - scale - EPSILON
    cx = random.uniform(safe_min, safe_max)
    cy = random.uniform(safe_min, safe_max)

    # Generate initial points ONLY within this localized bounding box
    local_pts = np.random.uniform(-scale, scale, size=(num_vertices * 5, 2))
    pts = local_pts + np.array([cx, cy])

    # Compute hull using a quick Graham-scan implementation.
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    pts = sorted(map(tuple, pts))
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    hull = np.array(hull, dtype=np.float32)

    def inside(pts_test: np.ndarray) -> np.ndarray:
        # Ray-casting algorithm for point-in-polygon. Vectorised for speed.
        x, y = pts_test[:, 0], pts_test[:, 1]
        inside_mask = np.zeros_like(x, dtype=bool)
        n = hull.shape[0]
        for i in range(n):
            x0, y0 = hull[i]
            x1, y1 = hull[(i + 1) % n]
            cond = ((y0 > y) != (y1 > y)) & (
                    x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
            )
            inside_mask ^= cond
        return inside_mask

    def sample_boundary(num: int) -> np.ndarray:
        verts = hull
        edges = np.stack([verts[(i + 1) % len(verts)] - verts[i] for i in range(len(verts))])
        lengths = np.linalg.norm(edges, axis=1)

        # Calculate true perpendicular normals for each edge.
        # If edge vector is (dx, dy), normal is (-dy, dx) normalized.
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=-1) / lengths[:, None]

        cum_lengths = np.cumsum(lengths)
        total_len = cum_lengths[-1]

        # Vectorized sampling
        distances = np.random.uniform(0, total_len, size=num)
        edge_indices = np.searchsorted(cum_lengths, distances)

        # Calculate interpolation factor 't' for all points simultaneously
        prev_lengths = np.where(edge_indices > 0, cum_lengths[edge_indices - 1], 0.0)
        t = (distances - prev_lengths) / lengths[edge_indices]

        starts = verts[edge_indices]
        points = starts + t[:, None] * edges[edge_indices]

        # Apply strict perpendicular jitter
        jitter_mags = np.random.uniform(-EPSILON, EPSILON, size=(num, 1))
        points += jitter_mags * normals[edge_indices]

        return np.clip(points, DOMAIN_MIN, DOMAIN_MAX).astype(np.float32)

    params = {"type": "polygon", "vertices": hull.tolist(), "center": [cx, cy], "scale": scale}
    return inside, sample_boundary, params


# ---------------------------------------------------------------------------
# Dataset construction ------------------------------------------------------
# ---------------------------------------------------------------------------

def _sample_shape_points(
    inside_fn: Callable[[np.ndarray], np.ndarray],
    boundary_sampler: Callable[[int], np.ndarray],
    points_per_shape: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``X`` and ``Y`` for a single shape.

    ``points_per_shape`` is split 50/50 between uniform and boundary samples.
    The ``boundary_sampler`` generates the required number of boundary points.
    """
    half = points_per_shape // 2
    uniform = _uniform_points(half)
    boundary = boundary_sampler(points_per_shape - half)
    X = np.concatenate([uniform, boundary], axis=0)
    # Store binary labels as uint8 to save memory; they will be cast to float in the PyTorch dataset loader.
    Y = inside_fn(X).astype(np.uint8)
    return X, Y


def generate_dataset(
    num_shapes: int = 10_000,
    points_per_shape: int = 5_000,
    output_path: Path | str = "functa_spatial_oracle.pt",
) -> None:
    """Generate the full dataset and write it to ``output_path``.

    The output is a ``torch.save`` of a dictionary ``{"X": X, "Y": Y, "meta": {...}}``
    where ``X`` has shape ``(N, M, 2)`` and ``Y`` has shape ``(N, M)``.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    X_all = []
    Y_all = []
    meta_shapes = []

    shape_factories: List[Callable[[], Tuple[Callable[[np.ndarray], np.ndarray], Callable[[int], np.ndarray], dict]]] = [
        _random_box,
        _random_circle,
        lambda: _random_convex_polygon(num_vertices=random.randint(4, 8)),
    ]

    for idx in tqdm(range(num_shapes), desc="Generating shapes", unit="shape"):
        # Choose a shape type uniformly at random.
        factory = random.choice(shape_factories)
        inside_fn, boundary_sampler, params = factory()
        X_shape, Y_shape = _sample_shape_points(inside_fn, boundary_sampler, points_per_shape)
        X_all.append(torch.from_numpy(X_shape))
        Y_all.append(torch.from_numpy(Y_shape))
        meta_shapes.append({"index": idx, **params})

    X_tensor = torch.stack(X_all)  # (N, M, 2)
    Y_tensor = torch.stack(Y_all)  # (N, M)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X_tensor, "Y": Y_tensor, "meta": meta_shapes}, output_path)
    print(f"Dataset saved to {output_path.resolve()}")


if __name__ == "__main__":
    generate_dataset()
