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

import argparse
import math
import os
import random
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Geometry helpers ----------------------------------------------------------
# ---------------------------------------------------------------------------

# Domain limits – the original plan uses a square around the origin.
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
    """Create a random axis‑aligned rectangle, with a fast boundary sampler.

    Returns a tuple ``(inside_fn, boundary_sampler, params)``. ``boundary_sampler``
    generates ``num`` points on the rectangle edges with a small jitter of
    ``±EPSILON``.
    """
    # Choose centre and half‑widths uniformly but keep the box inside the domain.
    cx = random.uniform(DOMAIN_MIN + 1.0, DOMAIN_MAX - 1.0)
    cy = random.uniform(DOMAIN_MIN + 1.0, DOMAIN_MAX - 1.0)
    hw = random.uniform(0.25, 3.5)
    hh = random.uniform(0.25, 3.5)
    x_min, x_max = cx - hw, cx + hw
    y_min, y_max = cy - hh, cy + hh

    def inside(pts: np.ndarray) -> np.ndarray:
        return (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) & (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)

    def sample_boundary(num: int) -> np.ndarray:
        # Randomly assign points to one of the 4 edges.
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
        # Apply jitter.
        jitter = np.random.uniform(-EPSILON, EPSILON, size=(num, 2))
        return np.clip(pts + jitter, DOMAIN_MIN, DOMAIN_MAX)

    params = {"type": "box", "center": [cx, cy], "half_width": hw, "half_height": hh}
    return inside, sample_boundary, params


def _random_circle() -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[int], np.ndarray], dict]:
    """Create a random circle within the domain, with a fast boundary sampler.

    Returns a tuple ``(inside_fn, boundary_sampler, params)`` where ``boundary_sampler``
    generates ``num`` points on (or near) the circle perimeter with a small jitter of
    ``±EPSILON``.
    """
    radius = random.uniform(0.5, 3)
    cx = random.uniform(DOMAIN_MIN + radius, DOMAIN_MAX - radius)
    cy = random.uniform(DOMAIN_MIN + radius, DOMAIN_MAX - radius)

    def inside(pts: np.ndarray) -> np.ndarray:
        return ((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) <= radius ** 2

    def sample_boundary(num: int) -> np.ndarray:
        # Sample random angles uniformly around the circle.
        theta = np.random.uniform(0, 2 * np.pi, size=(num,))
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        pts = np.stack([x, y], axis=-1)
        # Apply a small uniform jitter perpendicular to the boundary.
        jitter = np.random.uniform(-EPSILON, EPSILON, size=(num, 2))
        return np.clip(pts + jitter, DOMAIN_MIN, DOMAIN_MAX).astype(np.float32)

    params = {"type": "circle", "center": [cx, cy], "radius": radius}
    return inside, sample_boundary, params


def _random_convex_polygon(num_vertices: int = 5) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[int], np.ndarray], dict]:
    """Generate a simple convex polygon and a fast boundary sampler.

    Returns a tuple ``(inside_fn, boundary_sampler, params)``. The ``boundary_sampler``
    draws points uniformly along the polygon edges (weighted by edge length) and
    adds a small ``±EPSILON`` jitter.
    """
    num_vertices = max(3, min(num_vertices, 8))
    # Generate random points in the domain, compute their convex hull.
    pts = np.random.uniform(DOMAIN_MIN, DOMAIN_MAX, size=(num_vertices * 5, 2))
    # Compute hull using a quick Graham‑scan implementation.
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
    # Ensure the polygon is within the domain – clip any out‑of‑bounds vertices.
    hull = np.clip(hull, DOMAIN_MIN, DOMAIN_MAX)

    def inside(pts_test: np.ndarray) -> np.ndarray:
        # Ray‑casting algorithm for point‑in‑polygon. Vectorised for speed.
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
        # Compute edge vectors and lengths.
        verts = hull
        edges = np.stack([verts[(i + 1) % len(verts)] - verts[i] for i in range(len(verts))])
        lengths = np.linalg.norm(edges, axis=1)
        cum_lengths = np.cumsum(lengths)
        total_len = cum_lengths[-1]
        # Sample a random distance along the perimeter for each point.
        distances = np.random.uniform(0, total_len, size=num)
        points = np.empty((num, 2), dtype=np.float32)
        for i, d in enumerate(distances):
            edge_idx = np.searchsorted(cum_lengths, d)
            prev_len = cum_lengths[edge_idx - 1] if edge_idx > 0 else 0.0
            t = (d - prev_len) / lengths[edge_idx]
            start = verts[edge_idx]
            points[i] = start + t * edges[edge_idx]
        # Jitter perpendicular to the edge direction (approximate with uniform jitter).
        jitter = np.random.uniform(-EPSILON, EPSILON, size=(num, 2))
        return np.clip(points + jitter, DOMAIN_MIN, DOMAIN_MAX).astype(np.float32)

    params = {"type": "polygon", "vertices": hull.tolist()}
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

    for idx in range(num_shapes):
        # Choose a shape type uniformly at random.
        factory = random.choice(shape_factories)
        inside_fn, boundary_sampler, params = factory()
        X_shape, Y_shape = _sample_shape_points(inside_fn, boundary_sampler, points_per_shape)
        X_all.append(torch.from_numpy(X_shape))
        Y_all.append(torch.from_numpy(Y_shape))
        meta_shapes.append({"index": idx, **params})
        if (idx + 1) % 500 == 0:
            print(f"[Dataset] Generated {idx + 1}/{num_shapes} shapes")

    X_tensor = torch.stack(X_all)  # (N, M, 2)
    Y_tensor = torch.stack(Y_all)  # (N, M)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X_tensor, "Y": Y_tensor, "meta": meta_shapes}, output_path)
    print(f"Dataset saved to {output_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI entry point ----------------------------------------------------------
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Functa spatial oracle dataset (Phase 1)")
    parser.add_argument(
        "--num_shapes",
        type=int,
        default=10_000,
        help="Number of distinct geometric shapes to generate (default: 10 000)",
    )
    parser.add_argument(
        "--points_per_shape",
        type=int,
        default=5_000,
        help="Number of (x, y) samples per shape (default: 5 000)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/functa_spatial_oracle.pt",
        help="Path where the ``.pt`` file will be written",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_dataset(
        num_shapes=args.num_shapes,
        points_per_shape=args.points_per_shape,
        output_path=args.output_path,
    )
