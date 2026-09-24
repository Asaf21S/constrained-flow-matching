# -*- coding: utf-8 -*-
"""Simple (possibly non-convex) polygons as exact signed distance fields on the plane.

A polygon is an ordered ``(K, 2)`` vertex array in raw plane coordinates; the edge from the
last vertex back to the first is implicit. The feasible set is the interior, so the field
follows the polynomial convention ``d(x) <= 0`` inside.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def polygon_sdf(x: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Exact Euclidean signed distance to a simple polygon, negative inside.

    Magnitude is the distance to the nearest edge segment; the sign comes from an even-odd
    crossing test, so convexity is not required. Elementwise products only, never a matmul,
    so TF32 cannot flip the sign of points lying on an edge.

    Args:
        x: (..., 2) query points.
        vertices: (K, 2) polygon vertices in either orientation.

    Returns:
        (...) signed distances.
    """
    v = vertices.to(device=x.device, dtype=x.dtype)
    v_prev = torch.roll(v, shifts=1, dims=0)
    edge = v_prev - v                                            # (K, 2)
    rel = x.unsqueeze(-2) - v                                    # (..., K, 2)

    t = ((rel * edge).sum(-1) / (edge * edge).sum(-1)).clamp(0.0, 1.0)
    dist_sq = (rel - edge * t.unsqueeze(-1)).pow(2).sum(-1).amin(dim=-1)

    above = x[..., 1:2] >= v[:, 1]
    below_prev = x[..., 1:2] < v_prev[:, 1]
    left = edge[:, 0] * rel[..., 1] > edge[:, 1] * rel[..., 0]
    crossing = (above & below_prev & left) | (~above & ~below_prev & ~left)
    inside = crossing.sum(dim=-1) % 2 == 1

    dist = dist_sq.sqrt()
    return torch.where(inside, -dist, dist)


def sample_boundary_points(vertices: torch.Tensor, num: int,
                           device: torch.device | str | None = None) -> torch.Tensor:
    """(num, 2) points uniform in arc length along the polygon's edges."""
    v = vertices.to(device)
    edge = torch.roll(v, shifts=-1, dims=0) - v
    edge_idx = torch.multinomial(edge.norm(dim=-1), num, replacement=True)
    t = torch.rand(num, 1, device=v.device)
    return v[edge_idx] + t * edge[edge_idx]


def regular_polygon(num_sides: int, radius: float, center: Sequence[float] = (0.0, 0.0),
                    rotation_deg: float = 0.0) -> torch.Tensor:
    """(num_sides, 2) vertices on a circle of ``radius``, first vertex at ``rotation_deg``."""
    angles = torch.arange(num_sides, dtype=torch.float32) * (2.0 * math.pi / num_sides)
    angles = angles + math.radians(rotation_deg)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) * radius \
        + torch.tensor(center, dtype=torch.float32)


def star_polygon(num_points: int, outer_radius: float, inner_radius: float,
                 center: Sequence[float] = (0.0, 0.0), rotation_deg: float = 90.0) -> torch.Tensor:
    """(2 * num_points, 2) vertices alternating between the outer and inner radius."""
    angles = torch.arange(2 * num_points, dtype=torch.float32) * (math.pi / num_points)
    angles = angles + math.radians(rotation_deg)
    radii = torch.tensor([outer_radius, inner_radius], dtype=torch.float32).repeat(num_points)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) * radii.unsqueeze(-1) \
        + torch.tensor(center, dtype=torch.float32)


def rotated_rectangle(half_width: float, half_height: float,
                      center: Sequence[float] = (0.0, 0.0), rotation_deg: float = 0.0
                      ) -> torch.Tensor:
    """(4, 2) corners of a ``2 half_width x 2 half_height`` rectangle rotated about its centre."""
    corners = torch.tensor([[-half_width, -half_height], [half_width, -half_height],
                            [half_width, half_height], [-half_width, half_height]])
    theta = math.radians(rotation_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    rotated = torch.stack([cos * corners[:, 0] - sin * corners[:, 1],
                           sin * corners[:, 0] + cos * corners[:, 1]], dim=-1)
    return rotated + torch.tensor(center, dtype=torch.float32)


__all__ = ["polygon_sdf", "sample_boundary_points", "regular_polygon", "star_polygon",
           "rotated_rectangle"]
