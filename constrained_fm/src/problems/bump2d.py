# -*- coding: utf-8 -*-
"""Bump hunting on the plane: a 1% Gaussian signal buried in a falling exponential background.

The target lives on the box ``[0, L]^2`` and is the mixture

.. math::

    p(x) = (1 - w)\\, \\prod_{j} \\mathrm{Exp}_{[0,L]}(x_j; \\beta_j)
         + w\\, \\mathcal{N}_{[0,L]}(x; \\mu_s, \\Sigma_s),

with both components renormalised on the box, so ``p`` integrates to one exactly rather than
to within a truncation error. The constraint family is the intersection of random half-planes,
``C(x) = max_i (a_i . x - b_i) <= 0``, which is convex by construction and is already in the
form the SIREN encoder is asked to represent.

Sampling is exact for both components: the truncated exponential and the truncated normal are
both inverted through their CDFs, so no rejection loop is involved and the 1% signal fraction
is realised without variance from an accept/reject step.
"""

from __future__ import annotations

import math

import torch

from constrained_fm.src.consts import (BUMP_BACKGROUND_SCALES, BUMP_DOMAIN,
                                       BUMP_POLY_MAX_MASS, BUMP_POLY_MAX_VERTICES,
                                       BUMP_POLY_MIN_MASS, BUMP_POLY_MIN_VERTICES,
                                       BUMP_POLY_RADIUS_RANGE, BUMP_SIGNAL_MEAN,
                                       BUMP_SIGNAL_SIGMA, BUMP_SIGNAL_WEIGHT)
from constrained_fm.src.problems.base import AffineNormalizer, Constraint, Problem, Target

PROBLEM_NAME = "bump2d"

_LOG_2PI = math.log(2.0 * math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _as_pair(value: float | tuple[float, float] | list) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        return float(value[0]), float(value[1])
    return float(value), float(value)


class PolygonConstraint(Constraint):
    """``{x : max_i (a_i . x - b_i) <= 0}``, a convex polygon as a set of half-planes."""

    dim = 2

    def __init__(self, normals: torch.Tensor, offsets: torch.Tensor,
                 interior: torch.Tensor | None = None):
        self.normals = normals
        self.offsets = offsets
        self.interior = interior

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.normals.transpose(-1, -2) - self.offsets).amax(dim=-1)

    @property
    def interior_point(self) -> torch.Tensor | None:
        return self.interior

    @property
    def half_planes(self) -> torch.Tensor:
        """``(K, 3)`` rows ``[a_x, a_y, b]``."""
        return torch.cat([self.normals, self.offsets.unsqueeze(-1)], dim=-1)

    def to(self, device: torch.device | str) -> "PolygonConstraint":
        interior = None if self.interior is None else self.interior.to(device)
        return PolygonConstraint(self.normals.to(device), self.offsets.to(device), interior)


class BumpTarget(Target):
    """Truncated-exponential background plus a truncated-Gaussian signal on ``[0, L]^2``."""

    dim = 2

    def __init__(self, domain: float = BUMP_DOMAIN,
                 background_scales: tuple[float, float] = BUMP_BACKGROUND_SCALES,
                 signal_mean: tuple[float, float] = BUMP_SIGNAL_MEAN,
                 signal_sigma: float | tuple[float, float] = BUMP_SIGNAL_SIGMA,
                 signal_weight: float = BUMP_SIGNAL_WEIGHT):
        self.domain = float(domain)
        self.background_scales = _as_pair(background_scales)
        self.signal_mean = _as_pair(signal_mean)
        self.signal_sigma = _as_pair(signal_sigma)
        self.signal_weight = float(signal_weight)

    def _params(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kw = {"device": device, "dtype": dtype}
        return (torch.tensor(self.background_scales, **kw),
                torch.tensor(self.signal_mean, **kw),
                torch.tensor(self.signal_sigma, **kw))

    def in_domain(self, x: torch.Tensor) -> torch.Tensor:
        return ((x >= 0.0) & (x <= self.domain)).all(dim=-1)

    def _log_background(self, x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        log_norm = torch.log(beta) + torch.log(-torch.expm1(-self.domain / beta))
        return (-x / beta - log_norm).sum(dim=-1)

    def _log_signal(self, x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
        z = (x - mu) / sd
        mass = (torch.special.ndtr((self.domain - mu) / sd)
                - torch.special.ndtr(-mu / sd))
        return (-0.5 * z ** 2 - torch.log(sd) - 0.5 * _LOG_2PI - torch.log(mass)).sum(dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        beta, mu, sd = self._params(x.device, x.dtype)
        mixture = torch.logaddexp(
            math.log1p(-self.signal_weight) + self._log_background(x, beta),
            math.log(self.signal_weight) + self._log_signal(x, mu, sd))
        return torch.where(self.in_domain(x), mixture, torch.full_like(mixture, -math.inf))

    def marginal_components(self, t: torch.Tensor, axis: int = 0
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(log p_bg(x_axis), log p_sig(x_axis))``, each normalised on ``[0, L]`` separately.

        Both components are products over axes, so marginalising is exact and costs nothing.
        """
        beta, mu, sd = self._params(t.device, t.dtype)
        beta, mu, sd = beta[axis], mu[axis], sd[axis]
        log_bg = -t / beta - torch.log(beta) - torch.log(-torch.expm1(-self.domain / beta))
        mass = torch.special.ndtr((self.domain - mu) / sd) - torch.special.ndtr(-mu / sd)
        log_sig = (-0.5 * ((t - mu) / sd) ** 2 - torch.log(sd) - 0.5 * _LOG_2PI
                   - torch.log(mass))
        outside = ((t < 0.0) | (t > self.domain))
        fill = torch.full_like(t, -math.inf)
        return torch.where(outside, fill, log_bg), torch.where(outside, fill, log_sig)

    def marginal_log_prob(self, t: torch.Tensor, axis: int = 0) -> torch.Tensor:
        """``log p(x_axis)`` after integrating the other coordinate out analytically."""
        log_bg, log_sig = self.marginal_components(t, axis)
        return torch.logaddexp(math.log1p(-self.signal_weight) + log_bg,
                               math.log(self.signal_weight) + log_sig)

    def sample(self, num_points: int, device: torch.device | str | None = None) -> torch.Tensor:
        dtype = torch.get_default_dtype()
        beta, mu, sd = self._params(device, dtype)

        u = torch.rand(num_points, self.dim, device=device, dtype=dtype)
        background = -beta * torch.log1p(-u * (-torch.expm1(-self.domain / beta)))

        lo = torch.special.ndtr(-mu / sd)
        hi = torch.special.ndtr((self.domain - mu) / sd)
        v = torch.rand(num_points, self.dim, device=device, dtype=dtype)
        signal = mu + sd * torch.special.ndtri(lo + v * (hi - lo))

        pick = torch.rand(num_points, 1, device=device, dtype=dtype) < self.signal_weight
        return torch.where(pick, signal, background).clamp(0.0, self.domain)

    def mean_std(self, device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact per-axis mean and standard deviation of the mixture.

        Analytic rather than estimated from a pool so the normalising frame is identical on
        every device and every run, which a Monte Carlo estimate would not guarantee.
        """
        beta, mu, sd = self._params(device, dtype or torch.get_default_dtype())
        length = self.domain

        tail = torch.exp(-length / beta)
        norm = -torch.expm1(-length / beta)
        mean_bg = beta - length * tail / norm
        second_bg = 2.0 * beta ** 2 - (length ** 2 + 2.0 * length * beta) * tail / norm

        a, b = -mu / sd, (length - mu) / sd
        pdf_a, pdf_b = _INV_SQRT_2PI * torch.exp(-0.5 * a ** 2), _INV_SQRT_2PI * torch.exp(-0.5 * b ** 2)
        mass = torch.special.ndtr(b) - torch.special.ndtr(a)
        ratio = (pdf_a - pdf_b) / mass
        mean_sig = mu + sd * ratio
        var_sig = sd ** 2 * (1.0 + (a * pdf_a - b * pdf_b) / mass - ratio ** 2)

        weight = self.signal_weight
        mean = (1.0 - weight) * mean_bg + weight * mean_sig
        second = (1.0 - weight) * second_bg + weight * (var_sig + mean_sig ** 2)
        return mean, (second - mean ** 2).clamp_min(1e-12).sqrt()


def propose_polygons(num_candidates: int, domain: float = BUMP_DOMAIN,
                     min_vertices: int = BUMP_POLY_MIN_VERTICES,
                     max_vertices: int = BUMP_POLY_MAX_VERTICES,
                     radius_range: tuple[float, float] = BUMP_POLY_RADIUS_RANGE,
                     device: torch.device | str | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random convex polygons as ``(normals, offsets, active)`` of shapes (N, K, 2), (N, K), (N, K).

    Each polygon is the intersection of ``k ~ U{min, max}`` half-planes whose outward normals
    are angularly stratified, plus the four faces of the domain box. Stratifying the angles
    matters: ``k`` independent uniform directions frequently leave a gap wider than ``pi``,
    which makes the intersection unbounded, and the box faces would then be the only thing
    closing the region and every such polygon would look like a clipped wedge.
    """
    pad = max_vertices
    centres = torch.rand(num_candidates, 1, 2, device=device) * domain
    counts = torch.randint(min_vertices, max_vertices + 1, (num_candidates, 1), device=device)

    index = torch.arange(pad, device=device).unsqueeze(0)
    angle = 2.0 * math.pi * (index + torch.rand(num_candidates, pad, device=device)) / counts
    normals = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    radii = radius_range[0] + torch.rand(num_candidates, pad, device=device) * (
        radius_range[1] - radius_range[0])
    offsets = (normals * centres).sum(dim=-1) + radii
    active = index < counts

    box_normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=device)
    box_offsets = torch.tensor([domain, 0.0, domain, 0.0], device=device)
    normals = torch.cat([normals, box_normals.expand(num_candidates, -1, -1)], dim=1)
    offsets = torch.cat([offsets, box_offsets.expand(num_candidates, -1)], dim=1)
    active = torch.cat([active, torch.ones(num_candidates, 4, dtype=torch.bool, device=device)],
                       dim=1)
    return normals, offsets, active


def polygon_mass(normals: torch.Tensor, offsets: torch.Tensor, active: torch.Tensor,
                 pool: torch.Tensor, chunk_size: int = 200_000) -> torch.Tensor:
    """Fraction of *pool* inside each candidate polygon; (N, K, 2) and (M, 2) -> (N,).

    Folds the half-planes in one at a time and streams the pool. Contracting all three axes
    at once would materialise an (N, K, M) residual, which is hundreds of gigabytes at the
    pool sizes used to resolve a 1% signal.
    """
    counts = torch.zeros(normals.shape[0], device=pool.device, dtype=torch.long)
    for start in range(0, pool.shape[0], chunk_size):
        block = pool[start:start + chunk_size]
        inside = torch.ones(normals.shape[0], block.shape[0], dtype=torch.bool,
                            device=pool.device)
        for k in range(normals.shape[1]):
            satisfied = (normals[:, k, :] @ block.T) <= offsets[:, k].unsqueeze(-1)
            inside &= satisfied | ~active[:, k].unsqueeze(-1)
        counts += inside.sum(dim=1)
    return counts.to(pool.dtype) / pool.shape[0]


def sample_polygons(num_constraints: int, pool: torch.Tensor,
                    domain: float = BUMP_DOMAIN,
                    min_vertices: int = BUMP_POLY_MIN_VERTICES,
                    max_vertices: int = BUMP_POLY_MAX_VERTICES,
                    radius_range: tuple[float, float] = BUMP_POLY_RADIUS_RANGE,
                    min_mass: float = BUMP_POLY_MIN_MASS,
                    max_mass: float = BUMP_POLY_MAX_MASS,
                    batch_size: int = 256, max_rounds: int = 1000
                    ) -> tuple[list[PolygonConstraint], torch.Tensor]:
    """Polygons whose target mass, estimated on *pool*, lies in ``[min_mass, max_mass]``.

    Each kept polygon also carries the deepest pool point inside it, a discrete stand-in for
    the Chebyshev centre. The projection needs a reference that is strictly interior and, for
    a sliver, as far from every face as the pool allows; the feasible centroid would sit much
    closer to a wall.
    """
    device = pool.device
    kept: list[PolygonConstraint] = []
    masses: list[float] = []

    for _ in range(max_rounds):
        if len(kept) >= num_constraints:
            break
        normals, offsets, active = propose_polygons(batch_size, domain, min_vertices,
                                                    max_vertices, radius_range, device)
        normals, offsets = normals.to(pool.dtype), offsets.to(pool.dtype)
        mass = polygon_mass(normals, offsets, active, pool)

        keep = ((mass >= min_mass) & (mass <= max_mass)).nonzero(as_tuple=True)[0]
        for i in keep.tolist():
            if len(kept) >= num_constraints:
                break
            rows = active[i]
            faces, shifts = normals[i][rows].clone(), offsets[i][rows].clone()
            deepest = (pool @ faces.T - shifts).amax(dim=-1).argmin()
            kept.append(PolygonConstraint(faces, shifts, pool[deepest].clone()))
            masses.append(float(mass[i]))

    if len(kept) < num_constraints:
        raise RuntimeError(f"only {len(kept)}/{num_constraints} polygons fell in "
                           f"[{min_mass}, {max_mass}] after {max_rounds} rounds")
    return kept, torch.tensor(masses, dtype=torch.float64)


class BumpProblem(Problem):
    name = PROBLEM_NAME
    dim = 2

    def __init__(self, domain: float = BUMP_DOMAIN,
                 background_scales: tuple[float, float] = BUMP_BACKGROUND_SCALES,
                 signal_mean: tuple[float, float] = BUMP_SIGNAL_MEAN,
                 signal_sigma: float | tuple[float, float] = BUMP_SIGNAL_SIGMA,
                 signal_weight: float = BUMP_SIGNAL_WEIGHT,
                 min_mass: float = BUMP_POLY_MIN_MASS,
                 max_mass: float = BUMP_POLY_MAX_MASS,
                 mass_pool_size: int = 200_000):
        self.domain = float(domain)
        self.min_mass = float(min_mass)
        self.max_mass = float(max_mass)
        self.mass_pool_size = int(mass_pool_size)
        self._target = BumpTarget(domain, background_scales, signal_mean, signal_sigma,
                                  signal_weight)

    def target(self) -> BumpTarget:
        return self._target

    def sample_constraints(self, num_constraints: int,
                           device: torch.device | str | None = None) -> list[PolygonConstraint]:
        pool = self._target.sample(self.mass_pool_size, device=device)
        constraints, _ = sample_polygons(num_constraints, pool, domain=self.domain,
                                         min_mass=self.min_mass, max_mass=self.max_mass)
        return constraints

    def normalizer(self) -> AffineNormalizer:
        mean, std = self._target.mean_std()
        return AffineNormalizer(mean, std)


__all__ = ["PROBLEM_NAME", "PolygonConstraint", "BumpTarget", "BumpProblem",
           "propose_polygons", "sample_polygons", "polygon_mass"]
