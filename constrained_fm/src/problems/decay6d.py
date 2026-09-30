# -*- coding: utf-8 -*-
r"""Toy two-body decay in 6D Cartesian momenta, constrained by an axis-aligned box on particle 1.

Generative process, all in float64:

.. math::
    M \sim \mathcal N(m_0, \sigma_m^2),\quad \vec P \sim \mathcal N(0, \sigma_p^2 I),\quad
    u \sim \mathcal U(u_{lo}, u_{hi}),\quad \hat v \sim \mathcal U(\mathbb S^2),

    \vec p_1 = u (M \hat v + \vec P),\qquad \vec p_2 = (1 - u)(-M \hat v + \vec P).

For fixed ``u`` the map ``(\vec a = M \hat v, \vec P) \to (\vec p_1, \vec p_2)`` is linear with
``|J| = 8 u^3 (1-u)^3``, so the exact density is a 1D integral over the split:

.. math::
    p(x) = \frac{1}{u_{hi} - u_{lo}} \int_{u_{lo}}^{u_{hi}}
           \frac{p_a(\vec a_u)\, \mathcal N(\vec P_u; 0, \sigma_p^2 I)}{8 u^3 (1-u)^3}\, du,
    \qquad p_a(\vec a) = \frac{\mathcal N(r; m_0, \sigma_m^2) + \mathcal N(-r; m_0, \sigma_m^2)}
                              {4 \pi r^2},

with ``\vec a_u = (\vec p_1/u - \vec p_2/(1-u))/2``, ``\vec P_u = (\vec p_1/u + \vec p_2/(1-u))/2``
and ``r = \|\vec a_u\|``.
"""

from __future__ import annotations

import itertools
import math

import torch

from constrained_fm.src.consts import (DECAY_BEAM_SIGMA, DECAY_BOX_HALF_WIDTH_RANGE,
                                       DECAY_BOX_MASS_RANGE, DECAY_MASS_MEAN, DECAY_MASS_SIGMA,
                                       DECAY_MASS_TABLE_BINS, DECAY_MASS_TABLE_EXTENT,
                                       DECAY_MASS_TABLE_POOL, DECAY_MASS_TABLE_SEED,
                                       DECAY_QUADRATURE_NODES, DECAY_SPLIT_RANGE)
from constrained_fm.src.problems.base import AffineNormalizer, Constraint, Problem, Target

PROBLEM_NAME = "decay6d"
PARTICLE_DIM = 3

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_4PI = math.log(4.0 * math.pi)
_LOG_8 = math.log(8.0)
_RADIUS_FLOOR = 1e-12
_TABLE_CHUNK = 1_000_000


def _normal_log_pdf(x: torch.Tensor, mean: float, sigma: float) -> torch.Tensor:
    return -0.5 * ((x - mean) / sigma) ** 2 - math.log(sigma) - 0.5 * _LOG_2PI


def trapezoid_nodes(lo: float, hi: float, num_nodes: int, device: torch.device | str | None,
                    dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes on ``[lo, hi]`` and log trapezoid weights that integrate against ``U(lo, hi)``."""
    nodes = torch.linspace(lo, hi, num_nodes, device=device, dtype=dtype)
    weights = torch.full_like(nodes, 1.0 / (num_nodes - 1))
    weights[0] = weights[-1] = 0.5 / (num_nodes - 1)
    return nodes, weights.log()


class DecayTarget(Target):
    """``x = (\\vec p_1, \\vec p_2)`` from the toy decay, with an exact quadrature density."""

    dim = 6

    def __init__(self, mass_mean: float = DECAY_MASS_MEAN, mass_sigma: float = DECAY_MASS_SIGMA,
                 beam_sigma: float = DECAY_BEAM_SIGMA,
                 split_range: tuple[float, float] = DECAY_SPLIT_RANGE,
                 quadrature_nodes: int = DECAY_QUADRATURE_NODES):
        self.mass_mean = float(mass_mean)
        self.mass_sigma = float(mass_sigma)
        self.beam_sigma = float(beam_sigma)
        self.split_lo, self.split_hi = float(split_range[0]), float(split_range[1])
        self.quadrature_nodes = int(quadrature_nodes)

    def sample_with_split(self, num_points: int, device: torch.device | str | None = None,
                          generator: torch.Generator | None = None
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x, u)``: (N, 6) float64 momenta and the (N,) split that produced them."""
        opts = {"device": device, "dtype": torch.float64, "generator": generator}
        mass = self.mass_mean + self.mass_sigma * torch.randn(num_points, 1, **opts)
        beam = self.beam_sigma * torch.randn(num_points, PARTICLE_DIM, **opts)
        split = self.split_lo + (self.split_hi - self.split_lo) * torch.rand(num_points, 1, **opts)
        direction = torch.randn(num_points, PARTICLE_DIM, **opts)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        p1 = split * (mass * direction + beam)
        p2 = (1.0 - split) * (-mass * direction + beam)
        return torch.cat([p1, p2], dim=-1), split.squeeze(-1)

    def sample(self, num_points: int, device: torch.device | str | None = None,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """(N, 6) float64."""
        return self.sample_with_split(num_points, device, generator)[0]

    def log_joint_split(self, x: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """``log p(x | u)`` for every row of ``x`` (B, 6) at every node ``u`` (G,) -> (B, G)."""
        u = nodes[None, :, None]
        scaled_1 = x[:, None, :PARTICLE_DIM] / u
        scaled_2 = x[:, None, PARTICLE_DIM:] / (1.0 - u)
        a = 0.5 * (scaled_1 - scaled_2)
        beam = 0.5 * (scaled_1 + scaled_2)

        r = a.norm(dim=-1).clamp_min(_RADIUS_FLOOR)
        log_radial = torch.logaddexp(_normal_log_pdf(r, self.mass_mean, self.mass_sigma),
                                     _normal_log_pdf(-r, self.mass_mean, self.mass_sigma))
        log_a = log_radial - _LOG_4PI - 2.0 * r.log()
        log_beam = (-0.5 * beam.pow(2).sum(dim=-1) / self.beam_sigma ** 2
                    - PARTICLE_DIM * (math.log(self.beam_sigma) + 0.5 * _LOG_2PI))
        log_jac = _LOG_8 + PARTICLE_DIM * (nodes.log() + torch.log1p(-nodes))
        return log_a + log_beam - log_jac[None, :]

    def log_prob(self, x: torch.Tensor, chunk_size: int = 1024,
                 num_nodes: int | None = None) -> torch.Tensor:
        """Exact ``log p(x)`` in physical units; (N, 6) -> (N,) float64."""
        x = x.to(torch.float64)
        nodes, log_weights = trapezoid_nodes(self.split_lo, self.split_hi,
                                             num_nodes or self.quadrature_nodes, x.device)
        parts = [torch.logsumexp(self.log_joint_split(chunk, nodes) + log_weights, dim=1)
                 for chunk in x.split(chunk_size)]
        return torch.cat(parts) if parts else x.new_zeros(0)

    def _split_moment(self, reflect: bool) -> float:
        """``E[u^2]``, or ``E[(1-u)^2]`` when ``reflect``, for ``u ~ U(lo, hi)``."""
        lo, hi = (1.0 - self.split_hi, 1.0 - self.split_lo) if reflect else (self.split_lo,
                                                                               self.split_hi)
        return (hi ** 3 - lo ** 3) / (3.0 * (hi - lo))

    def mean_std(self, device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Analytic moments: mean 0, ``Var(p_{1,k}) = E[u^2](E[M^2] + 3\sigma_p^2)/3``."""
        rest = self.mass_mean ** 2 + self.mass_sigma ** 2 + PARTICLE_DIM * self.beam_sigma ** 2
        std_1 = math.sqrt(self._split_moment(False) * rest / PARTICLE_DIM)
        std_2 = math.sqrt(self._split_moment(True) * rest / PARTICLE_DIM)
        std = torch.tensor([std_1] * PARTICLE_DIM + [std_2] * PARTICLE_DIM,
                           device=device, dtype=dtype)
        return torch.zeros(self.dim, device=device, dtype=dtype), std


def boxes_contain(p1: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Elementwise containment, so no TF32 matmul can misclassify an edge sample; (..., 3)."""
    return ((p1 >= lo) & (p1 <= hi)).all(dim=-1)


OBSERVABLE_NAMES = ("p2_norm", "p2_z", "p2_tail")


def observables(x: torch.Tensor, tail_threshold: float) -> torch.Tensor:
    """``(|p2|, p2_z, 1[|p2| > tau])`` in physical units; (N, 6) -> (N, 3)."""
    p2 = x[:, PARTICLE_DIM:]
    norm = p2.norm(dim=-1)
    return torch.stack([norm, p2[:, -1], (norm > tail_threshold).to(x.dtype)], dim=-1)


def box_conditioning(centre: torch.Tensor, half_width: torch.Tensor) -> torch.Tensor:
    """``(centre, log half-width)`` in the normalized frame; (..., 3) x2 -> (..., 6)."""
    return torch.cat([centre, half_width.log()], dim=-1)


class BoxConstraint(Constraint):
    """``{x : lo <= \\vec p_1 <= hi}`` in physical units; ``\\vec p_2`` is unconstrained."""

    dim = 6

    def __init__(self, lo: torch.Tensor | list[float], hi: torch.Tensor | list[float]):
        self.lo = torch.as_tensor(lo, dtype=torch.float64)
        self.hi = torch.as_tensor(hi, dtype=torch.float64)

    @classmethod
    def from_centre(cls, centre: torch.Tensor | list[float],
                    half_width: torch.Tensor | list[float]) -> "BoxConstraint":
        centre = torch.as_tensor(centre, dtype=torch.float64)
        half_width = torch.as_tensor(half_width, dtype=torch.float64)
        return cls(centre - half_width, centre + half_width)

    @property
    def centre(self) -> torch.Tensor:
        return 0.5 * (self.lo + self.hi)

    @property
    def half_width(self) -> torch.Tensor:
        return 0.5 * (self.hi - self.lo)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        p1 = x[..., :PARTICLE_DIM]
        lo, hi = self.lo.to(x), self.hi.to(x)
        return torch.maximum(lo - p1, p1 - hi).amax(dim=-1)

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return boxes_contain(x[..., :PARTICLE_DIM], self.lo.to(x), self.hi.to(x))

    @property
    def params(self) -> torch.Tensor:
        return torch.cat([self.centre, self.half_width])

    def conditioning(self, normalizer: AffineNormalizer) -> torch.Tensor:
        """The (6,) network conditioning vector in the normalizer's frame and dtype."""
        mean = normalizer.mean[:PARTICLE_DIM]
        std = normalizer.std[:PARTICLE_DIM]
        centre = (self.centre.to(std) - mean) / std
        return box_conditioning(centre, self.half_width.to(std) / std)


def sample_anchored_boxes(p1: torch.Tensor,
                          half_width_range: tuple[float, float] = DECAY_BOX_HALF_WIDTH_RANGE,
                          generator: torch.Generator | None = None
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """One box per row of ``p1``, placed so ``p1`` is uniform inside it.

    The centre's density given ``p1`` is ``1[p1 in B] / vol(B)``, so it depends on ``x`` only through
    the indicator and the pairs ``(x, B)`` have conditional ``p(x | B) \\propto p(x) 1[p1 in B]``.
    """
    opts = {"device": p1.device, "dtype": p1.dtype, "generator": generator}
    log_lo, log_hi = math.log(half_width_range[0]), math.log(half_width_range[1])
    half_width = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(p1.shape, **opts))
    offset = torch.rand(p1.shape, **opts)
    return p1 + half_width * (1.0 - 2.0 * offset), half_width


class BoxMassTable:
    """Piecewise-linear 3D CDF of normalized ``\\vec p_1``: an approximate ``P(B)`` for the filter.

    Any deterministic function of ``B`` keeps the anchored scheme exact, so the approximation only
    moves which boxes are accepted, never the conditional the network learns.
    """

    def __init__(self, cdf: torch.Tensor, extent: float):
        self.cdf = cdf
        self.extent = float(extent)
        self.bins = cdf.shape[0] - 1

    @classmethod
    def from_target(cls, target: DecayTarget, normalizer: AffineNormalizer,
                    pool_size: int = DECAY_MASS_TABLE_POOL, bins: int = DECAY_MASS_TABLE_BINS,
                    extent: float = DECAY_MASS_TABLE_EXTENT, seed: int = DECAY_MASS_TABLE_SEED,
                    device: torch.device | str | None = None) -> "BoxMassTable":
        generator = torch.Generator(device=device).manual_seed(seed)
        counts = torch.zeros(bins ** 3, device=device, dtype=torch.float64)
        for start in range(0, pool_size, _TABLE_CHUNK):
            n = min(_TABLE_CHUNK, pool_size - start)
            p1 = normalizer.forward(target.sample(n, device, generator))[:, :PARTICLE_DIM]
            idx = ((p1 + extent) / (2.0 * extent) * bins).floor().long().clamp(0, bins - 1)
            flat = (idx[:, 0] * bins + idx[:, 1]) * bins + idx[:, 2]
            counts += torch.bincount(flat, minlength=bins ** 3).to(torch.float64)

        cdf = torch.zeros((bins + 1,) * 3, device=device, dtype=torch.float64)
        cdf[1:, 1:, 1:] = (counts.view(bins, bins, bins) / pool_size).cumsum(0).cumsum(1).cumsum(2)
        return cls(cdf, extent)

    def cdf_at(self, q: torch.Tensor) -> torch.Tensor:
        """Trilinear ``F(q) = P(\\vec p_1 <= q)``; (B, 3) -> (B,)."""
        scaled = ((q.to(self.cdf) + self.extent) / (2.0 * self.extent) * self.bins)
        scaled = scaled.clamp(0.0, float(self.bins))
        base = scaled.floor().long().clamp(max=self.bins - 1)
        frac = scaled - base

        total = torch.zeros(q.shape[0], device=q.device, dtype=self.cdf.dtype)
        for corner in itertools.product((0, 1), repeat=PARTICLE_DIM):
            step = torch.tensor(corner, device=q.device)
            idx = base + step
            weight = torch.where(step.bool(), frac, 1.0 - frac).prod(dim=-1)
            total = total + weight * self.cdf[idx[:, 0], idx[:, 1], idx[:, 2]]
        return total

    def mass(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """``P(lo <= \\vec p_1 <= hi)`` by inclusion-exclusion over the 8 corners; (B,)."""
        total = torch.zeros(lo.shape[0], device=lo.device, dtype=self.cdf.dtype)
        for corner in itertools.product((0, 1), repeat=PARTICLE_DIM):
            pick_hi = torch.tensor(corner, device=lo.device, dtype=torch.bool)
            sign = (-1.0) ** (PARTICLE_DIM - sum(corner))
            total = total + sign * self.cdf_at(torch.where(pick_hi, hi, lo))
        return total.clamp(0.0, 1.0)


def sample_conditioned_batch(target: DecayTarget, normalizer: AffineNormalizer,
                             table: BoxMassTable, batch_size: int,
                             half_width_range: tuple[float, float] = DECAY_BOX_HALF_WIDTH_RANGE,
                             mass_range: tuple[float, float] = DECAY_BOX_MASS_RANGE,
                             device: torch.device | str | None = None,
                             generator: torch.Generator | None = None, oversample: float = 2.0
                             ) -> tuple[torch.Tensor, torch.Tensor, float]:
    """``(x_n, box_n, acceptance)``: normalized samples of ``p(x | B)`` paired with their boxes.

    ``box_n`` is ``(centre, log half-width)``; the filter keeps boxes with table mass in
    ``mass_range``, a function of ``B`` alone, so the retained pairs stay exact.
    """
    xs, boxes = [], []
    drawn = accepted = 0
    while accepted < batch_size:
        n = int(oversample * batch_size)
        x = normalizer.forward(target.sample(n, device, generator))
        centre, half_width = sample_anchored_boxes(x[:, :PARTICLE_DIM], half_width_range, generator)
        mass = table.mass(centre - half_width, centre + half_width)
        keep = (mass >= mass_range[0]) & (mass <= mass_range[1])

        xs.append(x[keep])
        boxes.append(box_conditioning(centre[keep], half_width[keep]))
        drawn += n
        accepted += int(keep.sum())

    return (torch.cat(xs)[:batch_size], torch.cat(boxes)[:batch_size], accepted / drawn)


def conditioning_to_box(box_n: torch.Tensor, normalizer: AffineNormalizer) -> BoxConstraint:
    """Inverse of :meth:`BoxConstraint.conditioning` for one (6,) vector."""
    mean = normalizer.mean[:PARTICLE_DIM].to(box_n)
    std = normalizer.std[:PARTICLE_DIM].to(box_n)
    centre = box_n[:PARTICLE_DIM] * std + mean
    half_width = box_n[PARTICLE_DIM:].exp() * std
    return BoxConstraint.from_centre(centre.cpu(), half_width.cpu())


class DecayProblem(Problem):
    name = PROBLEM_NAME
    dim = 6

    def __init__(self, half_width_range: tuple[float, float] = DECAY_BOX_HALF_WIDTH_RANGE,
                 mass_range: tuple[float, float] = DECAY_BOX_MASS_RANGE, **target_kwargs):
        self.half_width_range = tuple(half_width_range)
        self.mass_range = tuple(mass_range)
        self._target = DecayTarget(**target_kwargs)

    def target(self) -> DecayTarget:
        return self._target

    def normalizer(self, dtype: torch.dtype = torch.float32) -> AffineNormalizer:
        mean, std = self._target.mean_std(dtype=dtype)
        return AffineNormalizer(mean, std)

    def mass_table(self, device: torch.device | str | None = None, **kwargs) -> BoxMassTable:
        return BoxMassTable.from_target(self._target, self.normalizer(torch.float64).to(device),
                                        device=device, **kwargs)

    def sample_constraints(self, num_constraints: int,
                           device: torch.device | str | None = None) -> list[BoxConstraint]:
        normalizer = self.normalizer(torch.float64).to(device)
        _, boxes, _ = sample_conditioned_batch(self._target, normalizer, self.mass_table(device),
                                               num_constraints, self.half_width_range,
                                               self.mass_range, device)
        return [conditioning_to_box(b, normalizer) for b in boxes]


__all__ = ["PROBLEM_NAME", "PARTICLE_DIM", "OBSERVABLE_NAMES", "DecayTarget", "DecayProblem",
           "BoxConstraint", "BoxMassTable", "boxes_contain", "box_conditioning",
           "conditioning_to_box", "observables", "sample_anchored_boxes",
           "sample_conditioned_batch", "trapezoid_nodes"]
