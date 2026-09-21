# -*- coding: utf-8 -*-
r"""Two massless particles in 6D Cartesian momenta, constrained by their pair invariant mass.

Each particle is generated in the coordinates the physics factorises in,

.. math::
    p_T \sim \mathrm{Exp}_{[10, 500]}(\beta),\quad
    \eta \sim \mathcal{N}_{[-3, 3]}(0, \sigma^2),\quad
    \phi \sim \mathcal{U}[-\pi, \pi],

and mapped to ``(p_x, p_y, p_z) = (p_T \cos\phi, p_T \sin\phi, p_T \sinh\eta)``. That map is a
diffeomorphism with ``|J| = p_T^2 \cosh\eta``, so the Cartesian density is exact rather than
estimated and NLL/KLD stay meaningful in 6D:

.. math::
    \log p(x) = \sum_i \big[\log p(p_{T,i}) + \log p(\eta_i) + \log p(\phi_i)
                            - 2\log p_{T,i} - \log\cosh\eta_i\big].

The feasible set is a mass shell ``{x : ||M(x) - M_\mathrm{target}| - \epsilon \le 0}``, which is
**not convex** -- it is the region between two nested surfaces. Two consequences are load
bearing: the constraint must not advertise an ``interior_point`` (the bisection fallback in
:mod:`constraint_projection` assumes a single boundary crossing), and ECI needs a damped
Newton step so a full step cannot jump the shell and land on the opposite wall.
"""

from __future__ import annotations

import math

import torch

from constrained_fm.src.consts import (KIN_ETA_RANGE, KIN_ETA_SIGMA, KIN_MASS_FLOOR,
                                       KIN_PT_RANGE, KIN_PT_SCALE, KIN_SHELL_MASS_BINS,
                                       KIN_SHELL_MAX_MASS, KIN_SHELL_MIN_MASS)
from constrained_fm.src.problems.base import AffineNormalizer, Constraint, Problem, Target

PROBLEM_NAME = "kinematics6d"

_LOG_2PI = math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


def _ndtr(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def _truncated_exponential_moments(lo: float, hi: float, beta: float) -> tuple[float, float]:
    r"""``(E[t], E[t^2])`` for a density ``\propto e^{-t/\beta}`` on ``[lo, hi]``.

    From ``\int t^k e^{-t/\beta} dt`` in closed form, so the normalising frame never depends on
    a sampled pool and is identical on every device.
    """
    e_lo, e_hi = math.exp(-lo / beta), math.exp(-hi / beta)
    norm = e_lo - e_hi
    first = ((lo + beta) * e_lo - (hi + beta) * e_hi) / norm
    second = ((lo * lo + 2.0 * beta * lo + 2.0 * beta * beta) * e_lo
              - (hi * hi + 2.0 * beta * hi + 2.0 * beta * beta) * e_hi) / norm
    return first, second


def _truncated_normal_mgf(t: float, sigma: float, lo: float, hi: float) -> float:
    r"""``E[e^{t\eta}]`` for ``\eta \sim \mathcal{N}_{[lo, hi]}(0, \sigma^2)``."""
    shift = t * sigma
    num = _ndtr(hi / sigma - shift) - _ndtr(lo / sigma - shift)
    den = _ndtr(hi / sigma) - _ndtr(lo / sigma)
    return math.exp(0.5 * sigma * sigma * t * t) * num / den


class KinematicsTarget(Target):
    """Two independent massless particles, reported as stacked Cartesian momenta."""

    dim = 6

    def __init__(self, pt_range: tuple[float, float] = KIN_PT_RANGE,
                 pt_scale: float = KIN_PT_SCALE,
                 eta_range: tuple[float, float] = KIN_ETA_RANGE,
                 eta_sigma: float = KIN_ETA_SIGMA):
        self.pt_lo, self.pt_hi = float(pt_range[0]), float(pt_range[1])
        self.pt_scale = float(pt_scale)
        self.eta_lo, self.eta_hi = float(eta_range[0]), float(eta_range[1])
        self.eta_sigma = float(eta_sigma)

        self._pt_norm = (math.exp(-self.pt_lo / self.pt_scale)
                         - math.exp(-self.pt_hi / self.pt_scale))
        self._eta_norm = (_ndtr(self.eta_hi / self.eta_sigma)
                          - _ndtr(self.eta_lo / self.eta_sigma))

    def to_cartesian(self, pt: torch.Tensor, eta: torch.Tensor,
                     phi: torch.Tensor) -> torch.Tensor:
        return torch.stack([pt * torch.cos(phi), pt * torch.sin(phi), pt * torch.sinh(eta)],
                           dim=-1)

    def to_spherical(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inverse of :meth:`to_cartesian`; ``(..., 3)`` -> three ``(...,)`` tensors."""
        pt = torch.linalg.norm(p[..., :2], dim=-1).clamp_min(KIN_MASS_FLOOR)
        return pt, torch.asinh(p[..., 2] / pt), torch.atan2(p[..., 1], p[..., 0])

    def in_support(self, x: torch.Tensor) -> torch.Tensor:
        pt, eta, _ = self.to_spherical(x.view(*x.shape[:-1], 2, 3))
        inside = ((pt >= self.pt_lo) & (pt <= self.pt_hi)
                  & (eta >= self.eta_lo) & (eta <= self.eta_hi))
        return inside.all(dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        pairs = x.view(*x.shape[:-1], 2, 3)
        pt, eta, _ = self.to_spherical(pairs)

        log_pt = -pt / self.pt_scale - math.log(self.pt_scale * self._pt_norm)
        log_eta = (-0.5 * (eta / self.eta_sigma) ** 2
                   - 0.5 * _LOG_2PI - math.log(self.eta_sigma * self._eta_norm))
        log_phi = -math.log(2.0 * math.pi)
        log_jac = 2.0 * torch.log(pt) + torch.log(torch.cosh(eta))

        total = (log_pt + log_eta + log_phi - log_jac).sum(dim=-1)
        return torch.where(self.in_support(x), total, torch.full_like(total, float("-inf")))

    def sample(self, num_points: int, device: torch.device | str | None = None) -> torch.Tensor:
        u = torch.rand(num_points, 2, 3, device=device, dtype=torch.float64)

        e_lo = math.exp(-self.pt_lo / self.pt_scale)
        pt = -self.pt_scale * torch.log(e_lo - u[..., 0] * self._pt_norm)

        lo_cdf = _ndtr(self.eta_lo / self.eta_sigma)
        eta = self.eta_sigma * torch.special.ndtri(lo_cdf + u[..., 1] * self._eta_norm)
        phi = (2.0 * u[..., 2] - 1.0) * math.pi

        pairs = self.to_cartesian(pt, eta, phi)
        return pairs.flatten(start_dim=-2).to(torch.float32)

    def invariant_mass(self, x: torch.Tensor) -> torch.Tensor:
        r"""``M = \sqrt{2 p_{T,1} p_{T,2} (\cosh\Delta\eta - \cos\Delta\phi)}``; (N, 6) -> (N,).

        Evaluated as ``2(E_1 E_2 - \vec p_1 \cdot \vec p_2)`` with ``E_i = \|\vec p_i\|``, the
        massless limit, so no particle mass ever has to be carried alongside the state.
        """
        p1, p2 = x[..., :3], x[..., 3:]
        energies = torch.linalg.norm(p1, dim=-1) * torch.linalg.norm(p2, dim=-1)
        m_squared = 2.0 * (energies - (p1 * p2).sum(dim=-1))
        return torch.sqrt(m_squared.clamp_min(KIN_MASS_FLOOR))

    def mass_scale(self) -> float:
        r"""``\sqrt{E[M^2]} = \sqrt{2} E[p_T] E[\cosh\eta]``, the conditioning unit for M.

        Exact because ``M^2`` factorises over the two particles: the cross terms are
        ``E[\sinh\eta]^2 = 0`` and ``E[\cos\phi]^2 + E[\sin\phi]^2 = 0`` by symmetry. Used
        instead of ``std(M)``, which has no closed form, so the conditioning scale is a fixed
        analytic constant rather than something measured off a pool.
        """
        mean_pt, _ = _truncated_exponential_moments(self.pt_lo, self.pt_hi, self.pt_scale)
        cosh_eta = _truncated_normal_mgf(1.0, self.eta_sigma, self.eta_lo, self.eta_hi)
        return _SQRT2 * mean_pt * cosh_eta

    def mean_std(self, device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Analytic per-axis moments of the Cartesian state.

        Every component is symmetric about zero, so only the spread matters:
        ``Var(p_x) = Var(p_y) = E[p_T^2] / 2`` and ``Var(p_z) = E[p_T^2] E[\sinh^2\eta]``, with
        ``E[\sinh^2\eta] = (E[\cosh 2\eta] - 1) / 2`` from the truncated-normal MGF.
        """
        _, second_pt = _truncated_exponential_moments(self.pt_lo, self.pt_hi, self.pt_scale)
        sinh_sq = 0.5 * (_truncated_normal_mgf(2.0, self.eta_sigma,
                                               self.eta_lo, self.eta_hi) - 1.0)

        transverse = math.sqrt(0.5 * second_pt)
        longitudinal = math.sqrt(second_pt * sinh_sq)
        std = torch.tensor([transverse, transverse, longitudinal] * 2,
                           device=device, dtype=dtype)
        return torch.zeros(self.dim, device=device, dtype=dtype), std


class MassWindowConstraint(Constraint):
    r"""``{x : |M(x) - M_\mathrm{target}| - \epsilon \le 0}``, a shell in invariant mass.

    Non-convex, so :attr:`interior_point` stays None and the projection must rely on its
    damped Newton loop alone. The absolute value is non-smooth only at ``M = M_target``, which
    sits strictly inside the shell at depth ``\epsilon``; the boundary itself is the pair of
    smooth surfaces ``M = M_target \pm \epsilon``, so the projection never differentiates the
    kink while it is correcting a violating point.
    """

    dim = 6

    def __init__(self, target: KinematicsTarget, mass_target: float, epsilon: float):
        self.target = target
        self.mass_target = float(mass_target)
        self.epsilon = float(epsilon)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return (self.target.invariant_mass(x) - self.mass_target).abs() - self.epsilon

    @property
    def params(self) -> torch.Tensor:
        r"""``(M_\mathrm{target} / s, \log(\epsilon / s))`` with ``s = \sqrt{E[M^2]}``.

        The width spans two decades across the benchmark, so it enters logarithmically; a raw
        ``\epsilon`` would let the widest shells dominate the conditioning input's scale.
        """
        scale = self.target.mass_scale()
        return torch.tensor([self.mass_target / scale, math.log(self.epsilon / scale)])


def shell_fraction(sorted_mass: torch.Tensor, centre: torch.Tensor,
                   epsilon: torch.Tensor) -> torch.Tensor:
    """Fraction of a sorted mass pool inside each window; all args broadcast to (B,)."""
    hi = torch.searchsorted(sorted_mass, (centre + epsilon).contiguous())
    lo = torch.searchsorted(sorted_mass, (centre - epsilon).contiguous())
    return (hi - lo).to(torch.float64) / sorted_mass.numel()


def solve_epsilon(sorted_mass: torch.Tensor, centre: torch.Tensor, fraction: torch.Tensor,
                  iterations: int = 48) -> torch.Tensor:
    """Bisects the half-width that puts ``fraction`` of the pool inside each window.

    The fraction is monotone in ``epsilon``, so bisection is the whole algorithm. It is run
    against the empirical CDF rather than an analytic one because ``M`` is a function of four
    of the six coordinates and its distribution has no closed form.
    """
    lo = torch.zeros_like(centre)
    hi = torch.full_like(centre, float(sorted_mass[-1] - sorted_mass[0]))

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        too_narrow = shell_fraction(sorted_mass, centre, mid) < fraction
        lo = torch.where(too_narrow, mid, lo)
        hi = torch.where(too_narrow, hi, mid)

    return 0.5 * (lo + hi)


def sample_mass_constraints(num_constraints: int, target: KinematicsTarget,
                            pool: torch.Tensor,
                            min_mass: float = KIN_SHELL_MIN_MASS,
                            max_mass: float = KIN_SHELL_MAX_MASS,
                            mass_bins: int = KIN_SHELL_MASS_BINS
                            ) -> tuple[list[MassWindowConstraint], torch.Tensor]:
    """Shells stratified uniformly over ``mass_bins`` log-spaced shell-mass levels.

    Centres are drawn as quantiles of the pool's own mass spectrum, so every shell sits where
    the target actually has support; a centre picked uniformly in ``M`` would mostly produce
    windows the sampler can never populate.
    """
    sorted_mass = target.invariant_mass(pool).to(torch.float64).sort().values
    levels = torch.logspace(math.log10(min_mass), math.log10(max_mass), mass_bins,
                            dtype=torch.float64, device=pool.device)

    fraction = levels.repeat_interleave(-(-num_constraints // mass_bins))[:num_constraints]
    quantiles = torch.rand(num_constraints, dtype=torch.float64, device=pool.device)
    centre = sorted_mass[(quantiles * (sorted_mass.numel() - 1)).long()]

    epsilon = solve_epsilon(sorted_mass, centre, fraction)
    achieved = shell_fraction(sorted_mass, centre, epsilon)

    constraints = [MassWindowConstraint(target, float(c), float(e))
                   for c, e in zip(centre.tolist(), epsilon.tolist())]
    return constraints, achieved.cpu()


class KinematicsProblem(Problem):
    name = PROBLEM_NAME
    dim = 6

    def __init__(self, pt_range: tuple[float, float] = KIN_PT_RANGE,
                 pt_scale: float = KIN_PT_SCALE,
                 eta_range: tuple[float, float] = KIN_ETA_RANGE,
                 eta_sigma: float = KIN_ETA_SIGMA,
                 min_mass: float = KIN_SHELL_MIN_MASS,
                 max_mass: float = KIN_SHELL_MAX_MASS,
                 mass_bins: int = KIN_SHELL_MASS_BINS,
                 mass_pool_size: int = 200_000):
        self.min_mass = float(min_mass)
        self.max_mass = float(max_mass)
        self.mass_bins = int(mass_bins)
        self.mass_pool_size = int(mass_pool_size)
        self._target = KinematicsTarget(pt_range, pt_scale, eta_range, eta_sigma)

    def target(self) -> KinematicsTarget:
        return self._target

    def sample_constraints(self, num_constraints: int,
                           device: torch.device | str | None = None
                           ) -> list[MassWindowConstraint]:
        pool = self._target.sample(self.mass_pool_size, device=device)
        constraints, _ = sample_mass_constraints(num_constraints, self._target, pool,
                                                 self.min_mass, self.max_mass, self.mass_bins)
        return constraints

    def normalizer(self) -> AffineNormalizer:
        mean, std = self._target.mean_std()
        return AffineNormalizer(mean, std)


__all__ = ["PROBLEM_NAME", "KinematicsTarget", "MassWindowConstraint", "KinematicsProblem",
           "sample_mass_constraints", "solve_epsilon", "shell_fraction"]
