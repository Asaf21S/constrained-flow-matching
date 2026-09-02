# -*- coding: utf-8 -*-
"""Inference-time operators for the polynomial feasible set {x : P(x) <= 0}.

ECI needs a projection onto that set; HardFlow needs a differentiable penalty whose gradient
can steer the velocity field. Both are built from P and its gradient in raw plane
coordinates, remembering that P is defined on the normalized coordinate x / scale.
"""

from __future__ import annotations

import torch

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly

DEFAULT_MARGIN = 1e-3


def poly_values(x: torch.Tensor, C: torch.Tensor, degree: int = POLYNOMIAL_DEGREE,
                scale: float = PLANE_SCALE) -> torch.Tensor:
    """P(x) for every row of ``x`` under a single coefficient matrix ``C``; returns (N,)."""
    x_pow, y_pow = compute_poly_features(x, degree=degree, scale=scale)
    C_batch = C.reshape(1, degree + 1, degree + 1).expand(x.shape[0], -1, -1)
    return evaluate_poly(x_pow, y_pow, C_batch).squeeze(-1)


def violation_penalty(x: torch.Tensor, C: torch.Tensor, degree: int = POLYNOMIAL_DEGREE,
                      scale: float = PLANE_SCALE, margin: float = 0.0) -> torch.Tensor:
    """Linear hinge on the constraint residual: zero wherever P(x) <= -margin, else P + margin.

    Deliberately linear rather than squared. A squared hinge has gradient proportional to the
    violation depth, so it vanishes exactly where guidance matters most -- on points sitting
    just outside the boundary, which are the ones that end up infeasible. The linear hinge
    keeps a push of magnitude ||grad P|| all the way to the level set, matching the
    distance-style penalty the reference HardFlow implementation uses.
    """
    return torch.relu(poly_values(x, C, degree, scale) + margin)


def poly_value_and_grad(x: torch.Tensor, C: torch.Tensor, degree: int = POLYNOMIAL_DEGREE,
                        scale: float = PLANE_SCALE) -> tuple[torch.Tensor, torch.Tensor]:
    """(P(x), grad_x P(x)) detached from the caller's graph; shapes (N,) and (N, 2)."""
    with torch.enable_grad():
        x_leaf = x.detach().requires_grad_(True)
        values = poly_values(x_leaf, C, degree, scale)
        (grads,) = torch.autograd.grad(values.sum(), x_leaf)
    return values.detach(), grads.detach()


def project_onto_polynomial_region(x: torch.Tensor, C: torch.Tensor,
                                   degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                                   margin: float = DEFAULT_MARGIN, max_iters: int = 16,
                                   step_clip: float = 1.0, backtrack_halvings: int = 8,
                                   overshoot_fraction: float = 0.25,
                                   eps: float = 1e-8) -> torch.Tensor:
    """Damped Newton projection of violating points onto the level set {P = -margin}.

    A cubic zero set admits no closed-form projection, so each violating point takes repeated
    Gauss-Newton steps ``x <- x - (P(x) + margin) * grad P / ||grad P||^2``, the first-order
    move onto the level set. Satisfied points are left untouched, so the operator is
    idempotent inside the region.

    The damping is what makes this a *projection* onto the boundary rather than a walk to
    anywhere feasible, and both halves matter. A step must reduce the violation, or it is
    halved. But it must also not dive far *past* the level set: the feasible set is unbounded,
    so an overshoot still satisfies the constraint and would terminate the loop with the sample
    stranded at an arbitrary interior depth. Rejecting overshoots turns those halvings into a
    bisection on the boundary crossing, which is what keeps the projected point next to the
    boundary instead of smeared through the interior.

    Args:
        x: (N, 2) raw-scale points to project.
        C: (degree+1, degree+1) coefficient matrix of a single constraint.
        margin: how far strictly inside the boundary to land, in units of P.
        max_iters: cap on Newton iterations; the loop exits as soon as nothing violates.
        step_clip: maximum length of a single Newton step, in raw plane units.
        backtrack_halvings: cap on step halvings per iteration.
        overshoot_fraction: how far past the level set a step may land, as a fraction of the
            violation it started from.

    Returns:
        (N, 2) projected points. Points whose Newton step diverges are left at their last
        finite position rather than propagating NaNs into the sampler.
    """
    x_proj = x.detach().clone()

    for _ in range(max_iters):
        values, grads = poly_value_and_grad(x_proj, C, degree, scale)
        residual = values + margin
        violating = residual > 0
        if not bool(violating.any()):
            break

        denom = (grads * grads).sum(dim=-1).clamp_min(eps)
        step = (residual / denom).unsqueeze(-1) * grads
        norms = step.norm(dim=-1, keepdim=True).clamp_min(eps)
        step = step * (norms.clamp(max=step_clip) / norms)

        candidate = x_proj - step
        for _ in range(backtrack_halvings):
            new_residual = poly_values(candidate, C, degree, scale) + margin
            # A NaN fails both comparisons, so non-finite candidates are rejected here too.
            accepted = ((new_residual < residual)
                        & (new_residual > -overshoot_fraction * residual))
            rejected = violating & ~accepted
            if not bool(rejected.any()):
                break
            step = torch.where(rejected.unsqueeze(-1), step * 0.5, step)
            candidate = x_proj - step

        accept = violating.unsqueeze(-1) & torch.isfinite(candidate).all(dim=-1, keepdim=True)
        x_proj = torch.where(accept, candidate, x_proj)

    return x_proj


__all__ = ["DEFAULT_MARGIN", "poly_values", "violation_penalty", "poly_value_and_grad",
           "project_onto_polynomial_region"]
