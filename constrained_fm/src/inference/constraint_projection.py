# -*- coding: utf-8 -*-
"""Inference-time operators for a feasible set {x : C(x) <= 0}.

ECI needs a projection onto that set; HardFlow needs a differentiable penalty whose gradient
can steer the velocity field. Both are built from a :class:`Constraint`'s value and gradient,
so they are independent of the constraint family and of the dimension of ``x``.
"""

from __future__ import annotations

import torch

from constrained_fm.src.problems.base import Constraint

DEFAULT_MARGIN = 1e-3
DEFAULT_BISECTIONS = 24


def _restore_feasibility(x: torch.Tensor, constraint: Constraint, interior: torch.Tensor,
                         margin: float, steps: int) -> torch.Tensor:
    """Bisects the segment from each point to ``interior`` for the boundary crossing.

    On a convex set the feasible fraction of that segment is the suffix ``[s*, 1]``, so a
    bisection on the feasibility predicate brackets ``s*`` and the upper end is feasible at
    every iteration. Used only for points the Newton loop could not fix, which is why moving
    along this ray instead of along the normal is an acceptable trade.
    """
    direction = interior.unsqueeze(0).to(x.dtype) - x
    lo = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    hi = torch.ones_like(lo)

    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        feasible = (constraint.value(x + mid * direction) + margin <= 0).unsqueeze(-1)
        hi = torch.where(feasible, mid, hi)
        lo = torch.where(feasible, lo, mid)

    return x + hi * direction


def project_onto_feasible_region(x: torch.Tensor, constraint: Constraint,
                                 margin: float = DEFAULT_MARGIN, max_iters: int = 16,
                                 step_clip: float = 1.0, backtrack_halvings: int = 8,
                                 overshoot_fraction: float = 0.25, damping: float = 1.0,
                                 bisections: int = DEFAULT_BISECTIONS,
                                 eps: float = 1e-8) -> torch.Tensor:
    """Damped Newton projection of violating points onto the level set {C = -margin}.

    A general constraint admits no closed-form projection, so each violating point takes
    repeated Gauss-Newton steps ``x <- x - (C(x) + margin) * grad C / ||grad C||^2``, the
    first-order move onto the level set. Satisfied points are left untouched, so the operator
    is idempotent inside the region.

    The damping is what makes this a *projection* onto the boundary rather than a walk to
    anywhere feasible, and both halves matter. A step must reduce the violation, or it is
    halved. But it must also not dive far *past* the level set: an overshoot still satisfies
    the constraint and would terminate the loop with the sample stranded at an arbitrary
    interior depth. Rejecting overshoots turns those halvings into a bisection on the boundary
    crossing, which is what keeps the projected point next to the boundary instead of smeared
    through the interior.

    That loop stalls where ``C`` is a max of several smooth pieces and the pieces meet at a
    sharp angle: the exact step onto the active piece pushes the point further outside its
    neighbour, so no step length reduces the violation and the backtracking collapses to a
    crawl. The iteration count cannot fix this, because the failure is geometric rather than
    a matter of budget. When the constraint can name an interior point -- which commits it to
    being convex -- the leftovers are instead bisected onto the boundary along the ray to that
    point, which terminates in a fixed number of evaluations regardless of the angle.

    Args:
        x: (N, dim) points to project, in the coordinates ``constraint`` is defined on.
        constraint: the feasible set; only ``value``, ``value_and_grad`` and the optional
            ``interior_point`` are used.
        margin: how far strictly inside the boundary to land, in units of C.
        max_iters: cap on Newton iterations; the loop exits as soon as nothing violates.
        step_clip: maximum length of a single Newton step.
        backtrack_halvings: cap on step halvings per iteration.
        overshoot_fraction: how far past the level set a step may land, as a fraction of the
            violation it started from.
        damping: scales every Newton step before clipping. Below 1 it trades iterations for
            stability on a non-convex feasible set, where a full step can cross the region and
            land on the opposite wall, leaving the loop oscillating instead of converging.
        bisections: budget for the convex fallback; ignored when there is no interior point.

    Returns:
        (N, dim) projected points. Points whose Newton step diverges are left at their last
        finite position rather than propagating NaNs into the sampler.
    """
    x_proj = x.detach().clone()

    for _ in range(max_iters):
        values, grads = constraint.value_and_grad(x_proj)
        residual = values + margin
        violating = residual > 0
        if not bool(violating.any()):
            break

        denom = (grads * grads).sum(dim=-1).clamp_min(eps)
        step = damping * (residual / denom).unsqueeze(-1) * grads
        norms = step.norm(dim=-1, keepdim=True).clamp_min(eps)
        step = step * (norms.clamp(max=step_clip) / norms)

        candidate = x_proj - step
        for _ in range(backtrack_halvings):
            new_residual = constraint.value(candidate) + margin
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

    interior = constraint.interior_point
    if interior is not None and bisections > 0:
        stuck = constraint.value(x_proj) + margin > 0
        if bool(stuck.any()):
            x_proj[stuck] = _restore_feasibility(x_proj[stuck], constraint, interior,
                                                 margin, bisections)

    return x_proj


__all__ = ["DEFAULT_BISECTIONS", "DEFAULT_MARGIN", "project_onto_feasible_region"]
