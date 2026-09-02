# -*- coding: utf-8 -*-
"""ECI and HardFlow: inference-time constraint enforcement on an *unconstrained* flow matcher.

Both take a velocity field trained on the full GMM with no knowledge of constraints, and
alter the Euler integration of dx/dt = v(x, t) so the terminal sample satisfies P(x) <= 0.
They differ in how the constraint enters:

  * **ECI** rewrites the **state**. At each step it extrapolates to the endpoint
    ``x1_hat = x_t + (1 - t) v``, projects that endpoint into the feasible set, and steps
    toward the projected endpoint by ``dt / (1 - t)``. That weight hits 1 on the last step, so
    the returned sample *is* the projected endpoint: satisfaction is exact up to the
    projection. Note this is the velocity form of the correction, not the literal
    ``x_t = (1 - t) x_0 + t x1_proj`` re-interpolation. Rebuilding the state from x_0 each step
    is a fixed-point iteration with amplification ``t * ||d x1_hat / d x||``, which sits just
    above 1 for a multi-modal target and compounds into a badly over-dispersed sample over a
    hundred steps. The velocity form reduces identically to Euler when the projection is
    inactive, so an unconstrained run reproduces the base model exactly.
  * **HardFlow** rewrites the velocity. It differentiates a constraint-violation penalty at
    the predicted endpoint w.r.t. the current state and subtracts that gradient from v, so
    the trajectory is steered toward the region without ever being teleported into it.
    Satisfaction is therefore approximate and depends on ``guidance_scale``.

Neither method leaves the model's probability-flow ODE intact, so likelihoods computed by
integrating the original field no longer describe the sampled distribution. Callers must
report NLL/KLD as undefined for these samplers.
"""

from __future__ import annotations

import torch

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.inference.constraint_projection import (DEFAULT_MARGIN,
                                                                project_onto_polynomial_region,
                                                                violation_penalty)

DEFAULT_STEPS = 100
# HardFlow backprops through the network at every step, so activations scale with the chunk.
DEFAULT_CHUNK = 20_000


def _as_matrix(coeffs: torch.Tensor, degree: int) -> torch.Tensor:
    return coeffs.reshape(degree + 1, degree + 1)


def _eci_chunk(model, x0: torch.Tensor, C: torch.Tensor, degree: int, scale: float, steps: int,
               correction_loops: int, margin: float, projection_iters: int) -> torch.Tensor:
    dt = 1.0 / steps
    x = x0

    for i in range(steps):
        t = i * dt
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        remaining = 1.0 - t
        # Reaches exactly 1 on the last step, so the sample lands on the projected endpoint.
        step_fraction = min(dt / remaining, 1.0)

        x_work = x
        for loop in range(correction_loops):
            with torch.no_grad():
                v = model(x_work, t_batch)
            x1_projected = project_onto_polynomial_region(x_work + remaining * v, C, degree=degree,
                                                          scale=scale, margin=margin,
                                                          max_iters=projection_iters)
            if loop < correction_loops - 1:
                x_work = x + step_fraction * (x1_projected - x)

        x = x + step_fraction * (x1_projected - x)

    return x


def _hardflow_chunk(model, x0: torch.Tensor, C: torch.Tensor, degree: int, scale: float,
                    steps: int, guidance_scale: float, margin: float) -> torch.Tensor:
    dt = 1.0 / steps
    x = x0

    for i in range(steps):
        t = i * dt
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)

        with torch.enable_grad():
            x_leaf = x.detach().requires_grad_(True)
            v = model(x_leaf, t_batch)
            x1_hat = x_leaf + (1.0 - t) * v
            penalty = violation_penalty(x1_hat, C, degree=degree, scale=scale, margin=margin).sum()
            (grad,) = torch.autograd.grad(penalty, x_leaf)

        v_guided = v.detach() - guidance_scale * grad
        x = x_leaf.detach() + v_guided * dt

    return x


def sample_eci(model, x0: torch.Tensor, coeffs: torch.Tensor,
               degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
               steps: int = DEFAULT_STEPS, correction_loops: int = 1,
               margin: float = DEFAULT_MARGIN, projection_iters: int = 16,
               chunk_size: int = DEFAULT_CHUNK) -> torch.Tensor:
    """Exact Constraint Injection sampling of ``x0`` under a single polynomial constraint.

    Args:
        model: unconstrained velocity field with signature ``model(x, t)``.
        x0: (N, 2) prior draws.
        coeffs: (degree+1, degree+1) or flat coefficient matrix of the constraint.
        correction_loops: extrapolate/project/interpolate repetitions per integration step.
        margin: how far strictly inside the boundary the projection aims.

    Returns:
        (N, 2) terminal samples.
    """
    model.eval()
    C = _as_matrix(coeffs, degree)
    chunks = [_eci_chunk(model, x0[i:i + chunk_size], C, degree, scale, steps, correction_loops,
                         margin, projection_iters)
              for i in range(0, x0.shape[0], chunk_size)]
    return torch.cat(chunks, dim=0)


def sample_hardflow(model, x0: torch.Tensor, coeffs: torch.Tensor,
                    degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                    steps: int = DEFAULT_STEPS, guidance_scale: float = 5.0,
                    margin: float = DEFAULT_MARGIN,
                    chunk_size: int = DEFAULT_CHUNK) -> torch.Tensor:
    """HardFlow gradient-guided sampling of ``x0`` under a single polynomial constraint.

    Args:
        model: unconstrained velocity field with signature ``model(x, t)``.
        x0: (N, 2) prior draws.
        coeffs: (degree+1, degree+1) or flat coefficient matrix of the constraint.
        guidance_scale: weight on the endpoint-penalty gradient subtracted from the velocity.
        margin: how far strictly inside the boundary the penalty stays active.

    Returns:
        (N, 2) terminal samples.
    """
    model.eval()
    C = _as_matrix(coeffs, degree)
    chunks = [_hardflow_chunk(model, x0[i:i + chunk_size], C, degree, scale, steps,
                              guidance_scale, margin)
              for i in range(0, x0.shape[0], chunk_size)]
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def sample_euler(model, x0: torch.Tensor, steps: int = DEFAULT_STEPS,
                 chunk_size: int = DEFAULT_CHUNK) -> torch.Tensor:
    """Plain Euler integration of the base field, with no constraint applied.

    The reference both methods deviate from: whatever ECI or HardFlow gain in feasibility is
    paid for out of the distribution this returns.

    Args:
        model: unconstrained velocity field with signature ``model(x, t)``.
        x0: (N, 2) prior draws.

    Returns:
        (N, 2) terminal samples.
    """
    model.eval()
    dt = 1.0 / steps
    chunks = []
    for start in range(0, x0.shape[0], chunk_size):
        x = x0[start:start + chunk_size]
        for i in range(steps):
            t_batch = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = x + model(x, t_batch) * dt
        chunks.append(x)
    return torch.cat(chunks, dim=0)


SAMPLERS = {"eci": sample_eci, "hardflow": sample_hardflow}

__all__ = ["sample_eci", "sample_hardflow", "sample_euler", "SAMPLERS", "DEFAULT_STEPS"]
