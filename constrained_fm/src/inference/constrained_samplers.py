# -*- coding: utf-8 -*-
"""ECI and HardFlow: inference-time constraint enforcement on an *unconstrained* flow matcher.

Both take a velocity field trained on the full target with no knowledge of constraints, on the
path ``x_t = (1 - t) x_0 + t x_1`` (``alpha_t = t``, ``beta_t = 1 - t``), and alter its Euler
integration so the terminal sample satisfies C(x) <= 0. Both reduce to the Euclidean
projection :func:`project_closest_point` for the constraint step, and both return the
projected point on the final step, so terminal feasibility is exact up to that projection.

  * **ECI** (Cheng et al., ICLR 2025, Alg. 2-3; ``eci_sample`` in the official code). Each
    Euler step runs ``M`` mixing iterations of extrapolation ``u_1 = u_t + (1 - t) v``,
    correction ``u_1 <- Proj(u_1)`` and interpolation ``u = (1 - t') u_0 + t' u_1``; the
    first ``M - 1`` use ``t' = t`` and the last advances to ``t' = t + dt``. The noise ``u_0``
    is the initial draw, redrawn from N(0, I) every ``R`` mixing iterations when ``R`` is set.
  * **HardFlow** (Li, Alim & Azizan, 2025, Alg. 1). Each active step takes the nominal Euler
    step ``x_bar = x_i + v(x_i, t_i) dt``, forms the posterior mean
    ``x_bar_N = x_bar + (1 - t') v(x_bar, t')`` and solves
    ``min_x C(x) + lambda_oc / (2 dt) t'^2 ||x - x_bar_N||^2 s.t. h(x) <= 0``. With no terminal
    cost (``C = 0``) the minimiser is the projection of ``x_bar_N`` for every ``lambda_oc``.
    The state is then rebuilt by the one-step fixed-point inverse of the posterior mean,
    ``x_{i+1} = t' x* + (1 - t') (x_bar - t' v(x_bar, t'))``, which equals ``x_bar`` whenever
    ``x_bar_N`` is already feasible. Steps before ``active_from * N`` are plain Euler.

Neither method leaves the model's probability-flow ODE intact, so likelihoods computed by
integrating the original field no longer describe the sampled distribution. Callers must
report NLL/KLD as undefined for these samplers.

Both are written against the :class:`Constraint` interface, so they are independent of the
constraint family and of the dimension of the state.
"""

from __future__ import annotations

import torch

from constrained_fm.src.inference.constraint_projection import (DEFAULT_MARGIN,
                                                                DEFAULT_PROJECTION_ITERS,
                                                                project_closest_point)
from constrained_fm.src.problems.base import Constraint

DEFAULT_STEPS = 100
DEFAULT_CHUNK = 20_000
DEFAULT_MIXING_ITERS = 1
# HardFlow's appendix solves the subproblem only in the second half of the sampling steps.
DEFAULT_ACTIVE_FROM = 0.5


def _time_batch(x: torch.Tensor, t: float) -> torch.Tensor:
    return torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)


@torch.no_grad()
def _eci_chunk(model, x0: torch.Tensor, constraint: Constraint, steps: int,
               mixing_iters: int, resample_interval: int | None, margin: float,
               projection_iters: int, projection_damping: float,
               generator: torch.Generator | None) -> torch.Tensor:
    noise = x0
    x = x0
    count = 0

    for i in range(steps):
        t = i / steps
        t_batch = _time_batch(x, t)
        for mix in range(mixing_iters):
            count += 1
            if resample_interval and count % resample_interval == 0:
                noise = torch.randn(x.shape, generator=generator, device=x.device,
                                    dtype=x.dtype)
            x1 = project_closest_point(x + (1.0 - t) * model(x, t_batch), constraint,
                                       margin=margin, max_iters=projection_iters,
                                       relaxation=projection_damping)
            t_interp = t if mix < mixing_iters - 1 else (i + 1) / steps
            x = t_interp * x1 + (1.0 - t_interp) * noise

    return x


@torch.no_grad()
def _hardflow_chunk(model, x0: torch.Tensor, constraint: Constraint, steps: int,
                    active_from: float, margin: float, projection_iters: int,
                    projection_damping: float) -> torch.Tensor:
    first_active = min(round(active_from * steps), steps - 1)
    x = x0

    for i in range(steps):
        t, t_next = i / steps, (i + 1) / steps
        x_bar = x + (t_next - t) * model(x, _time_batch(x, t))
        if i < first_active:
            x = x_bar
            continue

        v_bar = model(x_bar, _time_batch(x_bar, t_next))
        terminal = project_closest_point(x_bar + (1.0 - t_next) * v_bar, constraint,
                                         margin=margin, max_iters=projection_iters,
                                         relaxation=projection_damping)
        x = t_next * terminal + (1.0 - t_next) * (x_bar - t_next * v_bar)

    return x


def sample_eci(model, x0: torch.Tensor, constraint: Constraint,
               steps: int = DEFAULT_STEPS, mixing_iters: int = DEFAULT_MIXING_ITERS,
               resample_interval: int | None = None, margin: float = DEFAULT_MARGIN,
               projection_iters: int = DEFAULT_PROJECTION_ITERS,
               projection_damping: float = 1.0, seed: int | None = None,
               chunk_size: int = DEFAULT_CHUNK) -> torch.Tensor:
    """Extrapolation-Correction-Interpolation (ECI; Cheng et al., 2025) sampling of ``x0``.

    Args:
        model: unconstrained velocity field with signature ``model(x, t)``.
        x0: (N, dim) prior draws; also the interpolation noise until it is redrawn.
        constraint: the feasible set to inject.
        mixing_iters: ECI iterations per Euler step (``M``).
        resample_interval: redraw the interpolation noise every this many mixing iterations
            (``R``); ``None`` or 0 keeps ``x0`` throughout.
        margin: how far strictly inside the boundary the projection aims.
        projection_iters: SQP iterations of the closest-point projection.
        projection_damping: SQP relaxation in (0, 1].
        seed: seeds the noise redraws; ``None`` uses the global RNG.

    Returns:
        (N, dim) terminal samples.
    """
    model.eval()
    generator = None
    if resample_interval and seed is not None:
        generator = torch.Generator(device=x0.device)
        generator.manual_seed(seed)
    chunks = [_eci_chunk(model, x0[i:i + chunk_size], constraint, steps, mixing_iters,
                         resample_interval, margin, projection_iters, projection_damping,
                         generator)
              for i in range(0, x0.shape[0], chunk_size)]
    return torch.cat(chunks, dim=0)


def sample_hardflow(model, x0: torch.Tensor, constraint: Constraint,
                    steps: int = DEFAULT_STEPS, active_from: float = DEFAULT_ACTIVE_FROM,
                    margin: float = DEFAULT_MARGIN,
                    projection_iters: int = DEFAULT_PROJECTION_ITERS,
                    projection_damping: float = 1.0,
                    chunk_size: int = DEFAULT_CHUNK) -> torch.Tensor:
    """HardFlow (Li, Alim & Azizan, 2025, Alg. 1) sampling of ``x0`` with no terminal cost.

    Args:
        model: unconstrained velocity field with signature ``model(x, t)``.
        x0: (N, dim) prior draws.
        constraint: the feasible set ``h(x) <= 0``.
        active_from: fraction of the steps after which the subproblem is solved; earlier
            steps are plain Euler. The last step is always active.
        margin: how far strictly inside the boundary the projection aims.
        projection_iters: SQP iterations of the closest-point projection.
        projection_damping: SQP relaxation in (0, 1].

    Returns:
        (N, dim) terminal samples.
    """
    model.eval()
    chunks = [_hardflow_chunk(model, x0[i:i + chunk_size], constraint, steps, active_from,
                              margin, projection_iters, projection_damping)
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
        x0: (N, dim) prior draws.

    Returns:
        (N, dim) terminal samples.
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

__all__ = ["sample_eci", "sample_hardflow", "sample_euler", "SAMPLERS", "DEFAULT_STEPS",
           "DEFAULT_CHUNK", "DEFAULT_MIXING_ITERS", "DEFAULT_ACTIVE_FROM"]
