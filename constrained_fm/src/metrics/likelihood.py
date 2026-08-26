# -*- coding: utf-8 -*-
"""Exact negative log-likelihood of constraint-satisfying GT points under the flow matcher.

The distributional metrics (SWD/MMD/JSD) compare point clouds, so a model that covers the
right region with the wrong density can still score well. NLL scores the learned density
directly: it integrates the probability-flow ODE backwards from t=1 to t=0 while
accumulating the exact divergence of the velocity field, giving

    log p(x_1) = log N(x_0; 0, I) - integral_1^0 Tr(grad_x v_t) dt

In 2D the exact trace costs two backward passes per step, so no Hutchinson estimator is
needed. flow_matching's ODESolver.compute_likelihood already integrates this augmented
state, so this module only supplies the conditioning, batching and normalization.
"""

from __future__ import annotations

import math

import torch
from flow_matching.solver import ODESolver
from torch.distributions import Independent, Normal

from constrained_fm.src.datasets.gmm_target import compute_gmm_log_likelihood
from constrained_fm.src.solvers.ode_wrapper import WrappedModel


def exact_log_likelihood(model, x_1: torch.Tensor, bounds=None, coeffs: torch.Tensor | None = None,
                         z: torch.Tensor | None = None, step_size: float = 0.05,
                         chunk_size: int = 4000, device=None) -> torch.Tensor:
    """log p(x) under the model for each row of x_1, via the backward augmented ODE.

    Returns (N,) on x_1's device.
    """
    model.eval()
    solver = ODESolver(velocity_model=WrappedModel(model))
    num_points = x_1.shape[0]
    if num_points == 0:
        return torch.empty(0, device=x_1.device)

    prior_log_density = Independent(
        Normal(torch.zeros(2, device=device), torch.ones(2, device=device)), 1).log_prob

    if bounds is not None:
        cond_key, cond = "bounds", torch.as_tensor(
            bounds, dtype=torch.float32, device=device).view(1, -1).expand(num_points, -1)
    elif z is not None:
        cond_key, cond = "z", z.view(1, -1).expand(num_points, -1)
    elif coeffs is not None:
        cond_key, cond = "coeffs", coeffs.reshape(1, -1).expand(num_points, -1)
    else:
        cond_key, cond = None, None

    log_p_chunks = []
    for start in range(0, num_points, chunk_size):
        chunk_kwargs = {} if cond is None else {cond_key: cond[start:start + chunk_size]}
        _, log_p = solver.compute_likelihood(
            x_1=x_1[start:start + chunk_size],
            method="midpoint",
            step_size=step_size,
            exact_divergence=True,
            log_p0=prior_log_density,
            **chunk_kwargs,
        )
        log_p_chunks.append(log_p.detach())

    return torch.cat(log_p_chunks, dim=0)


def truncated_gmm_log_likelihood(x: torch.Tensor, mass: float, device=None) -> torch.Tensor:
    """log density of the GMM truncated to the constraint region, evaluated inside it.

    Truncation renormalizes by the constraint's probability mass, so the target density is
    p_gmm(x) / mass. Without this the reference entropy would be misattributed to the model.
    """
    if mass <= 0.0:
        return torch.full((x.shape[0],), float("nan"), device=x.device)
    return compute_gmm_log_likelihood(x, device=device) - math.log(mass)


def constraint_nll(model, x_true_valid: torch.Tensor, mass: float, bounds=None,
                   coeffs: torch.Tensor | None = None, z: torch.Tensor | None = None,
                   num_points: int = 5000, step_size: float = 0.05, chunk_size: int = 4000,
                   device=None) -> dict[str, float]:
    """Mean NLL of constraint-satisfying GT points, and the KL divergence it implies.

    x_true_valid must already be filtered to the constraint region; a random subset of
    num_points is scored.

    Subtracting the truncated GMM's own entropy turns the NLL into a Monte Carlo estimate of
    KL(p_true || p_model) >= 0, which unlike raw NLL is comparable across constraints of
    differing mass and so can be averaged over the benchmark. The estimate can dip slightly
    below zero, since `mass` is itself estimated from a finite pool.
    """
    if x_true_valid.shape[0] == 0:
        return {"nll": float("nan"), "kld": float("nan")}

    if x_true_valid.shape[0] > num_points:
        idx = torch.randperm(x_true_valid.shape[0], device=x_true_valid.device)[:num_points]
        x_true_valid = x_true_valid[idx]

    log_p_model = exact_log_likelihood(model, x_true_valid, bounds=bounds, coeffs=coeffs, z=z,
                                       step_size=step_size, chunk_size=chunk_size, device=device)
    finite = torch.isfinite(log_p_model)
    if not bool(finite.any()):
        return {"nll": float("inf"), "kld": float("inf")}

    nll = float(-log_p_model[finite].mean())
    log_p_true = truncated_gmm_log_likelihood(x_true_valid, mass, device=device)
    ideal_nll = float(-log_p_true[finite].mean())

    return {"nll": nll, "kld": nll - ideal_nll}


__all__ = ["exact_log_likelihood", "truncated_gmm_log_likelihood", "constraint_nll"]
