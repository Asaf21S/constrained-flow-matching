# -*- coding: utf-8 -*-
r"""Probability-flow ODE solves with exact divergence, for sampling and for densities.

Time runs from noise ``t = 0`` to data ``t = 1``. With the augmented state ``[x, \ell]`` and
``\dot\ell = \nabla \cdot v``,

.. math::
    \log q(x_1) = \log \mathcal N(x_0) - \int_0^1 \nabla \cdot v\, dt,

obtained either forward from a noise draw (sample and density from one trajectory) or backward
from a given ``x_1``. The adaptive step is controlled by the max over the batch, not the RMS, so
the tolerance holds for every sample rather than on average.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torchdiffeq import odeint

ODE_METHOD = "dopri5"
FALLBACK_METHOD = "rk4"


@dataclass
class SolveStats:
    nfe: int
    seconds: float


def standard_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * z.pow(2).sum(dim=-1) - 0.5 * z.shape[-1] * math.log(2.0 * math.pi)


def _max_norm(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.abs().max()


def exact_divergence(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``tr(dv/dx)`` per row via one VJP per dimension; rows must not interact."""
    dim = v.shape[1]
    div = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
    for i in range(dim):
        div = div + torch.autograd.grad(v[:, i].sum(), x, retain_graph=i < dim - 1)[0][:, i]
    return div


class _Field:
    def __init__(self, model: torch.nn.Module, cond: dict[str, torch.Tensor], with_divergence: bool):
        self.model = model
        self.cond = cond
        self.with_divergence = with_divergence
        self.nfe = 0

    def __call__(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        if not self.with_divergence:
            return self.model(state, t, **self.cond)
        with torch.enable_grad():
            x = state[:, :-1].detach().requires_grad_(True)
            v = self.model(x, t, **self.cond)
            div = exact_divergence(v, x)
        return torch.cat([v.detach(), div.detach()[:, None]], dim=1)


def _integrate(field: _Field, state: torch.Tensor, t0: float, t1: float, atol: float,
               rtol: float, method: str = ODE_METHOD,
               options: dict | None = None) -> tuple[torch.Tensor, SolveStats]:
    grid = torch.tensor([t0, t1], device=state.device, dtype=state.dtype)
    if state.is_cuda:
        torch.cuda.synchronize(state.device)
    start = time.perf_counter()
    with torch.no_grad():
        out = odeint(field, state, grid, method=method, atol=atol, rtol=rtol,
                     options=options or {"norm": _max_norm})[-1]
    if state.is_cuda:
        torch.cuda.synchronize(state.device)
    return out, SolveStats(field.nfe, time.perf_counter() - start)


def sample(model: torch.nn.Module, x0: torch.Tensor, cond: dict[str, torch.Tensor] | None = None,
           atol: float = 1e-5, rtol: float = 1e-5) -> tuple[torch.Tensor, SolveStats]:
    """Pushes noise ``x0`` to ``t = 1`` without tracking the density."""
    return _integrate(_Field(model, cond or {}, False), x0, 0.0, 1.0, atol, rtol)


def _integrate_isolating(field: _Field, state: torch.Tensor, t0: float, t1: float,
                         atol: float, rtol: float, fallback_steps: int
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    flagged = torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
    try:
        return _integrate(field, state, t0, t1, atol, rtol)[0], flagged
    except AssertionError as err:
        if "underflow" not in str(err):
            raise
        if state.shape[0] > 1:
            half = state.shape[0] // 2
            parts = [_integrate_isolating(field, part, t0, t1, atol, rtol, fallback_steps)
                     for part in (state[:half], state[half:])]
            return torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts])
        out, _ = _integrate(field, state, t0, t1, atol, rtol, FALLBACK_METHOD,
                            {"step_size": abs(t1 - t0) / fallback_steps})
        flagged[:] = True
        return out, flagged


def sample_isolating(model: torch.nn.Module, x0: torch.Tensor,
                     cond: dict[str, torch.Tensor] | None = None, atol: float = 1e-5,
                     rtol: float = 1e-5, fallback_steps: int = 1000
                     ) -> tuple[torch.Tensor, SolveStats, torch.Tensor]:
    """Sampling that isolates samples causing adaptive-step underflow; flags RK4 fallbacks."""
    field = _Field(model, cond or {}, False)
    if x0.is_cuda:
        torch.cuda.synchronize(x0.device)
    start = time.perf_counter()
    x1, flagged = _integrate_isolating(field, x0, 0.0, 1.0, atol, rtol, fallback_steps)
    if x0.is_cuda:
        torch.cuda.synchronize(x0.device)
    return x1, SolveStats(field.nfe, time.perf_counter() - start), flagged


def sample_with_log_prob(model: torch.nn.Module, x0: torch.Tensor,
                         cond: dict[str, torch.Tensor] | None = None, atol: float = 1e-5,
                         rtol: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, SolveStats]:
    """``(x_1, log q(x_1), stats)`` from one forward solve of the augmented ODE."""
    state = torch.cat([x0, x0.new_zeros(x0.shape[0], 1)], dim=1)
    out, stats = _integrate(_Field(model, cond or {}, True), state, 0.0, 1.0, atol, rtol)
    return out[:, :-1], standard_normal_log_prob(x0) - out[:, -1], stats


def sample_with_log_prob_isolating(model: torch.nn.Module, x0: torch.Tensor,
                                   cond: dict[str, torch.Tensor] | None = None,
                                   atol: float = 1e-5, rtol: float = 1e-5,
                                   fallback_steps: int = 1000
                                   ) -> tuple[torch.Tensor, torch.Tensor, SolveStats, torch.Tensor]:
    """Forward sample and log density, isolating adaptive-step underflows in the batch."""
    state = torch.cat([x0, x0.new_zeros(x0.shape[0], 1)], dim=1)
    field = _Field(model, cond or {}, True)
    if x0.is_cuda:
        torch.cuda.synchronize(x0.device)
    start = time.perf_counter()
    out, flagged = _integrate_isolating(field, state, 0.0, 1.0, atol, rtol, fallback_steps)
    if x0.is_cuda:
        torch.cuda.synchronize(x0.device)
    return (out[:, :-1], standard_normal_log_prob(x0) - out[:, -1],
            SolveStats(field.nfe, time.perf_counter() - start), flagged)


def log_prob(model: torch.nn.Module, x1: torch.Tensor, cond: dict[str, torch.Tensor] | None = None,
             atol: float = 1e-5, rtol: float = 1e-5) -> tuple[torch.Tensor, SolveStats]:
    """``(log q(x_1), stats)`` from one backward solve of the augmented ODE."""
    state = torch.cat([x1, x1.new_zeros(x1.shape[0], 1)], dim=1)
    out, stats = _integrate(_Field(model, cond or {}, True), state, 1.0, 0.0, atol, rtol)
    return standard_normal_log_prob(out[:, :-1]) + out[:, -1], stats


def log_prob_isolating(model: torch.nn.Module, x1: torch.Tensor, atol: float = 1e-5,
                       rtol: float = 1e-5, fallback_steps: int = 1000
                       ) -> tuple[torch.Tensor, SolveStats, torch.Tensor]:
    """Backward log-density solve that isolates adaptive-step underflows; flags RK4 fallbacks."""
    state = torch.cat([x1, x1.new_zeros(x1.shape[0], 1)], dim=1)
    field = _Field(model, {}, True)
    if x1.is_cuda:
        torch.cuda.synchronize(x1.device)
    start = time.perf_counter()
    out, flagged = _integrate_isolating(field, state, 1.0, 0.0, atol, rtol, fallback_steps)
    if x1.is_cuda:
        torch.cuda.synchronize(x1.device)
    log_q = standard_normal_log_prob(out[:, :-1]) + out[:, -1]
    return log_q, SolveStats(field.nfe, time.perf_counter() - start), flagged


__all__ = ["ODE_METHOD", "SolveStats", "standard_normal_log_prob", "exact_divergence", "sample",
           "sample_with_log_prob", "log_prob", "sample_isolating",
           "sample_with_log_prob_isolating", "log_prob_isolating"]
