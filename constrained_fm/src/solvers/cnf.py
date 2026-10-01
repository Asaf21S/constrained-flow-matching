# -*- coding: utf-8 -*-
r"""Probability-flow ODE solves with exact divergence, for sampling and for densities.

Time runs from noise ``t = 0`` to data ``t = 1``. With the augmented state ``[x, \ell]`` and
``\dot\ell = \nabla \cdot v``,

.. math::
    \log q(x_1) = \log \mathcal N(x_0) - \int_0^1 \nabla \cdot v\, dt,

obtained either forward from a noise draw (sample and density from one trajectory) or backward
from a given ``x_1``. The evaluation uses fixed-step midpoint solves (``*_fixed``); the adaptive
dopri5 step, controlled by the max over the batch, remains for training diagnostics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torchdiffeq import odeint

ODE_METHOD = "dopri5"
FIXED_METHOD = "midpoint"


@dataclass
class SolveStats:
    nfe: int
    seconds: float


def standard_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * z.pow(2).sum(dim=-1) - 0.5 * z.shape[-1] * math.log(2.0 * math.pi)


def _max_norm(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.abs().max()


def exact_divergence(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``tr(dv/dx)`` per row from one batched VJP over all basis vectors; rows must not interact."""
    dim = v.shape[1]
    basis = torch.eye(dim, device=v.device, dtype=v.dtype)[:, None, :].expand(dim, *v.shape)
    rows = torch.autograd.grad(v, x, grad_outputs=basis, is_grads_batched=True)[0]
    return rows.diagonal(dim1=0, dim2=2).sum(-1)


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


def _midpoint(field: _Field, state: torch.Tensor, t0: float, t1: float,
              steps: int) -> tuple[torch.Tensor, SolveStats]:
    dt = (t1 - t0) / steps
    if state.is_cuda:
        torch.cuda.synchronize(state.device)
    start = time.perf_counter()
    with torch.no_grad():
        for k in range(steps):
            t = state.new_tensor(t0 + k * dt)
            half = state + 0.5 * dt * field(t, state)
            state = state + dt * field(t + 0.5 * dt, half)
    if state.is_cuda:
        torch.cuda.synchronize(state.device)
    return state, SolveStats(field.nfe, time.perf_counter() - start)


def sample_fixed(model: torch.nn.Module, x0: torch.Tensor, steps: int,
                 cond: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, SolveStats]:
    """Pushes noise ``x0`` to ``t = 1`` with ``steps`` midpoint steps, no density."""
    return _midpoint(_Field(model, cond or {}, False), x0, 0.0, 1.0, steps)


def sample_with_log_prob_fixed(model: torch.nn.Module, x0: torch.Tensor, steps: int,
                               cond: dict[str, torch.Tensor] | None = None
                               ) -> tuple[torch.Tensor, torch.Tensor, SolveStats]:
    """``(x_1, log q(x_1), stats)``: the divergence is integrated along the sampling trajectory."""
    state = torch.cat([x0, x0.new_zeros(x0.shape[0], 1)], dim=1)
    out, stats = _midpoint(_Field(model, cond or {}, True), state, 0.0, 1.0, steps)
    return out[:, :-1], standard_normal_log_prob(x0) - out[:, -1], stats


def log_prob_fixed(model: torch.nn.Module, x1: torch.Tensor, steps: int,
                   cond: dict[str, torch.Tensor] | None = None
                   ) -> tuple[torch.Tensor, SolveStats]:
    """``(log p(x_1), stats)`` from one backward midpoint solve of the augmented ODE."""
    state = torch.cat([x1, x1.new_zeros(x1.shape[0], 1)], dim=1)
    out, stats = _midpoint(_Field(model, cond or {}, True), state, 1.0, 0.0, steps)
    return standard_normal_log_prob(out[:, :-1]) + out[:, -1], stats


def sample_with_log_prob(model: torch.nn.Module, x0: torch.Tensor,
                         cond: dict[str, torch.Tensor] | None = None, atol: float = 1e-5,
                         rtol: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, SolveStats]:
    """``(x_1, log q(x_1), stats)`` from one forward solve of the augmented ODE."""
    state = torch.cat([x0, x0.new_zeros(x0.shape[0], 1)], dim=1)
    out, stats = _integrate(_Field(model, cond or {}, True), state, 0.0, 1.0, atol, rtol)
    return out[:, :-1], standard_normal_log_prob(x0) - out[:, -1], stats


def log_prob(model: torch.nn.Module, x1: torch.Tensor, cond: dict[str, torch.Tensor] | None = None,
             atol: float = 1e-5, rtol: float = 1e-5) -> tuple[torch.Tensor, SolveStats]:
    """``(log q(x_1), stats)`` from one backward solve of the augmented ODE."""
    state = torch.cat([x1, x1.new_zeros(x1.shape[0], 1)], dim=1)
    out, stats = _integrate(_Field(model, cond or {}, True), state, 1.0, 0.0, atol, rtol)
    return standard_normal_log_prob(out[:, :-1]) + out[:, -1], stats


__all__ = ["ODE_METHOD", "FIXED_METHOD", "SolveStats", "standard_normal_log_prob",
           "exact_divergence", "sample", "sample_with_log_prob", "log_prob", "sample_fixed",
           "sample_with_log_prob_fixed", "log_prob_fixed"]
