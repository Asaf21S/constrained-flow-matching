# -*- coding: utf-8 -*-
"""Problem-agnostic interfaces: a target density, a feasible set, and the pairing of the two.

Every dimensionality and constraint-family assumption in the pipeline is meant to live behind
one of these objects, so adding a dataset is a new module here rather than an edit scattered
through the models, samplers and metrics.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch


class Constraint(abc.ABC):
    """A feasible set ``{x : C(x) <= 0}`` on ``R^dim``.

    Subclasses implement :meth:`value` alone; the gradient, hinge penalty and feasibility mask
    are all derived from it, so a new constraint family costs one method.
    """

    dim: int

    @abc.abstractmethod
    def value(self, x: torch.Tensor) -> torch.Tensor:
        """``C(x)`` for every row of ``x``: (N, dim) -> (N,), feasible where non-positive.

        Must stay autograd-transparent: HardFlow differentiates the penalty through it.
        """

    def value_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(C(x), grad_x C(x))``, detached from the caller's graph; (N,) and (N, dim)."""
        with torch.enable_grad():
            x_leaf = x.detach().requires_grad_(True)
            values = self.value(x_leaf)
            (grads,) = torch.autograd.grad(values.sum(), x_leaf)
        return values.detach(), grads.detach()

    def penalty(self, x: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        """Linear hinge ``relu(C(x) + margin)``: zero strictly inside, else the residual.

        Deliberately linear rather than squared. A squared hinge has gradient proportional to
        violation depth, so it vanishes exactly where guidance matters most -- on points
        sitting just outside the boundary, which are the ones that end up infeasible.
        """
        return torch.relu(self.value(x) + margin)

    def is_feasible(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x) <= 0

    def success_rate(self, x: torch.Tensor) -> float:
        """Percentage of rows of ``x`` inside the feasible set."""
        return self.is_feasible(x).float().mean().item() * 100.0

    @property
    def params(self) -> torch.Tensor | None:
        """Explicit conditioning vector (P,), or None when the constraint is encoded implicitly."""
        return None


class Target(abc.ABC):
    """The unconstrained data distribution the flow matcher learns, in physical coordinates."""

    dim: int

    @abc.abstractmethod
    def sample(self, num_points: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Draws (num_points, dim). Reproducibility comes from the global RNG, as elsewhere."""

    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Exact ``log p(x)``; (N, dim) -> (N,). Required for NLL/KLD to mean anything."""


@dataclass(frozen=True)
class AffineNormalizer:
    """``u = (x - mean) / std``, the frame the flow matcher's ``N(0, I)`` prior lives in."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def identity(cls, dim: int, device: torch.device | str | None = None) -> "AffineNormalizer":
        return cls(torch.zeros(dim, device=device), torch.ones(dim, device=device))

    @classmethod
    def from_samples(cls, x: torch.Tensor, eps: float = 1e-8) -> "AffineNormalizer":
        return cls(x.mean(dim=0), x.std(dim=0).clamp_min(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return u * self.std + self.mean

    @property
    def log_det_forward(self) -> torch.Tensor:
        """``log|det du/dx| = -sum log std``; normalized density is ``log p(x) - log_det_forward``."""
        return -torch.log(self.std).sum()

    def to(self, device: torch.device | str) -> "AffineNormalizer":
        return AffineNormalizer(self.mean.to(device), self.std.to(device))


class Problem(abc.ABC):
    """A target distribution paired with the constraint family evaluated against it."""

    name: str
    dim: int

    @abc.abstractmethod
    def target(self) -> Target:
        ...

    @abc.abstractmethod
    def sample_constraints(self, num_constraints: int,
                           device: torch.device | str | None = None) -> list[Constraint]:
        ...

    def normalizer(self) -> AffineNormalizer:
        """Identity unless the problem's physical units are badly scaled for an N(0, I) prior."""
        return AffineNormalizer.identity(self.dim)


__all__ = ["Constraint", "Target", "Problem", "AffineNormalizer"]
