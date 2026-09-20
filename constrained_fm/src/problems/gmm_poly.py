# -*- coding: utf-8 -*-
"""The original problem: a 4-peak 2D GMM constrained by bivariate polynomial sub-level sets.

Wraps the existing free functions rather than reimplementing them, so every number already
reported for this problem is reproduced bit-for-bit.
"""

from __future__ import annotations

import torch

from constrained_fm.src.consts import (GMM_COVS, GMM_MEANS, GMM_WEIGHTS, PLANE_SCALE,
                                       POLY_MAX_AREA_RATIO, POLY_MIN_AREA_RATIO,
                                       POLYNOMIAL_DEGREE)
from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.gmm_target import compute_gmm_log_likelihood, get_points
from constrained_fm.src.geometry.polynomials import compute_poly_features, evaluate_poly
from constrained_fm.src.problems.base import Constraint, Problem, Target

PROBLEM_NAME = "gmm_poly"


class PolynomialConstraint(Constraint):
    """``{x : P(x) <= 0}`` for the bivariate polynomial ``P(u) = u_x^T C u_y``, ``u = x / scale``."""

    dim = 2

    def __init__(self, coeffs: torch.Tensor, degree: int = POLYNOMIAL_DEGREE,
                 scale: float = PLANE_SCALE):
        self.degree = degree
        self.scale = scale
        self.coeffs = coeffs.reshape(degree + 1, degree + 1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        x_pow, y_pow = compute_poly_features(x, degree=self.degree, scale=self.scale)
        batched = self.coeffs.reshape(1, self.degree + 1, self.degree + 1).expand(x.shape[0], -1, -1)
        return evaluate_poly(x_pow, y_pow, batched).squeeze(-1)

    @property
    def params(self) -> torch.Tensor:
        return self.coeffs.reshape(-1)


class GMMTarget(Target):
    """The fixed 4-component Gaussian mixture on the plane."""

    dim = 2

    def __init__(self, means: list | torch.Tensor = GMM_MEANS,
                 covs: list | torch.Tensor = GMM_COVS,
                 weights: list | torch.Tensor = GMM_WEIGHTS):
        self.means = means
        self.covs = covs
        self.weights = weights

    def sample(self, num_points: int, device: torch.device | str | None = None) -> torch.Tensor:
        points, _ = get_points(num_points, self.means, self.covs, self.weights, device=device)
        return points

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return compute_gmm_log_likelihood(x, self.means, self.covs, self.weights, device=x.device)


class GMMPolyProblem(Problem):
    name = PROBLEM_NAME
    dim = 2

    def __init__(self, degree: int = POLYNOMIAL_DEGREE, scale: float = PLANE_SCALE,
                 min_area: float = POLY_MIN_AREA_RATIO, max_area: float = POLY_MAX_AREA_RATIO):
        self.degree = degree
        self.scale = scale
        self.min_area = min_area
        self.max_area = max_area

    def target(self) -> GMMTarget:
        return GMMTarget()

    def sample_constraints(self, num_constraints: int,
                           device: torch.device | str | None = None) -> list[PolynomialConstraint]:
        coeffs = sample_valid_polynomials(num_constraints, degree=self.degree, scale=self.scale,
                                          min_area=self.min_area, max_area=self.max_area,
                                          device=device)
        return [PolynomialConstraint(coeffs[i], self.degree, self.scale)
                for i in range(num_constraints)]


__all__ = ["PROBLEM_NAME", "PolynomialConstraint", "GMMTarget", "GMMPolyProblem"]
