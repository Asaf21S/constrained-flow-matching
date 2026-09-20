# -*- coding: utf-8 -*-
"""Dataset/constraint problems behind a common interface."""

from constrained_fm.src.problems.base import AffineNormalizer, Constraint, Problem, Target
from constrained_fm.src.problems.gmm_poly import (PROBLEM_NAME as GMM_POLY, GMMPolyProblem,
                                                  GMMTarget, PolynomialConstraint)
from constrained_fm.src.problems.registry import (available_problems, get_problem,
                                                  register_problem)

register_problem(GMM_POLY, GMMPolyProblem)

__all__ = ["AffineNormalizer", "Constraint", "Problem", "Target", "GMMPolyProblem", "GMMTarget",
           "PolynomialConstraint", "available_problems", "get_problem", "register_problem"]
