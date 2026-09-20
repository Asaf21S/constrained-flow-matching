# -*- coding: utf-8 -*-
"""Dataset/constraint problems behind a common interface."""

from constrained_fm.src.problems.base import (AffineNormalizer, Constraint, NormalizedConstraint,
                                              Problem, Target)
from constrained_fm.src.problems.bump2d import (PROBLEM_NAME as BUMP2D, BumpProblem, BumpTarget,
                                                PolygonConstraint)
from constrained_fm.src.problems.gmm_poly import (PROBLEM_NAME as GMM_POLY, GMMPolyProblem,
                                                  GMMTarget, PolynomialConstraint)
from constrained_fm.src.problems.registry import (available_problems, get_problem,
                                                  register_problem)

register_problem(GMM_POLY, GMMPolyProblem)
register_problem(BUMP2D, BumpProblem)

__all__ = ["AffineNormalizer", "Constraint", "NormalizedConstraint", "Problem", "Target",
           "GMMPolyProblem", "GMMTarget", "PolynomialConstraint", "BumpProblem", "BumpTarget",
           "PolygonConstraint", "available_problems", "get_problem", "register_problem"]
