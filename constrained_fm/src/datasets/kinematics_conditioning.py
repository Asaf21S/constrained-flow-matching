# -*- coding: utf-8 -*-
"""Conditional training data for the kinematics6d mass shells.

A shell constrains one scalar function of the state, so in a pool sorted by invariant mass
its feasible set is a *contiguous slice*. Drawing exact conditional samples is then two
binary searches and an index draw, with none of the rejection machinery the 2D polygons need:
there is no acceptance rate to degrade as the constraint tightens, so the thinnest shells in
the benchmark cost exactly what the widest ones do.
"""

from __future__ import annotations

import math

import torch

from constrained_fm.src.consts import KIN_SHELL_MAX_MASS, KIN_SHELL_MIN_MASS
from constrained_fm.src.problems.kinematics6d import KinematicsTarget, solve_epsilon


class ShellPool:
    """A mass-sorted pool of target samples, and conditional draws from it.

    Args:
        target: the unconstrained 6D target.
        pool_size: number of samples to draw once, up front.
        device: where the pool lives; it is used every training step.
    """

    def __init__(self, target: KinematicsTarget, pool_size: int, device: torch.device):
        self.target = target
        self.scale = target.mass_scale()

        pool = target.sample(pool_size, device=device)
        mass = target.invariant_mass(pool)
        order = mass.argsort()
        self.pool = pool[order]
        self.mass = mass[order].to(torch.float64)

    def draw_shells(self, num_shells: int, min_mass: float = KIN_SHELL_MIN_MASS,
                    max_mass: float = KIN_SHELL_MAX_MASS
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Centres and half-widths for shells log-uniform in the mass they enclose.

        Log-uniform rather than uniform because the benchmark spans two decades of shell
        mass; sampling linearly would spend almost every training step on the wide, easy
        windows and leave the tight ones underrepresented.
        """
        device = self.mass.device
        log_lo, log_hi = math.log(min_mass), math.log(max_mass)
        fraction = torch.rand(num_shells, dtype=torch.float64, device=device)
        fraction = (log_lo + fraction * (log_hi - log_lo)).exp()

        quantiles = torch.rand(num_shells, dtype=torch.float64, device=device)
        centre = self.mass[(quantiles * (self.mass.numel() - 1)).long()]
        return centre, solve_epsilon(self.mass, centre, fraction)

    def draw(self, num_shells: int, points_per_shell: int
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact conditional samples and their matching conditioning vectors.

        Returns:
            ``x_1`` of shape (num_shells * points_per_shell, 6) in physical units, and
            ``params`` of shape (num_shells * points_per_shell, 2).
        """
        centre, epsilon = self.draw_shells(num_shells)
        lo = torch.searchsorted(self.mass, (centre - epsilon).contiguous())
        hi = torch.searchsorted(self.mass, (centre + epsilon).contiguous())

        width = (hi - lo).clamp_min(1)
        offsets = torch.rand(num_shells, points_per_shell, device=self.mass.device,
                             dtype=torch.float64)
        index = (lo.unsqueeze(-1) + (offsets * width.unsqueeze(-1)).long()).clamp_(
            max=self.mass.numel() - 1)

        params = torch.stack([centre / self.scale, (epsilon / self.scale).log()], dim=-1)
        params = params.unsqueeze(1).expand(-1, points_per_shell, -1)
        return self.pool[index.reshape(-1)], params.reshape(-1, 2).to(torch.float32)
