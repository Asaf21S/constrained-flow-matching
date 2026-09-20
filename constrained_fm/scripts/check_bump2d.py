# -*- coding: utf-8 -*-
"""M1 gate: the bump2d target is a normalised density and its sampler matches it.

Checks, all in float64:

  A. ``\\int p = 1`` by midpoint quadrature on a fine grid, independently of the analytic
     normalising constants baked into ``log_prob``.
  B. The analytic 1D marginal agrees with numerically integrating the 2D density.
  C. The analytic mean/std used to build the normalising frame agree with a large sample.
  D. Polygon mass by Monte Carlo agrees with polygon mass by grid quadrature, which is the
     statement that the sampler and the density describe the same distribution.
  E. Diagnostics for the headline figure: how visible the bump is in the x1 marginal, and how
     pure the signal is inside a polygon drawn tightly around it.

    sbatch scripts/run_bump2d_check.sh
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.problems.bump2d import BumpProblem, PolygonConstraint, sample_polygons

DTYPE = torch.float64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the bump2d target and constraints.")
    parser.add_argument("--grid", type=int, default=4000, help="quadrature points per axis")
    parser.add_argument("--pool", type=int, default=2_000_000, help="Monte Carlo pool size")
    parser.add_argument("--num-polys", type=int, default=32)
    parser.add_argument("--figure-samples", type=int, default=100_000,
                        help="sample size the headline marginal will be drawn with")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser


def grid_density(target, grid: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Midpoint grid over the box: cell centres per axis, the density on it, and the cell area."""
    step = target.domain / grid
    centres = (torch.arange(grid, device=device, dtype=DTYPE) + 0.5) * step
    density = torch.empty(grid, grid, device=device, dtype=DTYPE)
    for start in range(0, grid, 128):
        rows = centres[start:start + 128]
        mesh = torch.stack(torch.meshgrid(rows, centres, indexing="ij"), dim=-1)
        density[start:start + 128] = target.log_prob(mesh.reshape(-1, 2)).reshape(rows.shape[0],
                                                                                  grid).exp()
    return centres, density, torch.tensor(step ** 2, device=device, dtype=DTYPE)


def check_normalisation(target, centres, density, cell, tol) -> list[str]:
    total = (density.sum() * cell).item()
    print(f"  integral of p over the box = {total:.10f}")
    if abs(total - 1.0) > tol:
        return [f"density integrates to {total:.10f}, not 1 within {tol}"]
    return []


def check_marginal(target, centres, density, cell, tol) -> list[str]:
    step = (target.domain / centres.shape[0])
    numeric = density.sum(dim=1) * step
    analytic = target.marginal_log_prob(centres, axis=0).exp()
    drift = (numeric - analytic).abs().max().item()
    print(f"  max |numeric - analytic| on the x1 marginal = {drift:.3e}")
    return [] if drift <= tol else [f"marginal drift {drift:.3e} exceeds {tol}"]


def check_moments(target, pool, tol) -> list[str]:
    mean, std = target.mean_std(device=pool.device, dtype=DTYPE)
    emp_mean, emp_std = pool.mean(dim=0), pool.std(dim=0)
    # Three standard errors of the mean, which is the honest bar for a Monte Carlo comparison.
    bar = 3.0 * std / math.sqrt(pool.shape[0])
    print(f"  analytic mean {mean.tolist()} vs sampled {emp_mean.tolist()}")
    print(f"  analytic std  {std.tolist()} vs sampled {emp_std.tolist()}")
    failures = []
    if (mean - emp_mean).abs().gt(bar).any():
        failures.append(f"mean drift {(mean - emp_mean).abs().tolist()} exceeds {bar.tolist()}")
    if (std - emp_std).abs().gt(0.02 * std).any():
        failures.append(f"std drift {(std - emp_std).abs().tolist()} exceeds 2%")
    return failures


def check_polygon_mass(constraints, centres, density, cell, pool, tol) -> list[str]:
    grid = torch.stack(torch.meshgrid(centres, centres, indexing="ij"), dim=-1).reshape(-1, 2)
    weights = density.reshape(-1) * cell
    worst = 0.0
    for constraint in constraints:
        mc = constraint.is_feasible(pool).to(DTYPE).mean().item()
        quad = 0.0
        for start in range(0, grid.shape[0], 1_000_000):
            block = slice(start, start + 1_000_000)
            quad += weights[block][constraint.is_feasible(grid[block])].sum().item()
        worst = max(worst, abs(mc - quad))
    bar = 3.0 / math.sqrt(pool.shape[0]) + tol
    print(f"  max |MC mass - quadrature mass| over {len(constraints)} polygons = {worst:.3e} "
          f"(bar {bar:.3e})")
    return [] if worst <= bar else [f"polygon mass drift {worst:.3e} exceeds {bar:.3e}"]


def report_visibility(target, args, device) -> None:
    """How large the bump is in the x1 marginal, in excess fraction and in Poisson sigmas."""
    edges = torch.linspace(0.0, target.domain, args.bins + 1, device=device, dtype=DTYPE)
    mids = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1] - edges[0]).item()

    log_bg, log_sig = target.marginal_components(mids, axis=0)
    weight = target.signal_weight
    background = (1.0 - weight) * log_bg.exp() * width * args.figure_samples
    signal = weight * log_sig.exp() * width * args.figure_samples

    peak = (log_sig.exp().max() * weight / ((1.0 - weight) * log_bg.exp()[signal.argmax()]))
    significance = (signal / background.clamp_min(1e-12).sqrt()).max().item()
    print(f"  peak excess over background = {peak.item() * 100:.2f}%")
    print(f"  max single-bin significance at N={args.figure_samples} = {significance:.2f} sigma")
    print(f"  signal events in the peak bin = {signal.max().item():.0f} on a background of "
          f"{background[signal.argmax()].item():.0f}")


def report_purity(target, pool, device) -> None:
    """Signal fraction inside a square drawn at +-3 sigma around the signal, as a polygon.

    Computed from the component masses rather than by labelling pool points: the mixture is
    sampled through a Bernoulli mask, so which component a given point came from is not
    recoverable after the fact.
    """
    mu = torch.tensor(target.signal_mean, device=device, dtype=DTYPE)
    sd = torch.tensor(target.signal_sigma, device=device, dtype=DTYPE)
    lo, hi = mu - 3.0 * sd, mu + 3.0 * sd

    background, signal = 1.0, 1.0
    for axis in range(target.dim):
        grid = torch.linspace(lo[axis].item(), hi[axis].item(), 20001, device=device,
                              dtype=DTYPE)
        step = (grid[1] - grid[0]).item()
        log_bg, log_sig = target.marginal_components(grid, axis=axis)
        background *= (log_bg.exp().sum().item() - 0.5 * (log_bg[0].exp().item()
                                                          + log_bg[-1].exp().item())) * step
        signal *= (log_sig.exp().sum().item() - 0.5 * (log_sig[0].exp().item()
                                                       + log_sig[-1].exp().item())) * step

    weight = target.signal_weight
    signal_mass = weight * signal
    background_mass = (1.0 - weight) * background
    purity = signal_mass / (signal_mass + background_mass)

    normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                           device=device, dtype=DTYPE)
    offsets = torch.stack([hi[0], -lo[0], hi[1], -lo[1]])
    mc_mass = PolygonConstraint(normals, offsets).is_feasible(pool).to(DTYPE).mean().item()

    print(f"  +-3 sigma window holds {(signal_mass + background_mass) * 100:.3f}% of the "
          f"target mass (MC {mc_mass * 100:.3f}%)")
    print(f"  signal purity inside that window = {purity * 100:.1f}% "
          f"(vs {weight * 100:.1f}% globally)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    set_seed(args.seed)

    problem = BumpProblem()
    target = problem.target()
    print(f"device {device} | grid {args.grid}^2 | pool {args.pool}")

    torch.set_default_dtype(DTYPE)
    pool = target.sample(args.pool, device=device).to(DTYPE)
    centres, density, cell = grid_density(target, args.grid, device)

    print("\n[A] normalisation")
    failures = check_normalisation(target, centres, density, cell, args.tol)

    print("\n[B] analytic marginal")
    failures += check_marginal(target, centres, density, cell, args.tol)

    print("\n[C] moments used by the normalising frame")
    failures += check_moments(target, pool, args.tol)

    print("\n[D] polygon mass, Monte Carlo vs quadrature")
    constraints, masses = sample_polygons(args.num_polys, pool)
    print(f"  masses span [{masses.min():.3f}, {masses.max():.3f}]")
    failures += check_polygon_mass(constraints, centres, density, cell, pool, args.tol)

    print("\n[E] headline figure diagnostics")
    report_visibility(target, args, device)
    report_purity(target, pool, device)

    if failures:
        print(f"\nFAILED: {len(failures)} problem(s)")
        for line in failures:
            print(f"  {line}")
        return 1

    print("\nPASSED: bump2d is a normalised density with a matching sampler")
    return 0


if __name__ == "__main__":
    sys.exit(main())
