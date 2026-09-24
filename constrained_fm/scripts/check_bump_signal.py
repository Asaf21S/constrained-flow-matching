# -*- coding: utf-8 -*-
"""Where the 1% signal survives, and where it is lost.

The signal is a narrow Gaussian carrying one percent of the target's mass, sitting far out
in the :math:`x_2` tail of the background. Any method that is going to be useful for a bump
hunt has to reproduce it *inside* a window that cuts the bulk away, where it stops being a
one-percent perturbation and becomes a third of the conditional mass.

This walks the chain and reports the signal fraction each stage recovers inside the
hand-drawn signal polygon of :mod:`plot_bumphunt`:

1. the target itself, as the reference,
2. the unconstrained base flow, filtered to the polygon,
3. the SIREN's reconstruction of the polygon, as a mass IoU,
4. the amortized conditional model, conditioned on the polygon.

The signal fraction is estimated by the posterior weight of the signal component under the
true mixture, averaged over samples, which needs no fit and no binning:

.. math::
    \\hat{w} = \\frac{1}{N} \\sum_n
        \\frac{w\\,\\mathcal{N}(x_n; \\mu_s, \\Sigma_s)}
             {w\\,\\mathcal{N}(x_n; \\mu_s, \\Sigma_s) + (1 - w)\\,p_{\\mathrm{bg}}(x_n)}.

    python -m constrained_fm.scripts.check_bump_signal
"""

from __future__ import annotations

import argparse

import torch

from constrained_fm.scripts.eval_bench1k import (conditioning, generate, load_models,
                                                 rejection_sample)
from constrained_fm.scripts.plot_bumphunt import SIGNAL_INDEX, bench_namespace, signal_polygon
from constrained_fm.src.consts import BUMP_SIREN_TAU
from constrained_fm.src.datasets.bump_conditioning import mass_iou, signal_fraction
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.problems.bump2d import BumpProblem

METHODS = ("gt", "functa", "eci", "hardflow")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit signal recovery inside the polygon.")
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--grid-size", type=int, default=256)
    return parser


def siren_iou(models: dict, constraint, problem, points: torch.Tensor, device) -> float:
    """Mass IoU between the SIREN's decoded region and the true polygon.

    Counted on target-distributed points, so an error is weighted by the probability mass it
    misplaces rather than by the area it covers.
    """
    cond = conditioning("functa", models, constraint, SIGNAL_INDEX, problem, device)
    planes = constraint.half_planes
    shapes = {"normals": planes[:, :2].unsqueeze(0),
              "offsets": planes[:, 2].unsqueeze(0),
              "active": torch.ones(1, planes.shape[0], dtype=torch.bool, device=device)}
    return float(mass_iou(models["siren"], cond["z"].unsqueeze(0), shapes, points,
                          domain=problem.domain).mean())


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device()
    problem = BumpProblem()
    normalizer = problem.normalizer().to(device)
    target = problem.target()
    bench = bench_namespace(METHODS)
    models = load_models(bench, problem, device)

    constraint = signal_polygon(problem.domain, device)
    pool = target.sample(1_000_000, device=device)
    mass = float(constraint.is_feasible(pool).double().mean())
    print(f"polygon mass {mass * 100:.2f}%  |  tau {models['functa_cfg']['tau']} "
          f"(consts {BUMP_SIREN_TAU})")
    print(f"unconditional signal fraction  {signal_fraction(target, pool):7.3f}%")

    truth = rejection_sample(constraint, problem, args.budget, SIGNAL_INDEX, bench, device)
    print(f"exact conditional              {signal_fraction(target, truth):7.3f}%   <- the target")

    base = models.get("base")
    if base is not None:
        with torch.no_grad():
            drawn = base.sample(num_points=500_000, step_size=bench.step_size, device=device)
        physical = normalizer.inverse(drawn)
        inside = physical[constraint.is_feasible(physical)]
        print(f"base flow, filtered to polygon {signal_fraction(target, inside):7.3f}%   "
              f"({inside.shape[0]} of 500000)")

    print(f"SIREN polygon reconstruction IoU "
          f"{siren_iou(models, constraint, problem, pool[:100_000], device):7.3f}")

    x0 = torch.randn(args.budget, problem.dim, device=device)
    for method in ("functa", "eci", "hardflow"):
        cond = conditioning(method, models, constraint, SIGNAL_INDEX, problem, device)
        with torch.no_grad():
            samples = generate(method, models, constraint, SIGNAL_INDEX, x0, problem,
                               normalizer, bench, device, cond)
        physical = normalizer.inverse(samples)
        kept = physical[constraint.is_feasible(physical)]
        rate = constraint.success_rate(physical)
        recovered = signal_fraction(target, kept) if kept.numel() else float("nan")
        print(f"{method:<30} {recovered:7.3f}%   (SR {rate:.2f}%)")


if __name__ == "__main__":
    main()
