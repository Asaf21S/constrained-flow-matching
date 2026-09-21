# -*- coding: utf-8 -*-
"""Stage 5 for kinematics6d: the spectrum, marginal and correlation panels.

Three figures, all for one showcase shell:

* ``mass_spectrum`` -- the invariant-mass spectra with the requested window shaded.
* ``kinematic_marginals`` -- the leading particle's :math:`(p_T, \\eta, \\phi)`.
* ``corner`` -- the six Cartesian momentum components.

The window pins one scalar function of six coordinates. The last two figures are what
separates a method that respects the conditional distribution from one that merely lands
inside the shell, which the acceptance rate alone cannot distinguish.

    sbatch scripts/run_bumphunt_plots.sh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from constrained_fm.scripts.eval_bench1k import (build_parser as bench_parser, conditioning,
                                                 generate, load_models, rejection_sample,
                                                 resolve_defaults)
from constrained_fm.src.datasets.benchmark_1k import constraints_from, load_benchmark_1k
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.problems.kinematics6d import KinematicsProblem
from constrained_fm.src.visualization.comparison import save_figure
from constrained_fm.src.visualization.kinematics import (plot_corner, plot_kinematic_marginals,
                                                         plot_mass_spectrum)

DEFAULT_OUTDIR = "constrained_fm/baselines/bench1k/kinematics6d"
DEFAULT_FIGURE_DIR = "constrained_fm/images/bench1k/kinematics6d"

PANEL_METHODS = ("explicit", "eci", "hardflow")
# The correlation panel takes the exact conditional, our method, and the baseline that
# distorts this geometry most; four contour sets per axis is unreadable.
CORNER_SERIES = ("truth", "explicit", "hardflow")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the kinematics6d spectrum and correlation panels.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--showcase", type=int, default=None,
                        help="benchmark constraint to draw (default: the tightest shell)")
    return parser


def bench_namespace(methods) -> argparse.Namespace:
    """The eval stage's argument set at its defaults, so sampling here matches scoring there."""
    args = bench_parser().parse_args([])
    args.problem = "kinematics6d"
    args.methods = list(methods)
    resolve_defaults(args)
    return args


def spherical_columns(target, x: torch.Tensor) -> dict[str, np.ndarray]:
    """The leading particle's ``(pT, eta, phi)`` as numpy columns."""
    pt, eta, phi = target.to_spherical(x[:, :3])
    return {"pt": pt.cpu().numpy(), "eta": eta.cpu().numpy(), "phi": phi.cpu().numpy()}


def sample_method(method: str, models: dict, constraint, index: int, problem, normalizer,
                  bench, device, budget: int) -> torch.Tensor:
    """One method's draw for one constraint, returned in physical units."""
    x0 = torch.randn(budget, problem.dim, device=device)
    cond = conditioning(method, models, constraint, index, problem, device)
    with torch.no_grad():
        samples = generate(method, models, constraint, index, x0, problem, normalizer, bench,
                           device, cond)
    return normalizer.inverse(samples).detach()


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.outdir)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    problem = KinematicsProblem()
    normalizer = problem.normalizer().to(device)
    target = problem.target()
    bench = bench_namespace(PANEL_METHODS)
    models = load_models(bench, problem, device)

    benchmark = load_benchmark_1k("kinematics6d", device=device)
    constraints = constraints_from(benchmark, problem, device=device)
    masses = benchmark["mass"].cpu().numpy()
    index = int(np.argmin(masses)) if args.showcase is None else int(args.showcase)
    constraint = constraints[index]

    draws = {"target": target.sample(args.budget, device=device),
             "truth": rejection_sample(constraint, problem, args.budget, index, bench, device)}
    for method in PANEL_METHODS:
        draws[method] = sample_method(method, models, constraint, index, problem, normalizer,
                                      bench, device, args.budget)

    window = (constraint.mass_target - constraint.epsilon,
              constraint.mass_target + constraint.epsilon)
    mass = {name: target.invariant_mass(values).cpu().numpy() for name, values in draws.items()}
    print(f"showcase shell {index} | M* {constraint.mass_target:.2f} GeV | "
          f"epsilon {constraint.epsilon:.3f} GeV | mass {masses[index] * 100:.2f}%")

    artifacts.save_arrays(out, shell_window=np.asarray(window),
                          shell_index=np.asarray([index]),
                          **{f"shell_mass_{name}": values for name, values in mass.items()})

    written = [save_figure(plot_mass_spectrum(mass, window), figure_dir / "mass_spectrum.png")]

    spherical = {name: spherical_columns(target, values) for name, values in draws.items()}
    written.append(save_figure(plot_kinematic_marginals(spherical),
                               figure_dir / "kinematic_marginals.png"))

    corner = {name: draws[name].cpu().numpy() for name in CORNER_SERIES if name in draws}
    written.append(save_figure(plot_corner(corner), figure_dir / "corner.png"))

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
