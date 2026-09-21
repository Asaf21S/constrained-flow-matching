# -*- coding: utf-8 -*-
"""Stage 5 for bump2d: the dataset and headline panels.

Four figures:

* ``target_density`` -- the mixture on a log colour axis, with the signal marked.
* ``polygon_gallery`` -- ten constraints drawn from the frozen benchmark.
* ``marginal_contrast`` -- the headline. One horizontal band isolating the signal, and the
  :math:`x_1` marginal under four treatments at an identical sample budget.
* ``conditional_panels`` -- every method on one benchmark constraint.

The band is built here rather than taken from the benchmark: the model was meta-trained on
random polygons only, so a hand-placed signal region is a generalisation test rather than a
search for a favourable case. Its mass lands inside the benchmark's own ``[0.02, 0.98]``
range, so it is a constraint of a kind the model was trained to handle.

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
from constrained_fm.src.consts import BUMP_SIGNAL_MEAN, BUMP_SIGNAL_SIGMA
from constrained_fm.src.datasets.benchmark_1k import constraints_from, load_benchmark_1k
from constrained_fm.src.datasets.bump_conditioning import signal_fraction
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.problems.bump2d import BumpProblem, PolygonConstraint
from constrained_fm.src.visualization.bumphunt import (plot_conditional_panels,
                                                       plot_marginal_contrast,
                                                       plot_polygon_gallery,
                                                       plot_target_density)
from constrained_fm.src.visualization.comparison import save_figure, short_label

DEFAULT_OUTDIR = "constrained_fm/baselines/bench1k/bump2d"
DEFAULT_FIGURE_DIR = "constrained_fm/images/bench1k/bump2d"

PANEL_METHODS = ("gt", "functa", "eci", "hardflow")
BASELINES = ("eci", "hardflow")
GALLERY_SIZE = 10
# Kept clear of the benchmark's 0..999 so the CAVIA query points cannot coincide with a
# scored constraint's.
BAND_INDEX = 900_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the bump2d dataset and headline panels.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--budget", type=int, default=20000,
                        help="sample budget shared by every panel of the headline figure")
    parser.add_argument("--band-halfwidth", type=float, default=2.9 * BUMP_SIGNAL_SIGMA,
                        help="half-height of the signal band, in units of x2")
    parser.add_argument("--showcase", type=int, default=None,
                        help="benchmark constraint for the per-method panels (default: the "
                             "tightest one, where the baselines are furthest apart)")
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--bins", type=int, default=60)
    return parser


def bench_namespace(methods) -> argparse.Namespace:
    """The eval stage's argument set at its defaults, so sampling here matches scoring there."""
    args = bench_parser().parse_args([])
    args.problem = "bump2d"
    args.methods = list(methods)
    resolve_defaults(args)
    return args


def signal_band(halfwidth: float, domain: float, device) -> PolygonConstraint:
    """An axis-aligned band in :math:`x_2` around the signal, left free in :math:`x_1`.

    Cutting only the coordinate the signal is rare in leaves the :math:`x_1` marginal intact,
    so the bump that appears there is the signal itself and not an artefact of the window.
    """
    lo, hi = BUMP_SIGNAL_MEAN[1] - halfwidth, BUMP_SIGNAL_MEAN[1] + halfwidth
    normals = torch.tensor([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]], device=device)
    offsets = torch.tensor([hi, -lo, domain, 0.0], device=device)
    interior = torch.tensor([BUMP_SIGNAL_MEAN[0], BUMP_SIGNAL_MEAN[1]], device=device)
    return PolygonConstraint(normals, offsets, interior)


def density_grid(target, domain: float, grid_size: int, device) -> np.ndarray:
    axis = torch.linspace(0.0, domain, grid_size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return target.log_prob(points).reshape(grid_size, grid_size).cpu().numpy()


def polygon_arrays(constraint) -> tuple[np.ndarray, np.ndarray]:
    planes = constraint.half_planes.cpu().numpy()
    return planes[:, :2], planes[:, 2]


def sample_method(method: str, models: dict, constraint, index: int, problem, normalizer,
                  bench, device, budget: int) -> torch.Tensor:
    """One method's draw for one constraint, returned in physical units."""
    x0 = torch.randn(budget, problem.dim, device=device)
    cond = conditioning(method, models, constraint, index, problem, device)
    with torch.no_grad():
        samples = generate(method, models, constraint, index, x0, problem, normalizer, bench,
                           device, cond)
    return normalizer.inverse(samples).detach()


def headline(models: dict, problem, normalizer, args, bench, device, out: Path,
             figure_dir: Path) -> list[Path]:
    """The panel the section is built around: one band, one budget, every treatment."""
    target = problem.target()
    constraint = signal_band(args.band_halfwidth, problem.domain, device)

    raw = target.sample(args.budget, device=device)
    inside = raw[constraint.is_feasible(raw)]
    truth = rejection_sample(constraint, problem, args.budget, BAND_INDEX, bench, device)

    draws = {"target": raw, "filtered": inside, "truth": truth}
    for method in ("functa",) + BASELINES:
        samples = sample_method(method, models, constraint, BAND_INDEX, problem, normalizer,
                                bench, device, args.budget)
        key = "generated" if method == "functa" else method
        draws[key] = samples[constraint.is_feasible(samples)]

    edges = np.linspace(0.0, problem.domain, args.bins + 1)
    series: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    annotations: dict[str, str] = {}
    for name, values in draws.items():
        column = values[:, 0].cpu().numpy()
        series[name], _ = np.histogram(column, bins=edges, density=True)
        counts[name] = int(column.size)
        annotations[name] = f"signal {signal_fraction(target, values):.1f}%"

    mass = float(constraint.is_feasible(target.sample(1_000_000, device=device))
                 .double().mean())
    artifacts.save_arrays(out, band_edges=edges, band_mass=np.asarray([mass]),
                          **{f"band_{name}": series[name] for name in series})

    print(f"signal band: mass {mass * 100:.2f}% | kept {counts['filtered']} of "
          f"{counts['target']} raw draws")
    for name in series:
        print(f"  {name:<10} {annotations[name]}  (N {counts[name]})")
    ylim = 1.25 * max(float(series[name].max()) for name in series if name != "hardflow")
    return [save_figure(plot_marginal_contrast(edges, series, counts, annotations, ylim),
                        figure_dir / "marginal_contrast.png")]


def showcase(models: dict, problem, normalizer, args, bench, constraints, masses, device,
             figure_dir: Path) -> list[Path]:
    """Every method on one benchmark constraint, so the failure modes sit side by side."""
    index = int(np.argmin(masses)) if args.showcase is None else int(args.showcase)
    constraint = constraints[index]
    normals, offsets = polygon_arrays(constraint)

    panels = []
    for method in PANEL_METHODS:
        samples = sample_method(method, models, constraint, index, problem, normalizer,
                                bench, device, args.budget)
        rate = constraint.success_rate(samples)
        panels.append({"label": short_label(method), "samples": samples.cpu().numpy(),
                       "normals": normals, "offsets": offsets,
                       "caption": f"AR {rate:.1f}%"})

    print(f"showcase constraint {index} | mass {masses[index] * 100:.2f}%")
    return [save_figure(plot_conditional_panels(panels, problem.domain),
                        figure_dir / "conditional_panels.png")]


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.outdir)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    problem = BumpProblem()
    normalizer = problem.normalizer().to(device)
    bench = bench_namespace(PANEL_METHODS)
    models = load_models(bench, problem, device)

    benchmark = load_benchmark_1k("bump2d", device=device)
    constraints = constraints_from(benchmark, problem, device=device)
    masses = benchmark["mass"].cpu().numpy()
    target = problem.target()

    log_density = density_grid(target, problem.domain, args.grid_size, device)
    artifacts.save_arrays(out, log_density=log_density)
    written = [save_figure(plot_target_density(log_density, problem.domain, BUMP_SIGNAL_MEAN),
                           figure_dir / "target_density.png")]

    picks = np.linspace(0, len(constraints) - 1, GALLERY_SIZE).astype(int)
    picks = picks[np.argsort(masses[picks])]
    gallery = [polygon_arrays(constraints[i]) for i in picks]
    background = target.sample(6000, device=device).cpu().numpy()
    written.append(save_figure(
        plot_polygon_gallery(gallery, masses[picks], background, problem.domain),
        figure_dir / "polygon_gallery.png"))

    written += headline(models, problem, normalizer, args, bench, device, out, figure_dir)
    written += showcase(models, problem, normalizer, args, bench, constraints, masses,
                        device, figure_dir)

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
