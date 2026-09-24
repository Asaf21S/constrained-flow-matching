# -*- coding: utf-8 -*-
"""Stage 5 for bump2d: the dataset and headline panels.

Figures:

* ``target_density`` -- the mixture on a linear and a log colour axis, with the signal marked.
* ``polygon_gallery`` -- ten constraints drawn from the frozen benchmark.
* ``signal_region`` -- the headline. One hand-drawn polygon around the signal, and the
  conditional density under five treatments at an identical sample budget.
* ``signal_region_benchmark`` -- the same panels for a benchmark polygon that contains the
  signal, chosen by a rule that looks at no method's output; ``signal_table.md`` summarises
  every such polygon.
* ``conditional_panels`` -- every method on one benchmark constraint.

The hand-drawn polygon is not in the benchmark: the model was meta-trained on random
polygons only, so it is a generalisation test rather than a search for a favourable case.

    sbatch scripts/run_bumphunt_plots.sh
"""

from __future__ import annotations

import argparse
import json
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
from constrained_fm.src.visualization.bumphunt import (half_plane_vertices,
                                                       plot_conditional_panels,
                                                       plot_polygon_gallery,
                                                       plot_signal_heatmaps,
                                                       plot_target_density)
from constrained_fm.src.visualization.comparison import save_figure, short_label

DEFAULT_OUTDIR = "constrained_fm/baselines/bench1k/bump2d"
DEFAULT_FIGURE_DIR = "constrained_fm/images/bench1k/bump2d"

PANEL_METHODS = ("gt", "functa", "eci", "hardflow")
GENERATIVE = ("functa", "eci", "hardflow")
HEATMAP_LABELS = {"truth": "Exact conditional", "filtered": "Rejection filtering",
                  "functa": "Ours", "eci": "ECI", "hardflow": "HardFlow"}
GALLERY_SIZE = 10
# Counter-clockwise, drawn by eye around the signal; convex, inside the box.
SIGNAL_POLYGON = ((0.3, 4.8), (3.8, 4.2), (4.6, 6.8), (3.0, 9.2), (0.4, 8.6))
# Kept clear of the benchmark's 0..999 so the CAVIA query points cannot coincide with a
# scored constraint's.
SIGNAL_INDEX = 900_000
FILTER_SEED = 60_000
SHOWCASE_MASS = 0.10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the bump2d dataset and headline panels.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--budget", type=int, default=20000,
                        help="sample budget shared by every panel of the headline figures")
    parser.add_argument("--min-signal", type=float, default=10.0,
                        help="exact signal fraction (%%) a benchmark polygon needs to enter "
                             "the signal table")
    parser.add_argument("--showcase", type=int, default=None,
                        help="benchmark constraint for the per-method panels (default: the "
                             "one whose mass is closest to 10%%)")
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--bins", type=int, default=48)
    return parser


def bench_namespace(methods) -> argparse.Namespace:
    """The eval stage's argument set at its defaults, so sampling here matches scoring there."""
    args = bench_parser().parse_args([])
    args.problem = "bump2d"
    args.methods = list(methods)
    resolve_defaults(args)
    return args


def signal_polygon(domain: float, device) -> PolygonConstraint:
    """The hand-drawn pentagon :data:`SIGNAL_POLYGON` plus the four box faces.

    For a counter-clockwise vertex list the outward normal of the edge
    :math:`v_i \\to v_{i+1}` is the edge vector rotated by :math:`-90^\\circ`, normalised;
    the offset is its projection of :math:`v_i`. The box faces match the benchmark's
    polygons, which always carry them.
    """
    vertices = torch.tensor(SIGNAL_POLYGON, device=device)
    edges = torch.roll(vertices, -1, dims=0) - vertices
    normals = torch.stack([edges[:, 1], -edges[:, 0]], dim=-1)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    offsets = (normals * vertices).sum(dim=-1)
    box_normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=device)
    box_offsets = torch.tensor([domain, 0.0, domain, 0.0], device=device)
    return PolygonConstraint(torch.cat([normals, box_normals]),
                             torch.cat([offsets, box_offsets]), vertices.mean(dim=0))


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


def safe_signal_fraction(target, x: torch.Tensor) -> float:
    return signal_fraction(target, x) if x.shape[0] else float("nan")


def heatmap_extent(vertices: np.ndarray, domain: float,
                   pad: float = 0.5) -> tuple[float, float, float, float]:
    """The polygon's bounding box, padded so leaked samples stay visible, clipped to the box."""
    lo = np.clip(vertices.min(axis=0) - pad, 0.0, domain)
    hi = np.clip(vertices.max(axis=0) + pad, 0.0, domain)
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def density_image(x: torch.Tensor, total: int, extent: tuple[float, float, float, float],
                  bins: int) -> np.ndarray:
    """Histogram density :math:`\\text{count} / (N \\cdot \\text{bin area})`, rows along x2.

    Dividing by the method's full budget ``total`` rather than by the points that land in
    the frame keeps mass a method places elsewhere from inflating the image.
    """
    points = x.cpu().numpy()
    counts, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=bins,
                                  range=[extent[2:], extent[:2]])
    area = (extent[1] - extent[0]) * (extent[3] - extent[2]) / bins ** 2
    return counts / (max(total, 1) * area)


def signal_panels(models: dict, constraint, index: int, problem, normalizer, bench, device,
                  budget: int, bins: int) -> tuple[list[dict], dict, tuple, float]:
    """Heat-map panels for one polygon, in the order the figure shows them.

    Exact conditional: rejection sampling repeated until ``budget`` points are accepted.
    Rejection filtering: ``budget`` unconstrained draws, of which only the accepted ones
    remain. The generative methods get ``budget`` draws each; every draw is shown, and the
    signal fraction is taken over the ones that satisfy the constraint.
    """
    target = problem.target()
    normals, offsets = polygon_arrays(constraint)
    extent = heatmap_extent(half_plane_vertices(normals, offsets), problem.domain)

    truth = rejection_sample(constraint, problem, budget, index, bench, device)
    raw = target.sample(budget, device=device)
    inside = raw[constraint.is_feasible(raw)]
    stats = {"truth": {"signal": safe_signal_fraction(target, truth), "n": budget},
             "filtered": {"signal": safe_signal_fraction(target, inside),
                          "n": int(inside.shape[0])}}
    images = {"truth": density_image(truth, budget, extent, bins),
              "filtered": density_image(inside, inside.shape[0], extent, bins)}
    captions = {"truth": f"N = {budget:,}\nsignal {stats['truth']['signal']:.1f}%",
                "filtered": (f"N = {inside.shape[0]:,} of {budget:,}\n"
                             f"signal {stats['filtered']['signal']:.1f}%")}

    for method in GENERATIVE:
        samples = sample_method(method, models, constraint, index, problem, normalizer,
                                bench, device, budget)
        kept = samples[constraint.is_feasible(samples)]
        rate = constraint.success_rate(samples)
        stats[method] = {"signal": safe_signal_fraction(target, kept), "n": budget,
                         "success_rate": rate}
        images[method] = density_image(samples, budget, extent, bins)
        captions[method] = f"SR {rate:.1f}%\nsignal {stats[method]['signal']:.1f}%"

    panels = [{"key": key, "label": HEATMAP_LABELS[key], "density": images[key],
               "caption": captions[key]} for key in HEATMAP_LABELS]
    return panels, stats, extent, float(images["truth"].max())


def render_heatmaps(panels: list[dict], constraint, extent: tuple, vmax: float,
                    path: Path) -> Path:
    normals, offsets = polygon_arrays(constraint)
    fig = plot_signal_heatmaps(panels, normals, offsets, extent, BUMP_SIGNAL_MEAN,
                               2.0 * BUMP_SIGNAL_SIGMA, vmax)
    return save_figure(fig, path)


def hand_drawn_region(models: dict, problem, normalizer, args, bench, device, out: Path,
                      figure_dir: Path) -> list[Path]:
    """The panel the section is built around: one polygon, one budget, every treatment."""
    constraint = signal_polygon(problem.domain, device)
    panels, stats, extent, vmax = signal_panels(models, constraint, SIGNAL_INDEX, problem,
                                                normalizer, bench, device, args.budget,
                                                args.bins)
    mass = float(constraint.is_feasible(problem.target().sample(1_000_000, device=device))
                 .double().mean())
    artifacts.save_arrays(out, signal_region_extent=np.asarray(extent),
                          signal_region_mass=np.asarray([mass]),
                          **{f"signal_region_{p['key']}": p["density"] for p in panels})

    print(f"hand-drawn polygon: mass {mass * 100:.2f}%")
    for key, row in stats.items():
        print(f"  {key:<10} signal {row['signal']:6.2f}%  (N {row['n']})"
              + (f"  SR {row['success_rate']:.2f}%" if "success_rate" in row else ""))
    return [render_heatmaps(panels, constraint, extent, vmax,
                            figure_dir / "signal_region.png")]


def filtered_signal(constraint, problem, index: int, budget: int, device) -> float:
    """Signal fraction of ``budget`` unconstrained draws after keeping the feasible ones."""
    with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
        torch.manual_seed(FILTER_SEED + index)
        raw = problem.target().sample(budget, device=device)
    return safe_signal_fraction(problem.target(), raw[constraint.is_feasible(raw)])


def signal_table(rows: dict[str, np.ndarray], exact: np.ndarray, rates: dict[str, np.ndarray],
                 count: int, min_signal: float) -> str:
    header = ("| method | median signal fraction (%) | median ratio to exact | "
              "ratio 25th-75th percentile | median SR (%) |")
    lines = [f"### Signal recovery on the {count} benchmark polygons that contain the signal "
             f"centre and hold at least {min_signal:g}% signal under the exact conditional\n",
             header, "|:---|:---|:---|:---|:---|"]
    for key, values in rows.items():
        ratio = values / exact
        rate = rates.get(key)
        lines.append(f"| {HEATMAP_LABELS[key]} | {np.nanmedian(values):.1f} | "
                     f"{np.nanmedian(ratio):.2f} | {np.nanpercentile(ratio, 25):.2f}-"
                     f"{np.nanpercentile(ratio, 75):.2f} | "
                     + ("--" if rate is None else f"{np.nanmedian(rate):.1f}") + " |")
    return "\n".join(lines) + "\n"


def benchmark_region(models: dict, problem, normalizer, args, bench, constraints, device,
                     out: Path, figure_dir: Path) -> list[Path]:
    """Every benchmark polygon containing the signal, and the median one of them drawn.

    Eligibility and the drawn polygon depend on the exact conditional only: the polygon is
    the one whose exact signal fraction is the median among the eligible ones.
    """
    per_method = json.loads((out / "metrics.json").read_text())["methods"]
    if "signal_fraction" not in per_method.get("gt", {}).get("per_shape", {}):
        print("skipping signal table: metrics.json has no signal_fraction -- re-run eval")
        return []
    signal = {m: np.asarray(per_method[m]["per_shape"]["signal_fraction"], dtype=float)
              for m in PANEL_METHODS}
    rates = {m: np.asarray(per_method[m]["per_shape"]["success_rate"], dtype=float)
             for m in GENERATIVE}

    centre = torch.tensor([BUMP_SIGNAL_MEAN], device=device)
    contains = np.array([bool(c.is_feasible(centre)[0]) for c in constraints])
    eligible = np.flatnonzero(contains & (signal["gt"] >= args.min_signal))
    if eligible.size == 0:
        print("skipping signal table: no eligible benchmark polygon")
        return []

    exact = signal["gt"][eligible]
    rows = {"truth": exact,
            "filtered": np.array([filtered_signal(constraints[i], problem, int(i), args.budget,
                                                  device) for i in eligible])}
    rows.update({m: signal[m][eligible] for m in GENERATIVE})
    table = signal_table(rows, exact, {m: rates[m][eligible] for m in GENERATIVE},
                         int(eligible.size), args.min_signal)
    table_path = figure_dir / "signal_table.md"
    table_path.write_text(table)
    print(f"{int(contains.sum())} benchmark polygons contain the signal centre\n{table}")

    index = int(eligible[np.argsort(exact)[eligible.size // 2]])
    constraint = constraints[index]
    panels, stats, extent, vmax = signal_panels(models, constraint, index, problem,
                                                normalizer, bench, device, args.budget,
                                                args.bins)
    print(f"median benchmark polygon {index}")
    for key, row in stats.items():
        print(f"  {key:<10} signal {row['signal']:6.2f}%  (N {row['n']})"
              + (f"  SR {row['success_rate']:.2f}%" if "success_rate" in row else ""))
    return [table_path, render_heatmaps(panels, constraint, extent, vmax,
                                        figure_dir / "signal_region_benchmark.png")]


def showcase(models: dict, problem, normalizer, args, bench, constraints, masses, device,
             figure_dir: Path) -> list[Path]:
    """Every method on one benchmark constraint, so the failure modes sit side by side.

    The default constraint is the one whose mass is closest to :data:`SHOWCASE_MASS`: a rule
    that looks at no method's output.
    """
    index = (int(np.argmin(np.abs(masses - SHOWCASE_MASS))) if args.showcase is None
             else int(args.showcase))
    constraint = constraints[index]
    normals, offsets = polygon_arrays(constraint)

    panels = []
    for method in PANEL_METHODS:
        samples = sample_method(method, models, constraint, index, problem, normalizer,
                                bench, device, args.budget)
        rate = constraint.success_rate(samples)
        panels.append({"label": short_label(method), "samples": samples.cpu().numpy(),
                       "normals": normals, "offsets": offsets,
                       "caption": f"SR {rate:.1f}%"})
        print(f"  {method:<10} SR {rate:.2f}%")

    print(f"showcase constraint {index} | mass {masses[index] * 100:.2f}%")
    return [save_figure(plot_conditional_panels(panels, problem.domain),
                        figure_dir / "conditional_panels.png")]


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.outdir)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

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

    written += hand_drawn_region(models, problem, normalizer, args, bench, device, out,
                                 figure_dir)
    written += benchmark_region(models, problem, normalizer, args, bench, constraints, device,
                                out, figure_dir)
    written += showcase(models, problem, normalizer, args, bench, constraints, masses,
                        device, figure_dir)

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
