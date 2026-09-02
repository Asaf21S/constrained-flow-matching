# -*- coding: utf-8 -*-
"""Benchmarks the ECI and HardFlow inference-time baselines on the 100-polynomial set.

Both reuse the single unconstrained checkpoint from ``train_base_fm.py`` and inject the
constraint only during integration, so no training happens here. Scored with the same
success rate / SWD / MMD / JSD protocol as every other model in the project.

NLL and KLD are deliberately reported as NaN: ECI overwrites the state and HardFlow
overwrites the velocity, so the sampled distribution is no longer the one whose density the
probability-flow ODE integrates. A likelihood computed from the base field would describe a
model that was never sampled from.

    sbatch scripts/run_eci_hardflow.sh
    python -m constrained_fm.scripts.eci_hardflow --methods eci --guidance-scale 20
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from tqdm import tqdm

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.experiment.registry import readme_table, summarize
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.inference.constrained_samplers import (DEFAULT_CHUNK, DEFAULT_STEPS,
                                                               sample_eci, sample_euler,
                                                               sample_hardflow)
from constrained_fm.src.inference.constraint_projection import DEFAULT_MARGIN
from constrained_fm.src.inference.evaluator import evaluate_validation_set_metrics
from constrained_fm.src.metrics.functa_fidelity import constraint_masses
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.visualization import diagnostics as diag

BASE_CKPT = "constrained_fm/baselines/base_fm/ckpt.pt"
DEFAULT_OUTDIR = "constrained_fm/baselines"
DEFAULT_FIGURE_DIR = "constrained_fm/images/functa/eci_hardflow"

# Methods that alter trajectories manually; exact density estimation does not survive them.
UNDEFINED_LIKELIHOOD_KEYS = ("nll", "kld")

# Constraints for the README figure, picked to span the range of feasible-region masses.
SHOWCASE_MASS_QUANTILES = (0.05, 0.35, 0.65, 0.95)
METHOD_LABELS = {"eci": "ECI", "hardflow": "HardFlow"}
SHOWCASE_POINTS = 4000

# Baselines pulled into the final table when their metrics.json already exists.
REFERENCE_BASELINES = [
    ("Coefficient-conditioned", "constrained_fm/baselines/poly_fm/metrics.json"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ECI and HardFlow on the validation set.")
    parser.add_argument("--methods", nargs="+", default=["eci", "hardflow"],
                        choices=["eci", "hardflow"])
    parser.add_argument("--ckpt", default=BASE_CKPT)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Euler integration steps")
    parser.add_argument("--correction-loops", type=int, default=1, help="ECI projections per step")
    parser.add_argument("--projection-iters", type=int, default=16, help="ECI Newton iterations")
    # Swept over 10/30/100/300: 100 is the knee, best on every distributional metric.
    parser.add_argument("--guidance-scale", type=float, default=100.0, help="HardFlow gradient weight")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help="how far inside P(x) = 0 both methods aim, in units of P")

    parser.add_argument("--num-polys", type=int, default=100)
    parser.add_argument("--num-x0", type=int, default=10000)
    parser.add_argument("--gmm-pool-size", type=int, default=100000)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--degree", type=int, default=POLYNOMIAL_DEGREE)
    parser.add_argument("--scale", type=float, default=PLANE_SCALE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    return parser


def load_base_model(args, device: torch.device) -> UnconstrainedFM:
    path = Path(args.ckpt)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/run_base_fm.sh first")
    model = UnconstrainedFM(time_dim=args.time_dim, hidden_dim=args.hidden_dim,
                            num_blocks=args.num_blocks).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def generate(method: str, model, x0: torch.Tensor, polys: torch.Tensor, args) -> torch.Tensor:
    """Samples every validation constraint with one method; returns (C, N, 2)."""
    per_shape = []
    for i in tqdm(range(polys.shape[0]), desc=f"{method} sampling"):
        if method == "eci":
            samples = sample_eci(model, x0, polys[i], degree=args.degree, scale=args.scale,
                                 steps=args.steps, correction_loops=args.correction_loops,
                                 margin=args.margin, projection_iters=args.projection_iters,
                                 chunk_size=args.chunk_size)
        else:
            samples = sample_hardflow(model, x0, polys[i], degree=args.degree, scale=args.scale,
                                      steps=args.steps, guidance_scale=args.guidance_scale,
                                      margin=args.margin, chunk_size=args.chunk_size)
        per_shape.append(samples.detach())
    return torch.stack(per_shape, dim=0)


def score(method: str, samples: torch.Tensor, gmm_pool: torch.Tensor, polys: torch.Tensor,
          mass: torch.Tensor, args, device: torch.device) -> dict:
    """Runs the shared benchmark, then blanks the metrics the method invalidates."""
    metrics = evaluate_validation_set_metrics(samples, x_true_pool=gmm_pool, coeffs=polys,
                                              degree=args.degree, scale=args.scale,
                                              model=None, nll_points=0, device=device)

    per_shape = {key: [float(v) for v in values] for key, values in metrics.items()}
    per_shape["mass"] = mass.cpu().tolist()
    for key in UNDEFINED_LIKELIHOOD_KEYS:
        per_shape[key] = [float("nan")] * polys.shape[0]

    return {
        "method": method,
        "model": "UnconstrainedFM + inference-time constraint",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "likelihood": "undefined -- trajectories are altered outside the probability-flow ODE",
        "sampling": {"steps": args.steps, "margin": args.margin,
                     "correction_loops": args.correction_loops if method == "eci" else None,
                     "projection_iters": args.projection_iters if method == "eci" else None,
                     "guidance_scale": args.guidance_scale if method == "hardflow" else None},
        "eval": {"num_polys": args.num_polys, "num_x0": args.num_x0,
                 "gmm_pool_size": args.gmm_pool_size},
        "per_shape": per_shape,
        "summary": summarize(per_shape),
    }


def render_figures(method: str, samples: torch.Tensor, polys: torch.Tensor,
                   per_shape: dict[str, list[float]], figure_dir: Path, args) -> None:
    """Best / quartile / worst constraints by success rate, so the failure mode is visible."""
    order = np.argsort(per_shape["success_rate"])
    picks = sorted({int(order[0]), int(order[len(order) // 4]), int(order[len(order) // 2]),
                    int(order[(3 * len(order)) // 4]), int(order[-1])})

    diag.save_figure(
        diag.plot_final_samples_gallery(
            [samples[i].cpu().numpy() for i in picks],
            [polys[i] for i in picks],
            [f"shape {i} | SR {per_shape['success_rate'][i]:.2f}%\n"
             f"SWD {per_shape['swd'][i]:.4f} | JSD {per_shape['jsd'][i]:.4f}" for i in picks],
            degree=args.degree, scale=args.scale),
        figure_dir / f"{method}_gallery.png")


def pick_showcase_shapes(mass: torch.Tensor) -> list[int]:
    """Constraint indices spanning the mass range, so the figure shows easy and hard regions."""
    order = torch.argsort(mass).tolist()
    return [order[min(int(q * len(order)), len(order) - 1)] for q in SHOWCASE_MASS_QUANTILES]


def render_comparison(showcase: dict[str, torch.Tensor], polys: torch.Tensor, picks: list[int],
                      mass: torch.Tensor, per_shape_by_method: dict[str, dict], figure_dir: Path,
                      args) -> None:
    """One row per showcase constraint, one column per method, with the unconstrained reference.

    Reads as a ledger: the leftmost column is the distribution the base model actually learned,
    and each method to its right shows what enforcing the boundary did to it.
    """
    columns = ["base"] + [m for m in args.methods if m in showcase]
    col_labels = ["Unconstrained (base FM)"] + [METHOD_LABELS[m] for m in columns[1:]]
    row_labels = [f"shape {i}\nmass {float(mass[i]):.2f}" for i in picks]

    grid, cells = [], []
    for row, shape in enumerate(picks):
        grid.append([showcase[m][row].numpy() for m in columns])
        cells.append([""] + [f"SR {per_shape_by_method[m]['success_rate'][shape]:.1f}% | "
                             f"SWD {per_shape_by_method[m]['swd'][shape]:.3f}"
                             for m in columns[1:]])

    diag.save_figure(
        diag.plot_samples_ablation_grid(grid, [polys[i] for i in picks], row_labels, col_labels,
                                        cell_labels=cells, degree=args.degree, scale=args.scale),
        figure_dir / "eci_hardflow_comparison.png")


def format_value(summary: dict[str, float], key: str, digits: int) -> str:
    value = summary.get(f"{key}_median", float("nan"))
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def comparison_table(rows: list[tuple[str, dict[str, float]]]) -> str:
    """Markdown table of median metrics, one column per approach."""
    labels = [label for label, _ in rows]
    lines = ["| Metric (median) | " + " | ".join(labels) + " |",
             "| :--- |" + " ---: |" * len(labels)]
    for key, name, digits in [("success_rate", "Success Rate (%)", 2), ("swd", "SWD", 4),
                              ("mmd", "MMD", 5), ("jsd", "JSD", 4), ("nll", "NLL", 4),
                              ("kld", "KLD", 4)]:
        cells = " | ".join(format_value(summary, key, digits) for _, summary in rows)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


def load_reference_rows() -> list[tuple[str, dict[str, float]]]:
    rows = []
    for label, path in REFERENCE_BASELINES:
        payload = Path(path)
        if payload.exists():
            rows.append((label, json.loads(payload.read_text()).get("summary", {})))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    model = load_base_model(args, device)
    print(f"device {device} | base model {args.ckpt}")

    gmm_pool, _ = get_points(args.gmm_pool_size, device=device)
    val_set = get_validation_set(device=device)
    polys = val_set["polynomials"][:args.num_polys].to(device)
    x0 = val_set["x0"][:args.num_x0].to(device)
    mass = constraint_masses(polys, gmm_pool, degree=args.degree, scale=args.scale)
    print(f"{polys.shape[0]} constraints | {x0.shape[0]} samples each | {args.steps} steps")

    picks = pick_showcase_shapes(mass)
    base = sample_euler(model, x0[:SHOWCASE_POINTS], steps=args.steps, chunk_size=args.chunk_size)
    showcase = {"base": base.cpu().repeat(len(picks), 1, 1)}
    per_shape_by_method = {}

    results = []
    for method in args.methods:
        samples = generate(method, model, x0, polys, args)
        record = score(method, samples, gmm_pool, polys, mass, args, device)

        out = Path(args.outdir) / method
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(record, indent=2))
        render_figures(method, samples, polys, record["per_shape"], figure_dir, args)
        showcase[method] = samples[picks, :SHOWCASE_POINTS].cpu()
        per_shape_by_method[method] = record["per_shape"]

        print(f"\n### {method}")
        print(readme_table(record["summary"]))
        print("NLL / KLD: undefined for this method (trajectories altered outside the ODE)")
        results.append((method.upper(), record["summary"]))
        del samples

    render_comparison(showcase, polys, picks, mass, per_shape_by_method, figure_dir, args)

    print("\n### comparison")
    print(comparison_table(load_reference_rows() + results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
