# -*- coding: utf-8 -*-
"""Inference sample-efficiency ablation: how far can the query budget N drop at test time?

The SIREN and the flow matcher are both frozen. Only N -- the number of query points fed
to the 15-step CAVIA inner loop at inference -- varies. This isolates how much of the
downstream generative quality is bought by the extraction query budget alone.

The inner loop is already N-invariant: extract_latents_batched reduces with .mean(dim=1)
over points and .sum() over shapes, so each z_i receives the gradient of its own mean MSE
and the effective step size does not move with N. Only the gradient's variance does.

Writes runs/<run_id>/ablations/query_points/ with metrics.json and four figures.

    python -m constrained_fm.scripts.ablate_query_points --run-id <run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.experiment.config import ExperimentConfig
from constrained_fm.src.experiment.registry import load_config, run_dir, write_json
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_siren,
                                                   resolve_device, set_seed)
from constrained_fm.src.geometry.polynomials import compute_poly_features_batched, evaluate_poly_batched
from constrained_fm.src.inference.evaluator import (evaluate_validation_set_metrics,
                                                    run_evaluation_inference)
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.metrics.functa_fidelity import constraint_masses, region_iou_batched
from constrained_fm.src.visualization import diagnostics as diag

DEFAULT_N_VALUES = [50, 100, 300, 500, 1000, 2000]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ablate the inference-time CAVIA query budget.")
    parser.add_argument("--run-id", required=True, help="run whose frozen SIREN + FM checkpoint to use")
    parser.add_argument("--num-points", type=int, nargs="+", default=DEFAULT_N_VALUES,
                        help="query-point counts N to sweep at inference")
    parser.add_argument("--num-polys", type=int, default=None,
                        help="validation polynomials to sweep; defaults to the run's eval setting")
    parser.add_argument("--num-x0", type=int, default=None,
                        help="particles per shape for the FM half; defaults to the run's eval setting")
    parser.add_argument("--no-flow-matching", action="store_true",
                        help="SIREN half only; skips all ODE sampling")
    parser.add_argument("--plot-shapes", type=int, default=4,
                        help="rows in the grid figures, chosen to span the IoU degradation range")
    parser.add_argument("--resolution", type=int, default=400, help="boundary rendering grid per axis")
    parser.add_argument("--smooth-sigma", type=float, default=2.0,
                        help="Gaussian blur in grid cells applied before tracing SIREN(x, z) = 0")
    parser.add_argument("--extraction-chunk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=None)
    return parser


def extract_at_budget(siren, cfg: ExperimentConfig, polys: torch.Tensor, num_points: int,
                      device: torch.device, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """CAVIA extraction for every polynomial using exactly `num_points` query points.

    Everything except N is held at the run's deployed setting, query_gmm_fraction included:
    the SIREN was meta-trained against one query distribution and only the budget is under test.
    """
    z_chunks, mse_chunks = [], []
    for start in range(0, polys.shape[0], chunk_size):
        C_chunk = polys[start:start + chunk_size]
        X_raw = sample_query_points(C_chunk.shape[0], num_points, scale=cfg.scale,
                                    gmm_fraction=cfg.extraction.query_gmm_fraction, device=device)
        x_pow, y_pow = compute_poly_features_batched(X_raw, degree=cfg.degree, scale=cfg.scale)
        Y = torch.tanh(evaluate_poly_batched(x_pow, y_pow, C_chunk))
        z_chunk, mse_chunk = extract_latents_batched(siren, X_raw / cfg.scale, Y,
                                                     lr=cfg.extraction.lr,
                                                     steps=cfg.extraction.steps)
        z_chunks.append(z_chunk)
        mse_chunks.append(mse_chunk)

    return torch.cat(z_chunks, dim=0), torch.cat(mse_chunks, dim=0)


def select_plot_shapes(iou_by_n: np.ndarray, count: int) -> list[int]:
    """Shapes spanning the degradation range: how much mass IoU each one loses at the
    smallest N relative to the largest. Picking only the worst would overstate the effect."""
    degradation = iou_by_n[-1] - iou_by_n[0]
    order = np.argsort(degradation)
    count = min(count, len(order))
    picks = np.linspace(0, len(order) - 1, count).round().astype(int)
    return sorted(int(order[p]) for p in picks)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config_or_die(args.run_id)
    device = resolve_device()
    ev = cfg.evaluation
    num_polys = args.num_polys or ev.num_polys
    num_x0 = args.num_x0 or ev.num_x0
    n_values = sorted(args.num_points)

    out = Path(args.outdir) if args.outdir else run_dir(cfg.run_id) / "ablations" / "query_points"
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    siren = load_siren(cfg, device)
    model = build_flow_matcher(cfg, siren, device)
    iteration = load_checkpoint(cfg, model, device)
    model.eval()
    print(f"run_id {cfg.run_id} | iteration {iteration} | device {device}")
    print(f"meta-trained budget 1000 pts | deployed budget {cfg.extraction.points_per_shape} pts | "
          f"sweeping {n_values}")
    print(f"extraction: {cfg.extraction.steps} steps, lr {cfg.extraction.lr}, "
          f"query_gmm_fraction {cfg.extraction.query_gmm_fraction}")

    gmm_pool, _ = get_points(ev.gmm_pool_size, device=device)
    val_set = get_validation_set(device=device)
    polys = val_set["polynomials"][:num_polys].to(device)
    x0 = val_set["x0"][:num_x0].to(device)

    mass = constraint_masses(polys, gmm_pool, degree=cfg.degree, scale=cfg.scale)
    iou_points = gmm_pool[torch.randperm(gmm_pool.shape[0], device=device)[:ev.iou_mass_samples]]

    # --- SIREN half: extraction quality per budget --------------------------------
    z_by_n, per_n = {}, {}
    shape_ids: list[int] = []

    def dump_metrics() -> None:
        """Rewritten after every N: a wall-clock kill still leaves the completed budgets."""
        write_json(out / "metrics.json", {
            "run_id": cfg.run_id,
            "iteration": iteration,
            "meta_trained_points_per_shape": 1000,
            "deployed_points_per_shape": cfg.extraction.points_per_shape,
            "extraction": {"steps": cfg.extraction.steps, "lr": cfg.extraction.lr,
                           "query_gmm_fraction": cfg.extraction.query_gmm_fraction},
            "num_polys": num_polys,
            "num_x0": 0 if args.no_flow_matching else num_x0,
            "n_values": n_values,
            "plot_shape_ids": shape_ids,
            "mass": mass.cpu().tolist(),
            "per_n": {str(n): per_n[n] for n in n_values if n in per_n},
        })

    for n in n_values:
        z_n, mse_n = extract_at_budget(siren, cfg, polys, n, device, args.extraction_chunk)
        iou_n = region_iou_batched(siren, z_n, polys, iou_points, degree=cfg.degree, scale=cfg.scale)
        z_by_n[n] = z_n
        per_n[n] = {"extraction_mse": mse_n.cpu().tolist(), "mass_iou": iou_n.cpu().tolist()}
        print(f"N={n:>5} | extraction MSE {mse_n.mean():.6f} | "
              f"mass IoU mean {iou_n.mean():.4f} median {iou_n.median():.4f} min {iou_n.min():.4f}",
              flush=True)
    dump_metrics()

    iou_by_n = np.array([per_n[n]["mass_iou"] for n in n_values])
    shape_ids = select_plot_shapes(iou_by_n, args.plot_shapes)
    print(f"plotting shapes {shape_ids}")

    # --- FM half: downstream generative quality per budget -------------------------
    samples_by_n = {}
    if not args.no_flow_matching:
        for n in n_values:
            samples = run_evaluation_inference(model, x0, z=z_by_n[n], step_size=ev.step_size,
                                               device=device)
            metrics = evaluate_validation_set_metrics(samples, x_true_pool=gmm_pool, coeffs=polys,
                                                      degree=cfg.degree, scale=cfg.scale,
                                                      device=device)
            per_n[n].update({k: [float(v) for v in metrics[k]]
                             for k in ("success_rate", "swd", "mmd", "jsd")})
            samples_by_n[n] = np.stack([samples[i] for i in shape_ids])
            sr = np.asarray(per_n[n]["success_rate"])
            print(f"N={n:>5} | success rate mean {sr.mean():.2f}% median {np.median(sr):.2f}% | "
                  f"SWD median {np.median(per_n[n]['swd']):.4f}", flush=True)
            dump_metrics()
            del samples
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # --- figures -------------------------------------------------------------------
    col_labels = [f"N = {n}" for n in n_values]
    row_labels = [f"shape {i}\nmass {mass[i]:.2f}" for i in shape_ids]
    coeffs_list = [polys[i] for i in shape_ids]

    z_grid = [[z_by_n[n][i] for n in n_values] for i in shape_ids]
    iou_cells = [[f"IoU {per_n[n]['mass_iou'][i]:.3f}" for n in n_values] for i in shape_ids]
    diag.save_figure(
        diag.plot_boundary_ablation_grid(siren, coeffs_list, z_grid, row_labels, col_labels,
                                         cell_labels=iou_cells, degree=cfg.degree, scale=cfg.scale,
                                         resolution=args.resolution,
                                         smooth_sigma=args.smooth_sigma),
        out / "siren_boundary_grid.png")

    metric_series = {"mass IoU": iou_by_n,
                     "extraction MSE": np.array([per_n[n]["extraction_mse"] for n in n_values])}

    if samples_by_n:
        sample_grid = [[samples_by_n[n][r] for n in n_values] for r in range(len(shape_ids))]
        sr_cells = [[f"SR {per_n[n]['success_rate'][i]:.1f}%" for n in n_values] for i in shape_ids]
        diag.save_figure(
            diag.plot_samples_ablation_grid(sample_grid, coeffs_list, row_labels, col_labels,
                                            cell_labels=sr_cells, degree=cfg.degree, scale=cfg.scale),
            out / "flow_matching_grid.png")
        metric_series["success rate (%)"] = np.array([per_n[n]["success_rate"] for n in n_values])
        metric_series["SWD"] = np.array([per_n[n]["swd"] for n in n_values])

    diag.save_figure(
        diag.plot_ablation_curves(n_values, metric_series, xlabel="inference query points N"),
        out / "degradation_curves.png")

    dump_metrics()

    # Keeps the figures re-renderable (resolution, smoothing, shape choice) without re-sweeping.
    torch.save({"n_values": n_values, "shape_ids": shape_ids,
                "z": {n: z_by_n[n].cpu() for n in n_values},
                "samples": {n: samples_by_n[n] for n in samples_by_n},
                "polys": polys.cpu()}, out / "latents.pt")

    print_summary(n_values, per_n)
    print(f"\nwrote {out}")
    return 0


def load_config_or_die(run_id: str) -> ExperimentConfig:
    cfg = load_config(run_id)
    if not cfg.siren_path().exists():
        raise SystemExit(f"{run_id} points at a missing SIREN checkpoint: {cfg.siren_path()}")
    return cfg


def print_summary(n_values, per_n) -> None:
    keys = [k for k in ("mass_iou", "success_rate", "swd", "extraction_mse") if k in per_n[n_values[0]]]
    header = f"{'N':>6}" + "".join(f"{k:>16}" for k in keys)
    print("\nmedian over shapes")
    print(header)
    print("-" * len(header))
    for n in n_values:
        row = f"{n:>6}"
        for k in keys:
            values = np.asarray(per_n[n][k], dtype=float)
            row += f"{np.median(values[np.isfinite(values)]):>16.4f}"
        print(row)


if __name__ == "__main__":
    raise SystemExit(main())
