# -*- coding: utf-8 -*-
"""Zero-shot transfer of the polynomial-trained Functa pipeline to polygon constraints.

The SIREN and the flow matcher are both frozen checkpoints trained only on degree-3
polynomial regions. Each polygon is handed to the unchanged CAVIA inner loop as the field
``tanh(g * d(x))``, with ``d`` its exact signed distance (negative inside) and ``g`` matched
to the median boundary slope ``||grad P||`` of the training polynomials, so the target has
the same local scale the SIREN was meta-trained on. The extracted latent then conditions
the flow matcher, whose samples are compared against GMM draws rejected to the polygon.
``--refine-steps`` continues with test-time Adam on ``z`` alone (weights still frozen), on
fresh query points concentrated around the polygon edges.

    siren_encoder/<name>.{png,pdf}   decoded field, true polygon dashed, decoded zero set solid
    siren_encoder/grid.{png,pdf}     every polygon in one boundary grid
    samples/<name>.{png,pdf}         ground truth | Functa FM density maps
    samples/grid.{png,pdf}           2 x K overview of the same maps

Arrays are written to ``<outdir>/artifacts/`` and ``--plot-only`` redraws everything from them.

    sbatch scripts/run_polygon_transfer.sh
    sbatch scripts/run_polygon_transfer.sh --field-gain 0.5
    sbatch scripts/run_polygon_transfer.sh --plot-only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.config import REPO_ROOT
from constrained_fm.src.experiment.registry import load_config, pin_baseline_run
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint,
                                                   load_siren, resolve_device, set_seed)
from constrained_fm.src.geometry.polygons import (polygon_sdf, regular_polygon,
                                                  rotated_rectangle, sample_boundary_points,
                                                  star_polygon)
from constrained_fm.src.geometry.polynomials import (compute_poly_features_batched,
                                                     evaluate_poly_batched)
from constrained_fm.src.inference.latent_extractor import extract_latents_batched, refine_latents
from constrained_fm.src.metrics.distributional import compute_jsd, compute_mmd, compute_swd
from constrained_fm.src.visualization import feasibility as fz
from constrained_fm.src.visualization import siren_encoder as se

FUNCTA_RUN = "siren-uniform-8d6375ab"
OUTDIR = "constrained_fm/baselines/polygon_transfer"
FIGURE_DIR = "constrained_fm/images/thesis_pool/polygon_transfer"
GAIN_CALIBRATION_POLYS = 512
GAIN_CALIBRATION_POINTS = 4000
GAIN_BOUNDARY_QUANTILE = 0.02   # fraction of lowest-|P| query points treated as on the boundary
SAMPLE_SEED_OFFSET = 1000


def polygon_catalogue() -> dict[str, torch.Tensor]:
    """Named test polygons in raw plane coordinates, spanning convex and non-convex shapes."""
    return {
        "triangle": torch.tensor([[-3.0, -2.8], [3.2, -2.2], [0.2, 3.2]]),
        "square": regular_polygon(4, 2.6, rotation_deg=45.0),
        "thin_rectangle": rotated_rectangle(3.2, 0.9, center=(0.2, 0.2), rotation_deg=35.0),
        "pentagon": regular_polygon(5, 2.4, center=(0.5, -0.3), rotation_deg=90.0),
        "hexagon": regular_polygon(6, 2.0, center=(-1.2, -0.8)),
        "star": star_polygon(5, 3.2, 1.3, center=(0.2, 0.2)),
        "l_shape": torch.tensor([[-3.0, -3.0], [1.5, -3.0], [1.5, -1.0], [-1.0, -1.0],
                                 [-1.0, 3.0], [-3.0, 3.0]]),
        "chevron": torch.tensor([[-3.0, -2.0], [0.0, 1.0], [3.0, -2.0], [3.0, 0.5],
                                 [0.0, 3.5], [-3.0, 0.5]]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", default=FUNCTA_RUN,
                        help="Functa FM run whose SIREN, checkpoint and extraction settings are used")
    parser.add_argument("--shapes", nargs="+", default=None,
                        help=f"subset of {sorted(polygon_catalogue())}; default is all")
    parser.add_argument("--field-gain", type=float, default=None,
                        help="g in tanh(g * d(x)); default calibrates to the polynomials' boundary slope")
    parser.add_argument("--num-samples", type=int, default=100000,
                        help="generated and ground-truth points per polygon for the density maps")
    parser.add_argument("--metric-samples", type=int, default=10000,
                        help="prefix of those points scored by SWD / MMD / JSD")
    parser.add_argument("--mass-points", type=int, default=1000000,
                        help="GMM draws backing constraint mass and decoded-region IoU")
    parser.add_argument("--resolution", type=int, default=600)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--refine-steps", type=int, default=0,
                        help="test-time Adam steps on z after CAVIA; 0 keeps the pure CAVIA latent")
    parser.add_argument("--refine-lr", type=float, default=3e-5,
                        help="Adam moves each coordinate ~lr per step; CAVIA latents have ||z|| ~ 0.01")
    parser.add_argument("--refine-points", type=int, default=2000,
                        help="fresh query points per shape per refinement step")
    parser.add_argument("--boundary-fraction", type=float, default=0.5,
                        help="share of refinement queries drawn near the polygon edges")
    parser.add_argument("--boundary-sigma", type=float, default=0.15,
                        help="Gaussian jitter (raw units) around the sampled edge points")
    parser.add_argument("--anchor-weight", type=float, default=0.0,
                        help="lambda in lambda ||z - z_cavia||^2")

    parser.add_argument("--encoder-style", default="paper", choices=sorted(se.STYLE_PRESETS))
    parser.add_argument("--samples-style", default="light", choices=sorted(fz.STYLE_PRESETS))
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--legend", action="store_true",
                        help="label the two boundaries inside each encoder panel")
    parser.add_argument("--no-captions", action="store_true",
                        help="drop the SR / SWD caption under the sample maps")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw from saved arrays; no checkpoint, no extraction, no ODE")
    return parser


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def calibrate_field_gain(cfg, device: torch.device) -> float:
    """Median ``||grad_x P||`` on the zero set of training-distribution polynomials.

    ``tanh(g d)`` has slope ``g`` across its boundary because ``||grad d|| = 1``, so this ``g``
    reproduces the boundary sharpness of the ``tanh(P)`` targets the SIREN was trained on.
    """
    polys = sample_valid_polynomials(GAIN_CALIBRATION_POLYS, degree=cfg.degree, scale=cfg.scale,
                                     min_area=cfg.pool.min_area, max_area=cfg.pool.max_area,
                                     device=device)
    X = ((torch.rand(GAIN_CALIBRATION_POLYS, GAIN_CALIBRATION_POINTS, 2, device=device) * 2 - 1)
         * cfg.scale).requires_grad_(True)
    x_pow, y_pow = compute_poly_features_batched(X, degree=cfg.degree, scale=cfg.scale)
    P = evaluate_poly_batched(x_pow, y_pow, polys)
    grad_norm = torch.autograd.grad(P.sum(), X)[0].norm(dim=-1)
    threshold = P.detach().abs().quantile(GAIN_BOUNDARY_QUANTILE, dim=1, keepdim=True)
    return float(grad_norm[P.detach().abs() <= threshold].median())


def reference_latent_norm(siren, cfg, device: torch.device, num_polys: int = 256) -> float:
    """Median ``||z||`` of CAVIA latents of training-distribution polynomials."""
    polys = sample_valid_polynomials(num_polys, degree=cfg.degree, scale=cfg.scale,
                                     min_area=cfg.pool.min_area, max_area=cfg.pool.max_area,
                                     device=device)
    X = sample_query_points(num_polys, cfg.extraction.points_per_shape, scale=cfg.scale,
                            gmm_fraction=cfg.extraction.query_gmm_fraction, device=device)
    Y = torch.tanh(evaluate_poly_batched(*compute_poly_features_batched(
        X, degree=cfg.degree, scale=cfg.scale), polys))
    z, _ = extract_latents_batched(siren, X / cfg.scale, Y, lr=cfg.extraction.lr,
                                   steps=cfg.extraction.steps)
    return float(z.norm(dim=-1).median())


def make_query_fn(polys: list[torch.Tensor], gain: float, scale: float, num_points: int,
                  boundary_fraction: float, boundary_sigma: float, device: torch.device):
    """Fresh (X / scale, tanh(g d(X))) per call: uniform box points plus jittered edge points."""
    num_boundary = int(round(num_points * boundary_fraction))
    num_uniform = num_points - num_boundary

    def query_fn() -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.stack([torch.cat([
            (torch.rand(num_uniform, 2, device=device) * 2 - 1) * scale,
            (sample_boundary_points(v, num_boundary, device=device)
             + boundary_sigma * torch.randn(num_boundary, 2, device=device)).clamp(-scale, scale),
        ]) for v in polys])
        Y = torch.stack([torch.tanh(gain * polygon_sdf(X[i], v)) for i, v in enumerate(polys)])
        return X / scale, Y

    return query_fn


def decode_fields(siren, z_batch: torch.Tensor, lattice: torch.Tensor, scale: float,
                  resolution: int, chunk_size: int) -> np.ndarray:
    """SIREN(x, z) on the lattice for each latent. Returns (B, R, R)."""
    fields = np.empty((z_batch.shape[0], resolution, resolution), dtype=np.float32)
    with torch.no_grad():
        for i, z in enumerate(z_batch):
            chunks = [siren(lattice[s:s + chunk_size] / scale, z).squeeze(-1)
                      for s in range(0, lattice.shape[0], chunk_size)]
            fields[i] = torch.cat(chunks).view(resolution, resolution).cpu().numpy()
    return fields


def rejection_sample(vertices: torch.Tensor, num: int, device: torch.device,
                     draw: int = 1000000) -> torch.Tensor:
    """``num`` GMM points inside the polygon, drawn in fixed-size proposal rounds."""
    kept, total = [], 0
    while total < num:
        x, _ = get_points(draw, device=device)
        inside = x[polygon_sdf(x, vertices) <= 0]
        kept.append(inside)
        total += inside.shape[0]
    return torch.cat(kept)[:num]


def decoded_values(siren, points: torch.Tensor, z: torch.Tensor, scale: float,
                   chunk_size: int) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([siren(points[s:s + chunk_size] / scale, z).squeeze(-1)
                          for s in range(0, points.shape[0], chunk_size)])


def pad_vertices(polys: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """(P, K_max, 2) NaN-padded vertex array plus the (P,) true vertex counts."""
    counts = np.asarray([p.shape[0] for p in polys], dtype=np.int32)
    padded = np.full((len(polys), counts.max(), 2), np.nan, dtype=np.float32)
    for i, p in enumerate(polys):
        padded[i, :p.shape[0]] = p.cpu().numpy()
    return padded, counts


def render(names: list[str], pred: np.ndarray, true: np.ndarray, gen: np.ndarray,
           gt: np.ndarray, per_shape: list[dict], scale: float, args) -> list[Path]:
    figure_dir = resolve_path(args.figure_dir)
    written: list[Path] = []

    enc_style = se.get_style(args.encoder_style, show_legend=args.legend,
                             gt_label=r"ground-truth polygon  $d(x) = 0$")
    for name, pred_field, true_field in zip(names, pred, true):
        fig = se.plot_encoder_panel(pred_field, true_field, scale=scale, style=enc_style)
        written += se.save_encoder_figure(fig, figure_dir / "siren_encoder" / name,
                                          formats=args.formats, dpi=args.dpi)

    cols = min(args.grid_cols, len(names))
    rows = -(-len(names) // cols)
    cells = rows * cols
    if cells == len(names):
        fig = se.plot_boundary_grid(pred, true, rows, cols, scale=scale,
                                    style=replace(enc_style, show_legend=False))
        written += se.save_encoder_figure(fig, figure_dir / "siren_encoder" / "grid",
                                          formats=args.formats, dpi=args.dpi)
    else:
        print(f"skipping encoder grid: {len(names)} shapes do not fill a {rows}x{cols} grid")

    smp_style = fz.get_style(args.samples_style, show_metrics=not args.no_captions,
                             boundary_label="polygon boundary")
    columns = []
    for name, true_field, g_pts, x_pts, record in zip(names, true, gt, gen, per_shape):
        panels = [fz.Panel("Ground truth", g_pts),
                  fz.Panel("Functa FM", x_pts, metrics=record, highlight=True)]
        fig = fz.plot_feasibility_row(panels, style=smp_style, scale=scale,
                                      boundary_field=true_field)
        written += se.save_encoder_figure(fig, figure_dir / "samples" / name,
                                          formats=args.formats, dpi=args.dpi)
        columns.append(panels)

    fig = fz.plot_feasibility_grid(columns, list(true), column_titles=names,
                                   style=smp_style, scale=scale)
    written += se.save_encoder_figure(fig, figure_dir / "samples" / "grid",
                                      formats=args.formats, dpi=args.dpi)
    return written


def replot(args) -> int:
    root = resolve_path(args.outdir)
    record = json.loads((root / "metrics.json").read_text())
    names = [item["name"] for item in record["per_shape"]]
    written = render(names, artifacts.load_array(root, "pred_fields"),
                     artifacts.load_array(root, "true_fields"),
                     artifacts.load_array(root, "samples"),
                     artifacts.load_array(root, "gt_samples"),
                     record["per_shape"], float(record["scale"]), args)
    print("\n".join(f"wrote {p}" for p in written))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plot_only:
        return replot(args)

    catalogue = polygon_catalogue()
    names = args.shapes or list(catalogue)
    unknown = sorted(set(names) - set(catalogue))
    if unknown:
        raise ValueError(f"unknown shapes {unknown}; choose from {sorted(catalogue)}")

    cfg = load_config(args.run_id)
    device = resolve_device()
    siren = load_siren(cfg, device)
    model = build_flow_matcher(cfg, siren, device)
    iteration = load_checkpoint(cfg, model, device)
    model.eval()
    polys = [catalogue[n].to(device) for n in names]

    set_seed(args.seed)
    gain = args.field_gain if args.field_gain is not None else calibrate_field_gain(cfg, device)
    print(f"run {cfg.run_id} | iteration {iteration} | device {device} | "
          f"{len(names)} polygons | field gain {gain:.4f}")

    X_raw = sample_query_points(len(names), cfg.extraction.points_per_shape, scale=cfg.scale,
                                gmm_fraction=cfg.extraction.query_gmm_fraction, device=device)
    Y = torch.stack([torch.tanh(gain * polygon_sdf(X_raw[i], v)) for i, v in enumerate(polys)])
    z_batch, extraction_mse = extract_latents_batched(siren, X_raw / cfg.scale, Y,
                                                      lr=cfg.extraction.lr,
                                                      steps=cfg.extraction.steps)
    z_cavia = z_batch
    refine_history: list[float] = []
    if args.refine_steps > 0:
        query_fn = make_query_fn(polys, gain, cfg.scale, args.refine_points,
                                 args.boundary_fraction, args.boundary_sigma, device)
        with torch.enable_grad():
            z_batch, refine_history = refine_latents(siren, z_cavia, query_fn,
                                                     steps=args.refine_steps, lr=args.refine_lr,
                                                     anchor_weight=args.anchor_weight)
    with torch.no_grad():
        fit_mse = ((siren(X_raw / cfg.scale, z_batch).squeeze(-1) - Y) ** 2).mean(dim=1)
    ref_norm = reference_latent_norm(siren, cfg, device)
    print(f"median polynomial ||z|| {ref_norm:.3f} | polygon ||z|| cavia "
          f"{[round(float(n), 3) for n in z_cavia.norm(dim=-1)]} -> final "
          f"{[round(float(n), 3) for n in z_batch.norm(dim=-1)]}")

    axis = torch.linspace(-cfg.scale, cfg.scale, args.resolution)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    lattice = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).to(device)
    pred = decode_fields(siren, z_batch, lattice, cfg.scale, args.resolution, args.chunk_size)
    true = np.stack([polygon_sdf(lattice, v).view(args.resolution, args.resolution).cpu().numpy()
                     for v in polys]).astype(np.float32)

    mass_points, _ = get_points(args.mass_points, device=device)
    samples, gt_samples, per_shape = [], [], []
    for i, (name, v) in enumerate(zip(names, polys)):
        z = z_batch[i]
        true_in = polygon_sdf(mass_points, v) <= 0
        pred_in = decoded_values(siren, mass_points, z, cfg.scale, args.chunk_size) <= 0
        mass = float(true_in.float().mean())
        iou = float((true_in & pred_in).sum() / (true_in | pred_in).sum().clamp(min=1))

        set_seed(args.seed + SAMPLE_SEED_OFFSET + i)
        gen = model.sample(num_points=args.num_samples, z=z, step_size=cfg.evaluation.step_size,
                           return_intermediates=False, device=device)
        if gen.ndim == 3:
            gen = gen[-1]
        gt = rejection_sample(v, args.num_samples, device)

        sr = float((polygon_sdf(gen, v) <= 0).float().mean()) * 100.0
        believed = float((decoded_values(siren, gen, z, cfg.scale, args.chunk_size) <= 0)
                         .float().mean()) * 100.0
        gen_m, gt_m = gen[:args.metric_samples], gt[:args.metric_samples]
        np.random.seed(args.seed + i)
        record = {"index": i, "name": name, "num_vertices": int(v.shape[0]), "mass": mass,
                  "mass_iou": iou, "extraction_mse": float(extraction_mse[i]),
                  "fit_mse": float(fit_mse[i]),
                  "z_norm_cavia": float(z_cavia[i].norm()), "z_norm": float(z.norm()),
                  "success_rate": sr, "believed_success_rate": believed,
                  "swd": compute_swd(gen_m, gt_m, seed=args.seed + i),
                  "mmd": compute_mmd(gen_m, gt_m), "jsd": compute_jsd(gen_m, gt_m)}
        per_shape.append(record)
        samples.append(gen.cpu().numpy())
        gt_samples.append(gt.cpu().numpy())
        print(f"[{i + 1}/{len(names)}] {name}: SR {sr:.2f}% | SWD {record['swd']:.4f}")

    root = resolve_path(args.outdir)
    run_id = pin_baseline_run(root, "polygon_transfer", args, extra={
        "config_run_id": cfg.run_id, "field_gain": gain})
    vertices, counts = pad_vertices(polys)
    artifacts.save_arrays(root, polygon_vertices=vertices, polygon_num_vertices=counts,
                          query_points=X_raw, query_targets=Y, latents=z_batch,
                          latents_cavia=z_cavia,
                          refine_history=np.asarray(refine_history, dtype=np.float32),
                          pred_fields=pred, true_fields=true, samples=np.stack(samples),
                          gt_samples=np.stack(gt_samples))
    artifacts.write_manifest(root, run_id=run_id, config_run_id=cfg.run_id, iteration=iteration,
                             names=names, field_gain=gain)
    (root / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "config_run_id": cfg.run_id, "iteration": iteration,
        "scale": cfg.scale, "field_gain": gain, "seed": args.seed,
        "refine_steps": args.refine_steps, "refine_lr": args.refine_lr,
        "anchor_weight": args.anchor_weight, "reference_z_norm": ref_norm,
        "per_shape": per_shape}, indent=2))

    print(f"\n| polygon | K | mass | mass IoU | CAVIA MSE | final MSE | z norm | SR (%) "
          f"| believed SR (%) | SWD | MMD | JSD |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in per_shape:
        print(f"| {r['name']} | {r['num_vertices']} | {r['mass']:.3f} | {r['mass_iou']:.4f} | "
              f"{r['extraction_mse']:.2e} | {r['fit_mse']:.2e} | {r['z_norm']:.3f} | "
              f"{r['success_rate']:.2f} | "
              f"{r['believed_success_rate']:.2f} | {r['swd']:.4f} | {r['mmd']:.1e} | "
              f"{r['jsd']:.4f} |")

    written = render(names, pred, true, np.stack(samples), np.stack(gt_samples), per_shape,
                     cfg.scale, args)
    print("\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
