# -*- coding: utf-8 -*-
"""Paper figures for the SIREN/CAVIA encoder: decoded constraint fields and latent interpolations.

Samples a batch of unseen polynomials, extracts each one's latent with the run's CAVIA inner
loop, decodes every field on a lattice, and renders

    encoder/shape<i>.{png,pdf}            one heatmap per constraint, both boundaries overlaid
    interpolation/pair<a>_<b>.{png,pdf}   a 1xK strip along z(t) = (1 - t) z_a + t z_b

Extraction MSE, decoded mass IoU and constraint mass are reported per shape in ``metrics.json``
so the best candidates can be picked without re-rendering.

Every decoded field is written to ``<outdir>/artifacts/``, so ``--plot-only`` restyles the whole
set with no checkpoint and no extraction.

    sbatch scripts/run_siren_encoder.sh
    sbatch scripts/run_siren_encoder.sh --num-shapes 8 --pairs 0:1 2:3 4:5 6:7
    python -m constrained_fm.scripts.plot_siren_encoder --plot-only --legend
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
from constrained_fm.src.experiment.runtime import load_siren, resolve_device, set_seed
from constrained_fm.src.geometry.polynomials import (compute_poly_features_batched,
                                                     evaluate_poly_batched)
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.metrics.functa_fidelity import region_iou_batched
from constrained_fm.src.visualization import siren_encoder as se

FUNCTA_RUN = "siren-uniform-8d6375ab"
OUTDIR = "constrained_fm/baselines/siren_encoder_figures"
FIGURE_DIR = "constrained_fm/images/thesis_pool/siren_encoder"
DEFAULT_PAIRS = ("0:1", "2:3", "4:5", "6:7")
DEFAULT_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", default=FUNCTA_RUN,
                        help="run whose SIREN weights and extraction settings are used")
    parser.add_argument("--num-shapes", type=int, default=8)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS),
                        help="interpolation endpoints as 'a:b' shape indices")
    parser.add_argument("--times", nargs="+", type=float, default=list(DEFAULT_TIMES))
    parser.add_argument("--resolution", type=int, default=600,
                        help="lattice per axis; higher removes jaggedness in the zero level set")
    parser.add_argument("--iou-points", type=int, default=100000,
                        help="GMM draws backing the mass-weighted decoded-region IoU")
    parser.add_argument("--chunk-size", type=int, default=65536,
                        help="lattice points per SIREN forward pass")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--style", default="paper", choices=sorted(se.STYLE_PRESETS))
    parser.add_argument("--cmap", default=None)
    parser.add_argument("--field-render", default=None, choices=["imshow", "contourf"])
    parser.add_argument("--pred-color", default=None, help="decoded boundary colour")
    parser.add_argument("--gt-color", default=None, help="ground-truth boundary colour")
    parser.add_argument("--smooth-sigma", type=float, default=None,
                        help="Gaussian blur in lattice cells applied before tracing a level set")
    parser.add_argument("--colorbar-label", default=None)
    parser.add_argument("--clip-to-unit", action="store_true",
                        help="fix the colour scale to [-1, 1] instead of the field's own range")
    parser.add_argument("--legend", action="store_true",
                        help="label the two boundaries inside each encoder panel")
    parser.add_argument("--no-ticks", action="store_true",
                        help="drop the axis ticks from the encoder panels too")
    parser.add_argument("--interp-colorbar", action="store_true",
                        help="add one shared colorbar to each interpolation strip")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw from saved fields; no checkpoint, no extraction")
    return parser


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_pairs(raw: list[str], num_shapes: int) -> list[tuple[int, int]]:
    pairs = []
    for item in raw:
        a, b = (int(v) for v in item.replace(",", ":").split(":"))
        if not (0 <= a < num_shapes and 0 <= b < num_shapes):
            raise ValueError(f"pair '{item}' is outside the {num_shapes} sampled shapes")
        pairs.append((a, b))
    return pairs


def encoder_style(args) -> se.EncoderStyle:
    overrides = {name: getattr(args, name) for name in
                 ("cmap", "field_render", "pred_color", "gt_color", "smooth_sigma",
                  "colorbar_label") if getattr(args, name) is not None}
    if args.clip_to_unit:
        overrides["clip_to_unit"] = True
    overrides["show_legend"] = args.legend
    if args.no_ticks:
        overrides["show_ticks"] = False
    return se.get_style(args.style, **overrides)


def interpolation_style(args) -> se.EncoderStyle:
    """Same visual language as the encoder panels, minus the per-panel axes furniture."""
    return replace(encoder_style(args), show_ticks=False, show_legend=False,
                   show_colorbar=args.interp_colorbar)


def decode_fields(siren, z_batch: torch.Tensor, points: torch.Tensor, scale: float,
                  resolution: int, chunk_size: int) -> np.ndarray:
    """SIREN(x, z) on the lattice for each latent, one shape at a time. Returns (B, R, R)."""
    fields = np.empty((z_batch.shape[0], resolution, resolution), dtype=np.float32)
    with torch.no_grad():
        for i, z in enumerate(z_batch):
            chunks = [siren(points[start:start + chunk_size] / scale, z).squeeze(-1)
                      for start in range(0, points.shape[0], chunk_size)]
            fields[i] = torch.cat(chunks).view(resolution, resolution).cpu().numpy()
    return fields


def true_fields(coeffs: torch.Tensor, points: torch.Tensor, degree: int, scale: float,
                resolution: int) -> np.ndarray:
    """P(x) on the lattice for each polynomial. Returns (B, R, R)."""
    grid = points.unsqueeze(0).expand(coeffs.shape[0], -1, -1)
    x_pow, y_pow = compute_poly_features_batched(grid, degree=degree, scale=scale)
    with torch.no_grad():
        values = evaluate_poly_batched(x_pow, y_pow, coeffs)
    return values.view(-1, resolution, resolution).cpu().numpy().astype(np.float32)


def render(pred: np.ndarray, true: np.ndarray, interp: np.ndarray,
           pairs: list[tuple[int, int]], times: list[float], scale: float, args) -> list[Path]:
    figure_dir = resolve_path(args.figure_dir)
    written: list[Path] = []

    style = encoder_style(args)
    for i, (pred_field, true_field) in enumerate(zip(pred, true)):
        fig = se.plot_encoder_panel(pred_field, true_field, scale=scale, style=style)
        written += se.save_encoder_figure(fig, figure_dir / "encoder" / f"shape{i}",
                                          formats=args.formats, dpi=args.dpi)

    strip_style = interpolation_style(args)
    for (a, b), fields in zip(pairs, interp):
        fig = se.plot_interpolation_row(fields, times, scale=scale, style=strip_style)
        written += se.save_encoder_figure(fig, figure_dir / "interpolation" / f"pair{a}_{b}",
                                          formats=args.formats, dpi=args.dpi)
    return written


def replot(args) -> int:
    """Redraws from the artifact store. Loads no checkpoint and extracts nothing."""
    root = resolve_path(args.outdir)
    record = json.loads((root / "metrics.json").read_text())

    pred = artifacts.load_array(root, "pred_fields")
    true = artifacts.load_array(root, "true_fields")
    interp = artifacts.load_array(root, "interp_fields")
    pairs = [tuple(int(v) for v in pair) for pair in artifacts.load_array(root, "interp_pairs")]
    times = [float(t) for t in artifacts.load_array(root, "interp_times")]

    written = render(pred, true, interp, pairs, times, float(record["scale"]), args)
    print("\n".join(f"wrote {p}" for p in written))
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plot_only:
        return replot(args)

    pairs = parse_pairs(args.pairs, args.num_shapes)
    cfg = load_config(args.run_id)
    device = resolve_device()
    siren = load_siren(cfg, device)
    print(f"siren from {cfg.run_id} | device {device} | {args.num_shapes} shapes "
          f"at {args.resolution}^2")

    set_seed(args.seed)
    polys = sample_valid_polynomials(args.num_shapes, degree=cfg.degree, scale=cfg.scale,
                                     min_area=cfg.pool.min_area, max_area=cfg.pool.max_area,
                                     device=device)

    X_raw = sample_query_points(args.num_shapes, cfg.extraction.points_per_shape,
                                scale=cfg.scale, gmm_fraction=cfg.extraction.query_gmm_fraction,
                                device=device)
    x_pow, y_pow = compute_poly_features_batched(X_raw, degree=cfg.degree, scale=cfg.scale)
    Y = torch.tanh(evaluate_poly_batched(x_pow, y_pow, polys))
    z_batch, extraction_mse = extract_latents_batched(siren, X_raw / cfg.scale, Y,
                                                      lr=cfg.extraction.lr,
                                                      steps=cfg.extraction.steps)

    axis = torch.linspace(-cfg.scale, cfg.scale, args.resolution)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    lattice = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).to(device)

    pred = decode_fields(siren, z_batch, lattice, cfg.scale, args.resolution, args.chunk_size)
    true = true_fields(polys, lattice, cfg.degree, cfg.scale, args.resolution)

    # Interpolating the latents, not the coefficients: the point is that the SIREN's own
    # latent space is traversable, so every intermediate must be decoded from z alone.
    times = torch.tensor(args.times, device=device, dtype=z_batch.dtype)
    interp = np.empty((len(pairs), len(args.times), args.resolution, args.resolution),
                      dtype=np.float32)
    for p, (a, b) in enumerate(pairs):
        z_path = torch.lerp(z_batch[a].unsqueeze(0), z_batch[b].unsqueeze(0),
                            times.view(-1, 1))
        interp[p] = decode_fields(siren, z_path, lattice, cfg.scale, args.resolution,
                                  args.chunk_size)

    mass_points, _ = get_points(args.iou_points, device=device)
    iou = region_iou_batched(siren, z_batch, polys, mass_points, degree=cfg.degree,
                             scale=cfg.scale)
    x_pow_m, y_pow_m = compute_poly_features_batched(
        mass_points.unsqueeze(0).expand(args.num_shapes, -1, -1), degree=cfg.degree,
        scale=cfg.scale)
    mass = (evaluate_poly_batched(x_pow_m, y_pow_m, polys) <= 0).float().mean(dim=1)

    root = resolve_path(args.outdir)
    tracked = {"config_run_id": cfg.run_id, "num_shapes": args.num_shapes,
               "resolution": args.resolution, "seed": args.seed, "pairs": args.pairs,
               "times": args.times, "points_per_shape": cfg.extraction.points_per_shape,
               "extraction_lr": cfg.extraction.lr, "extraction_steps": cfg.extraction.steps}
    run_id = pin_baseline_run(root, "siren_encoder_figures", tracked)

    artifacts.save_arrays(root, polynomials=polys, latents=z_batch, pred_fields=pred,
                          true_fields=true, interp_fields=interp,
                          interp_pairs=np.asarray(pairs, dtype=np.int32),
                          interp_times=np.asarray(args.times, dtype=np.float32))
    artifacts.write_manifest(root, run_id=run_id)

    record = {
        "run_id": run_id,
        "siren_run_id": cfg.run_id,
        "scale": cfg.scale,
        "degree": cfg.degree,
        "seed": args.seed,
        "pairs": [list(pair) for pair in pairs],
        "times": list(args.times),
        "per_shape": [
            {"index": i,
             "extraction_mse": float(extraction_mse[i]),
             "mass_iou": float(iou[i]),
             "mass": float(mass[i])}
            for i in range(args.num_shapes)
        ],
    }
    (root / "metrics.json").write_text(json.dumps(record, indent=2))

    print(f"{'shape':>5} {'mass':>8} {'mass IoU':>9} {'extract MSE':>12}")
    for item in record["per_shape"]:
        print(f"{item['index']:>5} {item['mass']:>8.3f} {item['mass_iou']:>9.4f} "
              f"{item['extraction_mse']:>12.2e}")

    written = render(pred, true, interp, pairs, list(args.times), cfg.scale, args)
    print("\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
