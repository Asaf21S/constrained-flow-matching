# -*- coding: utf-8 -*-
"""SIREN boundary grids: several rows x cols panels of decoded fields, one shape per panel.

Samples a pool of unseen polynomials, extracts their latents with the run's CAVIA inner
loop, decodes every field on a lattice, and picks ``--num-grids`` diverse subsets of
``rows * cols`` shapes -- one grid per subset -- so different candidate compositions can be
compared before choosing one for the paper.

Diversity is by constraint mass: the pool is sorted by mass and split into ``rows * cols``
equal-width bins, and each grid takes one shape from every bin, so every grid spans the full
mass range and only the specific shape drawn from each bin differs between grids.

    grid<g>_<rows>x<cols>.{png,pdf}   rows x cols heatmap panels, GT + decoded boundary overlaid

Every decoded pool field is written to ``<outdir>/artifacts/``, so ``--plot-only`` reselects
and restyles with no checkpoint and no extraction.

    sbatch scripts/run_siren_boundary_grid.sh
    sbatch scripts/run_siren_boundary_grid.sh --rows 2 --cols 3 --num-grids 4
    python -m constrained_fm.scripts.plot_siren_boundary_grid --plot-only --no-colorbar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from constrained_fm.scripts.plot_siren_encoder import decode_fields, resolve_path, true_fields
from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.registry import load_config, pin_baseline_run
from constrained_fm.src.experiment.runtime import load_siren, resolve_device, set_seed
from constrained_fm.src.geometry.polynomials import (compute_poly_features_batched,
                                                     evaluate_poly_batched)
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.metrics.functa_fidelity import region_iou_batched
from constrained_fm.src.visualization import siren_encoder as se

FUNCTA_RUN = "siren-uniform-8d6375ab"
OUTDIR = "constrained_fm/baselines/siren_boundary_grid"
FIGURE_DIR = "constrained_fm/images/thesis_pool/siren_encoder/boundary_grid"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", default=FUNCTA_RUN,
                        help="run whose SIREN weights and extraction settings are used")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--num-grids", type=int, default=3,
                        help="how many distinct diverse compositions to render")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="shapes sampled and decoded once; defaults to rows*cols*num-grids")
    parser.add_argument("--resolution", type=int, default=500,
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
    parser.add_argument("--no-colorbar", action="store_true",
                        help="drop the single grid-wide colorbar")
    parser.add_argument("--clip-to-unit", action="store_true",
                        help="fix the colour scale to [-1, 1] instead of the pool's own range")
    parser.add_argument("--legend", action="store_true",
                        help="label the two boundaries once, on the first panel of each grid")
    parser.add_argument("--no-ticks", action="store_true", help="drop axis ticks")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    parser.add_argument("--plot-only", action="store_true",
                        help="reselect and redraw from saved fields; no checkpoint, no extraction")
    return parser


def grid_style(args) -> se.EncoderStyle:
    overrides = {name: getattr(args, name) for name in
                 ("cmap", "field_render", "pred_color", "gt_color", "smooth_sigma",
                  "colorbar_label") if getattr(args, name) is not None}
    if args.clip_to_unit:
        overrides["clip_to_unit"] = True
    overrides["show_legend"] = args.legend
    overrides["show_colorbar"] = not args.no_colorbar
    if args.no_ticks:
        overrides["show_ticks"] = False
    return se.get_style(args.style, **overrides)


def diverse_selections(mass: np.ndarray, cells: int, num_grids: int,
                       seed: int) -> list[list[int]]:
    """One selection of ``cells`` pool indices per grid, each spanning the full mass range.

    The pool is sorted by mass and split into ``cells`` equal-size bins; every grid draws
    exactly one index from each bin, so a single grid never over-represents one mass range.
    Which specific pool member each grid gets from a bin is shuffled and then cycled, so
    successive grids differ in composition without dropping the mass-diversity guarantee.
    """
    order = np.argsort(mass)
    bins = np.array_split(order, cells)
    rng = np.random.default_rng(seed)
    for members in bins:
        rng.shuffle(members)

    return [[int(members[g % len(members)]) for members in bins] for g in range(num_grids)]


def render(pred_pool: np.ndarray, true_pool: np.ndarray, selections: list[list[int]],
          rows: int, cols: int, scale: float, args) -> list[Path]:
    figure_dir = resolve_path(args.figure_dir)
    style = grid_style(args)
    written: list[Path] = []

    for g, indices in enumerate(selections):
        fig = se.plot_boundary_grid(pred_pool[indices], true_pool[indices], rows, cols,
                                    scale=scale, style=style)
        stem = figure_dir / f"grid{g}_{rows}x{cols}"
        written += se.save_encoder_figure(fig, stem, formats=args.formats, dpi=args.dpi)
    return written


def replot(args) -> int:
    """Redraws from the artifact store. Loads no checkpoint and extracts nothing."""
    root = resolve_path(args.outdir)
    record = json.loads((root / "metrics.json").read_text())

    pred_pool = artifacts.load_array(root, "pool_pred_fields")
    true_pool = artifacts.load_array(root, "pool_true_fields")
    mass = np.asarray([item["mass"] for item in record["pool"]])
    cells = args.rows * args.cols
    if cells > mass.shape[0]:
        raise ValueError(f"{args.rows}x{args.cols} needs {cells} shapes, "
                         f"the saved pool only has {mass.shape[0]}")
    selections = diverse_selections(mass, cells, args.num_grids, args.seed)

    written = render(pred_pool, true_pool, selections, args.rows, args.cols,
                     float(record["scale"]), args)
    print("\n".join(f"wrote {p}" for p in written))
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = args.rows * args.cols
    if args.pool_size is None:
        args.pool_size = cells * args.num_grids
    if args.pool_size < cells:
        raise ValueError(f"--pool-size {args.pool_size} must be >= rows*cols ({cells})")

    if args.plot_only:
        return replot(args)

    cfg = load_config(args.run_id)
    device = resolve_device()
    siren = load_siren(cfg, device)
    print(f"siren from {cfg.run_id} | device {device} | pool {args.pool_size} shapes | "
          f"{args.num_grids} grids of {args.rows}x{args.cols} at {args.resolution}^2")

    set_seed(args.seed)
    polys = sample_valid_polynomials(args.pool_size, degree=cfg.degree, scale=cfg.scale,
                                     min_area=cfg.pool.min_area, max_area=cfg.pool.max_area,
                                     device=device)

    X_raw = sample_query_points(args.pool_size, cfg.extraction.points_per_shape,
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

    pred_pool = decode_fields(siren, z_batch, lattice, cfg.scale, args.resolution,
                              args.chunk_size)
    true_pool = true_fields(polys, lattice, cfg.degree, cfg.scale, args.resolution)

    mass_points, _ = get_points(args.iou_points, device=device)
    iou = region_iou_batched(siren, z_batch, polys, mass_points, degree=cfg.degree,
                             scale=cfg.scale)
    x_pow_m, y_pow_m = compute_poly_features_batched(
        mass_points.unsqueeze(0).expand(args.pool_size, -1, -1), degree=cfg.degree,
        scale=cfg.scale)
    mass = (evaluate_poly_batched(x_pow_m, y_pow_m, polys) <= 0).float().mean(dim=1)
    mass_np = mass.cpu().numpy()

    selections = diverse_selections(mass_np, cells, args.num_grids, args.seed)

    root = resolve_path(args.outdir)
    tracked = {"config_run_id": cfg.run_id, "pool_size": args.pool_size, "rows": args.rows,
               "cols": args.cols, "num_grids": args.num_grids, "resolution": args.resolution,
               "seed": args.seed, "points_per_shape": cfg.extraction.points_per_shape,
               "extraction_lr": cfg.extraction.lr, "extraction_steps": cfg.extraction.steps}
    run_id = pin_baseline_run(root, "siren_boundary_grid", tracked)

    artifacts.save_arrays(root, polynomials=polys, latents=z_batch,
                          pool_pred_fields=pred_pool, pool_true_fields=true_pool)
    artifacts.write_manifest(root, run_id=run_id)

    record = {
        "run_id": run_id,
        "siren_run_id": cfg.run_id,
        "scale": cfg.scale,
        "degree": cfg.degree,
        "seed": args.seed,
        "rows": args.rows,
        "cols": args.cols,
        "selections": selections,
        "pool": [
            {"index": i,
             "extraction_mse": float(extraction_mse[i]),
             "mass_iou": float(iou[i]),
             "mass": float(mass_np[i])}
            for i in range(args.pool_size)
        ],
    }
    (root / "metrics.json").write_text(json.dumps(record, indent=2))

    print(f"{'shape':>5} {'mass':>8} {'mass IoU':>9} {'extract MSE':>12}")
    for item in record["pool"]:
        print(f"{item['index']:>5} {item['mass']:>8.3f} {item['mass_iou']:>9.4f} "
              f"{item['extraction_mse']:>12.2e}")
    for g, indices in enumerate(selections):
        print(f"grid {g}: shapes {indices}")

    written = render(pred_pool, true_pool, selections, args.rows, args.cols, cfg.scale, args)
    print("\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
