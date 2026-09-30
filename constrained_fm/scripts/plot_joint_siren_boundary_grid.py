# -*- coding: utf-8 -*-
"""Render mass-diverse 2x4 boundary grids for the joint polynomial/polygon SIREN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from constrained_fm.scripts.plot_siren_boundary_grid import diverse_selections, grid_style
from constrained_fm.scripts.plot_siren_encoder import decode_fields, resolve_path
from constrained_fm.src.datasets import joint_conditioning as jc
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.config import REPO_ROOT
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.visualization import siren_encoder as se

SIREN_DIR = "constrained_fm/functa_dataset/joint_siren_sharp"
OUTDIR = "constrained_fm/baselines/joint_siren_boundary_grid"
FIGURE_DIR = "constrained_fm/images/thesis_pool/siren_encoder/joint_boundary_grid"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--siren-dir", default=SIREN_DIR)
    parser.add_argument("--checkpoint", default="siren_best.pt")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--num-grids", type=int, default=3)
    parser.add_argument("--pool-per-family", type=int, default=None,
                        help="shapes decoded per family; defaults to cols*num-grids")
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--iou-points", type=int, default=100000)
    parser.add_argument("--proxy-points", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--style", default="paper", choices=sorted(se.STYLE_PRESETS))
    parser.add_argument("--cmap", default=None)
    parser.add_argument("--field-render", default=None, choices=["imshow", "contourf"])
    parser.add_argument("--pred-color", default=None)
    parser.add_argument("--gt-color", default=None)
    parser.add_argument("--smooth-sigma", type=float, default=None)
    parser.add_argument("--colorbar-label", default=None)
    parser.add_argument("--no-colorbar", action="store_true")
    parser.add_argument("--clip-to-unit", action="store_true")
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--no-ticks", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    parser.add_argument("--plot-only", action="store_true")
    return parser


def render(pred: np.ndarray, true: np.ndarray, selections: list[list[int]],
           cols: int, scale: float, args) -> list[Path]:
    figure_dir = resolve_path(args.figure_dir)
    style = grid_style(args)
    written: list[Path] = []
    for index, selected in enumerate(selections):
        fig = se.plot_boundary_grid(pred[selected], true[selected], 2, cols,
                                    scale=scale, style=style)
        written += se.save_encoder_figure(
            fig, figure_dir / f"grid{index}_2x{cols}", formats=args.formats, dpi=args.dpi)
    return written


def choose_grids(mass: np.ndarray, family: np.ndarray, cols: int,
                 num_grids: int, seed: int) -> list[list[int]]:
    polygons = np.flatnonzero(family == jc.FAMILY_POLYGON)
    polynomials = np.flatnonzero(family == jc.FAMILY_POLYNOMIAL)
    if min(len(polygons), len(polynomials)) < cols:
        raise ValueError(f"each family needs at least {cols} pool members")
    polygon_rows = diverse_selections(mass[polygons], cols, num_grids, seed)
    polynomial_rows = diverse_selections(mass[polynomials], cols, num_grids, seed + 1)
    return [[*polygons[gon].tolist(), *polynomials[poly].tolist()]
            for gon, poly in zip(polygon_rows, polynomial_rows)]


def replot(args) -> int:
    root = resolve_path(args.outdir)
    record = json.loads((root / "metrics.json").read_text())
    pred = artifacts.load_array(root, "pool_pred_fields")
    true = artifacts.load_array(root, "pool_true_fields")
    family = artifacts.load_array(root, "family")
    mass = np.asarray([entry["mass"] for entry in record["pool"]])
    selections = choose_grids(mass, family, args.cols, args.num_grids, args.seed)
    written = render(pred, true, selections, args.cols, float(record["scale"]), args)
    print("\n".join(f"wrote {path}" for path in written))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plot_only:
        return replot(args)
    if args.cols < 1 or args.num_grids < 1:
        raise ValueError("--cols and --num-grids must be positive")

    device = resolve_device()
    siren_dir = resolve_path(args.siren_dir)
    siren, meta = jc.load_joint_siren(siren_dir, args.checkpoint, device)
    pool_size = args.pool_per_family or args.cols * args.num_grids
    if pool_size < args.cols:
        raise ValueError("--pool-per-family must be at least --cols")
    print(f"siren {meta['run_id']} | device {device} | {pool_size} shapes per family | "
          f"{args.num_grids} grids of 2x{args.cols} at {args.resolution}^2")

    set_seed(args.seed)
    proxy = jc.proxy_set(args.proxy_points, meta["degree"], meta["scale"], device)
    polygon_shapes = jc.sample_joint_shapes(
        pool_size, proxy, polygon_fraction=1.0, random_sign=False,
        degree=meta["degree"], scale=meta["scale"], min_mass=meta["min_mass"],
        max_mass=meta["max_mass"], device=device)
    polynomial_shapes = jc.sample_joint_shapes(
        pool_size, proxy, polygon_fraction=0.0, random_sign=False,
        degree=meta["degree"], scale=meta["scale"], min_mass=meta["min_mass"],
        max_mass=meta["max_mass"], device=device)
    shapes = {key: torch.cat([polygon_shapes[key], polynomial_shapes[key]])
              for key in jc.SHAPE_KEYS}

    x_raw = jc.sample_query_points(2 * pool_size, meta["points_per_shape"],
                                  scale=meta["scale"],
                                  gmm_fraction=meta["query_gmm_fraction"], device=device)
    x, y = jc.regression_targets(shapes, x_raw, meta["tau"], meta["degree"],
                                 meta["scale"], meta["poly_gain"])
    latents, extraction_mse = extract_latents_batched(
        siren, x, y, lr=meta["inner_lr"], steps=meta["inner_steps"])

    axis = torch.linspace(-meta["scale"], meta["scale"], args.resolution, device=device)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    lattice = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    pred = decode_fields(siren, latents, lattice, meta["scale"], args.resolution,
                         args.chunk_size)
    with torch.no_grad():
        truth = jc.constraint_values(
            shapes, lattice.unsqueeze(0).expand(2 * pool_size, -1, -1), meta["tau"],
            meta["degree"], meta["scale"], meta["poly_gain"])
    true = truth.reshape(2 * pool_size, args.resolution, args.resolution).cpu().numpy()

    mass_points, _ = jc.get_points(args.iou_points, device=device)
    masses = jc.constraint_mass(shapes, mass_points, meta["tau"], meta["degree"],
                                meta["scale"]).cpu().numpy()
    ious = jc.mass_iou(siren, latents, shapes, mass_points, meta["tau"],
                       meta["degree"], meta["scale"])
    ious_np = ious.numpy()
    family = shapes["family"].cpu().numpy()
    selections = choose_grids(masses, family, args.cols, args.num_grids, args.seed)

    root = resolve_path(args.outdir)
    run_id = pin_baseline_run(root, "joint_siren_boundary_grid", args,
                              extra={"siren_run_id": meta["run_id"], "tau": meta["tau"],
                                     "poly_gain": meta["poly_gain"]})
    artifacts.save_arrays(root, **{key: value.cpu() for key, value in shapes.items()},
                          latents=latents, pool_pred_fields=pred, pool_true_fields=true)
    artifacts.write_manifest(root, run_id=run_id, siren_run_id=meta["run_id"],
                             checkpoint=args.checkpoint)
    record = {
        "run_id": run_id, "siren_run_id": meta["run_id"], "checkpoint": args.checkpoint,
        "tau": meta["tau"], "poly_gain": meta["poly_gain"], "scale": meta["scale"],
        "degree": meta["degree"], "seed": args.seed, "cols": args.cols,
        "num_grids": args.num_grids, "pool_per_family": pool_size,
        "selections": selections,
        "pool": [{"index": i, "family": jc.FAMILY_NAMES[int(family[i])],
                  "mass": float(masses[i]), "mass_iou": float(ious[i]),
                  "extraction_mse": float(extraction_mse[i])}
                 for i in range(2 * pool_size)]}
    (root / "metrics.json").write_text(json.dumps(record, indent=2))

    for name, family_id in (("polygon", jc.FAMILY_POLYGON),
                            ("polynomial", jc.FAMILY_POLYNOMIAL)):
        keep = family == family_id
        print(f"{name}: IoU mean {ious_np[keep].mean():.4f} | mass range "
              f"{masses[keep].min():.3f}-{masses[keep].max():.3f}")
    written = render(pred, true, selections, args.cols, meta["scale"], args)
    print("\n".join(f"wrote {path}" for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
