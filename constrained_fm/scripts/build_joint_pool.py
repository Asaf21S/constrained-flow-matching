# -*- coding: utf-8 -*-
"""Builds the joint FM conditioning pool: both orientations of every polynomial and polygon.

Each entry stores the constraint once, with ``z_pos`` encoding ``{v <= 0}`` and ``z_neg`` its
exact complement ``{-v <= 0}``, so an FM batch can pick whichever orientation contains its
target sample. CAVIA settings and ``tau`` are read from the joint SIREN's ``metrics.json``.

    sbatch scripts/run_joint_pool.sh
    sbatch scripts/run_joint_pool.sh --siren-dir constrained_fm/functa_dataset/joint_siren_smoke --pool-size 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from constrained_fm.src.datasets import joint_conditioning as jc
from constrained_fm.src.experiment.config import REPO_ROOT
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device, set_seed

SIREN_DIR = "constrained_fm/functa_dataset/joint_siren"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--siren-dir", default=SIREN_DIR)
    parser.add_argument("--checkpoint", default="siren_best.pt")
    parser.add_argument("--pool-size", type=int, default=20000)
    parser.add_argument("--polygon-fraction", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--proxy-points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=None, help="default <siren-dir>/pool")
    return parser


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def family_summary(pool: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    summary = {}
    for index, name in enumerate(jc.FAMILY_NAMES):
        rows = pool["family"] == index
        if not bool(rows.any()):
            continue
        summary[name] = {
            "count": int(rows.sum()),
            "mass_min": float(pool["mass_pos"][rows].min()),
            "mass_median": float(pool["mass_pos"][rows].median()),
            "mass_max": float(pool["mass_pos"][rows].max()),
            "mse_pos_median": float(pool["mse_pos"][rows].median()),
            "mse_neg_median": float(pool["mse_neg"][rows].median()),
            "z_pos_norm_median": float(pool["z_pos"][rows].norm(dim=-1).median()),
            "z_neg_norm_median": float(pool["z_neg"][rows].norm(dim=-1).median()),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    siren_dir = resolve_path(args.siren_dir)
    siren, meta = jc.load_joint_siren(siren_dir, args.checkpoint, device)
    outdir = resolve_path(args.outdir) if args.outdir else siren_dir / "pool"

    set_seed(args.seed)
    run_id = pin_baseline_run(outdir, "joint_pool", args,
                              extra={"siren_run_id": meta["run_id"], "tau": meta["tau"]})
    print(f"run {run_id} | siren {meta['run_id']} ({args.checkpoint}) | tau {meta['tau']:.4f} "
          f"| {args.pool_size} constraints, polygon fraction {args.polygon_fraction}")

    proxy = jc.proxy_set(args.proxy_points, meta["degree"], meta["scale"], device)
    started = time.time()
    pool = jc.build_joint_pool(
        siren, proxy, meta["tau"], pool_size=args.pool_size,
        polygon_fraction=args.polygon_fraction, points_per_shape=meta["points_per_shape"],
        extraction_steps=meta["inner_steps"], extraction_lr=meta["inner_lr"],
        chunk_size=args.chunk_size, query_gmm_fraction=meta["query_gmm_fraction"],
        degree=meta["degree"], scale=meta["scale"], min_mass=meta["min_mass"],
        max_mass=meta["max_mass"], device=device)

    path = outdir / "pool.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save(pool, tmp)
    tmp.replace(path)

    summary = family_summary(pool)
    (outdir / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "siren_run_id": meta["run_id"], "checkpoint": args.checkpoint,
        "tau": meta["tau"], "pool_size": args.pool_size,
        "polygon_fraction": args.polygon_fraction,
        "build_seconds": round(time.time() - started, 1), "families": summary}, indent=2))

    for name, stats in summary.items():
        print(f"{name}: {stats['count']} | mass {stats['mass_min']:.3f}/{stats['mass_median']:.3f}/"
              f"{stats['mass_max']:.3f} | mse +{stats['mse_pos_median']:.2e} "
              f"-{stats['mse_neg_median']:.2e} | ||z|| +{stats['z_pos_norm_median']:.4f} "
              f"-{stats['z_neg_norm_median']:.4f}")
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
