# -*- coding: utf-8 -*-
"""Stage 1 of the v1k pipeline: build the 1000-constraint benchmark and its NLL point set.

Writes ``constrained_fm/benchmark/validation_set_v1k.pt`` (polynomials, exact Monte Carlo
masses, shared start points) and ``nll_eval_points_v1k.pt`` (the frozen ground-truth points
every exact-likelihood score is averaged over). Both are pure functions of their seeds, so a
deleted cache is reproduced bit-for-bit.

    sbatch scripts/run_val1k_build.sh
    python -m constrained_fm.scripts.build_val1k --rebuild
"""

from __future__ import annotations

import argparse
import os

import torch

from constrained_fm.src.consts import (PLANE_SCALE, POLYNOMIAL_DEGREE, VAL1K_MASS_BINS,
                                       VAL1K_MAX_MASS, VAL1K_MC_POOL_SIZE, VAL1K_MIN_MASS,
                                       VAL1K_NUM_POLYS, VAL1K_NUM_X0, VAL1K_SEED, VAL1K_SET_PATH)
from constrained_fm.src.datasets.validation_v1k import (build_validation_set_v1k,
                                                        save_validation_set_v1k)
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.metrics.eval_points import load_nll_eval_set_v1k


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the v1k validation benchmark.")
    parser.add_argument("--num-polys", type=int, default=VAL1K_NUM_POLYS)
    parser.add_argument("--min-mass", type=float, default=VAL1K_MIN_MASS)
    parser.add_argument("--max-mass", type=float, default=VAL1K_MAX_MASS)
    parser.add_argument("--mass-bins", type=int, default=VAL1K_MASS_BINS)
    parser.add_argument("--pool-size", type=int, default=VAL1K_MC_POOL_SIZE)
    parser.add_argument("--num-x0", type=int, default=VAL1K_NUM_X0)
    parser.add_argument("--degree", type=int, default=POLYNOMIAL_DEGREE)
    parser.add_argument("--scale", type=float, default=PLANE_SCALE)
    parser.add_argument("--seed", type=int, default=VAL1K_SEED)
    parser.add_argument("--rebuild", action="store_true",
                        help="regenerate even if the cached set already exists")
    parser.add_argument("--skip-nll-points", action="store_true")
    return parser


def print_mass_histogram(mass: torch.Tensor, num_bins: int, lo: float, hi: float) -> None:
    edges = torch.linspace(lo, hi, num_bins + 1, dtype=torch.float64)
    counts = torch.bucketize(mass, edges, right=True).clamp(1, num_bins) - 1
    print("\nmass bin occupancy")
    for b in range(num_bins):
        n = int((counts == b).sum())
        print(f"  [{edges[b]:.3f}, {edges[b + 1]:.3f})  {n:4d}  {'#' * min(n, 60)}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    print(f"device {device} | target {args.num_polys} polynomials | "
          f"mass in [{args.min_mass}, {args.max_mass}]", flush=True)

    if os.path.exists(VAL1K_SET_PATH) and not args.rebuild:
        print(f"{VAL1K_SET_PATH} already exists; pass --rebuild to regenerate.")
    else:
        val_set = build_validation_set_v1k(
            num_polys=args.num_polys, min_mass=args.min_mass, max_mass=args.max_mass,
            num_bins=args.mass_bins, pool_size=args.pool_size, num_x0=args.num_x0,
            degree=args.degree, scale=args.scale, seed=args.seed, device=device)
        save_validation_set_v1k(val_set)
        print(f"\nsaved {VAL1K_SET_PATH}")
        print(f"digest {val_set['poly_digest']} | polynomials {tuple(val_set['polynomials'].shape)}")
        print_mass_histogram(val_set["mass"], args.mass_bins, args.min_mass, args.max_mass)

    if not args.skip_nll_points:
        nll_set = load_nll_eval_set_v1k(degree=args.degree, scale=args.scale,
                                        rebuild=args.rebuild)
        short = int(nll_set["available"].min())
        print(f"\nNLL points  {nll_set['path']}")
        print(f"            {nll_set['num_points']} per constraint | pool {nll_set['pool_size']} | "
              f"digest {nll_set['poly_digest']}")
        if short < nll_set["num_points"]:
            print(f"warning: the smallest constraint only has {short} valid pool points")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
