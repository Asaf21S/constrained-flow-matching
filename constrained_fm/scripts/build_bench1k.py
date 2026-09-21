# -*- coding: utf-8 -*-
"""Stage 1 of the bench1k pipeline: freeze a 1000-constraint benchmark for one problem.

Writes ``constrained_fm/benchmark/benchmark_1k_<problem>.pt``: the stratified constraints,
their exact Monte Carlo masses, and the shared start points every method integrates from.

    sbatch scripts/run_bench1k_build.sh
    python -m constrained_fm.scripts.build_bench1k --problem bump2d --rebuild
"""

from __future__ import annotations

import argparse
import os

from constrained_fm.src.consts import (BENCH1K_MASS_BINS, BENCH1K_MC_POOL_SIZE,
                                       BENCH1K_NUM_CONSTRAINTS, BENCH1K_NUM_X0, BENCH1K_SEED)
from constrained_fm.src.datasets.benchmark_1k import (PROBLEM_NAMES, benchmark_path,
                                                      build_benchmark_1k, load_benchmark_1k,
                                                      mass_bin_histogram, save_benchmark_1k)
from constrained_fm.src.experiment.runtime import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a bench1k constraint benchmark.")
    parser.add_argument("--problem", nargs="+", default=list(PROBLEM_NAMES),
                        choices=list(PROBLEM_NAMES))
    parser.add_argument("--num-constraints", type=int, default=BENCH1K_NUM_CONSTRAINTS)
    parser.add_argument("--mass-bins", type=int, default=BENCH1K_MASS_BINS)
    parser.add_argument("--pool-size", type=int, default=BENCH1K_MC_POOL_SIZE)
    parser.add_argument("--num-x0", type=int, default=BENCH1K_NUM_X0)
    parser.add_argument("--seed", type=int, default=BENCH1K_SEED)
    parser.add_argument("--rebuild", action="store_true",
                        help="regenerate even if the cached benchmark already exists")
    return parser


def report(benchmark: dict) -> None:
    mass = benchmark["mass"]
    lo, hi = benchmark["mass_range"]
    print(f"digest {benchmark['digest']} | {mass.numel()} constraints | "
          f"mass in [{mass.min():.4f}, {mass.max():.4f}]")
    print("\nmass bin occupancy")
    for bin_lo, bin_hi, count in mass_bin_histogram(mass, benchmark["mass_bins"], lo, hi,
                                                    log_spaced=benchmark["problem"] != "bump2d"):
        print(f"  [{bin_lo:.4f}, {bin_hi:.4f})  {count:4d}  {'#' * min(count, 60)}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()

    for problem_name in args.problem:
        path = benchmark_path(problem_name)
        print(f"\n=== {problem_name} | device {device} ===", flush=True)

        if os.path.exists(path) and not args.rebuild:
            print(f"{path} already exists; pass --rebuild to regenerate.")
            report(load_benchmark_1k(problem_name))
            continue

        benchmark = build_benchmark_1k(problem_name, num_constraints=args.num_constraints,
                                       num_bins=args.mass_bins, pool_size=args.pool_size,
                                       num_x0=args.num_x0, seed=args.seed, device=device)
        print(f"\nsaved {save_benchmark_1k(benchmark)}")
        report(benchmark)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
