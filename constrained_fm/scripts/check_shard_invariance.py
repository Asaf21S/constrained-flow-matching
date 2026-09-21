# -*- coding: utf-8 -*-
"""Gate for the bench1k pipeline: a constraint must score the same however the shards are cut.

The sharded evaluation is only sound if slicing is bookkeeping. Every seed in
:mod:`eval_bench1k` is therefore keyed to the *global* constraint index rather than to the
position within a shard, and this check is what proves it: the same constraint range is
scored once as a single shard and again as two, and the two sets of rows must agree exactly.
Anything but an exact match means some metric is reading the shard boundary, and the merged
benchmark would silently depend on how the job array happened to be partitioned.

Run on a compute node -- it samples and scores like a real shard would.

    sbatch scripts/run_shard_invariance.sh
    python -m constrained_fm.scripts.check_shard_invariance --problem bump2d
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from constrained_fm.scripts.eval_bench1k import PROBLEM_METHODS, main as eval_main
from constrained_fm.src.datasets.benchmark_1k import PROBLEM_NAMES

DEFAULT_OUTDIR = "constrained_fm/baselines/bench1k_invariance"
COMPARED_KEYS = ("success_rate", "swd", "mmd", "jsd", "nll", "kld", "in_support_fraction",
                 "swd_noise_floor", "mmd_noise_floor",
                 "jsd_noise_floor", "truth_count", "compared_count", "mass")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check bench1k shard invariance.")
    parser.add_argument("--problem", nargs="+", default=list(PROBLEM_NAMES),
                        choices=list(PROBLEM_NAMES))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--num-constraints", type=int, default=8,
                        help="size of the range to re-cut; kept small, this is a GPU check")
    parser.add_argument("--num-x0", type=int, default=2000)
    parser.add_argument("--pool-size", type=int, default=200000)
    parser.add_argument("--gt-batch", type=int, default=200000)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--keep", action="store_true", help="leave the scratch shards on disk")
    return parser


def run_shard(problem: str, start: int, end: int, args) -> None:
    argv = ["--problem", problem, "--start-idx", str(start), "--end-idx", str(end),
            "--outdir", args.outdir, "--num-x0", str(args.num_x0),
            "--pool-size", str(args.pool_size), "--gt-batch", str(args.gt_batch)]
    if args.methods is not None:
        argv += ["--methods", *args.methods]
    eval_main(argv)


def rows_by_index(shard_dir: Path, method: str, cuts: list[tuple[int, int]]) -> dict:
    """``{constraint index: {metric: value}}`` gathered over one cut's shard files."""
    rows: dict[int, dict[str, float]] = {}
    for start, end in cuts:
        payload = json.loads((shard_dir / f"{method}__{start:05d}_{end:05d}.json").read_text())
        for position, index in enumerate(payload["indices"]):
            rows[index] = {key: float(values[position])
                           for key, values in payload["per_shape"].items()
                           if key in COMPARED_KEYS}
    return rows


def compare(whole: dict, split: dict, method: str) -> bool:
    """Exact agreement, metric by metric. Non-finite entries must match in kind, not value."""
    ok = True
    for key in COMPARED_KEYS:
        worst, worst_index, compared = 0.0, -1, 0
        for index in sorted(whole):
            a, b = whole[index].get(key), split[index].get(key)
            if a is None or b is None:
                continue
            if not (math.isfinite(a) and math.isfinite(b)):
                if math.isfinite(a) != math.isfinite(b):
                    print(f"  [{method}] {key}: constraint {index} is "
                          f"{a} in the whole cut and {b} in the split cut")
                    ok = False
                continue
            compared += 1
            if abs(a - b) > worst:
                worst, worst_index = abs(a - b), index
        if compared == 0:
            print(f"  [{method}] {key:16s} not compared")
            continue
        # An exactly matching metric leaves worst_index at -1, and must still be reported.
        status = "ok" if worst == 0.0 else f"DRIFT at constraint {worst_index}"
        print(f"  [{method}] {key:16s} {compared:3d} compared | "
              f"max |whole - split| {worst:.6e}  {status}")
        ok = ok and worst == 0.0
    return ok


def check_problem(problem: str, args) -> bool:
    n = args.num_constraints
    half = n // 2
    out = Path(args.outdir) / problem
    shutil.rmtree(out, ignore_errors=True)

    print(f"\n=== {problem}: scoring [0, {n}) as one shard ===", flush=True)
    run_shard(problem, 0, n, args)
    print(f"\n=== {problem}: re-scoring as [0, {half}) + [{half}, {n}) ===", flush=True)
    run_shard(problem, 0, half, args)
    run_shard(problem, half, n, args)

    methods = args.methods or list(PROBLEM_METHODS[problem])
    print(f"\n--- {problem} shard invariance ---")
    passed = True
    for method in methods:
        whole = rows_by_index(out / "shards", method, [(0, n)])
        split = rows_by_index(out / "shards", method, [(0, half), (half, n)])
        if set(whole) != set(split):
            print(f"  [{method}] the two cuts cover different constraints")
            passed = False
            continue
        passed = compare(whole, split, method) and passed

    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)
    return passed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = {problem: check_problem(problem, args) for problem in args.problem}

    print("\n=== summary ===")
    for problem, passed in results.items():
        print(f"  {problem:14s} {'PASSED' if passed else 'FAILED'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
