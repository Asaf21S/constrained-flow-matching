# -*- coding: utf-8 -*-
"""Why exact rejection sampling scores a success rate below 100% on the polygon benchmark.

Ground-truth points are accepted with ``C(x) <= 0`` in physical units, mapped into the
normalised frame, and later rescored through :class:`NormalizedConstraint`. Every point that
fails the rescoring is reported with its exact float64 constraint value, so the failures can
be attributed to one specific numerical step rather than guessed at.

Four rescoring variants isolate the candidates:

* ``as_scored``  -- the benchmark path: matmul value, after the normaliser round trip,
* ``no_trip``    -- matmul value on the physical points, no round trip,
* ``elementwise``-- round trip, but ``C`` evaluated with a multiply-and-sum instead of a matmul,
* ``float64``    -- round trip, then ``C`` evaluated in float64.

    python -m constrained_fm.scripts.check_bump_gt_sr
"""

from __future__ import annotations

import argparse
import os

import torch

from constrained_fm.scripts.eval_bench1k import rejection_sample
from constrained_fm.scripts.plot_bumphunt import bench_namespace
from constrained_fm.src.datasets.benchmark_1k import constraints_from, load_benchmark_1k
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.problems.base import NormalizedConstraint
from constrained_fm.src.problems.bump2d import BumpProblem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attribute ground-truth SR failures.")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=10000)
    return parser


def elementwise_value(constraint, x: torch.Tensor) -> torch.Tensor:
    """``max_i (a_i . x - b_i)`` without a matmul, so no reduced-precision GEMM path applies."""
    products = (x.unsqueeze(-2) * constraint.normals).sum(dim=-1)
    return (products - constraint.offsets).amax(dim=-1)


def float64_value(constraint, x: torch.Tensor) -> torch.Tensor:
    x64 = x.double()
    products = (x64.unsqueeze(-2) * constraint.normals.double()).sum(dim=-1)
    return (products - constraint.offsets.double()).amax(dim=-1)


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device()
    problem = BumpProblem()
    normalizer = problem.normalizer().to(device)
    bench = bench_namespace(("gt",))

    print(f"torch {torch.__version__} | device {device}")
    print(f"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE={os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')}")
    print(f"torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
    print(f"torch.get_float32_matmul_precision()={torch.get_float32_matmul_precision()}")

    benchmark = load_benchmark_1k("bump2d", device=device)
    constraints = constraints_from(benchmark, problem, device=device)
    indices = list(range(0, len(constraints), args.stride))

    totals = {"as_scored": 0, "no_trip": 0, "elementwise": 0, "float64": 0}
    truly_outside = 0
    exact_depths, matmul_errors, trip_shifts = [], [], []
    total_points = 0

    for index in indices:
        constraint = constraints[index]
        physical = rejection_sample(constraint, problem, args.num_samples, index, bench, device)
        tripped = normalizer.inverse(normalizer.forward(physical))
        wrapped = NormalizedConstraint(constraint, normalizer)
        total_points += physical.shape[0]

        fails = {
            "as_scored": ~wrapped.is_feasible(normalizer.forward(physical)),
            "no_trip": ~constraint.is_feasible(physical),
            "elementwise": elementwise_value(constraint, tripped) > 0,
            "float64": float64_value(constraint, tripped) > 0,
        }
        for key, mask in fails.items():
            totals[key] += int(mask.sum())

        exact = float64_value(constraint, physical)
        truly_outside += int((exact > 0).sum())
        matmul_errors.append((constraint.value(physical).double() - exact).abs().max())
        trip_shifts.append((tripped.double() - physical.double()).abs().max())

        bad = fails["as_scored"]
        if bool(bad.any()):
            exact_depths.append(exact[bad].cpu())
            print(f"constraint {index:4d}: {int(bad.sum())} fail | exact C(x) of failures "
                  f"{[f'{v:.2e}' for v in exact[bad].tolist()[:6]]}")

    print("\n--- failures per variant, out of", total_points, "accepted points ---")
    for key, count in totals.items():
        print(f"{key:<12} {count:6d}  ({count / total_points * 100:.4f}%)")
    print(f"accepted points truly outside in float64: {truly_outside}")
    print(f"max |C_matmul - C_float64| over accepted points: {max(matmul_errors):.3e}")
    print(f"max normaliser round-trip shift:              {max(trip_shifts):.3e}")
    if exact_depths:
        depths = torch.cat(exact_depths)
        print(f"exact C(x) of failing points: min {depths.min():.3e} max {depths.max():.3e}")


if __name__ == "__main__":
    main()
