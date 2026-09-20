# -*- coding: utf-8 -*-
"""M0 gate: the problem/constraint seam must be a pure refactor.

Two checks, neither of which writes to any baseline directory:

  A. ``PolynomialConstraint`` reproduces the legacy polynomial evaluation exactly, and its
     feasibility mask agrees with ``compute_success_rate_polynomial``.
  B. ECI and HardFlow, now driven by that constraint object, reproduce the per-shape success
     rates cached in ``baselines/{eci,hardflow}/metrics.json``. Those samplers carry no RNG,
     so given the frozen validation x0 the rates must match bit-for-bit. SWD/MMD/JSD are not
     compared: both subsample with an unseeded generator.

    sbatch scripts/run_m0_regression.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.experiment.config import REPO_ROOT
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.geometry.polynomials import (compute_poly_features_batched,
                                                     evaluate_poly_batched)
from constrained_fm.src.inference.constrained_samplers import sample_eci, sample_hardflow
from constrained_fm.src.inference.constraint_projection import DEFAULT_MARGIN
from constrained_fm.src.metrics.success_rates import compute_success_rate_polynomial
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.problems.gmm_poly import PolynomialConstraint

BASE_CKPT = "constrained_fm/baselines/base_fm/ckpt.pt"
CACHED_METRICS = {"eci": "constrained_fm/baselines/eci/metrics.json",
                  "hardflow": "constrained_fm/baselines/hardflow/metrics.json"}
# The settings the cached run recorded under its "sampling" key.
CACHED_STEPS = 100
CACHED_GUIDANCE = 100.0
CACHED_CORRECTION_LOOPS = 1
CACHED_PROJECTION_ITERS = 16


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M0 refactor regression gate.")
    parser.add_argument("--num-polys", type=int, default=20,
                        help="prefix of the validation set to re-sample")
    parser.add_argument("--num-x0", type=int, default=10000)
    parser.add_argument("--ckpt", default=BASE_CKPT)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    # 0.02 pp is two points in 10000: tight enough to catch any algorithmic change, loose
    # enough to survive a boundary point flipping on a 1-ULP difference in C(x).
    parser.add_argument("--tol", type=float, default=0.02,
                        help="allowed absolute drift in per-shape success rate (percentage points)")
    parser.add_argument("--rtol", type=float, default=1e-10,
                        help="relative tolerance on C(x) vs the batched einsum path in float64")
    return parser


def check_constraint_algebra(polys: torch.Tensor, points: torch.Tensor,
                             rtol: float) -> list[str]:
    """Part A: the wrapper must agree with the independent batched formula.

    Compared against ``evaluate_poly_batched``, which contracts with one einsum, whereas the
    wrapper contracts with two bmm calls. The two orderings are algebraically identical but
    not bitwise equal, and a degree-3 form evaluated against large coefficients cancels hard:
    in float32 the residual is ~1e-4 absolute purely from that cancellation. The assertion is
    therefore made in float64, where cancellation leaves ~1e-13 and any structural error (a
    transposed coefficient matrix, a swapped feature axis) still shows up as O(1).
    """
    failures = []
    scale_hint = polys.abs().amax(dim=(-2, -1))

    for dtype in (torch.float64, points.dtype):
        pts = points.to(dtype)
        cf = polys.to(dtype)
        x_pow, y_pow = compute_poly_features_batched(pts.unsqueeze(0).expand(cf.shape[0], -1, -1),
                                                     degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE)
        legacy = evaluate_poly_batched(x_pow, y_pow, cf)

        worst = 0.0
        for i in range(cf.shape[0]):
            constraint = PolynomialConstraint(cf[i], degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE)
            drift = (constraint.value(pts) - legacy[i]).abs().max().item()
            worst = max(worst, drift / max(scale_hint[i].item(), 1e-12))
            if dtype == torch.float64 and drift > rtol * scale_hint[i].item():
                failures.append(f"poly {i}: float64 C(x) drift {drift:.3e} exceeds "
                                f"{rtol * scale_hint[i].item():.3e}")
        verdict = "asserted" if dtype == torch.float64 else "informational"
        print(f"  {dtype} max drift / |coeff|_inf = {worst:.3e}  ({verdict})")

    for i in range(polys.shape[0]):
        constraint = PolynomialConstraint(polys[i], degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE)
        wrapper_sr = constraint.success_rate(points)
        legacy_sr = compute_success_rate_polynomial(points, polys[i], POLYNOMIAL_DEGREE,
                                                    PLANE_SCALE, points.device)
        if abs(wrapper_sr - float(legacy_sr)) > 1e-9:
            failures.append(f"poly {i}: success rate {wrapper_sr} vs legacy {legacy_sr}")

    return failures


def load_base_model(args, device: torch.device) -> UnconstrainedFM:
    path = REPO_ROOT / args.ckpt
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/run_base_fm.sh first")
    model = UnconstrainedFM(time_dim=args.time_dim, hidden_dim=args.hidden_dim,
                            num_blocks=args.num_blocks).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def check_samplers(args, polys: torch.Tensor, x0: torch.Tensor,
                   device: torch.device) -> list[str]:
    """Part B: per-shape success rates must match the cached baseline exactly."""
    failures = []
    model = load_base_model(args, device)

    for method, rel_path in CACHED_METRICS.items():
        cached = json.loads((REPO_ROOT / rel_path).read_text())["per_shape"]["success_rate"]

        rates = []
        for i in tqdm(range(polys.shape[0]), desc=f"{method} regression"):
            constraint = PolynomialConstraint(polys[i], degree=POLYNOMIAL_DEGREE,
                                              scale=PLANE_SCALE)
            if method == "eci":
                samples = sample_eci(model, x0, constraint, steps=CACHED_STEPS,
                                     correction_loops=CACHED_CORRECTION_LOOPS,
                                     margin=DEFAULT_MARGIN,
                                     projection_iters=CACHED_PROJECTION_ITERS,
                                     chunk_size=args.chunk_size)
            else:
                samples = sample_hardflow(model, x0, constraint, steps=CACHED_STEPS,
                                          guidance_scale=CACHED_GUIDANCE, margin=DEFAULT_MARGIN,
                                          chunk_size=args.chunk_size)
            rates.append(constraint.success_rate(samples.detach()))

        drift = [(i, rates[i], cached[i]) for i in range(len(rates))
                 if abs(rates[i] - cached[i]) > args.tol]
        worst = max((abs(r - c) for r, c in zip(rates, cached[:len(rates)])), default=0.0)
        print(f"  {method}: {len(rates)} shapes, max |drift| = {worst:.3e} pp")
        for i, got, want in drift:
            failures.append(f"{method} poly {i}: success rate {got:.4f} vs cached {want:.4f}")

    return failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    set_seed(args.seed)

    val_set = get_validation_set(device=device)
    polys = val_set["polynomials"][:args.num_polys].to(device)
    x0 = val_set["x0"][:args.num_x0].to(device)
    print(f"device {device} | {polys.shape[0]} constraints | {x0.shape[0]} samples each")

    print("\n[A] constraint algebra")
    failures = check_constraint_algebra(polys, val_set["x1"][:50000].to(device), args.rtol)
    print(f"  {len(failures)} mismatch(es)")

    print("\n[B] ECI / HardFlow vs cached baselines")
    failures += check_samplers(args, polys, x0, device)

    if failures:
        print(f"\nFAILED: {len(failures)} regression(s)")
        for line in failures[:40]:
            print(f"  {line}")
        return 1

    print("\nPASSED: the problem/constraint seam is a pure refactor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
