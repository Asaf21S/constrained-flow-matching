# -*- coding: utf-8 -*-
"""Diagnoses why the ECI projection leaves points outside a polygon, and sweeps its budget.

ECI's last integration step has ``step_fraction == 1``, so the returned sample *is* the
projected endpoint and any infeasibility is a failure of the Newton loop alone. A polygon is
a max of half-planes, so a point near a corner is projected onto one face and pushed off
another; the loop then behaves like alternating projection, whose linear rate degrades as the
corner sharpens. This script checks that picture by counting how many half-planes the failed
points violate, and measures what projection budget clears the gate.

    python -m constrained_fm.scripts.check_eci_projection --num-polys 12
"""

from __future__ import annotations

import argparse

import torch

from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.inference.constrained_samplers import DEFAULT_STEPS, sample_eci
from constrained_fm.src.inference.constraint_projection import (DEFAULT_MARGIN,
                                                                project_onto_feasible_region)
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.problems.base import NormalizedConstraint
from constrained_fm.src.problems.bump2d import BumpProblem, sample_polygons

BASE_CKPT = "constrained_fm/baselines/bump2d_base_fm/ckpt.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep the ECI projection budget on bump2d.")
    parser.add_argument("--ckpt", default=BASE_CKPT)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--num-polys", type=int, default=12)
    parser.add_argument("--pool-polys", type=int, default=100)
    parser.add_argument("--num-x0", type=int, default=4000)
    parser.add_argument("--pool-size", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--damping", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--trace-iters", type=int, default=40)
    return parser


def trace(points: torch.Tensor, constraint: NormalizedConstraint, polygon, normalizer,
          margin: float, iters: int) -> None:
    """Replays the Newton loop one iteration at a time on points known to be infeasible.

    Isolates the projection from the integrator: if the residual stalls here, the budget is
    not the problem and the step-acceptance rule is.
    """
    x = points.clone()
    for k in range(iters):
        residual = constraint.value(x) + margin
        bad = residual > 0.0
        if not bool(bad.any()):
            print(f"    iter {k:3d} | cleared")
            return
        faces = ((normalizer.inverse(x[bad]) @ polygon.normals.T - polygon.offsets) > 0.0)
        if k < 8 or k % 8 == 0:
            print(f"    iter {k:3d} | {int(bad.sum()):5d} bad | max res {residual.max():.5f} "
                  f"| mean res {residual[bad].mean():.5f} "
                  f"| {faces.sum(dim=-1).float().mean():.2f} faces")
        x = project_onto_feasible_region(x, constraint, margin=margin, max_iters=1)
    print(f"    iter {iters:3d} | {int((constraint.value(x) + margin > 0).sum()):5d} still bad")


def anatomy(samples: torch.Tensor, constraint: NormalizedConstraint,
            polygon, normalizer, margin: float) -> tuple[int, float, float]:
    """Number of half-planes each failed point violates, and how far outside it sits."""
    bad = samples[constraint.value(samples) > 0.0]
    if bad.numel() == 0:
        return 0, 0.0, 0.0
    physical = normalizer.inverse(bad)
    residuals = physical @ polygon.normals.T - polygon.offsets
    violated = (residuals > 0.0).sum(dim=-1).float()
    return bad.shape[0], violated.mean().item(), residuals.amax(dim=-1).max().item()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()

    problem = BumpProblem()
    normalizer = problem.normalizer().to(device)
    model = UnconstrainedFM(input_dim=2, time_dim=args.time_dim, hidden_dim=args.hidden_dim,
                            num_blocks=args.num_blocks).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    set_seed(args.seed)
    pool = problem.target().sample(args.pool_size, device=device)
    polygons, mass = sample_polygons(args.pool_polys, pool, domain=problem.domain,
                                     min_mass=problem.min_mass, max_mass=problem.max_mass)

    # Hardest first: the earlier sweep showed success rate falls monotonically with mass.
    order = torch.argsort(mass)[:args.num_polys].tolist()
    set_seed(args.seed + 1)
    x0 = torch.randn(args.num_x0, problem.dim, device=device)

    print(f"device {device} | {len(order)} lowest-mass polygons | {args.num_x0} points each")
    print(f"mass {[round(float(mass[i]), 3) for i in order]}\n")

    for damping in args.damping:
        for iters in args.iters:
            rates, faces, worst, counts = [], [], 0.0, 0
            for i in order:
                wrapped = NormalizedConstraint(polygons[i], normalizer)
                samples = sample_eci(model, x0, wrapped, steps=args.steps, margin=args.margin,
                                     projection_iters=iters, projection_damping=damping)
                rates.append(wrapped.success_rate(samples))
                n_bad, mean_faces, max_res = anatomy(samples, wrapped, polygons[i],
                                                     normalizer, args.margin)
                counts += n_bad
                if n_bad:
                    faces.append(mean_faces)
                worst = max(worst, max_res)

            rates_sorted = sorted(rates)
            face_mean = sum(faces) / len(faces) if faces else 0.0
            print(f"damping {damping:.2f} | iters {iters:4d} | SR min {rates_sorted[0]:6.2f} "
                  f"median {rates_sorted[len(rates_sorted) // 2]:6.2f} "
                  f"mean {sum(rates) / len(rates):6.2f} | {counts:6d} failed pts "
                  f"| {face_mean:.2f} faces violated each | worst residual {worst:.4f}",
                  flush=True)

    worst_poly = order[0]
    wrapped = NormalizedConstraint(polygons[worst_poly], normalizer)
    samples = sample_eci(model, x0, wrapped, steps=args.steps, margin=args.margin,
                         projection_iters=16, projection_damping=1.0)
    failed = samples[wrapped.value(samples) > 0.0]
    print(f"\nre-projecting {failed.shape[0]} failed points of polygon {worst_poly} "
          f"(mass {float(mass[worst_poly]):.3f}, {polygons[worst_poly].offsets.shape[0]} faces)")
    trace(failed, wrapped, polygons[worst_poly], normalizer, args.margin, args.trace_iters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
