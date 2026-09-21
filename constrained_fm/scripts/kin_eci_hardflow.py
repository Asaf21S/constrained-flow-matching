# -*- coding: utf-8 -*-
"""Scores ECI and HardFlow against the kinematics6d mass shells.

The shell ``|M - M_target| <= epsilon`` is not convex, so the bisection fallback that rescued
the 2D polygons is unavailable by construction -- ``interior_point`` is None here and the
projection has nothing but its damped Newton loop. A full Newton step can cross the shell
entirely and land on the opposite wall, so the step scale is swept rather than assumed, and
every infeasible sample is attributed to the wall it escaped through.

    python -m constrained_fm.scripts.kin_eci_hardflow
    python -m constrained_fm.scripts.kin_eci_hardflow --damping-sweep 1.0 0.5 0.25 0.1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from constrained_fm.src.consts import KIN_MMD_GAMMA
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.registry import pin_baseline_run, summarize
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.inference.constrained_samplers import (DEFAULT_CHUNK, DEFAULT_STEPS,
                                                               sample_eci, sample_hardflow)
from constrained_fm.src.metrics.distributional import compute_mmd, compute_swd
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.problems.base import NormalizedConstraint
from constrained_fm.src.problems.kinematics6d import KinematicsProblem, sample_mass_constraints

BASE_CKPT = "constrained_fm/baselines/kin6d_base_fm/ckpt.pt"
DEFAULT_OUTDIR = "constrained_fm/baselines"
SAVED_SAMPLES = 2000
SWD_PROJECTIONS = 200
UNDEFINED_LIKELIHOOD_KEYS = ("nll", "kld")
# The thinnest shell is 0.004 wide in units of C, so the 2D default of 1e-3 would land
# projected points a quarter of the way into it.
DEFAULT_MARGIN = 1e-4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECI and HardFlow on kinematics6d.")
    parser.add_argument("--methods", nargs="+", default=["eci", "hardflow"],
                        choices=["eci", "hardflow"])
    parser.add_argument("--ckpt", default=BASE_CKPT)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--correction-loops", type=int, default=1)
    parser.add_argument("--projection-iters", type=int, default=32)
    parser.add_argument("--projection-damping", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=100.0)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--num-shells", type=int, default=100)
    parser.add_argument("--num-x0", type=int, default=10000)
    parser.add_argument("--pool-size", type=int, default=500000)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--damping-sweep", type=float, nargs="+", default=None,
                        help="success-rate-only sweep of the Newton step scale, then exit")
    parser.add_argument("--sweep-shells", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser


def load_base_model(args, problem: KinematicsProblem, device: torch.device) -> UnconstrainedFM:
    path = Path(args.ckpt)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/run_kin_fm.sh first")
    model = UnconstrainedFM(input_dim=problem.dim, time_dim=args.time_dim,
                            hidden_dim=args.hidden_dim, num_blocks=args.num_blocks).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def build_benchmark(args, problem: KinematicsProblem, device: torch.device):
    """Pool, shells and shared initial noise, each from its own seed.

    Seeding the draws separately keeps the shell set identical when only the number of initial
    points changes, which would otherwise shift the global RNG.
    """
    set_seed(args.seed)
    target = problem.target()
    pool = target.sample(args.pool_size, device=device)
    constraints, mass = sample_mass_constraints(args.num_shells, target, pool)

    set_seed(args.seed + 1)
    x0 = torch.randn(args.num_x0, problem.dim, device=device)
    return pool, constraints, mass, x0


def run_sampler(method: str, model, x0: torch.Tensor, wrapped, args,
                damping: float | None = None) -> torch.Tensor:
    if method == "eci":
        return sample_eci(model, x0, wrapped, steps=args.steps,
                          correction_loops=args.correction_loops, margin=args.margin,
                          projection_iters=args.projection_iters,
                          projection_damping=args.projection_damping if damping is None
                          else damping,
                          chunk_size=args.chunk_size)
    return sample_hardflow(model, x0, wrapped, steps=args.steps,
                           guidance_scale=args.guidance_scale, margin=args.margin,
                           chunk_size=args.chunk_size)


def wall_split(constraint, physical: torch.Tensor) -> tuple[float, float]:
    """Fraction of samples escaping through the low-mass and the high-mass wall.

    A non-convex shell fails asymmetrically: a projection that oscillates between the two
    walls leaves roughly balanced debris, whereas a velocity field that simply cannot reach a
    thin window piles up on whichever side the unconstrained model prefers.
    """
    mass = constraint.target.invariant_mass(physical)
    below = (mass < constraint.mass_target - constraint.epsilon).float().mean().item()
    above = (mass > constraint.mass_target + constraint.epsilon).float().mean().item()
    return below * 100.0, above * 100.0


def sweep_damping(args, model, constraints, normalizer, x0) -> int:
    """Success rate against the Newton step scale on the tightest shells."""
    order = sorted(range(len(constraints)), key=lambda i: constraints[i].epsilon)
    subset = [constraints[i] for i in order[:args.sweep_shells]]
    print(f"sweeping {len(subset)} tightest shells | epsilon "
          f"{subset[0].epsilon:.3f} to {subset[-1].epsilon:.3f}\n")

    for damping in args.damping_sweep:
        rates, below, above = [], [], []
        for constraint in tqdm(subset, desc=f"damping {damping}"):
            wrapped = NormalizedConstraint(constraint, normalizer)
            samples = run_sampler("eci", model, x0, wrapped, args, damping=damping)
            rates.append(wrapped.success_rate(samples))
            lo, hi = wall_split(constraint, normalizer.inverse(samples))
            below.append(lo)
            above.append(hi)

        rates_t = torch.tensor(rates)
        print(f"  damping {damping:5.3f} | SR median {rates_t.median():7.3f} "
              f"| mean {rates_t.mean():7.3f} | min {rates_t.min():7.3f} "
              f"| low-wall {np.mean(below):.3f}% | high-wall {np.mean(above):.3f}%")
    return 0


def score(method: str, samples: torch.Tensor, pool_u: torch.Tensor, constraints, normalizer,
          mass: torch.Tensor, args) -> dict:
    keys = ("swd", "mmd", "swd_noise_floor", "mmd_noise_floor", "success_rate",
            "wall_low", "wall_high", "truth_count")
    per_shape: dict[str, list[float]] = {k: [] for k in keys}

    for i, constraint in enumerate(tqdm(constraints, desc=f"{method} scoring")):
        wrapped = NormalizedConstraint(constraint, normalizer)
        truth = pool_u[wrapped.is_feasible(pool_u)]
        half = truth.shape[0] // 2

        per_shape["swd"].append(compute_swd(samples[i], truth,
                                            num_projections=SWD_PROJECTIONS))
        per_shape["mmd"].append(compute_mmd(samples[i], truth, gamma=KIN_MMD_GAMMA))
        per_shape["swd_noise_floor"].append(
            compute_swd(truth[:half], truth[half:], num_projections=SWD_PROJECTIONS))
        per_shape["mmd_noise_floor"].append(
            compute_mmd(truth[:half], truth[half:], gamma=KIN_MMD_GAMMA))
        per_shape["success_rate"].append(wrapped.success_rate(samples[i]))

        low, high = wall_split(constraint, normalizer.inverse(samples[i]))
        per_shape["wall_low"].append(low)
        per_shape["wall_high"].append(high)
        per_shape["truth_count"].append(float(truth.shape[0]))

    per_shape["mass"] = mass.tolist()
    per_shape["epsilon"] = [c.epsilon for c in constraints]
    for key in UNDEFINED_LIKELIHOOD_KEYS:
        per_shape[key] = [float("nan")] * len(constraints)

    return {
        "method": method,
        "problem": "kinematics6d",
        "model": "UnconstrainedFM + inference-time constraint",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "likelihood": "undefined -- trajectories are altered outside the probability-flow ODE",
        "frame": "metrics in the normalised frame; success rate is frame-invariant",
        "sampling": {"steps": args.steps, "margin": args.margin,
                     "correction_loops": args.correction_loops if method == "eci" else None,
                     "projection_iters": args.projection_iters if method == "eci" else None,
                     "projection_damping": args.projection_damping if method == "eci" else None,
                     "guidance_scale": args.guidance_scale if method == "hardflow" else None},
        "eval": {"num_shells": args.num_shells, "num_x0": args.num_x0,
                 "pool_size": args.pool_size, "swd_projections": SWD_PROJECTIONS,
                 "mmd_gamma": KIN_MMD_GAMMA},
        "per_shape": per_shape,
        "summary": summarize(per_shape),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()

    problem = KinematicsProblem()
    normalizer = problem.normalizer().to(device)
    model = load_base_model(args, problem, device)
    print(f"device {device} | base model {args.ckpt}")

    pool, constraints, mass, x0 = build_benchmark(args, problem, device)
    pool_u = normalizer.forward(pool)
    print(f"{len(constraints)} shells | mass in [{mass.min():.4f}, {mass.max():.4f}] "
          f"| {x0.shape[0]} initial points")

    if args.damping_sweep is not None:
        return sweep_damping(args, model, constraints, normalizer, x0)

    for method in args.methods:
        out_samples = torch.empty(len(constraints), x0.shape[0], problem.dim, device=device)
        for i, constraint in enumerate(tqdm(constraints, desc=f"{method} sampling")):
            wrapped = NormalizedConstraint(constraint, normalizer)
            out_samples[i] = run_sampler(method, model, x0, wrapped, args)

        record = score(method, out_samples, pool_u, constraints, normalizer, mass, args)

        out = Path(args.outdir) / f"kin6d_{method}"
        run_id = pin_baseline_run(out, f"kin6d_{method}", args, extra={"base_ckpt": args.ckpt})
        record["run_id"] = run_id
        (out / "metrics.json").write_text(json.dumps(record, indent=2))

        physical = normalizer.inverse(out_samples[:, :SAVED_SAMPLES])
        artifacts.save_arrays(out, samples=physical.cpu().numpy(),
                              mass=mass.numpy().astype(np.float64))
        artifacts.write_manifest(out, run_id=run_id, method=f"kin6d_{method}")

        summary = record["summary"]
        print(f"\n### kinematics6d {method} ({run_id})")
        for key in ("success_rate", "swd", "swd_noise_floor", "mmd", "mmd_noise_floor",
                    "wall_low", "wall_high"):
            print(f"  {key:16s} median {summary[f'{key}_median']:.4f} | "
                  f"mean {summary[f'{key}_mean']:.4f} | p5 {summary[f'{key}_p5']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
