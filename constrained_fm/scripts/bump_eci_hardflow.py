# -*- coding: utf-8 -*-
"""Scores ECI and HardFlow against the bump2d problem.

Both samplers run in the frame the model was trained in, with the polygons pushed into that
frame by :class:`NormalizedConstraint` rather than by rescaling half-planes by hand. Success
rate is frame-invariant; SWD, MMD and JSD are reported in the normalised frame so their
magnitudes sit on the same scale as the numbers already published for the GMM problem.

    python -m constrained_fm.scripts.bump_eci_hardflow
    python -m constrained_fm.scripts.bump_eci_hardflow --methods hardflow --guidance-scale 20
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.registry import pin_baseline_run, summarize
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.inference.constrained_samplers import (DEFAULT_CHUNK, DEFAULT_STEPS,
                                                               sample_eci, sample_hardflow)
from constrained_fm.src.inference.constraint_projection import DEFAULT_MARGIN
from constrained_fm.src.metrics.distributional import compute_jsd, compute_mmd, compute_swd
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.problems.base import NormalizedConstraint
from constrained_fm.src.problems.bump2d import BumpProblem, sample_polygons

BASE_CKPT = "constrained_fm/baselines/bump2d_base_fm/ckpt.pt"
DEFAULT_OUTDIR = "constrained_fm/baselines"
SAVED_SAMPLES = 2000
UNDEFINED_LIKELIHOOD_KEYS = ("nll", "kld")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECI and HardFlow on the bump2d problem.")
    parser.add_argument("--methods", nargs="+", default=["eci", "hardflow"],
                        choices=["eci", "hardflow"])
    parser.add_argument("--ckpt", default=BASE_CKPT)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--correction-loops", type=int, default=1)
    parser.add_argument("--projection-iters", type=int, default=16)
    parser.add_argument("--projection-damping", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=100.0)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--num-polys", type=int, default=100)
    parser.add_argument("--num-x0", type=int, default=10000)
    parser.add_argument("--pool-size", type=int, default=200000)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser


def load_base_model(args, device: torch.device) -> UnconstrainedFM:
    path = Path(args.ckpt)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/run_bump_fm.sh first")
    model = UnconstrainedFM(input_dim=2, time_dim=args.time_dim, hidden_dim=args.hidden_dim,
                            num_blocks=args.num_blocks).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def build_benchmark(args, problem: BumpProblem, device: torch.device):
    """Pool, constraints and shared initial noise, each from its own seed.

    Seeding the three draws separately keeps the constraint set identical when only the
    number of initial points changes, which would otherwise shift the global RNG.
    """
    set_seed(args.seed)
    target = problem.target()
    pool = target.sample(args.pool_size, device=device)
    constraints, mass = sample_polygons(args.num_polys, pool, domain=problem.domain,
                                        min_mass=problem.min_mass, max_mass=problem.max_mass)

    set_seed(args.seed + 1)
    x0 = torch.randn(args.num_x0, problem.dim, device=device)
    return pool, constraints, mass, x0


def generate(method: str, model, x0: torch.Tensor, constraints, normalizer, args) -> torch.Tensor:
    out = torch.empty(len(constraints), x0.shape[0], x0.shape[1], device=x0.device)
    for i, constraint in enumerate(tqdm(constraints, desc=f"{method} sampling")):
        wrapped = NormalizedConstraint(constraint, normalizer)
        if method == "eci":
            out[i] = sample_eci(model, x0, wrapped, steps=args.steps,
                                correction_loops=args.correction_loops, margin=args.margin,
                                projection_iters=args.projection_iters,
                                projection_damping=args.projection_damping,
                                chunk_size=args.chunk_size)
        else:
            out[i] = sample_hardflow(model, x0, wrapped, steps=args.steps,
                                     guidance_scale=args.guidance_scale, margin=args.margin,
                                     chunk_size=args.chunk_size)
    return out


def score(method: str, samples: torch.Tensor, pool_u: torch.Tensor, constraints, normalizer,
          mass: torch.Tensor, args) -> dict:
    per_shape: dict[str, list[float]] = {k: [] for k in ("swd", "mmd", "jsd", "success_rate")}
    for i, constraint in enumerate(tqdm(constraints, desc=f"{method} scoring")):
        wrapped = NormalizedConstraint(constraint, normalizer)
        truth = pool_u[wrapped.is_feasible(pool_u)]
        per_shape["swd"].append(compute_swd(samples[i], truth))
        per_shape["mmd"].append(compute_mmd(samples[i], truth))
        per_shape["jsd"].append(compute_jsd(samples[i], truth))
        per_shape["success_rate"].append(wrapped.success_rate(samples[i]))

    per_shape["mass"] = mass.tolist()
    for key in UNDEFINED_LIKELIHOOD_KEYS:
        per_shape[key] = [float("nan")] * len(constraints)

    return {
        "method": method,
        "problem": "bump2d",
        "model": "UnconstrainedFM + inference-time constraint",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "likelihood": "undefined -- trajectories are altered outside the probability-flow ODE",
        "frame": "metrics computed in the normalised frame; success rate is frame-invariant",
        "sampling": {"steps": args.steps, "margin": args.margin,
                     "correction_loops": args.correction_loops if method == "eci" else None,
                     "projection_iters": args.projection_iters if method == "eci" else None,
                     "projection_damping": args.projection_damping if method == "eci" else None,
                     "guidance_scale": args.guidance_scale if method == "hardflow" else None},
        "eval": {"num_polys": args.num_polys, "num_x0": args.num_x0,
                 "pool_size": args.pool_size},
        "per_shape": per_shape,
        "summary": summarize(per_shape),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()

    problem = BumpProblem()
    normalizer = problem.normalizer().to(device)
    model = load_base_model(args, device)
    print(f"device {device} | base model {args.ckpt}")

    pool, constraints, mass, x0 = build_benchmark(args, problem, device)
    pool_u = normalizer.forward(pool)
    print(f"{len(constraints)} polygons | mass in [{mass.min():.3f}, {mass.max():.3f}] "
          f"| {x0.shape[0]} initial points")

    for method in args.methods:
        samples = generate(method, model, x0, constraints, normalizer, args)
        record = score(method, samples, pool_u, constraints, normalizer, mass, args)

        out = Path(args.outdir) / f"bump2d_{method}"
        run_id = pin_baseline_run(out, f"bump2d_{method}", args, extra={"base_ckpt": args.ckpt})
        record["run_id"] = run_id
        (out / "metrics.json").write_text(json.dumps(record, indent=2))

        physical = normalizer.inverse(samples[:, :SAVED_SAMPLES])
        artifacts.save_arrays(out, samples=physical.cpu().numpy(),
                              mass=mass.numpy().astype(np.float64))
        artifacts.write_manifest(out, run_id=run_id, method=f"bump2d_{method}")

        summary = record["summary"]
        print(f"\n### bump2d {method} ({run_id})")
        for key in ("success_rate", "swd", "mmd", "jsd"):
            print(f"  {key:14s} median {summary[f'{key}_median']:.4f} | "
                  f"mean {summary[f'{key}_mean']:.4f} | p5 {summary[f'{key}_p5']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
