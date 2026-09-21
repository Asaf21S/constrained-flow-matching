# -*- coding: utf-8 -*-
"""Stage 2 of the bench1k pipeline: score every method on a slice of one problem's benchmark.

Methods: rejection sampling (the achievable noise floor), the amortized conditional model,
and the two inference-time hacks ECI and HardFlow. All of them integrate from the same frozen
start points and are scored against the same fixed reference pool, so the only difference
between two rows is the method.

A thousand constraints times four methods does not fit in one job, so the work is sliced by
constraint index and each slice writes its own shard file. Everything a metric depends on is
keyed off the *global* constraint index -- the rejection sampling seed, and the RNG the SWD
projections and the MMD subsample draw from -- while the reference pool and the start points
come from the frozen benchmark. A constraint therefore scores identically no matter which
slice it lands in, and the shards can be re-cut freely.

    sbatch scripts/run_bench1k_eval.sh --problem kinematics6d
    python -m constrained_fm.scripts.eval_bench1k --problem bump2d --start-idx 0 --end-idx 50
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from constrained_fm.src.consts import KIN_MMD_GAMMA
from constrained_fm.src.datasets.benchmark_1k import (PROBLEM_NAMES, constraints_from,
                                                      load_benchmark_1k)
from constrained_fm.src.experiment.registry import pin_baseline_run, summarize
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.inference.constrained_samplers import (DEFAULT_CHUNK, DEFAULT_STEPS,
                                                               sample_eci, sample_hardflow)
from constrained_fm.src.metrics.distributional import (compute_jsd, compute_jsd_1d, compute_mmd,
                                                       compute_swd)
from constrained_fm.src.metrics.likelihood import conditional_nll
from constrained_fm.src.models.constrained_mass import MassWindowConstrainedFM
from constrained_fm.src.models.unconstrained import UnconstrainedFM
from constrained_fm.src.problems.base import NormalizedConstraint
from constrained_fm.src.problems.bump2d import BumpProblem
from constrained_fm.src.problems.kinematics6d import KinematicsProblem

# The amortized method differs by problem: bump2d encodes its polygon through a SIREN latent,
# kinematics6d feeds the shell parameters straight in. Both are "the model that was trained
# on the constraint", as opposed to the two that bolt it on at inference time.
PROBLEM_METHODS = {
    "bump2d": ("gt", "eci", "hardflow"),
    "kinematics6d": ("gt", "eci", "hardflow", "explicit"),
}
BASE_CKPT = {
    "bump2d": "constrained_fm/baselines/bump2d_base_fm/ckpt.pt",
    "kinematics6d": "constrained_fm/baselines/kin6d_base_fm/ckpt.pt",
}
EXPLICIT_CKPT = "constrained_fm/baselines/kin6d_explicit/ckpt.pt"
DEFAULT_OUTDIR = "constrained_fm/baselines/bench1k"

METRIC_KEYS = ("success_rate", "swd", "mmd", "jsd", "nll", "kld", "in_support_fraction",
               "swd_noise_floor", "mmd_noise_floor", "jsd_noise_floor",
               "truth_count", "compared_count")
# ECI and HardFlow alter the trajectory outside the probability-flow ODE, so no density of
# theirs exists and their NLL/KLD stay NaN; rejection sampling's KLD is identically zero and
# is left NaN rather than reported as a result.
NLL_METHODS = {"bump2d": (), "kinematics6d": ("explicit",)}
# The divergence trace costs two backward passes per step per point, so the NLL is a mean
# over a subset of the same reference points the distributional metrics already use.
NLL_POINTS = 4000
NLL_STEP_SIZE = 0.05

SWD_PROJECTIONS = 200
# The 2D problems keep gamma = 1.0 so their numbers stay comparable with the polynomial
# benchmark; in the normalised 6D frame that kernel is degenerate and the pinned median
# heuristic is used instead.
MMD_GAMMA = {"bump2d": 1.0, "kinematics6d": KIN_MMD_GAMMA}
# The thinnest kinematics shell is 0.004 wide in units of C, so the 2D default of 1e-3 would
# land projected points a quarter of the way into it.
MARGIN = {"bump2d": 1e-3, "kinematics6d": 1e-4}
# A size-matched floor needs twice the generated sample count inside the *tightest*
# constraint, so the pool is sized as 2 * num_x0 / min_mass: 0.02 for a polygon, 0.01 for a
# shell. Undersizing it silently shrinks the comparison for exactly the constraints that
# matter most.
POOL_SIZE = {"bump2d": 1_000_000, "kinematics6d": 3_000_000}
PROJECTION_ITERS = {"bump2d": 16, "kinematics6d": 32}

# Independent RNG streams, so seeding one never shifts another.
REFERENCE_POOL_SEED = 20_000
GT_SAMPLE_SEED = 30_000
METRIC_SEED = 50_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score one slice of a bench1k benchmark.")
    parser.add_argument("--problem", default="kinematics6d", choices=list(PROBLEM_NAMES))
    parser.add_argument("--start-idx", "--start_idx", dest="start_idx", type=int, default=0,
                        help="first constraint index of this shard, inclusive")
    parser.add_argument("--end-idx", "--end_idx", dest="end_idx", type=int, default=None,
                        help="last constraint index of this shard, exclusive (default: all)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="default: every method the problem supports")

    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--explicit-ckpt", default=EXPLICIT_CKPT)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--num-frequencies", type=int, default=16)

    parser.add_argument("--num-x0", type=int, default=None,
                        help="samples per constraint (default: all of the benchmark's x0)")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="reference pool the metrics compare against")
    parser.add_argument("--gt-batch", type=int, default=500000,
                        help="draw size per rejection round")
    parser.add_argument("--step-size", type=float, default=0.05)

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="ECI/HardFlow steps")
    parser.add_argument("--correction-loops", type=int, default=1)
    parser.add_argument("--projection-iters", type=int, default=None)
    parser.add_argument("--projection-damping", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=100.0)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser


def resolve_defaults(args) -> None:
    """Fills the per-problem defaults, so one flag set covers both benchmarks."""
    if args.methods is None:
        args.methods = list(PROBLEM_METHODS[args.problem])
    unknown = set(args.methods) - set(PROBLEM_METHODS[args.problem])
    if unknown:
        raise ValueError(f"{args.problem} does not support {sorted(unknown)}; "
                         f"available: {PROBLEM_METHODS[args.problem]}")
    if args.base_ckpt is None:
        args.base_ckpt = BASE_CKPT[args.problem]
    if args.margin is None:
        args.margin = MARGIN[args.problem]
    if args.pool_size is None:
        args.pool_size = POOL_SIZE[args.problem]
    if args.projection_iters is None:
        args.projection_iters = PROJECTION_ITERS[args.problem]


def seed_metric_rng(index: int) -> None:
    """Pins the MMD subsample to the constraint, not to shard order.

    POT's SWD does not read this state -- it constructs its own generator -- so the same seed
    is handed to :func:`compute_swd` explicitly.
    """
    seed = METRIC_SEED + index
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


# --- sampling -----------------------------------------------------------------------------


def reference_pool(problem, args, device: torch.device) -> torch.Tensor:
    """One fixed pool for the whole sweep, drawn under a forked RNG.

    Shared across shards by construction: the seed is a constant, not a function of the
    slice, so two shards score against numerically identical truth.
    """
    with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
        torch.manual_seed(REFERENCE_POOL_SEED)
        return problem.target().sample(args.pool_size, device=device)


def rejection_sample(constraint, problem, num_samples: int, index: int, args,
                     device: torch.device) -> torch.Tensor:
    """Target draws kept where the constraint holds, refilled until ``num_samples``.

    Drawn under a forked RNG seeded by the global constraint index, so these points are
    reproducible and independent of the reference pool the metrics score against.
    """
    target = problem.target()
    batch = max(args.gt_batch, num_samples * 2)
    kept, collected, attempt = [], 0, 0

    while collected < num_samples:
        with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
            torch.manual_seed(GT_SAMPLE_SEED + index * 100 + attempt)
            pool = target.sample(batch, device=device)
        inside = pool[constraint.is_feasible(pool)]
        kept.append(inside)
        collected += int(inside.shape[0])
        attempt += 1
        if attempt > 200:
            raise RuntimeError(f"constraint {index} is too small to reach {num_samples} samples")

    return torch.cat(kept, dim=0)[:num_samples]


def generate(method: str, models: dict, constraint, index: int, x0: torch.Tensor, problem,
             normalizer, args, device: torch.device) -> torch.Tensor:
    """Samples for one constraint, always returned in the normalised frame."""
    if method == "gt":
        physical = rejection_sample(constraint, problem, x0.shape[0], index, args, device)
        return normalizer.forward(physical)

    if method == "explicit":
        return models["explicit"].sample(num_points=x0.shape[0],
                                         params=constraint.params.to(device),
                                         step_size=args.step_size, device=device, x_init=x0)

    wrapped = NormalizedConstraint(constraint, normalizer)
    if method == "eci":
        return sample_eci(models["base"], x0, wrapped, steps=args.steps,
                          correction_loops=args.correction_loops, margin=args.margin,
                          projection_iters=args.projection_iters,
                          projection_damping=args.projection_damping,
                          chunk_size=args.chunk_size)
    return sample_hardflow(models["base"], x0, wrapped, steps=args.steps,
                           guidance_scale=args.guidance_scale, margin=args.margin,
                           chunk_size=args.chunk_size)


# --- scoring ------------------------------------------------------------------------------


def jensen_shannon(problem_name: str, gen_u: torch.Tensor, truth_u: torch.Tensor,
                   normalizer, target) -> float:
    """Planar KDE on the plane, and a binned invariant-mass histogram in six dimensions."""
    if problem_name == "bump2d":
        return compute_jsd(gen_u, truth_u)
    return compute_jsd_1d(target.invariant_mass(normalizer.inverse(gen_u)),
                          target.invariant_mass(normalizer.inverse(truth_u)))


def in_support_fraction(target, normalizer, samples: torch.Tensor) -> float:
    """Fraction of generated points inside the target's hard support, in physical units.

    Quoted beside KLD because the divergence is measured in the reverse direction, at
    ground-truth points: that keeps it finite but blind to mass the model puts where the
    data density is exactly zero, which this number exposes.
    """
    if not hasattr(target, "in_support"):
        return float("nan")
    inside = target.in_support(normalizer.inverse(samples))
    return float(inside.double().mean()) * 100.0


def likelihood_row(method: str, models: dict, constraint, truth_u: torch.Tensor, normalizer,
                   target, mass: float, args, device) -> dict[str, float]:
    """Exact NLL and ``KL(p_true || p_model)`` for the methods that own a density."""
    if method not in NLL_METHODS[args.problem]:
        return {}

    u_true = truth_u[:NLL_POINTS].to(device)
    log_p_true = target.log_prob(normalizer.inverse(u_true).double())
    return conditional_nll(models[method], u_true, log_p_true, mass,
                           float(normalizer.log_det_forward),
                           params=constraint.params.to(device),
                           step_size=NLL_STEP_SIZE, device=device)


def score_one(problem_name: str, samples: torch.Tensor, truth: torch.Tensor, wrapped,
              normalizer, target, gamma: float, seed: int) -> dict[str, float]:
    """One metric row, with a truth-against-truth floor beside every discrepancy.

    The floor is the same statistic with the generator replaced by a second, disjoint draw
    from the conditional truth. Both draws are cut to exactly the same size as the generated
    set and scored against the same reference, because the empirical Wasserstein and JS
    estimators carry an ``O(N^{-1/2})`` bias: a floor computed on more points than the method
    was given measures sample count, not fidelity, and would report exact rejection sampling
    as three times its own floor.

    A method sitting on its floor is indistinguishable from exact conditional sampling at
    this sample count, and nothing below it is meaningful. The measurement and its floor are
    given the same SWD projection directions, so their ratio does not carry the projection
    noise twice.
    """
    n = min(samples.shape[0], truth.shape[0] // 2)
    generated, reference, second_draw = samples[:n], truth[:n], truth[n:2 * n]

    row = {
        "success_rate": wrapped.success_rate(samples),
        "in_support_fraction": in_support_fraction(target, normalizer, samples),
        "swd": compute_swd(generated, reference, num_projections=SWD_PROJECTIONS, seed=seed),
        "mmd": compute_mmd(generated, reference, gamma=gamma),
        "jsd": jensen_shannon(problem_name, generated, reference, normalizer, target),
        "swd_noise_floor": compute_swd(second_draw, reference,
                                       num_projections=SWD_PROJECTIONS, seed=seed),
        "mmd_noise_floor": compute_mmd(second_draw, reference, gamma=gamma),
        "jsd_noise_floor": jensen_shannon(problem_name, second_draw, reference, normalizer,
                                          target),
        "truth_count": float(truth.shape[0]),
        "compared_count": float(n),
    }
    return {key: row.get(key, float("nan")) for key in METRIC_KEYS}


# --- models -------------------------------------------------------------------------------


def load_models(args, problem, device: torch.device) -> dict:
    """Loads only the checkpoints the requested methods actually need."""
    models: dict = {}

    if {"eci", "hardflow"} & set(args.methods):
        path = Path(args.base_ckpt)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- train the base flow matcher first")
        model = UnconstrainedFM(input_dim=problem.dim, time_dim=args.time_dim,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        models["base"] = model.eval()

    if "explicit" in args.methods:
        path = Path(args.explicit_ckpt)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run scripts/run_kin_constrained.sh")
        normalizer = problem.normalizer()
        model = MassWindowConstrainedFM(
            input_dim=problem.dim, time_dim=args.time_dim, hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks, num_frequencies=args.num_frequencies,
            frame_mean=normalizer.mean, frame_std=normalizer.std,
            mass_scale=problem.target().mass_scale()).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        models["explicit"] = model.eval()

    return models


# --- shards -------------------------------------------------------------------------------


def shard_path(out: Path, method: str, start: int, end: int) -> Path:
    return out / "shards" / f"{method}__{start:05d}_{end:05d}.json"


# Which slice a job happens to run is not part of what the result is.
_SHARD_ARGS = ("start_idx", "end_idx", "methods")


def pin_once(out: Path, args, digest: str) -> str:
    """One run id for the whole sweep: every shard fingerprints the same settings.

    Reuses the id already on disk so a running array never rewrites provenance underneath
    its siblings.
    """
    provenance = out / "provenance.json"
    if provenance.exists():
        return json.loads(provenance.read_text())["run_id"]
    settings = {k: v for k, v in vars(args).items() if k not in _SHARD_ARGS}
    return pin_baseline_run(out, f"bench1k_{args.problem}", settings,
                            extra={"benchmark_digest": digest})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_defaults(args)
    device = resolve_device()

    problem = BumpProblem() if args.problem == "bump2d" else KinematicsProblem()
    normalizer = problem.normalizer().to(device)
    target = problem.target()

    benchmark = load_benchmark_1k(args.problem, device=device)
    constraints = constraints_from(benchmark, problem, device=device)
    end_idx = benchmark["mass"].numel() if args.end_idx is None else args.end_idx
    indices = list(range(args.start_idx, min(end_idx, benchmark["mass"].numel())))
    if not indices:
        raise ValueError(f"empty shard [{args.start_idx}, {end_idx})")

    x0 = benchmark["x0"][:args.num_x0] if args.num_x0 else benchmark["x0"]
    pool_u = normalizer.forward(reference_pool(problem, args, device))
    models = load_models(args, problem, device)

    out = Path(args.outdir) / args.problem
    run_id = pin_once(out, args, benchmark["digest"])
    (out / "shards").mkdir(parents=True, exist_ok=True)
    print(f"device {device} | {args.problem} | digest {benchmark['digest']} | run {run_id}\n"
          f"constraints [{indices[0]}, {indices[-1]}] | {x0.shape[0]} samples each "
          f"| methods {args.methods}", flush=True)

    for method in args.methods:
        per_shape: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}

        for index in tqdm(indices, desc=f"{method} [{args.start_idx}, {end_idx})"):
            constraint = constraints[index]
            wrapped = NormalizedConstraint(constraint, normalizer)

            seed_metric_rng(index)
            samples = generate(method, models, constraint, index, x0, problem, normalizer,
                               args, device)
            truth = pool_u[wrapped.is_feasible(pool_u)]

            seed_metric_rng(index)
            row = score_one(args.problem, samples.detach(), truth, wrapped, normalizer,
                            target, MMD_GAMMA[args.problem], METRIC_SEED + index)
            row.update(likelihood_row(method, models, constraint, truth, normalizer, target,
                                      float(benchmark["mass"][index]), args, device))
            for key in METRIC_KEYS:
                per_shape[key].append(row[key])

        per_shape["mass"] = benchmark["mass"][indices].tolist()

        record = {
            "method": method,
            "problem": args.problem,
            "run_id": run_id,
            "benchmark_digest": benchmark["digest"],
            "start_idx": indices[0],
            "end_idx": indices[-1] + 1,
            "indices": indices,
            "scored_at": datetime.now().isoformat(timespec="seconds"),
            "frame": "metrics in the normalised frame; success rate is frame-invariant",
            "eval": {"num_x0": x0.shape[0], "pool_size": args.pool_size,
                     "swd_projections": SWD_PROJECTIONS,
                     "mmd_gamma": MMD_GAMMA[args.problem], "step_size": args.step_size,
                     "steps": args.steps, "margin": args.margin,
                     "projection_iters": args.projection_iters,
                     "projection_damping": args.projection_damping,
                     "guidance_scale": args.guidance_scale},
            "per_shape": per_shape,
            "summary": summarize(per_shape),
        }
        path = shard_path(out, method, indices[0], indices[-1] + 1)
        path.write_text(json.dumps(record, indent=2))

        summary = record["summary"]
        print(f"\n### {method} -> {path}")
        for key in ("success_rate", "swd", "swd_noise_floor", "mmd", "mmd_noise_floor",
                    "jsd", "jsd_noise_floor"):
            if f"{key}_median" in summary:
                print(f"  {key:16s} median {summary[f'{key}_median']:.4f} | "
                      f"mean {summary[f'{key}_mean']:.4f} | p5 {summary[f'{key}_p5']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
