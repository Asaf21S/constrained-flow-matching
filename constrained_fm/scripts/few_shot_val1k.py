# -*- coding: utf-8 -*-
"""Few-shot unconstrained baseline over the full 1000-constraint validation set.

For every constraint this rejection-samples exactly N valid GMM points, trains an
*unconditional* flow matcher from scratch on just those points, and scores it against the
same reference pool the other five v1k methods were scored against. One model per
constraint, so a full pass trains 1000 models.

Everything a metric depends on is keyed off the *global* constraint index -- the training
seed, the rejection sampling, the early-stopping draw, the NLL subset and the RNG the SWD
projections read -- so a constraint scores identically no matter which slice it lands in.

Shards are written into the v1k shard directory under a method name, which is all
``merge_val1k`` and ``plot_val1k`` need to pick the baseline up. Each shot budget must carry
its own method name, since merge keys on it and two budgets sharing one name would be read
as a single method covering each constraint twice.

Per-constraint results are checkpointed to ``results/`` before the shard file is written, so
a preempted job resumes instead of retraining what it already finished.

    sbatch scripts/run_val1k_fewshot.sh
    N=100 METHOD=fewshot_N100 sbatch scripts/run_val1k_fewshot.sh
    python -m constrained_fm.scripts.few_shot_val1k --start-idx 0 --end-idx 50 --num-points 2000
    python -m constrained_fm.scripts.few_shot_val1k --start-idx 0 --end-idx 50 --assemble-only
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from tqdm import tqdm

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.datasets.validation_v1k import seeded_gmm_pool
from constrained_fm.src.experiment.registry import pin_baseline_run, summarize
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.geometry.polynomials import compute_poly_features
from constrained_fm.src.inference.evaluator import evaluate_single_configuration
from constrained_fm.src.metrics.eval_points import load_nll_eval_set_v1k
from constrained_fm.src.metrics.likelihood import constraint_nll
# Shared with the 100-constraint sweep so both baselines train by identical rules.
from constrained_fm.scripts.few_shot_unconstrained import rejection_sample, train_few_shot

METHOD = "fewshot"
METRIC_KEYS = ("success_rate", "swd", "mmd", "jsd", "nll", "kld")

DEFAULT_OUTDIR = "constrained_fm/baselines/val1k"
DEFAULT_WORKDIR = "constrained_fm/baselines/few_shot_val1k"

# Must match eval_val1k, or the metrics are measured against a different ground truth.
REFERENCE_POOL_SEED = 20_000
METRIC_SEED = 50_000
TRAIN_SEED = 60_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Few-shot baseline on a slice of the v1k set.")
    parser.add_argument("--start-idx", "--start_idx", dest="start_idx", type=int, default=0,
                        help="first constraint index of this shard, inclusive")
    parser.add_argument("--end-idx", "--end_idx", dest="end_idx", type=int, default=None,
                        help="last constraint index of this shard, exclusive (default: all)")
    parser.add_argument("--num-points", type=int, default=2000,
                        help="shot budget N: valid samples the baseline is allowed to see")

    parser.add_argument("--hidden-dim", type=int, default=1024,
                        help="matches ConstrainedFlowMatcher")
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)

    parser.add_argument("--iterations", type=int, default=20000,
                        help="upper bound; early stopping decides")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-points", type=int, default=10000,
                        help="held-out constraint-satisfying points driving early stopping")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=12,
                        help="evaluations without improvement")

    parser.add_argument("--num-x0", type=int, default=10000, help="samples per constraint")
    parser.add_argument("--gmm-pool-size", type=int, default=100000,
                        help="reference pool the metrics compare against")
    parser.add_argument("--nll-points", type=int, default=5000)
    parser.add_argument("--step-size", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--degree", type=int, default=POLYNOMIAL_DEGREE)
    parser.add_argument("--scale", type=float, default=PLANE_SCALE)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="v1k directory whose shards/ this baseline joins")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR,
                        help="per-constraint results and provenance for this baseline")
    parser.add_argument("--method", default=METHOD,
                        help="method name the shards carry; must be unique per shot budget")
    parser.add_argument("--assemble-only", action="store_true",
                        help="write the shard file from existing per-constraint results")
    parser.add_argument("--save-samples", action="store_true",
                        help="also persist the raw (N, 2) sample tensor per constraint")
    return parser


def seed_metric_rng(index: int) -> None:
    """Pins the SWD projections and the MMD subsample to the constraint, not to shard order."""
    seed = METRIC_SEED + index
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def result_path(workdir: Path, index: int, n_points: int) -> Path:
    return workdir / "results" / f"idx{index:05d}_N{n_points}.json"


def shard_path(out: Path, method: str, start: int, end: int) -> Path:
    return out / "shards" / f"{method}__{start:05d}_{end:05d}.json"


# Which slice a job happens to run is not part of what the result is.
_SHARD_ARGS = ("start_idx", "end_idx", "assemble_only", "save_samples", "outdir")


def pin_once(workdir: Path, args) -> str:
    """One run id per shot budget, reused so concurrent shards agree on provenance."""
    budget_dir = workdir / args.method
    provenance = budget_dir / "provenance.json"
    if provenance.exists():
        return json.loads(provenance.read_text())["run_id"]
    settings = {k: v for k, v in vars(args).items() if k not in _SHARD_ARGS}
    return pin_baseline_run(budget_dir, "few_shot_val1k", settings)


def run_item(index: int, C: torch.Tensor, gmm_pool: torch.Tensor, pool_features,
             nll_points: torch.Tensor, mass: float, args, device) -> tuple[dict, np.ndarray]:
    """Trains one specialist on N valid points and scores it like every other v1k method."""
    set_seed(TRAIN_SEED + args.seed + index)
    started = time.time()

    x_train = rejection_sample(C, args.num_points, args.degree, args.scale, device)
    x_val = rejection_sample(C, args.val_points, args.degree, args.scale, device)

    model, train_info = train_few_shot(x_train, x_val, args, device)
    train_seconds = time.time() - started

    samples = model.sample(num_points=args.num_x0, step_size=args.step_size, device=device)
    if samples.ndim == 3:
        samples = samples[-1]

    seed_metric_rng(index)
    metrics = evaluate_single_configuration(
        samples=samples, x_true_pool=gmm_pool, coeffs=C, degree=args.degree, scale=args.scale,
        x_pow_true=pool_features[0], y_pow_true=pool_features[1], device=device)

    # The ODE this model was sampled from is intact, so its density is defined. No
    # conditioning is passed: the constraint reached the model only through its training set.
    if args.nll_points > 0:
        metrics.update(constraint_nll(model, nll_points.to(device), mass,
                                      num_points=args.nll_points, step_size=args.step_size,
                                      subset_seed=index, device=device))

    record = {key: float(metrics.get(key, float("nan"))) for key in METRIC_KEYS}
    record.update({"index": index, "n_points": args.num_points, "mass": mass,
                   "train_seconds": train_seconds,
                   "best_iteration": train_info["best_iteration"],
                   "stopped_at": train_info["stopped_at"]})
    return record, samples.detach().cpu().numpy().astype(np.float32)


def assemble_shard(out: Path, workdir: Path, indices: list[int], run_id: str, digest: str,
                   args) -> Path:
    """Scatters the per-constraint files of this range into one shard merge_val1k accepts."""
    records = []
    missing = []
    for index in indices:
        path = result_path(workdir, index, args.num_points)
        if path.exists():
            records.append(json.loads(path.read_text()))
        else:
            missing.append(index)
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(indices)} constraints have no result "
            f"(first few: {missing[:10]}). Re-run this shard to fill them in.")

    per_shape = {key: [record[key] for record in records] for key in METRIC_KEYS}
    per_shape = {key: values for key, values in per_shape.items()
                 if any(np.isfinite(values))}
    per_shape["mass"] = [record["mass"] for record in records]

    payload = {
        "run_id": run_id,
        "method": args.method,
        "validation_set": "v1k",
        "poly_digest": digest,
        "start_idx": indices[0],
        "end_idx": indices[-1] + 1,
        "indices": indices,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "eval": {"n_points": args.num_points, "num_x0": args.num_x0,
                 "gmm_pool_size": args.gmm_pool_size, "step_size": args.step_size,
                 "nll_points": args.nll_points, "val_points": args.val_points,
                 "reference_pool_seed": REFERENCE_POOL_SEED,
                 "median_train_seconds": float(np.median([r["train_seconds"] for r in records]))},
        "per_shape": per_shape,
        "summary": summarize(per_shape),
    }
    path = shard_path(out, args.method, indices[0], indices[-1] + 1)
    path.write_text(json.dumps(payload, indent=2))
    return path


def main(argv: list[str] | None = None) -> int:
    from constrained_fm.src.datasets.validation_v1k import get_validation_set_v1k

    args = build_parser().parse_args(argv)
    device = resolve_device()
    out = Path(args.outdir)
    workdir = Path(args.workdir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    (workdir / "results").mkdir(parents=True, exist_ok=True)
    run_id = pin_once(workdir, args)

    val_set = get_validation_set_v1k(device=device)
    total = val_set["polynomials"].shape[0]
    start = max(0, args.start_idx)
    end = total if args.end_idx is None else min(args.end_idx, total)
    if start >= end:
        raise ValueError(f"empty shard: start {start} >= end {end} (set holds {total})")
    indices = list(range(start, end))

    if args.assemble_only:
        path = assemble_shard(out, workdir, indices, run_id, val_set["poly_digest"], args)
        print(f"wrote {path}")
        return 0

    polys = val_set["polynomials"].to(device)

    print(f"run_id {run_id} | device {device} | constraints [{start}, {end}) of {total}")
    print(f"digest {val_set['poly_digest']} | N {args.num_points} | "
          f"model: hidden {args.hidden_dim}, {args.num_blocks} blocks", flush=True)

    # One fixed reference pool for every shard and every method: the SWD/MMD/JSD of two
    # methods are only comparable if they were measured against the same ground truth.
    gmm_pool = seeded_gmm_pool(args.gmm_pool_size, REFERENCE_POOL_SEED, device=device)
    pool_features = compute_poly_features(gmm_pool, degree=args.degree, scale=args.scale)
    nll_set = load_nll_eval_set_v1k(num_points=args.nll_points, degree=args.degree,
                                    scale=args.scale, device="cpu")

    for index in tqdm(indices, desc=f"{args.method} N={args.num_points}"):
        path = result_path(workdir, index, args.num_points)
        if path.exists():
            continue

        # One pathological constraint must not cost the whole shard; it is retried on rerun.
        try:
            record, samples = run_item(index, polys[index], gmm_pool, pool_features,
                                       nll_set["points"][index], float(nll_set["mass"][index]),
                                       args, device)
        except Exception as exc:
            print(f"constraint {index}: FAILED ({type(exc).__name__}: {exc})", flush=True)
            continue

        path.write_text(json.dumps({"run_id": run_id, **record}, indent=2))
        if args.save_samples:
            (workdir / "samples").mkdir(parents=True, exist_ok=True)
            np.save(workdir / "samples" / f"idx{index:05d}_N{args.num_points}.npy", samples)

        print(f"[{index}] AR {record['success_rate']:6.2f}% | SWD {record['swd']:.4f} | "
              f"KLD {record['kld']:.4f} | stopped {record['stopped_at']} | "
              f"{record['train_seconds']:.0f}s", flush=True)

    path = assemble_shard(out, workdir, indices, run_id, val_set["poly_digest"], args)
    print(f"\nshard [{start}, {end}) finished. wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
