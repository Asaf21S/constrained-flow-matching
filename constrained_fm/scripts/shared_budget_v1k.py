# -*- coding: utf-8 -*-
"""Functa vs. fine-tuned few-shot when both methods see the same N points, on the v1k set.

For every constraint, one fixed draw of GMM points is the shared information budget; budget
N is its first N points, so every budget is a prefix of the next and the number of points
inside the constraint never decreases with N.

* Functa labels all N points with tanh(P), runs the frozen SIREN's CAVIA extraction on them
  and samples the frozen latent-conditioned flow matcher.
* Few-shot fine-tunes the base unconditional flow matcher (the ECI/HardFlow base) on the
  points with P(x) <= 0 only. A constraint with none inside is not trained and scores NaN.

Both are scored against the v1k reference pool with the v1k metric seeds, so every number is
comparable with the main table. Few-shot keeps the 10k held-out early-stopping set of the
existing sweeps, which favours the baseline.

Shards land in ``<outdir>/shards`` under ``functa_N<N>`` / ``fewshot_ft_N<N>``, which
``merge_val1k --outdir`` stitches. Few-shot results are checkpointed per constraint, so a
preempted task resumes.

    sbatch scripts/run_shared_budget.sh
    python -m constrained_fm.scripts.shared_budget_v1k --start-idx 0 --end-idx 50 --num-points 50
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from constrained_fm.scripts.eval_query_budget import FUNCTA_RUN_ID, as_batched_samples
from constrained_fm.scripts.few_shot_unconstrained import rejection_sample, train_few_shot
from constrained_fm.scripts.few_shot_val1k import (BASE_CKPT, FINETUNE_EVAL_EVERY, FINETUNE_LR,
                                                   METRIC_KEYS, REFERENCE_POOL_SEED, TRAIN_SEED,
                                                   load_base_state, seed_metric_rng)
from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.datasets.validation_v1k import seeded_gmm_pool
from constrained_fm.src.experiment.registry import load_config, pin_baseline_run, summarize
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_siren,
                                                   resolve_device, set_seed)
from constrained_fm.src.geometry.polynomials import (compute_poly_features,
                                                     compute_poly_features_batched,
                                                     evaluate_poly_batched)
from constrained_fm.src.inference.evaluator import (evaluate_single_configuration,
                                                    run_evaluation_inference)
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.metrics.eval_points import load_nll_eval_set_v1k
from constrained_fm.src.metrics.functa_fidelity import true_region_mask
from constrained_fm.src.metrics.likelihood import constraint_nll

FUNCTA = "functa"
FEWSHOT = "fewshot_ft"
METHODS = (FUNCTA, FEWSHOT)
DEFAULT_N_VALUES = [50, 100, 300, 500, 1000, 2000]
DEFAULT_OUTDIR = "constrained_fm/baselines/shared_budget_v1k"

# Independent of the reference-pool, query-point, metric and training streams.
SHARED_POINT_SEED = 70_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared-N Functa vs few-shot on a v1k slice.")
    parser.add_argument("--start-idx", "--start_idx", dest="start_idx", type=int, default=0,
                        help="first constraint index of this shard, inclusive")
    parser.add_argument("--end-idx", "--end_idx", dest="end_idx", type=int, default=None,
                        help="last constraint index of this shard, exclusive (default: all)")
    parser.add_argument("--num-points", type=int, nargs="+", default=DEFAULT_N_VALUES,
                        help="budgets N; each is a prefix of the --pool-points draw")
    parser.add_argument("--pool-points", type=int, default=max(DEFAULT_N_VALUES),
                        help="size of the per-constraint draw every budget is a prefix of")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))

    parser.add_argument("--functa-run-id", default=FUNCTA_RUN_ID)
    parser.add_argument("--extraction-chunk", type=int, default=128)

    parser.add_argument("--base-ckpt", default=BASE_CKPT)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20000,
                        help="upper bound; early stopping decides")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=FINETUNE_LR)
    parser.add_argument("--val-points", type=int, default=10000,
                        help="held-out constraint-satisfying points driving early stopping")
    parser.add_argument("--eval-every", type=int, default=FINETUNE_EVAL_EVERY)
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
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser


def method_name(method: str, n_points: int) -> str:
    return f"{method}_N{n_points}"


def shard_path(out: Path, method: str, n_points: int, start: int, end: int) -> Path:
    return out / "shards" / f"{method_name(method, n_points)}__{start:05d}_{end:05d}.json"


def result_path(out: Path, index: int, n_points: int) -> Path:
    return out / "results" / FEWSHOT / f"idx{index:05d}_N{n_points}.json"


# Which slice or budget a task runs is not part of what the sweep is.
_SHARD_ARGS = ("start_idx", "end_idx", "num_points", "methods")


def pin_once(out: Path, args) -> str:
    """One run id for the whole sweep, reused from disk so array tasks agree on it."""
    provenance = out / "provenance.json"
    if provenance.exists():
        return json.loads(provenance.read_text())["run_id"]
    settings = {k: v for k, v in vars(args).items() if k not in _SHARD_ARGS}
    return pin_baseline_run(out, "shared_budget_v1k", settings)


def shared_points(index: int, pool_points: int, device: torch.device) -> torch.Tensor:
    """The (pool_points, 2) GMM draw whose prefixes are every budget of one constraint."""
    return seeded_gmm_pool(pool_points, SHARED_POINT_SEED + index, device=device)


def inside_counts(polys: torch.Tensor, points: torch.Tensor, args) -> list[int]:
    return [int(true_region_mask(polys[j], points[j], degree=args.degree,
                                 scale=args.scale).sum()) for j in range(polys.shape[0])]


def functa_latents(siren, cfg, polys: torch.Tensor, points: torch.Tensor,
                   chunk_size: int) -> torch.Tensor:
    """CAVIA extraction from the shared points, labelled with tanh(P) as at meta-training.

    ``points`` is (B, N, 2) raw-scale; it is clamped into the SIREN's domain like every
    GMM-drawn query point.
    """
    X_raw = points.clamp(-cfg.scale, cfg.scale)
    z_chunks = []
    for start in range(0, X_raw.shape[0], chunk_size):
        X = X_raw[start:start + chunk_size]
        x_pow, y_pow = compute_poly_features_batched(X, degree=cfg.degree, scale=cfg.scale)
        Y = torch.tanh(evaluate_poly_batched(x_pow, y_pow, polys[start:start + chunk_size]))
        z_chunk, _ = extract_latents_batched(siren, X / cfg.scale, Y, lr=cfg.extraction.lr,
                                             steps=cfg.extraction.steps)
        z_chunks.append(z_chunk)
    return torch.cat(z_chunks, dim=0)


def run_functa(models: dict, polys: torch.Tensor, indices: list[int], points: torch.Tensor,
               x0: torch.Tensor, gmm_pool: torch.Tensor, pool_features, nll_set: dict | None,
               args, device: torch.device) -> dict[str, list[float]]:
    """One metric row per constraint for the shard at one budget."""
    model = models["functa"]
    z = functa_latents(models["siren"], models["functa_cfg"], polys, points,
                       args.extraction_chunk)
    raw = run_evaluation_inference(model, x0, z=z, step_size=args.step_size, device=device)
    samples = as_batched_samples(raw, polys.shape[0], device)

    scores: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    for j, index in enumerate(tqdm(indices, desc="functa scoring", leave=False)):
        seed_metric_rng(index)
        metrics = evaluate_single_configuration(
            samples=samples[j], x_true_pool=gmm_pool, coeffs=polys[j], degree=args.degree,
            scale=args.scale, x_pow_true=pool_features[0], y_pow_true=pool_features[1],
            model=model if nll_set is not None else None, z=z[j],
            nll_points=args.nll_points if nll_set is not None else 0,
            nll_step_size=args.step_size,
            nll_eval_points=None if nll_set is None else nll_set["points"][index].to(device),
            nll_mass=None if nll_set is None else float(nll_set["mass"][index]),
            device=device)
        for key in METRIC_KEYS:
            scores[key].append(float(metrics.get(key, float("nan"))))
    return scores


def run_fewshot_item(index: int, C: torch.Tensor, x_train: torch.Tensor,
                     base_state: dict, gmm_pool: torch.Tensor, pool_features,
                     nll_set: dict | None, args, device: torch.device) -> dict:
    """Fine-tunes the base model on the inside points and scores it like every v1k method."""
    nan = float("nan")
    if x_train.shape[0] == 0:
        return {**{key: nan for key in METRIC_KEYS}, "train_seconds": nan,
                "best_iteration": nan, "stopped_at": nan, "best_val_loss": nan,
                "initial_val_loss": nan}

    # Same seed as few_shot_val1k, so the held-out set matches the existing sweeps.
    set_seed(TRAIN_SEED + args.seed + index)
    started = time.time()
    x_val = rejection_sample(C, args.val_points, args.degree, args.scale, device)
    model, train_info = train_few_shot(x_train, x_val, args, device, init_state=base_state)
    train_seconds = time.time() - started

    samples = model.sample(num_points=args.num_x0, step_size=args.step_size, device=device)
    if samples.ndim == 3:
        samples = samples[-1]

    seed_metric_rng(index)
    metrics = evaluate_single_configuration(
        samples=samples, x_true_pool=gmm_pool, coeffs=C, degree=args.degree, scale=args.scale,
        x_pow_true=pool_features[0], y_pow_true=pool_features[1], device=device)
    if nll_set is not None:
        metrics.update(constraint_nll(model, nll_set["points"][index].to(device),
                                      float(nll_set["mass"][index]),
                                      num_points=args.nll_points, step_size=args.step_size,
                                      subset_seed=index, device=device))

    record = {key: float(metrics.get(key, nan)) for key in METRIC_KEYS}
    record.update({"train_seconds": train_seconds,
                   "best_iteration": train_info["best_iteration"],
                   "stopped_at": train_info["stopped_at"],
                   "best_val_loss": train_info["best_val_loss"],
                   "initial_val_loss": train_info["initial_val_loss"]})
    return record


def run_fewshot(out: Path, run_id: str, polys: torch.Tensor, indices: list[int],
                points: torch.Tensor, n_points: int, base_state: dict, gmm_pool: torch.Tensor,
                pool_features, nll_set: dict | None, args,
                device: torch.device) -> tuple[dict[str, list[float]], dict]:
    """Per-constraint fine-tuning with on-disk checkpoints; returns metric rows and run stats."""
    records = []
    for j, index in enumerate(tqdm(indices, desc=f"{FEWSHOT} N={n_points}")):
        path = result_path(out, index, n_points)
        if path.exists():
            records.append(json.loads(path.read_text()))
            continue

        x_train = points[j][true_region_mask(polys[j], points[j], degree=args.degree,
                                             scale=args.scale)]
        record = run_fewshot_item(index, polys[j], x_train, base_state, gmm_pool,
                                  pool_features, nll_set, args, device)
        record.update({"run_id": run_id, "index": index, "n_points": n_points,
                       "n_inside": int(x_train.shape[0])})
        path.write_text(json.dumps(record, indent=2))
        records.append(record)
        print(f"[{index}] N_in {record['n_inside']:4d} | AR {record['success_rate']:6.2f}% | "
              f"SWD {record['swd']:.4f} | KLD {record['kld']:.4f} | "
              f"stopped {record['stopped_at']}", flush=True)

    scores = {key: [record[key] for record in records] for key in METRIC_KEYS}
    stats = {"median_train_seconds": float(np.nanmedian([r["train_seconds"] for r in records])),
             "median_best_iteration": float(np.nanmedian([r["best_iteration"]
                                                          for r in records])),
             "num_untrained": sum(r["n_inside"] == 0 for r in records)}
    return scores, stats


def write_shard(out: Path, method: str, n_points: int, indices: list[int],
                scores: dict[str, list[float]], n_inside: list[int], mass: list[float],
                run_id: str, digest: str, eval_block: dict) -> Path:
    per_shape = {key: values for key, values in scores.items() if any(np.isfinite(values))}
    per_shape["mass"] = mass
    per_shape["n_inside"] = n_inside
    payload = {
        "run_id": run_id,
        "method": method_name(method, n_points),
        "validation_set": "v1k",
        "poly_digest": digest,
        "start_idx": indices[0],
        "end_idx": indices[-1] + 1,
        "indices": indices,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "eval": eval_block,
        "per_shape": per_shape,
        "summary": summarize(per_shape),
    }
    path = shard_path(out, method, n_points, indices[0], indices[-1] + 1)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_functa(args, device: torch.device) -> dict:
    cfg = load_config(args.functa_run_id)
    siren = load_siren(cfg, device)
    model = build_flow_matcher(cfg, siren, device)
    load_checkpoint(cfg, model, device)
    return {"functa": model.eval(), "siren": siren, "functa_cfg": cfg}


def main(argv: list[str] | None = None) -> int:
    from constrained_fm.src.datasets.validation_v1k import get_validation_set_v1k

    args = build_parser().parse_args(argv)
    n_values = sorted(set(args.num_points))
    if n_values[-1] > args.pool_points:
        raise ValueError(f"budget {n_values[-1]} exceeds --pool-points {args.pool_points}")

    device = resolve_device()
    out = Path(args.outdir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    (out / "results" / FEWSHOT).mkdir(parents=True, exist_ok=True)
    run_id = pin_once(out, args)

    val_set = get_validation_set_v1k(device=device)
    total = val_set["polynomials"].shape[0]
    start = max(0, args.start_idx)
    end = total if args.end_idx is None else min(args.end_idx, total)
    if start >= end:
        raise ValueError(f"empty shard: start {start} >= end {end} (set holds {total})")

    indices = list(range(start, end))
    polys = val_set["polynomials"][start:end].to(device)
    mass = val_set["mass"][start:end].tolist()
    x0 = val_set["x0"][:args.num_x0].to(device)
    digest = val_set["poly_digest"]
    pool = torch.stack([shared_points(index, args.pool_points, device) for index in indices])

    print(f"run_id {run_id} | device {device} | constraints [{start}, {end}) of {total}")
    print(f"digest {digest} | budgets {n_values} | methods {args.methods}", flush=True)

    gmm_pool = seeded_gmm_pool(args.gmm_pool_size, REFERENCE_POOL_SEED, device=device)
    pool_features = compute_poly_features(gmm_pool, degree=args.degree, scale=args.scale)
    nll_set = load_nll_eval_set_v1k(num_points=args.nll_points, degree=args.degree,
                                    scale=args.scale, device="cpu") if args.nll_points > 0 else None

    models = load_functa(args, device) if FUNCTA in args.methods else {}
    base_state = load_base_state(args, device) if FEWSHOT in args.methods else None

    common = {"pool_points": args.pool_points, "shared_point_seed": SHARED_POINT_SEED,
              "num_x0": args.num_x0, "gmm_pool_size": args.gmm_pool_size,
              "step_size": args.step_size, "nll_points": args.nll_points,
              "reference_pool_seed": REFERENCE_POOL_SEED}

    for n in n_values:
        points = pool[:, :n]
        n_inside = inside_counts(polys, points, args)
        print(f"N={n:>5} | inside points median {int(np.median(n_inside))}, "
              f"min {min(n_inside)}, zero for {sum(c == 0 for c in n_inside)}", flush=True)

        if FUNCTA in args.methods and not shard_path(out, FUNCTA, n, start, end).exists():
            cfg = models["functa_cfg"]
            scores = run_functa(models, polys, indices, points, x0, gmm_pool, pool_features,
                                nll_set, args, device)
            eval_block = {**common, "n_points": n, "functa_run_id": args.functa_run_id,
                          "extraction_steps": cfg.extraction.steps,
                          "extraction_lr": cfg.extraction.lr, "query_distribution": "gmm"}
            path = write_shard(out, FUNCTA, n, indices, scores, n_inside, mass, run_id,
                               digest, eval_block)
            print(f"wrote {path}", flush=True)

        if FEWSHOT in args.methods and not shard_path(out, FEWSHOT, n, start, end).exists():
            scores, stats = run_fewshot(out, run_id, polys, indices, points, n, base_state,
                                        gmm_pool, pool_features, nll_set, args, device)
            eval_block = {**common, "n_points": n, "init": args.base_ckpt, "lr": args.lr,
                          "eval_every": args.eval_every, "patience": args.patience,
                          "val_points": args.val_points, **stats}
            path = write_shard(out, FEWSHOT, n, indices, scores, n_inside, mass, run_id,
                               digest, eval_block)
            print(f"wrote {path}", flush=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
