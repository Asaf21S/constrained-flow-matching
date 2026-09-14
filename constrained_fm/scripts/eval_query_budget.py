# -*- coding: utf-8 -*-
"""Inference query-budget ablation on the v1k set, sliced by constraint index.

The SIREN and the flow matcher are both frozen. Only N -- the number of query points fed to
the 15-step CAVIA inner loop at inference -- varies, so this isolates how much of the
pipeline's quality the extraction budget alone is buying. The 100-polynomial version of this
sweep lives in ``ablate_query_points.py``; this one runs the same measurement over all 1000
stratified constraints, which does not fit in a single job.

Everything a number depends on is keyed off the *global* constraint index -- the reference
pool seed, the CAVIA query points, and the RNG the SWD projections and the MMD subsample draw
from -- so a constraint scores identically no matter which slice it lands in, and shards can
be re-cut freely. The query-point seed also carries the budget, so N = 10 and N = 2000 are
independent draws rather than a prefix relationship that would understate the small budgets'
variance.

NLL/KLD are deliberately not scored here: the exact-divergence integration dominates the
runtime and none of the four reported panels use it.

    sbatch scripts/run_query_budget_eval.sh                                # the full array
    python -m constrained_fm.scripts.eval_query_budget --start-idx 0 --end-idx 50
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

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.datasets.validation_v1k import resolve_validation_set, seeded_gmm_pool
from constrained_fm.src.experiment.config import ExperimentConfig
from constrained_fm.src.experiment.registry import load_config, pin_baseline_run, summarize
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_siren,
                                                   resolve_device)
from constrained_fm.src.geometry.polynomials import (compute_poly_features,
                                                     compute_poly_features_batched,
                                                     evaluate_poly_batched)
from constrained_fm.src.inference.evaluator import (evaluate_single_configuration,
                                                    run_evaluation_inference)
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.metrics.functa_fidelity import constraint_masses, region_iou_batched

DEFAULT_N_VALUES = [10, 25, 50, 100, 300, 1000, 2000]
METRIC_KEYS = ("mass_iou", "extraction_mse", "success_rate", "swd", "mmd", "jsd")

FUNCTA_RUN_ID = "siren-uniform-8d6375ab"
DEFAULT_OUTDIR = "constrained_fm/baselines/query_budget"

# Shared with eval_val1k so the reference pool and the metric RNG are literally the same
# streams: the N = 1000 column of this sweep must reproduce the val1k Functa row.
REFERENCE_POOL_SEED = 20_000
QUERY_POINT_SEED = 40_000
METRIC_SEED = 50_000
# Offsets the query-point stream per budget, so the budgets are independent draws.
BUDGET_SEED_STRIDE = 1_000_003


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep the inference query budget on a v1k slice.")
    parser.add_argument("--start-idx", "--start_idx", dest="start_idx", type=int, default=0,
                        help="first constraint index of this shard, inclusive")
    parser.add_argument("--end-idx", "--end_idx", dest="end_idx", type=int, default=None,
                        help="last constraint index of this shard, exclusive (default: all)")
    parser.add_argument("--num-points", type=int, nargs="+", default=DEFAULT_N_VALUES,
                        help="query-point counts N to sweep at inference")

    parser.add_argument("--run-id", default=FUNCTA_RUN_ID,
                        help="run whose frozen SIREN + flow-matcher checkpoint to use")
    parser.add_argument("--validation-set", default="v1k", choices=("v1k", "legacy100"))
    parser.add_argument("--no-flow-matching", action="store_true",
                        help="SIREN half only; skips all ODE sampling")

    parser.add_argument("--num-x0", type=int, default=10000, help="samples per constraint")
    parser.add_argument("--gmm-pool-size", type=int, default=100000,
                        help="reference pool the metrics compare against")
    parser.add_argument("--iou-mass-samples", type=int, default=20000,
                        help="GMM points the mass-weighted region IoU is measured over")
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--extraction-chunk", type=int, default=128)

    parser.add_argument("--degree", type=int, default=POLYNOMIAL_DEGREE)
    parser.add_argument("--scale", type=float, default=PLANE_SCALE)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser


def seed_metric_rng(index: int) -> None:
    """Pins the SWD projections and the MMD subsample to the constraint, not to shard order."""
    seed = METRIC_SEED + index
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def seeded_query_points(cfg: ExperimentConfig, index: int, num_points: int,
                        device: torch.device) -> torch.Tensor:
    """CAVIA query coordinates for one constraint at one budget, keyed to its global index.

    At ``num_points = cfg.extraction.points_per_shape`` and stride 0 this would coincide with
    the val1k stream; the stride is applied unconditionally so no budget is a special case.
    """
    seed = QUERY_POINT_SEED + index + BUDGET_SEED_STRIDE * num_points
    with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
        torch.manual_seed(seed)
        return sample_query_points(1, num_points, scale=cfg.scale,
                                   gmm_fraction=cfg.extraction.query_gmm_fraction,
                                   device=device)[0]


def extract_at_budget(siren, cfg: ExperimentConfig, polys: torch.Tensor, indices: list[int],
                      num_points: int, device: torch.device,
                      chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """CAVIA extraction for the shard at exactly ``num_points`` query points.

    Everything except N is held at the run's deployed setting, ``query_gmm_fraction``
    included: the SIREN was meta-trained against one query distribution and only the budget
    is under test. Returns the latents and the per-constraint final extraction MSE.
    """
    X_raw = torch.stack([seeded_query_points(cfg, i, num_points, device) for i in indices])
    z_chunks, mse_chunks = [], []
    for start in range(0, X_raw.shape[0], chunk_size):
        X = X_raw[start:start + chunk_size]
        C = polys[start:start + chunk_size]
        x_pow, y_pow = compute_poly_features_batched(X, degree=cfg.degree, scale=cfg.scale)
        Y = torch.tanh(evaluate_poly_batched(x_pow, y_pow, C))
        z_chunk, mse_chunk = extract_latents_batched(siren, X / cfg.scale, Y,
                                                     lr=cfg.extraction.lr,
                                                     steps=cfg.extraction.steps)
        z_chunks.append(z_chunk)
        mse_chunks.append(mse_chunk)
    return torch.cat(z_chunks, dim=0), torch.cat(mse_chunks, dim=0)


def score_samples(samples: torch.Tensor, polys: torch.Tensor, indices: list[int],
                  gmm_pool: torch.Tensor, pool_features, args,
                  device: torch.device) -> dict[str, list[float]]:
    """One SR / SWD / MMD / JSD row per constraint."""
    keys = ("success_rate", "swd", "mmd", "jsd")
    scores: dict[str, list[float]] = {key: [] for key in keys}
    for j, index in enumerate(tqdm(indices, desc="scoring", leave=False)):
        seed_metric_rng(index)
        metrics = evaluate_single_configuration(
            samples=samples[j], x_true_pool=gmm_pool, coeffs=polys[j], degree=args.degree,
            scale=args.scale, x_pow_true=pool_features[0], y_pow_true=pool_features[1],
            device=device)
        for key in keys:
            scores[key].append(float(metrics.get(key, float("nan"))))
    return scores


def as_batched_samples(raw, num_conditions: int, device: torch.device) -> torch.Tensor:
    """``run_evaluation_inference`` drops the leading axis for a single condition."""
    array = np.asarray(raw, dtype=np.float32).reshape(num_conditions, -1, 2)
    return torch.from_numpy(array).to(device)


def shard_path(out: Path, num_points: int, start: int, end: int) -> Path:
    return out / "shards" / f"n{num_points:05d}__{start:05d}_{end:05d}.json"


# Which slice a job happens to run is not part of what the result is.
_SHARD_ARGS = ("start_idx", "end_idx")


def pin_once(out: Path, args) -> str:
    """One run id for the whole sweep, reused from disk so array tasks never race."""
    provenance = out / "provenance.json"
    if provenance.exists():
        return json.loads(provenance.read_text())["run_id"]
    settings = {k: v for k, v in vars(args).items() if k not in _SHARD_ARGS}
    return pin_baseline_run(out, "query_budget", settings)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    out = Path(args.outdir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    run_id = pin_once(out, args)
    n_values = sorted(args.num_points)

    cfg = load_config(args.run_id)
    val_set = resolve_validation_set(args.validation_set, device=device)
    total = val_set["polynomials"].shape[0]
    start = max(0, args.start_idx)
    end = total if args.end_idx is None else min(args.end_idx, total)
    if start >= end:
        raise ValueError(f"empty shard: start {start} >= end {end} (set holds {total})")

    indices = list(range(start, end))
    polys = val_set["polynomials"][start:end].to(device)
    x0 = val_set["x0"][:args.num_x0].to(device)

    siren = load_siren(cfg, device)
    model = None
    if not args.no_flow_matching:
        model = build_flow_matcher(cfg, siren, device)
        load_checkpoint(cfg, model, device)
        model.eval()

    print(f"run_id {run_id} | device {device} | constraints [{start}, {end}) of {total}")
    print(f"digest {val_set['poly_digest']} | budgets {n_values} | "
          f"meta-trained at 1000 pts, deployed at {cfg.extraction.points_per_shape}")
    print(f"extraction: {cfg.extraction.steps} steps, lr {cfg.extraction.lr}, "
          f"query_gmm_fraction {cfg.extraction.query_gmm_fraction}", flush=True)

    # One fixed reference pool for every shard and every budget: SWD/MMD/JSD are only
    # comparable across budgets if they were measured against the same ground truth.
    gmm_pool = seeded_gmm_pool(args.gmm_pool_size, REFERENCE_POOL_SEED, device=device)
    pool_features = compute_poly_features(gmm_pool, degree=args.degree, scale=args.scale)
    iou_points = gmm_pool[:args.iou_mass_samples]

    # legacy100 carries no cached mass, so estimate it against the same reference pool.
    mass = (val_set["mass"][start:end].tolist() if val_set["mass"] is not None else
            constraint_masses(polys, gmm_pool, degree=args.degree, scale=args.scale).cpu().tolist())

    for n in n_values:
        z, mse = extract_at_budget(siren, cfg, polys, indices, n, device, args.extraction_chunk)
        iou = region_iou_batched(siren, z, polys, iou_points, degree=args.degree, scale=args.scale)
        per_shape = {"mass_iou": iou.cpu().tolist(), "extraction_mse": mse.cpu().tolist()}
        print(f"N={n:>5} | extraction MSE {mse.mean():.6f} | mass IoU mean {iou.mean():.4f} "
              f"median {iou.median():.4f} min {iou.min():.4f}", flush=True)

        if model is not None:
            raw = run_evaluation_inference(model, x0, z=z, step_size=args.step_size, device=device)
            samples = as_batched_samples(raw, polys.shape[0], device)
            per_shape.update(score_samples(samples, polys, indices, gmm_pool, pool_features,
                                           args, device))
            sr = np.asarray(per_shape["success_rate"])
            print(f"N={n:>5} | success rate mean {sr.mean():.2f}% | "
                  f"SWD median {np.median(per_shape['swd']):.4f}", flush=True)
            del raw, samples

        per_shape["mass"] = mass
        payload = {
            "run_id": run_id,
            "num_points": n,
            "validation_set": args.validation_set,
            "poly_digest": val_set["poly_digest"],
            "start_idx": start,
            "end_idx": end,
            "indices": indices,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "eval": {"num_x0": 0 if model is None else args.num_x0,
                     "gmm_pool_size": args.gmm_pool_size,
                     "iou_mass_samples": args.iou_mass_samples,
                     "step_size": args.step_size,
                     "extraction_steps": cfg.extraction.steps,
                     "extraction_lr": cfg.extraction.lr,
                     "query_gmm_fraction": cfg.extraction.query_gmm_fraction,
                     "meta_trained_points_per_shape": 1000,
                     "deployed_points_per_shape": cfg.extraction.points_per_shape,
                     "reference_pool_seed": REFERENCE_POOL_SEED},
            "per_shape": per_shape,
            "summary": summarize(per_shape),
        }
        path = shard_path(out, n, start, end)
        path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {path}", flush=True)

        del z
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
