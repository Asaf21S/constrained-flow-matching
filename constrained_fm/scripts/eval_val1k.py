# -*- coding: utf-8 -*-
"""Stage 2 of the v1k pipeline: score five methods on a slice of the 1000-constraint set.

Methods: rejection sampling (the achievable noise floor), the coefficient-conditioned oracle,
the Functa model, and the two inference-time hacks ECI and HardFlow. Every method is scored
against the same fixed reference pool with the same SR / SWD / MMD / JSD protocol; NLL and
KLD are additionally reported for the two methods whose sampled density the probability-flow
ODE actually integrates.

A thousand constraints times five methods does not fit in one job, so the work is sliced by
constraint index and each slice writes its own shard file. Everything that a metric depends
on is keyed off the *global* constraint index -- the reference pool seed, the rejection
sampling seed, the CAVIA query points, and the RNG the SWD projections and the MMD subsample
draw from -- so a constraint scores identically no matter which slice it lands in, and shards
can be re-cut freely.

    sbatch scripts/run_val1k_eval.sh                       # the full 20-shard array
    python -m constrained_fm.scripts.eval_val1k --start-idx 0 --end-idx 50
    python -m constrained_fm.scripts.eval_val1k --start_idx 0 --end_idx 50 --methods eci
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
from constrained_fm.src.datasets.validation_v1k import seeded_gmm_pool
from constrained_fm.src.experiment.config import ExperimentConfig
from constrained_fm.src.experiment.registry import load_config, pin_baseline_run, summarize
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_siren,
                                                   resolve_device)
from constrained_fm.src.geometry.polynomials import (compute_poly_features,
                                                     compute_poly_features_batched,
                                                     evaluate_poly_batched)
from constrained_fm.src.inference.constrained_samplers import (DEFAULT_CHUNK, DEFAULT_STEPS,
                                                               sample_eci, sample_hardflow)
from constrained_fm.src.inference.constraint_projection import DEFAULT_MARGIN
from constrained_fm.src.inference.evaluator import (evaluate_single_configuration,
                                                    run_evaluation_inference)
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.metrics.eval_points import load_nll_eval_set_v1k
from constrained_fm.src.metrics.functa_fidelity import true_region_mask
from constrained_fm.src.models.constrained_poly import PolynomialConstrainedFM
from constrained_fm.src.models.unconstrained import UnconstrainedFM

METHODS = ("gt", "coeff", "functa", "eci", "hardflow")
# The two methods whose samples are the pushforward of the ODE whose density we integrate.
LIKELIHOOD_METHODS = ("coeff", "functa")
METRIC_KEYS = ("success_rate", "swd", "mmd", "jsd", "nll", "kld")

FUNCTA_RUN_ID = "siren-uniform-8d6375ab"
COEFF_CKPT = "constrained_fm/baselines/poly_fm_b1024/ckpt.pt"
BASE_CKPT = "constrained_fm/baselines/base_fm/ckpt.pt"
DEFAULT_OUTDIR = "constrained_fm/baselines/val1k"

# Independent RNG streams, so seeding one never shifts another.
REFERENCE_POOL_SEED = 20_000
GT_SAMPLE_SEED = 30_000
QUERY_POINT_SEED = 40_000
METRIC_SEED = 50_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score five methods on a slice of the v1k set.")
    parser.add_argument("--start-idx", "--start_idx", dest="start_idx", type=int, default=0,
                        help="first constraint index of this shard, inclusive")
    parser.add_argument("--end-idx", "--end_idx", dest="end_idx", type=int, default=None,
                        help="last constraint index of this shard, exclusive (default: all)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))

    parser.add_argument("--functa-run-id", default=FUNCTA_RUN_ID)
    parser.add_argument("--coeff-ckpt", default=COEFF_CKPT)
    parser.add_argument("--base-ckpt", default=BASE_CKPT)
    parser.add_argument("--coeff-hidden-dim", type=int, default=1024)
    parser.add_argument("--base-hidden-dim", type=int, default=1024)
    parser.add_argument("--base-num-blocks", type=int, default=4)
    parser.add_argument("--base-time-dim", type=int, default=128)

    parser.add_argument("--num-x0", type=int, default=10000, help="samples per constraint")
    parser.add_argument("--gmm-pool-size", type=int, default=100000,
                        help="reference pool the metrics compare against")
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--nll-points", type=int, default=5000)
    parser.add_argument("--extraction-chunk", type=int, default=128)

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="ECI/HardFlow Euler steps")
    parser.add_argument("--correction-loops", type=int, default=1)
    parser.add_argument("--projection-iters", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=100.0)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)

    parser.add_argument("--degree", type=int, default=POLYNOMIAL_DEGREE)
    parser.add_argument("--scale", type=float, default=PLANE_SCALE)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--save-samples", action="store_true",
                        help="also persist the raw (C, N, 2) sample tensors for this shard")
    return parser


def seed_metric_rng(index: int) -> None:
    """Pins the SWD projections and the MMD subsample to the constraint, not to shard order."""
    seed = METRIC_SEED + index
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


# --- sampling -----------------------------------------------------------------------------


def rejection_sample(coeffs: torch.Tensor, num_samples: int, index: int, args,
                     device: torch.device) -> torch.Tensor:
    """Draws GMM points and keeps those inside the constraint, refilling until ``num_samples``.

    The pool is drawn under a forked RNG seeded by the global constraint index, so these
    points are reproducible and independent of the reference pool the metrics score against.
    """
    batch = max(args.gmm_pool_size, num_samples * 2)
    kept, collected, attempt = [], 0, 0
    while collected < num_samples:
        pool = seeded_gmm_pool(batch, GT_SAMPLE_SEED + index * 100 + attempt, device=device)
        inside = pool[true_region_mask(coeffs, pool, degree=args.degree, scale=args.scale)]
        kept.append(inside)
        collected += int(inside.shape[0])
        attempt += 1
        if attempt > 200:
            raise RuntimeError(f"constraint {index} is too small to reach {num_samples} samples")
    return torch.cat(kept, dim=0)[:num_samples]


def seeded_query_points(cfg: ExperimentConfig, index: int, device: torch.device) -> torch.Tensor:
    """CAVIA query coordinates for one constraint, keyed to its global index."""
    with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
        torch.manual_seed(QUERY_POINT_SEED + index)
        return sample_query_points(1, cfg.extraction.points_per_shape, scale=cfg.scale,
                                   gmm_fraction=cfg.extraction.query_gmm_fraction,
                                   device=device)[0]


def extract_latents(siren, cfg: ExperimentConfig, polys: torch.Tensor, indices: list[int],
                    device: torch.device, chunk_size: int) -> torch.Tensor:
    """Per-constraint CAVIA extraction, batched but seeded per global index."""
    X_raw = torch.stack([seeded_query_points(cfg, i, device) for i in indices])
    z_chunks = []
    for start in range(0, X_raw.shape[0], chunk_size):
        X = X_raw[start:start + chunk_size]
        C = polys[start:start + chunk_size]
        x_pow, y_pow = compute_poly_features_batched(X, degree=cfg.degree, scale=cfg.scale)
        Y = torch.tanh(evaluate_poly_batched(x_pow, y_pow, C))
        z_chunk, _ = extract_latents_batched(siren, X / cfg.scale, Y, lr=cfg.extraction.lr,
                                             steps=cfg.extraction.steps)
        z_chunks.append(z_chunk)
    return torch.cat(z_chunks, dim=0)


def sample_inference_hack(method: str, model, x0: torch.Tensor, polys: torch.Tensor,
                          args) -> torch.Tensor:
    """ECI / HardFlow inject the constraint per constraint, so there is no batched form."""
    per_shape = []
    for i in tqdm(range(polys.shape[0]), desc=f"{method} sampling"):
        if method == "eci":
            samples = sample_eci(model, x0, polys[i], degree=args.degree, scale=args.scale,
                                 steps=args.steps, correction_loops=args.correction_loops,
                                 margin=args.margin, projection_iters=args.projection_iters,
                                 chunk_size=args.chunk_size)
        else:
            samples = sample_hardflow(model, x0, polys[i], degree=args.degree, scale=args.scale,
                                      steps=args.steps, guidance_scale=args.guidance_scale,
                                      margin=args.margin, chunk_size=args.chunk_size)
        per_shape.append(samples.detach())
    return torch.stack(per_shape, dim=0)


def generate(method: str, models: dict, polys: torch.Tensor, indices: list[int], x0: torch.Tensor,
             args, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Returns (C, N, 2) samples for the shard, plus the latents when the method uses them."""
    if method == "gt":
        samples = torch.stack([rejection_sample(polys[j], args.num_x0, i, args, device)
                               for j, i in enumerate(tqdm(indices, desc="gt rejection sampling"))])
        return samples, None

    if method == "coeff":
        out = run_evaluation_inference(models["coeff"], x0, coeffs=polys,
                                       step_size=args.step_size, device=device)
        return as_batched_samples(out, polys.shape[0], device), None

    if method == "functa":
        cfg, siren = models["functa_cfg"], models["siren"]
        z = extract_latents(siren, cfg, polys, indices, device, args.extraction_chunk)
        out = run_evaluation_inference(models["functa"], x0, z=z, step_size=args.step_size,
                                       device=device)
        return as_batched_samples(out, polys.shape[0], device), z

    return sample_inference_hack(method, models["base"], x0, polys, args), None


# --- scoring ------------------------------------------------------------------------------


def score(method: str, samples: torch.Tensor, polys: torch.Tensor, indices: list[int],
          latents: torch.Tensor | None, gmm_pool: torch.Tensor, pool_features, models: dict,
          nll_set: dict | None, args, device: torch.device) -> dict[str, list[float]]:
    """One metric row per constraint, with NLL/KLD only where the density is defined."""
    scores_by_key: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    likelihood_model = models.get(method) if method in LIKELIHOOD_METHODS else None
    nll_points = args.nll_points if likelihood_model is not None and nll_set is not None else 0

    for j, index in enumerate(tqdm(indices, desc=f"{method} scoring")):
        seed_metric_rng(index)
        metrics = evaluate_single_configuration(
            samples=samples[j], x_true_pool=gmm_pool, coeffs=polys[j], degree=args.degree,
            scale=args.scale, x_pow_true=pool_features[0], y_pow_true=pool_features[1],
            model=likelihood_model,
            z=None if latents is None else latents[j],
            nll_points=nll_points, nll_step_size=args.step_size,
            nll_eval_points=None if nll_points == 0 else nll_set["points"][index].to(device),
            nll_mass=None if nll_points == 0 else float(nll_set["mass"][index]),
            device=device)
        for key in METRIC_KEYS:
            scores_by_key[key].append(float(metrics.get(key, float("nan"))))

    return scores_by_key


# --- models -------------------------------------------------------------------------------


def load_models(methods: list[str], args, device: torch.device) -> dict:
    """Loads only the checkpoints the requested methods actually need."""
    models: dict = {}

    if "coeff" in methods:
        path = Path(args.coeff_ckpt)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run scripts/run_poly_fm.sh first")
        model = PolynomialConstrainedFM(degree=args.degree, hidden_dim=args.coeff_hidden_dim,
                                        scale_factor=args.scale).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        models["coeff"] = model.eval()

    if "functa" in methods:
        cfg = load_config(args.functa_run_id)
        siren = load_siren(cfg, device)
        model = build_flow_matcher(cfg, siren, device)
        load_checkpoint(cfg, model, device)
        models.update({"functa": model.eval(), "siren": siren, "functa_cfg": cfg})

    if {"eci", "hardflow"} & set(methods):
        path = Path(args.base_ckpt)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run scripts/run_base_fm.sh first")
        model = UnconstrainedFM(time_dim=args.base_time_dim, hidden_dim=args.base_hidden_dim,
                                num_blocks=args.base_num_blocks).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        models["base"] = model.eval()

    return models


def shard_path(out: Path, method: str, start: int, end: int) -> Path:
    return out / "shards" / f"{method}__{start:05d}_{end:05d}.json"


# Which slice a job happens to run is not part of what the result is.
_SHARD_ARGS = ("start_idx", "end_idx", "save_samples")


def pin_once(out: Path, args) -> str:
    """One run id for the whole sweep: every shard fingerprints the same settings.

    Reuses the id already on disk so a running array never rewrites provenance underneath
    its siblings.
    """
    provenance = out / "provenance.json"
    if provenance.exists():
        return json.loads(provenance.read_text())["run_id"]
    settings = {k: v for k, v in vars(args).items() if k not in _SHARD_ARGS}
    return pin_baseline_run(out, "val1k", settings)


def as_batched_samples(raw, num_conditions: int, device: torch.device) -> torch.Tensor:
    """``run_evaluation_inference`` drops the leading axis for a single condition."""
    array = np.asarray(raw, dtype=np.float32).reshape(num_conditions, -1, 2)
    return torch.from_numpy(array).to(device)


def main(argv: list[str] | None = None) -> int:
    from constrained_fm.src.datasets.validation_v1k import get_validation_set_v1k

    args = build_parser().parse_args(argv)
    device = resolve_device()
    out = Path(args.outdir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
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

    print(f"run_id {run_id} | device {device} | constraints [{start}, {end}) of {total}")
    print(f"digest {val_set['poly_digest']} | methods {args.methods} | "
          f"{args.num_x0} samples per constraint", flush=True)

    # One fixed reference pool for every shard and every method: the SWD/MMD/JSD of two
    # methods are only comparable if they were measured against the same ground truth.
    gmm_pool = seeded_gmm_pool(args.gmm_pool_size, REFERENCE_POOL_SEED, device=device)
    pool_features = compute_poly_features(gmm_pool, degree=args.degree, scale=args.scale)

    needs_likelihood = bool(set(args.methods) & set(LIKELIHOOD_METHODS)) and args.nll_points > 0
    nll_set = load_nll_eval_set_v1k(num_points=args.nll_points, degree=args.degree,
                                    scale=args.scale, device="cpu") if needs_likelihood else None

    models = load_models(args.methods, args, device)

    for method in args.methods:
        samples, latents = generate(method, models, polys, indices, x0, args, device)
        scores_by_key = score(method, samples, polys, indices, latents, gmm_pool, pool_features,
                              models, nll_set, args, device)

        per_shape = {key: values for key, values in scores_by_key.items()
                     if any(np.isfinite(values))}
        per_shape["mass"] = mass
        payload = {
            "run_id": run_id,
            "method": method,
            "validation_set": "v1k",
            "poly_digest": val_set["poly_digest"],
            "start_idx": start,
            "end_idx": end,
            "indices": indices,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "eval": {"num_x0": args.num_x0, "gmm_pool_size": args.gmm_pool_size,
                     "step_size": args.step_size,
                     "nll_points": args.nll_points if method in LIKELIHOOD_METHODS else 0,
                     "reference_pool_seed": REFERENCE_POOL_SEED},
            "per_shape": per_shape,
            "summary": summarize(per_shape),
        }
        path = shard_path(out, method, start, end)
        path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {path}", flush=True)

        if args.save_samples:
            samples_dir = out / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            np.save(samples_dir / f"{method}__{start:05d}_{end:05d}.npy",
                    samples.cpu().numpy().astype(np.float32))

        del samples, latents
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
