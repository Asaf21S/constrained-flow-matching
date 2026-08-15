# -*- coding: utf-8 -*-
"""Attributes a run's residual error to the SIREN conditioning or to the flow matcher.

The flow matcher only ever sees z, so its ceiling is the region the SIREN decodes from z.
Scoring the samples twice - once against the true constraint, once against the believed
region - splits the error into a part the flow matcher owns and a part the encoder owns:

  believed_sr    % of generated samples inside {SIREN(x, z) <= 0}   -> FM obedience
  oracle_sr      % of GMM mass in the believed region that is also
                 inside the true region                             -> encoder ceiling
  swd_vs_believed SWD(samples, GMM restricted to believed region)   -> FM distribution error
  swd_oracle     SWD(GMM believed, GMM true)                        -> encoder ceiling

A run where believed_sr is near 100 and swd_vs_believed is near zero, while true_sr and
swd track oracle_sr and swd_oracle, is bottlenecked by the encoder, not the flow matcher.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import torch

from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.experiment.registry import load_config, run_dir, write_json
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_siren,
                                                   resolve_device, set_seed)
from constrained_fm.src.inference.evaluator import run_evaluation_inference
from constrained_fm.src.metrics.distributional import compute_swd
from constrained_fm.src.metrics.functa_fidelity import decode_region, true_region_mask
from constrained_fm.src.metrics.success_rates import compute_success_rate_polynomial
from constrained_fm.scripts.eval_fm import extract_validation_latents

REPORT_NAME = "bottleneck.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attribute residual error to encoder vs flow matcher.")
    parser.add_argument("--run-id", required=True, help="run id of an evaluated run")
    parser.add_argument("--num-polys", type=int, default=100, help="validation shapes to score")
    parser.add_argument("--num-x0", type=int, default=10000, help="samples per shape")
    parser.add_argument("--pool-samples", type=int, default=20000,
                        help="GMM points used for the oracle reference sets")
    return parser


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)
    return {
        "median": ordered[n // 2],
        "mean": sum(ordered) / n,
        "p5": ordered[max(0, int(0.05 * (n - 1)))],
        "p95": ordered[min(n - 1, int(0.95 * (n - 1)))],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.run_id)
    device = resolve_device()
    ev = cfg.evaluation
    print(f"run_id  {cfg.run_id}")
    print(f"device  {device}")

    set_seed(ev.seed)
    siren = load_siren(cfg, device)
    model = build_flow_matcher(cfg, siren, device)
    iteration = load_checkpoint(cfg, model, device)
    model.eval()

    gmm_pool, _ = get_points(ev.gmm_pool_size, device=device)
    subset = torch.randperm(gmm_pool.shape[0], device=device)[:args.pool_samples]
    gmm_pool = gmm_pool[subset]

    val_set = get_validation_set(device=device)
    val_polys = val_set["polynomials"][:args.num_polys].to(device)
    val_x0 = val_set["x0"][:args.num_x0].to(device)

    z_val, _ = extract_validation_latents(siren, cfg, val_polys, device)
    samples = run_evaluation_inference(model, val_x0, z=z_val, step_size=ev.step_size, device=device)

    per_shape: dict[str, list[float]] = {key: [] for key in
                                         ["true_sr", "believed_sr", "oracle_sr",
                                          "swd_vs_true", "swd_vs_believed", "swd_oracle"]}

    for i in range(val_polys.shape[0]):
        C = val_polys[i]
        z = z_val[i]
        gen = torch.as_tensor(samples[i], dtype=torch.float32, device=device)

        true_sr = compute_success_rate_polynomial(gen, C, cfg.degree, cfg.scale, device)
        believed_sr = float((decode_region(siren, z, gen, scale=cfg.scale) <= 0).float().mean()) * 100.0

        pool_true = true_region_mask(C, gmm_pool, degree=cfg.degree, scale=cfg.scale)
        pool_believed = decode_region(siren, z, gmm_pool, scale=cfg.scale) <= 0

        gmm_true = gmm_pool[pool_true]
        gmm_believed = gmm_pool[pool_believed]

        # The best any z-conditioned sampler could do: perfectly fill the believed region.
        oracle_sr = float(pool_true[pool_believed].float().mean()) * 100.0 if pool_believed.any() else 0.0

        per_shape["true_sr"].append(float(true_sr))
        per_shape["believed_sr"].append(believed_sr)
        per_shape["oracle_sr"].append(oracle_sr)
        per_shape["swd_vs_true"].append(compute_swd(gen, gmm_true))
        per_shape["swd_vs_believed"].append(compute_swd(gen, gmm_believed))
        per_shape["swd_oracle"].append(compute_swd(gmm_believed, gmm_true))

    summary = {key: quantiles(values) for key, values in per_shape.items()}

    write_json(run_dir(cfg.run_id) / REPORT_NAME, {
        "run_id": cfg.run_id,
        "iteration": iteration,
        "diagnosed_at": datetime.now().isoformat(timespec="seconds"),
        "num_polys": args.num_polys,
        "num_x0": args.num_x0,
        "per_shape": per_shape,
        "summary": summary,
    })

    print()
    print(f"{'metric':<18}{'median':>10}{'mean':>10}{'p5':>10}{'p95':>10}")
    for key, stats in summary.items():
        print(f"{key:<18}{stats['median']:>10.4f}{stats['mean']:>10.4f}"
              f"{stats['p5']:>10.4f}{stats['p95']:>10.4f}")

    print()
    print("attribution on the worst-10 shapes by true success rate")
    print(f"{'shape':>6}{'trueSR':>9}{'believSR':>10}{'oracleSR':>10}"
          f"{'swdTrue':>10}{'swdBelvd':>10}{'swdOracle':>11}")
    order = sorted(range(len(per_shape["true_sr"])), key=lambda i: per_shape["true_sr"][i])[:10]
    for i in order:
        print(f"{i:>6}{per_shape['true_sr'][i]:>9.2f}{per_shape['believed_sr'][i]:>10.2f}"
              f"{per_shape['oracle_sr'][i]:>10.2f}{per_shape['swd_vs_true'][i]:>10.4f}"
              f"{per_shape['swd_vs_believed'][i]:>10.4f}{per_shape['swd_oracle'][i]:>11.4f}")

    print(f"\nwritten to {run_dir(cfg.run_id) / REPORT_NAME}")
    print(json.dumps(summary["believed_sr"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
