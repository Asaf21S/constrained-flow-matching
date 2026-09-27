# -*- coding: utf-8 -*-
"""Selects ECI's mixing iterations M and noise-resample interval R for the v1k benchmark.

The grid is scored on a small tuning set built by the v1k generator under a different seed,
so it shares v1k's mass stratification but none of its constraints or start points. Every
configuration runs through exactly the sampling and metric code of ``eval_val1k``.

Selection rule (the one ``tune_bench1k`` applies):

* eligible: median success rate at least ``min(SR_TARGET, best median SR - SR_SLACK)``;
* score: the mean of ``log(median SWD)`` and ``log(median MMD)``;
* selected: the lowest score among the eligible, ties going to the smaller M.

Writes ``<outdir>/selected.json``, which ``eval_val1k --eci-selected`` reads back.

    sbatch scripts/run_val1k_tune_eci.sh
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import torch

from constrained_fm.scripts.eval_val1k import (REFERENCE_POOL_SEED, build_parser as eval_parser,
                                               load_models, sample_inference_hack, score)
from constrained_fm.src.consts import VAL1K_SEED
from constrained_fm.src.datasets.validation_v1k import (build_validation_set_v1k,
                                                        seeded_gmm_pool)
from constrained_fm.src.experiment.registry import pin_baseline_run, summarize
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.geometry.polynomials import compute_poly_features

MIXING_ITERS = (1, 2, 5, 10)
RESAMPLE_INTERVALS = (1, 5, None)
NUM_TUNE_POLYS = 20
TUNE_SET_SEED = VAL1K_SEED + 1
# Keeps the tuning seeds (ECI noise, metric RNG) clear of the benchmark's 0..999.
TUNE_INDEX_OFFSET = 100_000
SR_TARGET = 95.0
SR_SLACK = 1.0
DEFAULT_OUTDIR = "constrained_fm/baselines/val1k_v2/tuning"


def grid() -> list[dict]:
    return [{"mixing_iters": m, "resample_interval": r}
            for m, r in product(MIXING_ITERS, RESAMPLE_INTERVALS)]


def config_row(settings: dict, per_shape: dict[str, list[float]]) -> dict:
    sr = float(np.nanmedian(per_shape["success_rate"]))
    swd = float(np.nanmedian(per_shape["swd"]))
    mmd = float(np.nanmedian(per_shape["mmd"]))
    return {"settings": settings, "cost": settings["mixing_iters"], "success_rate": sr,
            "swd": swd, "mmd": mmd,
            "score": 0.5 * (math.log(max(swd, 1e-12)) + math.log(max(mmd, 1e-12)))}


def select(rows: list[dict]) -> tuple[dict, float]:
    threshold = min(SR_TARGET, max(r["success_rate"] for r in rows) - SR_SLACK)
    for r in rows:
        r["eligible"] = r["success_rate"] >= threshold
    best = min((r for r in rows if r["eligible"]), key=lambda r: (r["score"], r["cost"]))
    return best, threshold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune ECI (M, R) on a v1k-style tuning set.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--num-polys", type=int, default=NUM_TUNE_POLYS)
    args, eval_argv = parser.parse_known_args(argv)
    bench = eval_parser().parse_args(["--methods", "eci", *eval_argv])

    device = resolve_device()
    out = Path(args.outdir)
    run_id = pin_baseline_run(out, "val1k_eci_tune",
                              {**vars(bench), "num_polys": args.num_polys,
                               "tune_set_seed": TUNE_SET_SEED, "grid": grid()})

    tune_set = build_validation_set_v1k(num_polys=args.num_polys, seed=TUNE_SET_SEED,
                                        device=device)
    torch.save(tune_set, out / "tune_set.pt")
    polys = tune_set["polynomials"].to(device)
    x0 = tune_set["x0"][:bench.num_x0].to(device)
    indices = [TUNE_INDEX_OFFSET + j for j in range(polys.shape[0])]
    print(f"run_id {run_id} | tuning digest {tune_set['poly_digest']} | "
          f"{polys.shape[0]} constraints x {x0.shape[0]} samples", flush=True)

    gmm_pool = seeded_gmm_pool(bench.gmm_pool_size, REFERENCE_POOL_SEED, device=device)
    pool_features = compute_poly_features(gmm_pool, degree=bench.degree, scale=bench.scale)
    models = load_models(["eci"], bench, device)

    rows = []
    for settings in grid():
        bench.mixing_iters = settings["mixing_iters"]
        bench.resample_interval = settings["resample_interval"] or 0
        samples = sample_inference_hack("eci", models["base"], x0, polys, indices, bench)
        per_shape = score("eci", samples, polys, indices, None, gmm_pool, pool_features,
                          models, None, bench, device)
        per_shape = {k: v for k, v in per_shape.items() if any(np.isfinite(v))}
        row = config_row(settings, per_shape)
        row["summary"] = summarize(per_shape)
        rows.append(row)
        print(f"{settings} | SR {row['success_rate']:.2f} | SWD {row['swd']:.4f} | "
              f"MMD {row['mmd']:.5f} | score {row['score']:.3f}", flush=True)
        del samples
        torch.cuda.empty_cache()

    best, threshold = select(rows)
    print(f"\nSR threshold {threshold:.1f}% -> selected {best['settings']}")
    print("| M | R | median SR % | median SWD | median MMD | score | eligible |")
    print("|---:|---:|---:|---:|---:|---:|:---:|")
    for r in sorted(rows, key=lambda r: (not r["eligible"], r["score"])):
        mark = " **<-**" if r is best else ""
        print(f"| {r['settings']['mixing_iters']} | {r['settings']['resample_interval']} "
              f"| {r['success_rate']:.2f} | {r['swd']:.4f} | {r['mmd']:.5f} "
              f"| {r['score']:.3f} | {'y' if r['eligible'] else 'n'}{mark} |")

    path = out / "selected.json"
    path.write_text(json.dumps({
        "run_id": run_id, "tuning_digest": tune_set["poly_digest"],
        "rule": {"sr_target": SR_TARGET, "sr_slack": SR_SLACK,
                 "score": "mean of log median-SWD and log median-MMD",
                 "tie_break": "smaller mixing_iters"},
        "selected": best["settings"], "threshold": threshold, "rows": rows}, indent=1))
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
