# -*- coding: utf-8 -*-
"""Selects every method's inference-time hyperparameters on a held-out constraint set.

Each array task scores one (method, configuration) pair from :func:`grid` on the tuning split
(``benchmark_tune_<problem>.pt``: its own constraints, start points and seeds, never a
benchmark constraint), through exactly the sampling and metric code the benchmark uses.
``--select`` then applies one rule to every method and writes ``selected.json``, which
``eval_bench1k`` reads back as the per-method sampler settings.

Selection rule, identical for every method:

* eligible: configurations whose median success rate is at least
  ``min(SR_TARGET, best median success rate of that method - SR_SLACK)``, so a method is
  never tuned into trading feasibility for fidelity, yet one that cannot reach the target
  is still compared on its best-feasibility settings;
* score: the mean of ``log(median SWD / median SWD floor)`` and
  ``log(median MMD / median MMD floor)``, the distance from exact conditional sampling;
* selected: the lowest score among the eligible, ties going to the cheaper sampler.

    sbatch scripts/run_bench1k_build.sh --split tune
    PROBLEM=kinematics6d sbatch --array=0-76%16 scripts/run_tune.sh
    PROBLEM=kinematics6d sbatch scripts/run_tune.sh --select
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from constrained_fm.scripts.eval_bench1k import (MARGIN, METRIC_SEED, MMD_GAMMA,
                                                 PROJECTION_ITERS, SAMPLER_KEYS, TUNING_DIR,
                                                 build_parser as bench_parser, conditioning,
                                                 generate, load_models, reference_pool,
                                                 resolve_defaults, score_one,
                                                 seed_metric_rng)
from constrained_fm.src.datasets.benchmark_1k import (PROBLEM_NAMES, constraints_from,
                                                      load_benchmark_1k)
from constrained_fm.src.experiment.registry import summarize
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.problems.base import NormalizedConstraint
from constrained_fm.src.problems.bump2d import BumpProblem
from constrained_fm.src.problems.kinematics6d import KinematicsProblem

OURS = {"bump2d": "functa", "kinematics6d": "explicit"}
STEP_SIZES = (0.1, 0.05, 0.02, 0.01, 0.005)
ECI_STEPS = (50, 100, 200)
ECI_LOOPS = (1, 2, 3)
# A polygon's faces are planes, so the undamped Newton step is already exact there.
ECI_DAMPING = {"bump2d": (1.0,), "kinematics6d": (1.0, 0.5)}
HARDFLOW_SCALES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
HARDFLOW_STEPS = (100, 200, 400)
MARGIN_FACTORS = (1.0, 10.0)

SR_TARGET = 95.0
SR_SLACK = 1.0
# Keeps tuning seeds (CAVIA query points, metric RNG) clear of the benchmark's 0..999.
TUNE_INDEX_OFFSET = 100_000
TUNE_KEYS = ("success_rate", "swd", "mmd", "jsd", "swd_noise_floor", "mmd_noise_floor",
             "jsd_noise_floor", "in_support_fraction", "signal_fraction")


def grid(problem: str) -> list[tuple[str, dict]]:
    """Every (method, sampler settings) pair tried; the array task id indexes this list."""
    margin, iters = MARGIN[problem], PROJECTION_ITERS[problem]
    configs = [(OURS[problem], {"step_size": s}) for s in STEP_SIZES]
    for steps, loops, damping, factor in product(ECI_STEPS, ECI_LOOPS, ECI_DAMPING[problem],
                                                 MARGIN_FACTORS):
        # A damped Newton step covers less ground, so it gets proportionally more iterations.
        configs.append(("eci", {"steps": steps, "correction_loops": loops,
                                "projection_iters": round(iters / damping),
                                "projection_damping": damping, "margin": margin * factor}))
    for scale, steps, factor in product(HARDFLOW_SCALES, HARDFLOW_STEPS, MARGIN_FACTORS):
        configs.append(("hardflow", {"steps": steps, "guidance_scale": scale,
                                     "margin": margin * factor}))
    for method, settings in configs:
        assert set(settings) == set(SAMPLER_KEYS[method]), (method, settings)
    return configs


def cost(method: str, settings: dict) -> float:
    """Velocity-network evaluations per sample; HardFlow's each also carry a backward pass."""
    if method in ("functa", "explicit"):
        return 2.0 / settings["step_size"]
    if method == "eci":
        return float(settings["steps"] * settings["correction_loops"])
    return float(settings["steps"])


def config_path(problem: str, task_id: int, method: str) -> Path:
    return Path(TUNING_DIR) / problem / "configs" / f"{task_id:03d}__{method}.json"


def run_task(problem_name: str, task_id: int, device: torch.device) -> Path:
    method, settings = grid(problem_name)[task_id]
    bench = bench_parser().parse_args(["--problem", problem_name, "--methods", method,
                                       "--no-tuned"])
    resolve_defaults(bench)
    bench.settings[method] = dict(settings)

    problem = BumpProblem() if problem_name == "bump2d" else KinematicsProblem()
    normalizer = problem.normalizer().to(device)
    target = problem.target()
    benchmark = load_benchmark_1k(problem_name, device=device, split="tune")
    constraints = constraints_from(benchmark, problem, device=device)
    x0 = benchmark["x0"]
    pool_u = normalizer.forward(reference_pool(problem, bench, device))
    models = load_models(bench, problem, device)
    print(f"{problem_name} task {task_id}: {method} {settings} | {len(constraints)} tuning "
          f"constraints x {x0.shape[0]} samples | digest {benchmark['digest']}", flush=True)

    per_shape: dict[str, list[float]] = {key: [] for key in TUNE_KEYS}
    for position, constraint in enumerate(tqdm(constraints, desc=f"{method} #{task_id}")):
        index = TUNE_INDEX_OFFSET + position
        wrapped = NormalizedConstraint(constraint, normalizer)
        seed_metric_rng(index)
        cond = conditioning(method, models, constraint, index, problem, device)
        samples = generate(method, models, constraint, index, x0, problem, normalizer, bench,
                           device, cond)
        truth = pool_u[wrapped.is_feasible(pool_u)]
        seed_metric_rng(index)
        row = score_one(problem_name, samples.detach(), truth, wrapped, normalizer, target,
                        MMD_GAMMA[problem_name], METRIC_SEED + index)
        for key in TUNE_KEYS:
            per_shape[key].append(row[key])

    path = config_path(problem_name, task_id, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"problem": problem_name, "task_id": task_id, "method": method,
                                "settings": settings, "cost": cost(method, settings),
                                "tuning_digest": benchmark["digest"],
                                "per_shape": per_shape, "summary": summarize(per_shape)},
                               indent=1))
    print(f"wrote {path}", flush=True)
    return path


def config_row(record: dict) -> dict:
    """The quantities the rule compares, all medians over the tuning constraints."""
    shape = {k: np.asarray(v, dtype=float) for k, v in record["per_shape"].items()}
    swd = np.nanmedian(shape["swd"]) / max(np.nanmedian(shape["swd_noise_floor"]), 1e-12)
    mmd = np.nanmedian(shape["mmd"]) / max(np.nanmedian(shape["mmd_noise_floor"]), 1e-12)
    return {"task_id": record["task_id"], "method": record["method"],
            "settings": record["settings"], "cost": record["cost"],
            "success_rate": float(np.nanmedian(shape["success_rate"])),
            "success_rate_p5": float(np.nanpercentile(shape["success_rate"], 5)),
            "swd_ratio": float(swd), "mmd_ratio": float(mmd),
            "score": 0.5 * (math.log(max(swd, 1e-12)) + math.log(max(mmd, 1e-12)))}


def select(problem: str) -> Path:
    configs = grid(problem)
    records = {}
    for task_id, (method, _) in enumerate(configs):
        path = config_path(problem, task_id, method)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- re-run array task {task_id}")
        records[task_id] = json.loads(path.read_text())
    digests = {r["tuning_digest"] for r in records.values()}
    if len(digests) != 1:
        raise RuntimeError(f"tuning results come from different tuning sets: {digests}")

    rows = [config_row(r) for r in records.values()]
    selected, tables = {}, {}
    for method in dict.fromkeys(m for m, _ in configs):
        mine = [r for r in rows if r["method"] == method]
        threshold = min(SR_TARGET, max(r["success_rate"] for r in mine) - SR_SLACK)
        for r in mine:
            r["eligible"] = r["success_rate"] >= threshold
        best = min((r for r in mine if r["eligible"]), key=lambda r: (r["score"], r["cost"]))
        selected[method] = best["settings"]
        tables[method] = {"threshold": threshold, "selected_task": best["task_id"],
                          "rows": sorted(mine, key=lambda r: (not r["eligible"], r["score"]))}

        print(f"\n### {problem} {method}: SR threshold {threshold:.1f}% -> task "
              f"{best['task_id']} {best['settings']}")
        print("| task | settings | cost | median SR % | p5 SR % | SWD / floor | MMD / floor "
              "| score | eligible |")
        print("|---:|:---|---:|---:|---:|---:|---:|---:|:---:|")
        for r in tables[method]["rows"]:
            mark = " **<-**" if r["task_id"] == best["task_id"] else ""
            print(f"| {r['task_id']} | {r['settings']} | {r['cost']:g} | {r['success_rate']:.2f} "
                  f"| {r['success_rate_p5']:.2f} | {r['swd_ratio']:.2f} | {r['mmd_ratio']:.1f} "
                  f"| {r['score']:.3f} | {'y' if r['eligible'] else 'n'}{mark} |")

    path = Path(TUNING_DIR) / problem / "selected.json"
    path.write_text(json.dumps({
        "problem": problem, "tuning_digest": digests.pop(),
        "rule": {"sr_target": SR_TARGET, "sr_slack": SR_SLACK,
                 "score": "mean of log median-SWD/floor and log median-MMD/floor",
                 "tie_break": "lower cost"},
        "selected": selected, "tables": tables}, indent=1))
    print(f"\nwrote {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune sampler hyperparameters per method.")
    parser.add_argument("--problem", required=True, choices=list(PROBLEM_NAMES))
    parser.add_argument("--task-id", type=int, default=None,
                        help="index into grid(problem); one configuration per array task")
    parser.add_argument("--select", action="store_true",
                        help="apply the selection rule to finished tasks and write selected.json")
    parser.add_argument("--list", action="store_true", help="print the grid and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for task_id, (method, settings) in enumerate(grid(args.problem)):
            print(task_id, method, settings)
        return 0
    if args.select:
        select(args.problem)
        return 0
    if args.task_id is None:
        raise ValueError("pass --task-id, --select or --list")
    run_task(args.problem, args.task_id, resolve_device())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
