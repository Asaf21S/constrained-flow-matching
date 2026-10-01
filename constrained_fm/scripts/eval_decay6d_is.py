# -*- coding: utf-8 -*-
r"""Importance-sampling evaluation of ``E[f(p2) | p1 in B]`` on the fixed decay6d boxes.

The box-conditioned flow ``q`` is the proposal and ``w = p 1_B / q`` corrects it, with ``p`` the
unconstrained flow (learned) or the quadrature density (exact). Estimators per box and ``N``:

    q_raw        mean of f over all q samples, violators included
    q_filtered   mean of f over the in-box q samples
    is_learned   self-normalized IS with p = p_uncon
    is_exact     self-normalized IS with the exact p
    rej_equal_n / rej_equal_time / rej_equal_nfe
                 rejection from p_uncon with N samples, or with the sample count whose cost
                 matches is_learned in wall-clock or in network evaluations

``(1/N) sum w`` estimates ``P(B)`` and checks that both flows share one normalization.

The work is a SLURM array. Tasks ``[0, boxes * blocks_per_box)`` each draw one block of ``q``
samples for one box; the next ``uncon_blocks`` tasks draw ``p_uncon`` samples and record per
minibatch, per box in-box counts and sums of f. ``--stage merge`` forms the estimates from
disjoint slices of those blocks and writes metrics and plotting artifacts.

    python -m constrained_fm.scripts.eval_decay6d_is --stage shard --task-id 0
    python -m constrained_fm.scripts.eval_decay6d_is --stage merge
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from constrained_fm.scripts.train_decay6d_fm import build_model, freeze_fp64
from constrained_fm.src.consts import DECAY_ODE_ATOL, DECAY_ODE_FALLBACK_STEPS, DECAY_ODE_RTOL
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.problems.decay6d import (OBSERVABLE_NAMES, PARTICLE_DIM, BoxConstraint,
                                                 DecayProblem, boxes_contain, observables)
from constrained_fm.src.solvers import cnf

ROOT = "constrained_fm/baselines/decay6d_is"
SMOKE_ROOT = "constrained_fm/baselines/decay6d_is/smoke"
SHARDS_DIR = "shards"
ESTIMATORS = ("q_raw", "q_filtered", "is_learned", "is_exact",
              "rej_equal_n", "rej_equal_time", "rej_equal_nfe")
WEIGHTS = ("learned", "exact")
_EVAL_UNTRACKED = frozenset({"stage", "task_id"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="decay6d CFM + importance sampling evaluation.")
    parser.add_argument("--stage", choices=("shard", "merge"), required=True)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--box-ckpt", default=None)
    parser.add_argument("--uncon-ckpt", default=None)
    parser.add_argument("--boxes", default=None)
    parser.add_argument("--blocks-per-box", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=500_000)
    parser.add_argument("--chunk", type=int, default=10_000,
                        help="ODE batch for every solve, q and p_uncon alike")
    parser.add_argument("--uncon-blocks", type=int, default=8)
    parser.add_argument("--uncon-block-size", type=int, default=10_000_000)
    parser.add_argument("--minibatch", type=int, default=1000,
                        help="granularity at which rejection counts are stored")
    parser.add_argument("--calibration-chunks", type=int, default=5,
                        help="p_uncon sampling chunks timed inside every IS task")
    parser.add_argument("--n-values", type=int, nargs="+", default=[1000, 10_000, 100_000])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--plot-cap", type=int, default=500_000)
    parser.add_argument("--atol", type=float, default=DECAY_ODE_ATOL)
    parser.add_argument("--rtol", type=float, default=DECAY_ODE_RTOL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    root = SMOKE_ROOT if args.smoke else ROOT
    if args.smoke:
        args.blocks_per_box, args.block_size, args.chunk = 1, 2000, 1000
        args.uncon_blocks, args.uncon_block_size, args.calibration_chunks = 1, 20_000, 1
        args.n_values, args.reps, args.plot_cap = [1000], 2, 2000
    args.box_ckpt = args.box_ckpt or f"{root}/box/ckpt.pt"
    args.uncon_ckpt = args.uncon_ckpt or f"{root}/uncon/ckpt.pt"
    args.boxes = args.boxes or f"{root}/benchmark/boxes.json"
    args.outdir = args.outdir or f"{root}/eval"
    if args.block_size % args.chunk or args.uncon_block_size % args.chunk \
            or args.chunk % args.minibatch:
        raise ValueError("block sizes must be multiples of --chunk, --chunk of --minibatch")
    return args


def load_checkpoint(path: str, problem: DecayProblem, device) -> tuple[torch.nn.Module, str]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    std = torch.tensor(ckpt["normalizer_std"], dtype=torch.float64)
    if not torch.allclose(std, problem.normalizer(torch.float64).std):
        raise ValueError(f"{path} was trained with a different normalizer")
    model = build_model(ckpt["mode"], ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    return freeze_fp64(model.to(device)), ckpt["run_id"]


def shard_path(out: Path, task_id: int, num_is_tasks: int, blocks_per_box: int) -> Path:
    if task_id < num_is_tasks:
        box, block = divmod(task_id, blocks_per_box)
        return out / SHARDS_DIR / f"is_box{box}_block{block}.npz"
    return out / SHARDS_DIR / f"uncon_block{task_id - num_is_tasks}.npz"


def _uncon_chunk(model, n: int, generator, device, atol: float, rtol: float):
    x0 = torch.randn(n, 6, device=device, dtype=torch.float64, generator=generator)
    return cnf.sample_isolating(model, x0, None, atol, rtol, DECAY_ODE_FALLBACK_STEPS)


def _timed_log_prob(target, x: torch.Tensor) -> tuple[torch.Tensor, float]:
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    start = time.perf_counter()
    log_p = target.log_prob(x)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    return log_p, time.perf_counter() - start


def run_is_task(args, box: dict, box_model, uncon_model, problem, generator, device) -> dict:
    target = problem.target()
    normalizer = problem.normalizer(torch.float64).to(device)
    log_det = normalizer.log_det_forward.item()
    constraint = BoxConstraint(box["lo"], box["hi"])
    cond = {"box": torch.tensor(box["conditioning"], device=device, dtype=torch.float64)[None]}

    fields = {k: [] for k in ("log_q", "log_p_learned", "log_p_exact", "inside", "f", "p2",
                              "q_fallback", "p_fallback")}
    timing = {k: [] for k in ("q_seconds", "q_nfe", "p_seconds", "p_nfe", "exact_seconds")}
    for _ in range(args.block_size // args.chunk):
        x0 = torch.randn(args.chunk, 6, device=device, dtype=torch.float64, generator=generator)
        x_n, log_q, q_stats, q_fallback = cnf.sample_with_log_prob_isolating(
            box_model, x0, cond, args.atol, args.rtol, DECAY_ODE_FALLBACK_STEPS)
        log_p, p_stats, fallback = cnf.log_prob_isolating(uncon_model, x_n, args.atol, args.rtol,
                                                          DECAY_ODE_FALLBACK_STEPS)
        x = normalizer.inverse(x_n)
        log_exact, exact_seconds = _timed_log_prob(target, x)
        if fallback.any():
            print(f"p_uncon fallback on {int(fallback.sum())} samples, "
                  f"{int((fallback & constraint.contains(x)).sum())} in box", flush=True)

        fields["log_q"].append(log_q + log_det)
        fields["log_p_learned"].append(log_p + log_det)
        fields["log_p_exact"].append(log_exact)
        fields["inside"].append(constraint.contains(x))
        fields["p_fallback"].append(fallback)
        fields["q_fallback"].append(q_fallback)
        fields["f"].append(observables(x, box["tail_threshold"]))
        fields["p2"].append(x[:, PARTICLE_DIM:].float())
        timing["q_seconds"].append(q_stats.seconds)
        timing["q_nfe"].append(q_stats.nfe)
        timing["p_seconds"].append(p_stats.seconds)
        timing["p_nfe"].append(p_stats.nfe)
        timing["exact_seconds"].append(exact_seconds)

    calib = [_uncon_chunk(uncon_model, args.chunk, generator, device, args.atol, args.rtol)[1]
             for _ in range(args.calibration_chunks)]
    out = {k: torch.cat(v).cpu().numpy() for k, v in fields.items()}
    out.update({k: np.asarray(v) for k, v in timing.items()})
    out["calib_seconds"] = np.asarray([s.seconds for s in calib])
    out["calib_nfe"] = np.asarray([s.nfe for s in calib])
    return out


def run_uncon_task(args, boxes: list[dict], uncon_model, problem, generator, device) -> dict:
    normalizer = problem.normalizer(torch.float64).to(device)
    lo = torch.tensor([b["lo"] for b in boxes], device=device, dtype=torch.float64)
    hi = torch.tensor([b["hi"] for b in boxes], device=device, dtype=torch.float64)
    per_chunk = args.chunk // args.minibatch

    counts, sums, seconds, nfe, fallbacks = [], [], [], [], []
    for _ in range(args.uncon_block_size // args.chunk):
        x_n, stats, fallback = _uncon_chunk(uncon_model, args.chunk, generator, device,
                                            args.atol, args.rtol)
        x = normalizer.inverse(x_n)
        inside = boxes_contain(x[:, None, :PARTICLE_DIM], lo, hi).to(x.dtype)
        counts.append(inside.view(per_chunk, args.minibatch, -1).sum(1))
        # where, not a product: out-of-box fallback samples may be NaN and 0 * NaN = NaN.
        sums.append(torch.stack([
            torch.where(inside[:, b, None] > 0, observables(x, box["tail_threshold"]), 0.0)
            .view(per_chunk, args.minibatch, -1).sum(1) for b, box in enumerate(boxes)], dim=1))
        seconds.append(stats.seconds)
        nfe.append(stats.nfe)
        fallbacks.append(fallback.sum().item())
    return {"counts": torch.cat(counts).cpu().numpy(), "sums": torch.cat(sums).cpu().numpy(),
            "seconds": np.asarray(seconds), "nfe": np.asarray(nfe),
            "fallbacks": np.asarray(fallbacks)}


def run_shard(args, run_id: str, out: Path, device) -> None:
    problem = DecayProblem()
    bench = json.loads(Path(args.boxes).read_text())
    boxes = bench["boxes"]
    num_is = len(boxes) * args.blocks_per_box
    if args.task_id is None or not 0 <= args.task_id < num_is + args.uncon_blocks:
        raise ValueError(f"--task-id must lie in [0, {num_is + args.uncon_blocks})")

    uncon_model, _ = load_checkpoint(args.uncon_ckpt, problem, device)
    generator = torch.Generator(device=device).manual_seed(args.seed + args.task_id)
    if args.task_id < num_is:
        box_model, _ = load_checkpoint(args.box_ckpt, problem, device)
        payload = run_is_task(args, boxes[args.task_id // args.blocks_per_box], box_model,
                              uncon_model, problem, generator, device)
    else:
        payload = run_uncon_task(args, boxes, uncon_model, problem, generator, device)

    path = shard_path(out, args.task_id, num_is, args.blocks_per_box)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, run_id=np.asarray(run_id), **payload)
    print(f"task {args.task_id} -> {path}")


def _load_shard(path: Path, run_id: str) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing shard {path}; rerun that array task")
    with np.load(path) as data:
        shard = {k: data[k] for k in data.files}
    if str(shard.pop("run_id")) != run_id:
        raise ValueError(f"{path} belongs to a different run than {run_id}")
    return shard


def _weight_stats(log_w: np.ndarray, f: np.ndarray, n: int) -> dict[str, float | np.ndarray]:
    """SNIS estimate, ``(1/N) sum w``, ESS / N and max normalized weight of one subset."""
    finite = np.isfinite(log_w)
    if not finite.any():
        return {"estimate": np.full(f.shape[1], np.nan), "mass": 0.0, "ess_frac": 0.0,
                "max_weight": np.nan}
    shift = log_w[finite].max()
    w = np.where(finite, np.exp(log_w - shift), 0.0)
    total = w.sum()
    f = np.where(w[:, None] > 0, f, 0.0)
    return {"estimate": w @ f / total, "mass": math.exp(shift) * total / n,
            "ess_frac": total ** 2 / (w @ w) / n, "max_weight": w.max() / total}


def _cumulative(per_chunk: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(per_chunk, dtype=np.float64)])


def _span_cost(cum: np.ndarray, start: int, stop: int, chunk: int) -> float:
    """Cost of samples ``[start, stop)``, pro-rating chunks that are only partly used."""
    grid = np.arange(cum.size)
    return float(np.interp(stop / chunk, grid, cum) - np.interp(start / chunk, grid, cum))


def _summary(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.nanmean(values)), "median": float(np.nanmedian(values)),
            "std": float(np.nanstd(values))}


def _triple(stats: dict[str, float], fmt: str) -> str:
    return " / ".join(format(stats[k], fmt) for k in ("mean", "median", "std"))


def markdown_tables(report: dict) -> str:
    """One combined cost and accuracy table per box, with values ordered by N."""
    n_values = report["n_values"]
    n_order = "/".join(f"{n:,}" for n in n_values)

    def triplet(values, fmt: str) -> str:
        return "<br>".join(format(value, fmt).replace("e-", "e&#8209;") for value in values)

    lines = [
        "# Decay6D per-box accuracy and cost",
        "",
        f"Each box has one combined table. Its values are ordered by $N={n_order}$, one value per line within each cell.",
        "One repetition uses $N$ proposal samples or its listed rejection draw budget to produce",
        "one estimate. Each accuracy cell gives the standard deviation of absolute error across",
        "valid repetitions; RMSE is one value per box, estimator, and observable, computed across",
        "those repetitions. `valid reps` gives finite estimates out of 20, ordered by observable",
        "(norm / z / tail) and then by N (one line per N). A repetition is valid for an observable",
        "only if its estimate is finite. Raw $q$ includes every proposal, so a non-finite fallback",
        "output can invalidate its norm and z estimates; the tail indicator can remain numerically",
        "finite because a NaN threshold comparison is false. Thus valid means finite, not",
        "necessarily fallback-free. Filtered $q$ and IS exclude leaked proposals.",
        "",
        "Cost columns show mean values only, except draw/evaluation budgets which are fixed per",
        "repetition. `q draws` is the number from the box-conditioned proposal; `p_uncon draws`",
        "is the number from the unconstrained model. Density evals are learned / exact. Samples",
        "used are all proposals for raw $q$, in-box proposals for filtered $q$ and IS, and accepted",
        "events for rejection. NFE counts velocity-network calls. Exact IS adds quadrature time but",
        "no extra network NFE; all times include the estimator's density work.",
        "",
        "Equal-time and equal-NFE draw budgets are calibrated from median per-chunk costs, then",
        "rounded to 1,000-draw minibatches. The table reports mean realized costs, which need not",
        "match exactly: adaptive solver work and fallback trajectories vary between repetitions.",
        "At $N=100{,}000$, equal-time windows contain 1.4--1.9 million draws; the observed",
        "60 fallbacks in 80 million draws imply about 1.1--1.5 fallback trajectories per such",
        "window on average, which can raise realized cost above the median-chunk target.",
        "Timing/NFE costs for partial 10,000-sample chunks are prorated by sample count.",
        "",
        "#### Summary figures",
        "",
        "RMSE at $N=1,000$ against $P(\\mathcal B)$ for $\\lVert\\vec p_2\\rVert$, $p_{2z}$, and the tail",
        "probability; the mass axis is reversed, so boxes become rarer to the right.",
        "",
        *(f"![RMSE vs constraint mass, {obs}](../../../images/thesis_pool/decay6d_is/"
          f"rmse_vs_mass_{obs}_n1000.png)" for obs in OBSERVABLE_NAMES),
        "",
        f"small_offcentre error-cost frontier at IS budgets $N={n_order}$, using mean time per estimate;",
        "point labels give each estimator's own model draws per estimate (rejection draws far more",
        "than $N$ to match IS time or NFE). Means include occasional fallback trajectories.",
        "",
        "![RMSE vs time, small_offcentre](../../../images/thesis_pool/decay6d_is/"
        "rmse_vs_time_small_offcentre_p2_norm.png)",
        "",
    ]

    for name, box in report["boxes"].items():
        per_n = [box["by_n"][str(n)] for n in n_values]
        lines += [f"#### {name}: $P(\\mathcal B)={box['gt_mass']:.4f}$, "
                  f"GT events={box['gt_count']:,}", "",
                  "| estimator | $q$ draws | $p_{\\rm uncon}$ draws | density evals (learned/exact) "
                  "| samples used (mean) | NFE (mean) | time [s] (mean) | valid reps (norm/z/tail per N) "
                  "| $\\lVert\\vec p_2\\rVert$ abs-error std | $\\lVert\\vec p_2\\rVert$ RMSE "
                  "| $p_{2z}$ abs-error std | $p_{2z}$ RMSE "
                  "| tail abs-error std | tail RMSE |",
                  "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
                  "| ---: | ---: | ---: | ---: |"]
        for estimator in ESTIMATORS:
            costs = [entry["cost"][estimator] for entry in per_n]
            scores = [entry["estimators"][estimator] for entry in per_n]
            q_draws = triplet([c["q_draws"] for c in costs], ",")
            p_draws = triplet([c["uncon_draws"] for c in costs], ",")
            density_evals = "<br>".join(
                f"{c['learned_density_evals']:,} / {c['exact_density_evals']:,}" for c in costs)
            samples = triplet([c["used"]["mean"] for c in costs], ",.0f")
            nfe = triplet([c["nfe"]["mean"] for c in costs], ",.0f")
            seconds = triplet([c["seconds"]["mean"] for c in costs], ",.1f")
            reps = "<br>".join(" / ".join(str(s[k]["valid_reps"])
                                         for k in OBSERVABLE_NAMES) for s in scores)
            cells = []
            for observable in OBSERVABLE_NAMES:
                cells.extend([
                    triplet([s[observable]["abs_err"]["std"] for s in scores], ".2e"),
                    triplet([s[observable]["rmse"] for s in scores], ".2e"),
                ])
            lines.append("| " + " | ".join(
                [estimator, q_draws, p_draws, density_evals, samples, nfe, seconds, reps] + cells
            ) + " |")
        lines.append("")
    return "\n".join(lines)


def merge(args, run_id: str, out: Path) -> None:
    bench = json.loads(Path(args.boxes).read_text())
    boxes = bench["boxes"]
    n_values = sorted(args.n_values)
    num_b, num_f, num_n, reps = len(boxes), len(OBSERVABLE_NAMES), len(n_values), args.reps
    num_is = num_b * args.blocks_per_box

    uncon = [_load_shard(shard_path(out, num_is + k, num_is, args.blocks_per_box), run_id)
             for k in range(args.uncon_blocks)]
    rej_counts = np.concatenate([s["counts"] for s in uncon])
    rej_sums = np.concatenate([s["sums"] for s in uncon])
    uncon_nfe = float(np.median(np.concatenate([s["nfe"] for s in uncon])))
    uncon_fallbacks = int(np.sum([s.get("fallbacks", np.zeros(1)).sum() for s in uncon]))
    rej_cum = {k: _cumulative(np.concatenate([s[k] for s in uncon])) for k in ("seconds", "nfe")}

    estimates = np.full((num_b, len(ESTIMATORS), num_f, num_n, reps), np.nan)
    mass = np.full((num_b, len(WEIGHTS), num_n, reps), np.nan)
    ess_frac = np.full_like(mass, np.nan)
    max_weight = np.full_like(mass, np.nan)
    rej_budget = np.zeros((num_b, 3, num_n), dtype=np.int64)
    rej_reps = np.zeros_like(rej_budget)
    rep_seconds = np.full((num_b, len(ESTIMATORS), num_n, reps), np.nan)
    rep_nfe = np.full_like(rep_seconds, np.nan)
    rep_used = np.full_like(rep_seconds, np.nan)
    gt_mean = np.array([[b["gt"][k]["mean"] for k in OBSERVABLE_NAMES] for b in boxes])
    gt_se = np.array([[b["gt"][k]["se"] for k in OBSERVABLE_NAMES] for b in boxes])
    box_report, plot_arrays = {}, {}

    for b, box in enumerate(boxes):
        blocks = [_load_shard(out / SHARDS_DIR / f"is_box{b}_block{k}.npz", run_id)
                  for k in range(args.blocks_per_box)]
        data = {k: np.concatenate([blk[k] for blk in blocks])
            for k in ("log_q", "log_p_learned", "log_p_exact", "inside", "f", "p2",
                  "p_fallback")}
        inside, f = data["inside"], data["f"]
        p_fallback = data["p_fallback"]
        q_fallback = np.concatenate([blk.get("q_fallback", np.zeros(blk["inside"].shape,
                                          dtype=bool))
                         for blk in blocks])
        log_w = {"learned": np.where(inside, data["log_p_learned"] - data["log_q"], -np.inf),
                 "exact": np.where(inside, data["log_p_exact"] - data["log_q"], -np.inf)}

        timing = {k: np.concatenate([blk[k] for blk in blocks]) for k in
                  ("q_seconds", "q_nfe", "p_seconds", "p_nfe", "exact_seconds", "calib_seconds",
                   "calib_nfe")}
        # Median per chunk: one stalled solve (see cnf.log_prob_isolating) must not set the cost.
        med = {k: float(np.median(v)) for k, v in timing.items()}
        is_seconds = (med["q_seconds"] + med["p_seconds"]) / args.chunk
        is_nfe = med["q_nfe"] + med["p_nfe"]
        rej_seconds = med["calib_seconds"] / args.chunk
        cost_ratio = {"equal_n": 1.0, "equal_time": is_seconds / rej_seconds,
                      "equal_nfe": is_nfe / med["calib_nfe"]}
        cum = {k: _cumulative(timing[k]) for k in
               ("q_seconds", "q_nfe", "p_seconds", "p_nfe", "exact_seconds")}

        for j, n in enumerate(n_values):
            for r in range(min(reps, inside.shape[0] // n)):
                sl = slice(r * n, (r + 1) * n)
                span = {k: _span_cost(c, sl.start, sl.stop, args.chunk) for k, c in cum.items()}
                rep_seconds[b, :4, j, r] = [span["q_seconds"], span["q_seconds"],
                                            span["q_seconds"] + span["p_seconds"],
                                            span["q_seconds"] + span["exact_seconds"]]
                rep_nfe[b, :4, j, r] = [span["q_nfe"], span["q_nfe"],
                                        span["q_nfe"] + span["p_nfe"], span["q_nfe"]]
                rep_used[b, 0, j, r] = n
                rep_used[b, 1:4, j, r] = inside[sl].sum()
                estimates[b, 0, :, j, r] = f[sl].mean(0)
                if inside[sl].any():
                    estimates[b, 1, :, j, r] = f[sl][inside[sl]].mean(0)
                for w_idx, name in enumerate(WEIGHTS):
                    stats = _weight_stats(log_w[name][sl], f[sl], n)
                    estimates[b, 2 + w_idx, :, j, r] = stats["estimate"]
                    mass[b, w_idx, j, r] = stats["mass"]
                    ess_frac[b, w_idx, j, r] = stats["ess_frac"]
                    max_weight[b, w_idx, j, r] = stats["max_weight"]

            for v, ratio in enumerate(cost_ratio.values()):
                k = max(1, round(n * ratio / args.minibatch))
                rej_budget[b, v, j] = k * args.minibatch
                rej_reps[b, v, j] = min(reps, rej_counts.shape[0] // k)
                for r in range(rej_reps[b, v, j]):
                    count = rej_counts[r * k:(r + 1) * k, b].sum()
                    draws = (r * k * args.minibatch, (r + 1) * k * args.minibatch)
                    rep_seconds[b, 4 + v, j, r] = _span_cost(rej_cum["seconds"], *draws, args.chunk)
                    rep_nfe[b, 4 + v, j, r] = _span_cost(rej_cum["nfe"], *draws, args.chunk)
                    rep_used[b, 4 + v, j, r] = count
                    if count > 0:
                        estimates[b, 4 + v, :, j, r] = rej_sums[r * k:(r + 1) * k, b].sum(0) / count

        gap = (data["log_p_learned"] - data["log_p_exact"])[inside]
        finite_gap = gap[np.isfinite(gap)]
        box_report[box["name"]] = {
            "gt_mass": box["gt_mass"],
            "gt_count": box["gt_count"],
            "leakage": float(1.0 - inside.mean()),
            "p_uncon_fallback": {"total": int(p_fallback.sum()),
                                 "in_box": int((p_fallback & inside).sum())},
            "q_fallback": int(q_fallback.sum()),
            "log_p_uncon_minus_exact_on_q": {
                "mean": float(finite_gap.mean()) if finite_gap.size else float("nan"),
                "std": float(finite_gap.std()) if finite_gap.size else float("nan"),
                "p1": float(np.percentile(finite_gap, 1)) if finite_gap.size else float("nan"),
                "p99": float(np.percentile(finite_gap, 99)) if finite_gap.size else float("nan"),
                "finite_samples": int(finite_gap.size),
                "total_samples": int(gap.size)},
            "cost_ratio_vs_rejection": cost_ratio,
            "seconds_per_sample": {"is_learned": float(is_seconds),
                                   "exact_density": med["exact_seconds"] / args.chunk,
                                   "uncon_sample": float(rej_seconds)},
            "nfe_per_chunk": {"q": med["q_nfe"], "p_uncon_backward": med["p_nfe"],
                              "p_uncon_sample": med["calib_nfe"]},
        }

        cap = min(args.plot_cap, inside.shape[0])
        plot_arrays.update({f"q_p2_box{b}": data["p2"][:cap], f"q_inside_box{b}": inside[:cap],
                            f"log_w_learned_box{b}": log_w["learned"][:cap],
                            f"log_w_exact_box{b}": log_w["exact"][:cap]})

    err = estimates - gt_mean[:, None, :, None, None]
    bias = np.nanmean(err, axis=-1)
    rmse = np.sqrt(np.nanmean(err ** 2, axis=-1))
    spread = np.nanstd(estimates, axis=-1)
    valid = np.isfinite(estimates).sum(-1)

    for b, box in enumerate(boxes):
        per_n = {}
        for j, n in enumerate(n_values):
            per_n[str(n)] = {
                "estimators": {e: {fname: {"bias": float(bias[b, i, k, j]),
                                           "std": float(spread[b, i, k, j]),
                                           "rmse": float(rmse[b, i, k, j]),
                                           "abs_err": _summary(np.abs(err[b, i, k, j])),
                                           "valid_reps": int(valid[b, i, k, j])}
                                   for k, fname in enumerate(OBSERVABLE_NAMES)}
                               for i, e in enumerate(ESTIMATORS)},
                "cost": {e: {"q_draws": n if i < 4 else 0,
                              "uncon_draws": int(rej_budget[b, i - 4, j]) if i >= 4 else 0,
                              "learned_density_evals": n if e == "is_learned" else 0,
                              "exact_density_evals": n if e == "is_exact" else 0,
                              "used": _summary(rep_used[b, i, j]),
                              "nfe": _summary(rep_nfe[b, i, j]),
                              "seconds": _summary(rep_seconds[b, i, j])}
                         for i, e in enumerate(ESTIMATORS)},
                "rejection_budget": dict(zip(("equal_n", "equal_time", "equal_nfe"),
                                             rej_budget[b, :, j].tolist())),
                "mass": {w: {"mean": float(np.nanmean(mass[b, i, j])),
                             "std": float(np.nanstd(mass[b, i, j]))}
                         for i, w in enumerate(WEIGHTS)},
                "ess_frac": {w: float(np.nanmean(ess_frac[b, i, j])) for i, w in enumerate(WEIGHTS)},
                "max_weight": {w: float(np.nanmean(max_weight[b, i, j]))
                               for i, w in enumerate(WEIGHTS)},
            }
        box_report[box["name"]]["by_n"] = per_n

    artifacts.save_arrays(out, estimates=estimates, gt_mean=gt_mean, gt_se=gt_se,
                          n_values=np.asarray(n_values), mass=mass, ess_frac=ess_frac,
                          max_weight=max_weight, rej_budget=rej_budget, rej_reps=rej_reps,
                          rep_seconds=rep_seconds, rep_nfe=rep_nfe, rep_used=rep_used,
                          gt_mass=np.array([b["gt_mass"] for b in boxes]), **plot_arrays)
    artifacts.write_manifest(out, run_id=run_id, estimators=list(ESTIMATORS),
                             weights=list(WEIGHTS), observables=list(OBSERVABLE_NAMES),
                             boxes=[b["name"] for b in boxes],
                             benchmark_dir=str(Path(args.boxes).parent))

    report = {
        "run_id": run_id,
        "benchmark_run_id": bench["run_id"],
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "estimators": list(ESTIMATORS),
        "observables": list(OBSERVABLE_NAMES),
        "n_values": n_values,
        "reps": reps,
        "uncon_pool_samples": int(rej_counts.shape[0] * args.minibatch),
        "uncon_nfe_per_chunk": uncon_nfe,
        "uncon_sample_fallbacks": uncon_fallbacks,
        "boxes": box_report,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    (out / "tables.md").write_text(markdown_tables(report))

    print(f"### decay6d IS ({run_id})")
    for b, box in enumerate(boxes):
        j = num_n - 1
        row = "  ".join(f"{e}={rmse[b, i, 0, j]:.2e}" for i, e in enumerate(ESTIMATORS))
        print(f"  {box['name']:16s} N={n_values[j]} RMSE |p2|: {row}")


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
    warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
    args = resolve_args(build_parser().parse_args(argv))
    device = resolve_device()
    out = Path(args.outdir)

    extra = {"box_ckpt_run_id": torch.load(args.box_ckpt, map_location="cpu",
                                           weights_only=True)["run_id"],
             "uncon_ckpt_run_id": torch.load(args.uncon_ckpt, map_location="cpu",
                                             weights_only=True)["run_id"],
             "benchmark_run_id": json.loads(Path(args.boxes).read_text())["run_id"]}
    settings = {k: v for k, v in vars(args).items() if k not in _EVAL_UNTRACKED}
    run_id = pin_baseline_run(out, "decay6d_is", settings, extra)

    if args.stage == "shard":
        run_shard(args, run_id, out, device)
    else:
        merge(args, run_id, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
