# -*- coding: utf-8 -*-
r"""Importance-sampling evaluation of ``E[f(p2) | p1 in B]`` on the fixed decay6d boxes.

The box-conditioned flow ``q`` is the proposal and ``w = p 1_B / q`` corrects it, with ``p`` the
unconstrained flow (learned) or the quadrature density (exact). Every ODE solve uses ``--steps``
fixed midpoint steps. Estimators per box, ``N`` and repetition:

    q_raw           mean of f over all q samples, violators included
    q_filtered      mean of f over the in-box q samples
    is_learned      self-normalized IS with p = p_uncon
    is_exact        self-normalized IS with the exact p
    rej_equal_n     rejection from p_uncon with N draws
    rej_equal_time  rejection from p_uncon for the measured wall time of this is_learned repetition
    rej_equal_nfe   rejection from p_uncon until its network calls match this is_learned repetition

``(1/N) sum w`` estimates ``P(B)`` and checks that both flows share one normalization.

One SLURM array task per box runs every repetition, IS and rejection alternating in one process on
one GPU. ``--stage merge`` collects the per-box shards into metrics, tables and plotting artifacts
under ``eval/steps<S>``.

    python -m constrained_fm.scripts.eval_decay6d_is --stage shard --task-id 0 --steps 32
    python -m constrained_fm.scripts.eval_decay6d_is --stage merge --steps 32
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
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.problems.decay6d import (OBSERVABLE_NAMES, PARTICLE_DIM, BoxConstraint,
                                                 DecayProblem, observables)
from constrained_fm.src.solvers import cnf

ROOT = "constrained_fm/baselines/decay6d_is"
SMOKE_ROOT = "constrained_fm/baselines/decay6d_is/smoke"
SHARDS_DIR = "shards"
ESTIMATORS = ("q_raw", "q_filtered", "is_learned", "is_exact",
              "rej_equal_n", "rej_equal_time", "rej_equal_nfe")
WEIGHTS = ("learned", "exact")
DEFAULT_STEPS = 32
CALIBRATION_REPEATS = 3
_EVAL_UNTRACKED = frozenset({"stage", "task_id"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="decay6d CFM + importance sampling evaluation.")
    parser.add_argument("--stage", choices=("shard", "merge"), required=True)
    parser.add_argument("--task-id", type=int, default=None, help="box index")
    parser.add_argument("--box-ckpt", default=None)
    parser.add_argument("--uncon-ckpt", default=None)
    parser.add_argument("--boxes", default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="fixed midpoint steps for every ODE solve")
    parser.add_argument("--chunk", type=int, default=10_000,
                        help="largest ODE batch, q and p_uncon alike")
    parser.add_argument("--n-values", type=int, nargs="+", default=[1000, 10_000, 100_000])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--plot-cap", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    root = SMOKE_ROOT if args.smoke else ROOT
    if args.smoke:
        args.chunk, args.n_values, args.reps, args.plot_cap = 1000, [1000, 3000], 2, 2000
    args.box_ckpt = args.box_ckpt or f"{root}/box/ckpt.pt"
    args.uncon_ckpt = args.uncon_ckpt or f"{root}/uncon/ckpt.pt"
    args.boxes = args.boxes or f"{root}/benchmark/boxes.json"
    args.outdir = args.outdir or f"{root}/eval/steps{args.steps}"
    return args


def load_checkpoint(path: str, problem: DecayProblem, device) -> tuple[torch.nn.Module, str]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    std = torch.tensor(ckpt["normalizer_std"], dtype=torch.float64)
    if not torch.allclose(std, problem.normalizer(torch.float64).std):
        raise ValueError(f"{path} was trained with a different normalizer")
    model = build_model(ckpt["mode"], ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    return freeze_fp64(model.to(device)), ckpt["run_id"]


def shard_path(out: Path, box_index: int) -> Path:
    return out / SHARDS_DIR / f"box{box_index}.npz"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pieces(n: int, chunk: int) -> list[int]:
    return [chunk] * (n // chunk) + ([n % chunk] if n % chunk else [])


def _timed_log_prob(target, x: torch.Tensor) -> tuple[torch.Tensor, float]:
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    start = time.perf_counter()
    log_p = target.log_prob(x)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    return log_p, time.perf_counter() - start


class BoxRunner:
    """Timed IS repetitions and p_uncon rejection runs for one box on one device."""

    def __init__(self, args, box: dict, box_model, uncon_model, problem, device) -> None:
        self.args, self.box_model, self.uncon_model, self.device = args, box_model, uncon_model, device
        self.target = problem.target()
        self.normalizer = problem.normalizer(torch.float64).to(device)
        self.log_det = self.normalizer.log_det_forward.item()
        self.constraint = BoxConstraint(box["lo"], box["hi"])
        self.tau = box["tail_threshold"]
        self.cond = {"box": torch.tensor(box["conditioning"], device=device,
                                         dtype=torch.float64)[None]}
        # Separate streams keep the IS draws independent of how many draws rejection consumed.
        seed = 2 * (1000 * args.seed + args.task_id)
        self.gen_q = torch.Generator(device=device).manual_seed(seed)
        self.gen_p = torch.Generator(device=device).manual_seed(seed + 1)
        self.sizes = self.costs = None

    def _noise(self, m: int, generator) -> torch.Tensor:
        return torch.randn(m, 6, device=self.device, dtype=torch.float64, generator=generator)

    def is_rep(self, n: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """N proposals with log q (forward), log p_uncon (backward) and the exact log p."""
        steps = self.args.steps
        parts = {k: [] for k in ("log_q", "log_p", "log_exact", "inside", "finite", "f", "p2")}
        cost = {"p_seconds": 0.0, "exact_seconds": 0.0, "q_nfe": 0, "p_nfe": 0}
        _sync(self.device)
        start = time.perf_counter()
        for m in _pieces(n, self.args.chunk):
            x_n, log_q, q_stats = cnf.sample_with_log_prob_fixed(
                self.box_model, self._noise(m, self.gen_q), steps, self.cond)
            log_p, p_stats = cnf.log_prob_fixed(self.uncon_model, x_n, steps)
            x = self.normalizer.inverse(x_n)
            log_exact, exact_seconds = _timed_log_prob(self.target, x)
            parts["log_q"].append(log_q + self.log_det)
            parts["log_p"].append(log_p + self.log_det)
            parts["log_exact"].append(log_exact)
            parts["inside"].append(self.constraint.contains(x))
            parts["finite"].append(torch.isfinite(x).all(1))
            parts["f"].append(observables(x, self.tau))
            parts["p2"].append(x[:, PARTICLE_DIM:].float())
            cost["p_seconds"] += p_stats.seconds
            cost["exact_seconds"] += exact_seconds
            cost["q_nfe"] += q_stats.nfe
            cost["p_nfe"] += p_stats.nfe
        _sync(self.device)
        cost["total"] = time.perf_counter() - start
        return {k: torch.cat(v).cpu().numpy() for k, v in parts.items()}, cost

    def draw_uncon(self, m: int) -> tuple[int, torch.Tensor, int, int]:
        """One p_uncon batch: in-box count, in-box sum of f, NFE, non-finite samples."""
        x_n, stats = cnf.sample_fixed(self.uncon_model, self._noise(m, self.gen_p),
                                      self.args.steps)
        x = self.normalizer.inverse(x_n)
        inside = self.constraint.contains(x)
        # where, not a product: a non-finite out-of-box sample would give 0 * NaN = NaN.
        f_sum = torch.where(inside[:, None], observables(x, self.tau), 0.0).sum(0)
        return int(inside.sum()), f_sum, stats.nfe, int((~torch.isfinite(x).all(1)).sum())

    def calibrate(self) -> None:
        """Median wall time of one p_uncon batch per size, used to size the final batch."""
        chunk = self.args.chunk
        self.sizes = np.unique([1, max(1, chunk // 100), max(1, chunk // 10), max(1, chunk // 4),
                                max(1, chunk // 2), chunk])
        costs = []
        for m in self.sizes:
            runs = []
            for _ in range(CALIBRATION_REPEATS):
                _sync(self.device)
                start = time.perf_counter()
                self.draw_uncon(int(m))
                _sync(self.device)
                runs.append(time.perf_counter() - start)
            costs.append(float(np.median(runs)))
        self.costs = np.maximum.accumulate(costs)

    def _fit(self, remaining: float) -> int:
        """Largest batch whose calibrated time fits in ``remaining`` seconds."""
        if remaining < self.costs[0]:
            return 0
        return int(np.interp(remaining, self.costs, self.sizes))

    def rejection(self, piece: int, *, draws: int | None = None, nfe: int | None = None,
                  seconds: float | None = None) -> tuple[np.ndarray, dict[str, float]]:
        """Rejection from p_uncon under one budget: draws, network calls, or wall time."""
        count = drawn = calls = nonfinite = 0
        f_sum = torch.zeros(len(OBSERVABLE_NAMES), device=self.device, dtype=torch.float64)
        _sync(self.device)
        start = time.perf_counter()
        while True:
            if draws is not None:
                m = min(piece, draws - drawn)
            elif nfe is not None:
                m = piece if calls < nfe else 0
            else:
                m = min(piece, self._fit(seconds - (time.perf_counter() - start)))
            if m <= 0:
                break
            c, s, k, bad = self.draw_uncon(m)
            count, drawn, calls, nonfinite = count + c, drawn + m, calls + k, nonfinite + bad
            f_sum += s
        _sync(self.device)
        elapsed = time.perf_counter() - start
        estimate = (f_sum / count).cpu().numpy() if count else np.full(len(OBSERVABLE_NAMES), np.nan)
        return estimate, {"used": count, "draws": drawn, "nfe": calls, "seconds": elapsed,
                          "nonfinite": nonfinite}


def run_box(args, runner: BoxRunner) -> dict[str, np.ndarray]:
    n_values = sorted(args.n_values)
    num_e, num_f, num_n, reps = len(ESTIMATORS), len(OBSERVABLE_NAMES), len(n_values), args.reps
    col = {e: i for i, e in enumerate(ESTIMATORS)}
    estimates = np.full((num_e, num_f, num_n, reps), np.nan)
    cost = {k: np.full((num_e, num_n, reps), np.nan) for k in ("seconds", "nfe", "used", "draws")}
    weight = {k: np.full((len(WEIGHTS), num_n, reps), np.nan)
              for k in ("mass", "ess_frac", "max_weight")}
    counts = {k: 0 for k in ("q_total", "q_in_box", "q_nonfinite", "p_uncon_nonfinite_in_box",
                             "rejection_nonfinite")}
    plot = {k: [] for k in ("q_p2", "q_inside", "log_w_learned", "log_w_exact")}
    plotted = 0

    runner.is_rep(min(n_values[0], args.chunk))  # warm-up: kernels, allocator, autograd graph
    runner.draw_uncon(args.chunk)
    runner.calibrate()

    for j, n in enumerate(n_values):
        start = time.perf_counter()
        for r in range(reps):
            s, t = runner.is_rep(n)
            inside, f = s["inside"], s["f"]
            log_w = {"learned": np.where(inside, s["log_p"] - s["log_q"], -np.inf),
                     "exact": np.where(inside, s["log_exact"] - s["log_q"], -np.inf)}
            estimates[col["q_raw"], :, j, r] = f.mean(0)
            if inside.any():
                estimates[col["q_filtered"], :, j, r] = f[inside].mean(0)
            for w, (kind, e) in enumerate(zip(WEIGHTS, ("is_learned", "is_exact"))):
                stats = _weight_stats(log_w[kind], f, n)
                estimates[col[e], :, j, r] = stats["estimate"]
                for k in weight:
                    weight[k][w, j, r] = stats[k]

            q_seconds = t["total"] - t["p_seconds"] - t["exact_seconds"]
            for e, sec, calls in (("q_raw", q_seconds, t["q_nfe"]),
                                  ("q_filtered", q_seconds, t["q_nfe"]),
                                  ("is_learned", t["total"] - t["exact_seconds"],
                                   t["q_nfe"] + t["p_nfe"]),
                                  ("is_exact", t["total"] - t["p_seconds"], t["q_nfe"])):
                cost["seconds"][col[e], j, r], cost["nfe"][col[e], j, r] = sec, calls
                cost["used"][col[e], j, r] = n if e == "q_raw" else inside.sum()
                cost["draws"][col[e], j, r] = n

            budgets = {"rej_equal_n": (args.chunk, {"draws": n}),
                       "rej_equal_time": (args.chunk, {"seconds": cost["seconds"][col["is_learned"], j, r]}),
                       "rej_equal_nfe": (min(n, args.chunk), {"nfe": t["q_nfe"] + t["p_nfe"]})}
            for e, (piece, budget) in budgets.items():
                estimates[col[e], :, j, r], c = runner.rejection(piece, **budget)
                for k in cost:
                    cost[k][col[e], j, r] = c[k]
                counts["rejection_nonfinite"] += c["nonfinite"]

            counts["q_total"] += n
            counts["q_in_box"] += int(inside.sum())
            counts["q_nonfinite"] += int((~s["finite"]).sum())
            counts["p_uncon_nonfinite_in_box"] += int((inside & ~np.isfinite(s["log_p"])).sum())
            if j == num_n - 1 and plotted < args.plot_cap:
                take = min(n, args.plot_cap - plotted)
                for k, v in (("q_p2", s["p2"]), ("q_inside", inside),
                             ("log_w_learned", log_w["learned"]), ("log_w_exact", log_w["exact"])):
                    plot[k].append(v[:take])
                plotted += take
        ratio = cost["seconds"][col["rej_equal_time"], j] / cost["seconds"][col["is_learned"], j]
        print(f"N={n}: {reps} reps in {time.perf_counter() - start:.1f}s, is_learned "
              f"{np.median(cost['seconds'][col['is_learned'], j]):.3f}s, equal-time ratio "
              f"{ratio.min():.3f}..{ratio.max():.3f}", flush=True)

    return {"estimates": estimates, **{f"rep_{k}": v for k, v in cost.items()}, **weight,
            **{k: np.asarray(v) for k, v in counts.items()},
            **{k: np.concatenate(v) for k, v in plot.items()},
            "calib_sizes": runner.sizes, "calib_seconds": runner.costs}


def run_shard(args, run_id: str, out: Path, device) -> None:
    problem = DecayProblem()
    boxes = json.loads(Path(args.boxes).read_text())["boxes"]
    if args.task_id is None or not 0 <= args.task_id < len(boxes):
        raise ValueError(f"--task-id must lie in [0, {len(boxes)})")

    box_model, _ = load_checkpoint(args.box_ckpt, problem, device)
    uncon_model, _ = load_checkpoint(args.uncon_ckpt, problem, device)
    box = boxes[args.task_id]
    print(f"box {args.task_id} {box['name']}: P(B)={box['gt_mass']:.4g}, steps={args.steps}",
          flush=True)
    payload = run_box(args, BoxRunner(args, box, box_model, uncon_model, problem, device))

    path = shard_path(out, args.task_id)
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


def _summary(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.nanmean(values)), "median": float(np.nanmedian(values)),
            "std": float(np.nanstd(values))}


def _triple(stats: dict[str, float], fmt: str) -> str:
    return " / ".join(format(stats[k], fmt) for k in ("mean", "median", "std"))


def markdown_tables(report: dict) -> str:
    """One combined cost and accuracy table per box, with values ordered by N."""
    n_values = report["n_values"]
    n_order = "/".join(f"{n:,}" for n in n_values)
    steps = report["steps"]
    figures = f"../../../../images/thesis_pool/decay6d_is/steps{steps}"
    rarest = min(report["boxes"], key=lambda name: report["boxes"][name]["gt_mass"])

    def triplet(values, fmt: str) -> str:
        return "<br>".join(format(value, fmt).replace("e-", "e&#8209;") for value in values)

    lines = [
        f"# Decay6D per-box accuracy and cost ({steps} midpoint steps)",
        "",
        f"Every ODE solve (q sampling with its log-density, the backward $p_{{\\rm uncon}}$ density,",
        f"and $p_{{\\rm uncon}}$ sampling for rejection) uses {steps} fixed midpoint steps, i.e.",
        f"{2 * steps} velocity calls per batch. Each box is one SLURM task on one GPU, and its IS",
        "and rejection repetitions alternate in the same process.",
        "",
        f"Each box has one combined table. Its values are ordered by $N={n_order}$, one value per line within each cell.",
        "Each accuracy cell gives the standard deviation of absolute error across valid",
        "repetitions; RMSE is computed across those repetitions. `valid reps` gives finite",
        f"estimates out of {report['reps']}, ordered by observable (norm / z / tail) and then by N.",
        "A rejection repetition with no accepted event has no estimate and is not valid.",
        "",
        "Cost columns show means over repetitions. `q draws` is the number from the box-conditioned",
        "proposal; `p_uncon draws` is the mean number from the unconstrained model. Samples used",
        "are all proposals for raw $q$, in-box proposals for filtered $q$ and IS, and accepted",
        "events for rejection. NFE counts velocity-network calls. Times are measured wall clock",
        "for each repetition and include the estimator's own density work only.",
        "",
        "Budgets are per repetition, not calibrated averages: `rej_equal_time` samples",
        "$p_{\\rm uncon}$ until the measured wall time of the same repetition's `is_learned` run is",
        "used up, sizing its last batch from the measured batch-time curve so it does not overrun;",
        "`rej_equal_nfe` draws batches of $\\min(N, 10^4)$ until its velocity calls reach the",
        "`is_learned` count, which is $2N$ draws; `rej_equal_n` draws exactly $N$.",
        "",
        "#### Summary figures",
        "",
        "RMSE against $P(\\mathcal B)$; rows are $\\Vert\\vec p_2\\Vert$, $p_{2z}$ and the tail",
        "probability, columns are $N$. The mass axis is reversed, so boxes become rarer to the right.",
        "",
        f"![RMSE vs constraint mass]({figures}/rmse_vs_mass_grid.png)",
        "",
        f"{rarest} error-cost frontier at IS budgets $N={n_order}$, using mean time per estimate;",
        "point labels give each estimator's own model draws per estimate.",
        "",
        f"![RMSE vs time, {rarest}]({figures}/rmse_vs_time_{rarest}_p2_norm.png)",
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
            p_draws = triplet([c["uncon_draws"] for c in costs], ",.0f")
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
    shards = [_load_shard(shard_path(out, b), run_id) for b in range(len(boxes))]
    estimates = np.stack([s["estimates"] for s in shards])
    rep = {k: np.stack([s[f"rep_{k}"] for s in shards]) for k in ("seconds", "nfe", "used", "draws")}
    mass, ess_frac, max_weight = (np.stack([s[k] for s in shards])
                                  for k in ("mass", "ess_frac", "max_weight"))
    gt_mean = np.array([[b["gt"][k]["mean"] for k in OBSERVABLE_NAMES] for b in boxes])
    gt_se = np.array([[b["gt"][k]["se"] for k in OBSERVABLE_NAMES] for b in boxes])
    col = {e: i for i, e in enumerate(ESTIMATORS)}

    err = estimates - gt_mean[:, None, :, None, None]
    bias = np.nanmean(err, axis=-1)
    rmse = np.sqrt(np.nanmean(err ** 2, axis=-1))
    spread = np.nanstd(estimates, axis=-1)
    valid = np.isfinite(estimates).sum(-1)

    box_report, plot_arrays = {}, {}
    for b, (box, shard) in enumerate(zip(boxes, shards)):
        inside = shard["q_inside"]
        gap = (shard["log_w_learned"] - shard["log_w_exact"])[inside]
        finite_gap = gap[np.isfinite(gap)]
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
                              "uncon_draws": float(np.mean(rep["draws"][b, i, j])) if i >= 4 else 0,
                              "learned_density_evals": n if e == "is_learned" else 0,
                              "exact_density_evals": n if e == "is_exact" else 0,
                              **{k: _summary(v[b, i, j]) for k, v in rep.items()}}
                         for i, e in enumerate(ESTIMATORS)},
                "equal_time_ratio": _summary(rep["seconds"][b, col["rej_equal_time"], j]
                                             / rep["seconds"][b, col["is_learned"], j]),
                "mass": {w: {"mean": float(np.nanmean(mass[b, i, j])),
                             "std": float(np.nanstd(mass[b, i, j]))}
                         for i, w in enumerate(WEIGHTS)},
                "ess_frac": {w: float(np.nanmean(ess_frac[b, i, j])) for i, w in enumerate(WEIGHTS)},
                "max_weight": {w: float(np.nanmean(max_weight[b, i, j]))
                               for i, w in enumerate(WEIGHTS)},
            }
        box_report[box["name"]] = {
            "gt_mass": box["gt_mass"],
            "gt_count": box["gt_count"],
            "leakage": float(1.0 - shard["q_in_box"] / shard["q_total"]),
            "nonfinite": {"q": int(shard["q_nonfinite"]),
                          "p_uncon_on_q_in_box": int(shard["p_uncon_nonfinite_in_box"]),
                          "rejection": int(shard["rejection_nonfinite"])},
            "log_p_uncon_minus_exact_on_q": {
                "mean": float(finite_gap.mean()) if finite_gap.size else float("nan"),
                "std": float(finite_gap.std()) if finite_gap.size else float("nan"),
                "p1": float(np.percentile(finite_gap, 1)) if finite_gap.size else float("nan"),
                "p99": float(np.percentile(finite_gap, 99)) if finite_gap.size else float("nan"),
                "finite_samples": int(finite_gap.size),
                "total_samples": int(gap.size)},
            "uncon_batch_seconds": dict(zip(map(str, shard["calib_sizes"].tolist()),
                                            shard["calib_seconds"].tolist())),
            "by_n": per_n,
        }
        plot_arrays.update({f"{k}_box{b}": shard[k] for k in
                            ("q_p2", "q_inside", "log_w_learned", "log_w_exact")})

    artifacts.save_arrays(out, estimates=estimates, gt_mean=gt_mean, gt_se=gt_se,
                          n_values=np.asarray(n_values), mass=mass, ess_frac=ess_frac,
                          max_weight=max_weight,
                          **{f"rep_{k}": v for k, v in rep.items()},
                          gt_mass=np.array([b["gt_mass"] for b in boxes]), **plot_arrays)
    artifacts.write_manifest(out, run_id=run_id, estimators=list(ESTIMATORS),
                             weights=list(WEIGHTS), observables=list(OBSERVABLE_NAMES),
                             boxes=[b["name"] for b in boxes],
                             benchmark_dir=str(Path(args.boxes).parent))

    report = {
        "run_id": run_id,
        "benchmark_run_id": bench["run_id"],
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "ode_method": cnf.FIXED_METHOD,
        "steps": args.steps,
        "estimators": list(ESTIMATORS),
        "observables": list(OBSERVABLE_NAMES),
        "n_values": n_values,
        "reps": args.reps,
        "boxes": box_report,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    (out / "tables.md").write_text(markdown_tables(report))

    print(f"### decay6d IS ({run_id}, {args.steps} midpoint steps)")
    for b, box in enumerate(boxes):
        j = len(n_values) - 1
        row = "  ".join(f"{e}={rmse[b, i, 0, j]:.2e}" for i, e in enumerate(ESTIMATORS))
        print(f"  {box['name']:14s} N={n_values[j]} RMSE |p2|: {row}")


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
