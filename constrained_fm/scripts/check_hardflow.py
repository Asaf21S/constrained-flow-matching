# -*- coding: utf-8 -*-
"""Diagnostic: what HardFlow actually does on the two physics benchmarks.

bump2d -- why the signal is sometimes over-represented. Every HardFlow sample is paired with
the plain-Euler endpoint of the *same* start point, which splits the feasible samples into
those the base flow already put inside the polygon ("inside") and those the projection had to
move in ("rescued"). The signal share of each group, and of the polygon population as a
whole, says whether the signal comes from the base flow or from the projection.

kinematics6d -- whether the failure is the base model, the step count, or the projection. A
sweep over steps, activation point and SQP damping on a few shells, with the plain base flow
filtered to the shell as a no-projection reference.

Nothing here changes a benchmark number; results go to stdout and a JSON beside the scores.

    sbatch scripts/run_hardflow_check.sh
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from constrained_fm.scripts.eval_bench1k import (METRIC_SEED, MMD_GAMMA, SWD_PROJECTIONS,
                                                 build_parser as bench_parser, load_models,
                                                 rejection_sample, resolve_defaults)
from constrained_fm.scripts.plot_bumphunt import SIGNAL_INDEX, signal_polygon
from constrained_fm.src.consts import BUMP_SIGNAL_MEAN, BUMP_SIGNAL_SIGMA
from constrained_fm.src.datasets.benchmark_1k import constraints_from, load_benchmark_1k
from constrained_fm.src.datasets.bump_conditioning import signal_fraction
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.inference.constrained_samplers import (DEFAULT_CHUNK, sample_euler,
                                                               sample_hardflow)
from constrained_fm.src.inference.constraint_projection import project_closest_point
from constrained_fm.src.metrics.distributional import compute_mmd, compute_swd
from constrained_fm.src.problems.base import NormalizedConstraint
from constrained_fm.src.problems.bump2d import BumpProblem
from constrained_fm.src.problems.kinematics6d import KinematicsProblem

OUT_ROOT = Path("constrained_fm/baselines/bench1k")
BUMP_ACTIVE_FROM = (0.0, 0.25, 0.5, 0.75)
KIN_SHOWCASE = 38
# (steps, active_from, projection damping)
KIN_VARIANTS = ((100, 0.5, 1.0), (100, 0.0, 1.0), (100, 0.75, 1.0), (400, 0.5, 1.0),
                (100, 0.5, 0.5), (100, 0.5, 0.25))
T_BUCKETS = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose HardFlow on bump2d/kinematics6d.")
    parser.add_argument("--problems", nargs="+", default=["bump2d", "kinematics6d"])
    parser.add_argument("--num-x0", type=int, default=10000)
    return parser


def bench_args(problem: str) -> argparse.Namespace:
    args = bench_parser().parse_args(["--problem", problem, "--methods", "gt", "hardflow"])
    resolve_defaults(args)
    return args


# --- instrumented sampler -----------------------------------------------------------------


@torch.no_grad()
def traced_hardflow(model, x0: torch.Tensor, constraint, steps: int, active_from: float,
                    margin: float, projection_iters: int, projection_damping: float = 1.0,
                    side_fn=None, chunk_size: int = DEFAULT_CHUNK) -> dict[str, torch.Tensor]:
    """``sample_hardflow`` with per-sample bookkeeping and identical numerics.

    Returns the endpoints, the (steps, N) mask of steps whose posterior mean violated the
    constraint, the summed absolute per-coordinate displacement due to the projection and to
    the nominal Euler drift, and, if ``side_fn`` is given, its (steps, N) value at the
    posterior mean of each active step.
    """
    parts: dict[str, list[torch.Tensor]] = {"x": [], "active": [], "guide": [], "drift": [],
                                            "side": []}
    first_active = min(round(active_from * steps), steps - 1)
    for start in range(0, x0.shape[0], chunk_size):
        x = x0[start:start + chunk_size]
        active = torch.zeros(steps, x.shape[0], dtype=torch.bool, device=x.device)
        side = torch.zeros(steps, x.shape[0], dtype=torch.int8, device=x.device)
        guide, drift = torch.zeros_like(x), torch.zeros_like(x)

        for i in range(steps):
            t, t_next = i / steps, (i + 1) / steps
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            x_bar = x + (t_next - t) * model(x, t_batch)
            drift += (x_bar - x).abs()
            if i < first_active:
                x = x_bar
                continue

            t_next_batch = torch.full((x.shape[0],), t_next, device=x.device, dtype=x.dtype)
            v_bar = model(x_bar, t_next_batch)
            posterior = x_bar + (1.0 - t_next) * v_bar
            active[i] = constraint.penalty(posterior, margin=margin) > 0
            if side_fn is not None:
                side[i] = side_fn(posterior).to(torch.int8)
            terminal = project_closest_point(posterior, constraint, margin=margin,
                                             max_iters=projection_iters,
                                             relaxation=projection_damping)
            x = t_next * terminal + (1.0 - t_next) * (x_bar - t_next * v_bar)
            guide += (x - x_bar).abs()

        for key, value in (("x", x), ("active", active), ("guide", guide), ("drift", drift),
                           ("side", side)):
            parts[key].append(value)

    return {"x": torch.cat(parts["x"]), "active": torch.cat(parts["active"], dim=1),
            "guide": torch.cat(parts["guide"]), "drift": torch.cat(parts["drift"]),
            "side": torch.cat(parts["side"], dim=1)}


def active_profile(active: torch.Tensor) -> list[float]:
    """Percentage of samples with a violating posterior mean, averaged over ``T_BUCKETS`` bins."""
    return [float(chunk.float().mean()) * 100.0 for chunk in active.chunk(T_BUCKETS, dim=0)]


def wall_flips(side: torch.Tensor) -> torch.Tensor:
    """Per sample, how often the posterior mean jumps from one violated wall to the other."""
    last = torch.zeros_like(side[0])
    flips = torch.zeros(side.shape[1], device=side.device)
    for row in side:
        flips += ((row != 0) & (last != 0) & (row != last)).float()
        last = torch.where(row != 0, row, last)
    return flips


def check_parity(model, x0, wrapped, args) -> float:
    """Max deviation of the traced sampler from the production one on a small batch."""
    reference = sample_hardflow(model, x0, wrapped, steps=args.steps,
                                active_from=args.active_from, margin=args.margin,
                                projection_iters=args.projection_iters,
                                projection_damping=args.projection_damping)
    traced = traced_hardflow(model, x0, wrapped, args.steps, args.active_from, args.margin,
                             args.projection_iters, args.projection_damping)
    return float((reference - traced["x"]).abs().max())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def share(target, x: torch.Tensor) -> float:
    return signal_fraction(target, x) if x.shape[0] else float("nan")


# --- bump2d -------------------------------------------------------------------------------


def bump_polygon_row(model, constraint, x0, base_u, normalizer, target, active_from: float,
                     args) -> dict:
    """HardFlow on one polygon, decomposed by where the base flow would have put each sample."""
    wrapped = NormalizedConstraint(constraint, normalizer)
    run = traced_hardflow(model, x0, wrapped, args.steps, active_from, args.margin,
                          args.projection_iters, args.projection_damping)
    x = normalizer.inverse(run["x"])
    base = normalizer.inverse(base_u)

    feasible = constraint.is_feasible(x)
    base_inside = constraint.is_feasible(base)
    inside = feasible & base_inside
    rescued = feasible & ~base_inside
    mu = torch.tensor(BUMP_SIGNAL_MEAN, device=x.device)
    core = feasible & ((x - mu).norm(dim=-1) < 2.0 * BUMP_SIGNAL_SIGMA)
    moved = (x - base).norm(dim=-1)

    return {
        "sr": float(feasible.float().mean()) * 100.0,
        "signal": share(target, x[feasible]),
        "base_inside": float(base_inside.float().mean()) * 100.0,
        "inside_share": float(inside.float().sum() / feasible.float().sum().clamp_min(1)) * 100,
        "signal_inside": share(target, x[inside]),
        "signal_rescued": share(target, x[rescued]),
        "rescued_to_core": float(rescued[core].float().mean()) * 100.0 if core.any() else math.nan,
        "core_share": float(core.float().sum() / feasible.float().sum().clamp_min(1)) * 100.0,
        "moved_inside": float(moved[base_inside].median()) if base_inside.any() else math.nan,
        "rescued_origin": base[rescued & core].mean(dim=0).tolist() if (rescued & core).any()
        else None,
        "active": active_profile(run["active"]),
        "ever_active": float(run["active"].any(dim=0).float().mean()) * 100.0,
    }


def run_bump(args_cli, device) -> dict:
    args = bench_args("bump2d")
    problem = BumpProblem()
    target = problem.target()
    normalizer = problem.normalizer().to(device)
    benchmark = load_benchmark_1k("bump2d", device=device)
    constraints = constraints_from(benchmark, problem, device=device)
    model = load_models(args, problem, device)["base"]
    x0 = benchmark["x0"][:args_cli.num_x0]
    base_u = sample_euler(model, x0, steps=args.steps)

    wrapped_hand = NormalizedConstraint(signal_polygon(problem.domain, device), normalizer)
    print(f"parity traced vs production HardFlow: max |dx| = "
          f"{check_parity(model, x0[:2000], wrapped_hand, args):.2e}", flush=True)

    scores = json.loads((OUT_ROOT / "bump2d" / "metrics.json").read_text())["methods"]
    exact = np.asarray(scores["gt"]["per_shape"]["signal_fraction"], dtype=float)
    scored_hf = np.asarray(scores["hardflow"]["per_shape"]["signal_fraction"], dtype=float)
    mu = torch.tensor([BUMP_SIGNAL_MEAN], device=device)
    depth = np.array([-float(c.value(mu)[0]) for c in constraints])
    eligible = np.flatnonzero((depth >= 0) & (exact >= 10.0))
    ratio = scored_hf[eligible] / exact[eligible]

    # --- every eligible polygon at the production settings
    rows = []
    for index in eligible:
        row = bump_polygon_row(model, constraints[index], x0, base_u, normalizer, target,
                               args.active_from, args)
        row.update(index=int(index), exact=float(exact[index]), depth=float(depth[index]),
                   mass=float(benchmark["mass"][index]), scored=float(scored_hf[index]))
        row["ratio"] = row["signal"] / row["exact"]
        rows.append(row)
    rows.sort(key=lambda r: r["ratio"])

    print(f"\n## bump2d: {len(rows)} eligible polygons, HardFlow active from "
          f"{args.active_from:g}, {args.steps} steps\n")
    print("| poly | mass % | depth(mu_s) | exact sig % | HF sig % | ratio | SR % | base inside % "
          "| inside share of feas % | sig inside % | sig rescued % | core from rescued % |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['index']} | {r['mass'] * 100:.1f} | {r['depth']:.2f} | {r['exact']:.1f} | "
              f"{r['signal']:.1f} | {r['ratio']:.2f} | {r['sr']:.1f} | {r['base_inside']:.1f} | "
              f"{r['inside_share']:.1f} | {r['signal_inside']:.1f} | {r['signal_rescued']:.1f} | "
              f"{r['rescued_to_core']:.1f} |")

    table = {k: np.array([r[k] for r in rows], dtype=float)
             for k in ("ratio", "mass", "depth", "exact", "base_inside", "inside_share",
                       "signal_inside", "signal_rescued")}
    print("\nSpearman correlation with the HardFlow/exact ratio:")
    for key in ("mass", "depth", "exact", "base_inside", "inside_share", "signal_inside",
                "signal_rescued"):
        print(f"  {key:<16} {spearman(table['ratio'], table[key]):+.2f}")
    print(f"recomputed vs scored ratio: median {np.median(table['ratio']):.2f} vs "
          f"{np.median(ratio):.2f}", flush=True)

    # --- the activation-point sweep on the illustrative polygons
    focus = {"hand-drawn": (signal_polygon(problem.domain, device), SIGNAL_INDEX),
             "median (455)": (constraints[455], 455),
             f"lowest ratio ({rows[0]['index']})": (constraints[rows[0]['index']],
                                                    rows[0]["index"]),
             f"highest ratio ({rows[-1]['index']})": (constraints[rows[-1]['index']],
                                                      rows[-1]["index"])}
    sweep = {}
    for name, (constraint, index) in focus.items():
        truth = rejection_sample(constraint, problem, x0.shape[0], index, args, device)
        base = normalizer.inverse(base_u)
        base_kept = base[constraint.is_feasible(base)]
        print(f"\n### {name}: exact signal {share(target, truth):.1f}%, base flow filtered "
              f"{share(target, base_kept):.1f}% (base inside {base_kept.shape[0] / x0.shape[0] * 100:.1f}%)")
        print("| active from | SR % | signal % | sig inside % | sig rescued % | inside share % "
              "| core share % | core from rescued % | rescued->core origin | ever active % "
              "| active by t-quarter % | median move of inside |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|:---|---:|")
        sweep[name] = {}
        for active_from in BUMP_ACTIVE_FROM:
            r = bump_polygon_row(model, constraint, x0, base_u, normalizer, target, active_from,
                                 args)
            sweep[name][active_from] = r
            origin = ("--" if r["rescued_origin"] is None
                      else "(" + ", ".join(f"{v:.2f}" for v in r["rescued_origin"]) + ")")
            print(f"| {active_from:g} | {r['sr']:.1f} | {r['signal']:.1f} | "
                  f"{r['signal_inside']:.1f} | "
                  f"{r['signal_rescued']:.1f} | {r['inside_share']:.1f} | {r['core_share']:.1f} | "
                  f"{r['rescued_to_core']:.1f} | {origin} | {r['ever_active']:.1f} | "
                  + " / ".join(f"{a:.0f}" for a in r["active"])
                  + f" | {r['moved_inside']:.3f} |", flush=True)

    return {"polygons": rows, "sweep": sweep}


# --- kinematics6d -------------------------------------------------------------------------


def kinematic_summary(target, x: torch.Tensor, constraint) -> dict[str, float]:
    """Marginal fingerprints in physical units: the quantities the marginal plot shows."""
    pt, eta, phi = target.to_spherical(x.view(-1, 2, 3))
    mass = target.invariant_mass(x)
    lo = constraint.mass_target - constraint.epsilon
    hi = constraint.mass_target + constraint.epsilon
    resultant = torch.stack([phi.cos().mean(), phi.sin().mean()]).norm()
    corr = torch.corrcoef(torch.stack([eta[:, 0], eta[:, 1]]))[0, 1]
    return {"pt_median": float(pt.median()), "pt_q90": float(pt.flatten().quantile(0.9)),
            "eta_edge": float((eta.abs() > 2.5).float().mean()) * 100.0,
            "eta_abs_mean": float(eta.abs().mean()),
            "eta_corr": float(corr), "phi_resultant": float(resultant),
            "below": float((mass < lo).float().mean()) * 100.0,
            "above": float((mass > hi).float().mean()) * 100.0}


def distance_row(samples_u: torch.Tensor, truth_u: torch.Tensor, index: int) -> dict:
    n = min(samples_u.shape[0], truth_u.shape[0] // 2)
    if n < 2:
        return {"n": n, "swd": math.nan, "swd_floor": math.nan, "mmd": math.nan,
                "mmd_floor": math.nan}
    gen, ref, second = samples_u[:n], truth_u[:n], truth_u[n:2 * n]
    seed = METRIC_SEED + index
    gamma = MMD_GAMMA["kinematics6d"]
    torch.manual_seed(seed)
    return {"n": n,
            "swd": compute_swd(gen, ref, num_projections=SWD_PROJECTIONS, seed=seed),
            "swd_floor": compute_swd(second, ref, num_projections=SWD_PROJECTIONS, seed=seed),
            "mmd": compute_mmd(gen, ref, gamma=gamma),
            "mmd_floor": compute_mmd(second, ref, gamma=gamma)}


def run_kinematics(args_cli, device) -> dict:
    args = bench_args("kinematics6d")
    problem = KinematicsProblem()
    target = problem.target()
    normalizer = problem.normalizer().to(device)
    benchmark = load_benchmark_1k("kinematics6d", device=device)
    constraints = constraints_from(benchmark, problem, device=device)
    model = load_models(args, problem, device)["base"]
    x0 = benchmark["x0"][:args_cli.num_x0]
    base_u = sample_euler(model, x0, steps=args.steps)
    base = normalizer.inverse(base_u)

    std = normalizer.std
    print(f"\nnormaliser std (GeV): {[round(float(s), 1) for s in std]}; "
          f"longitudinal/transverse = {float(std[2] / std[0]):.2f}", flush=True)

    masses = benchmark["mass"].double().cpu()
    windows = {"showcase": KIN_SHOWCASE,
               "thinnest": int((masses - 0.01).abs().argmin()),
               "widest": int((masses - 0.5).abs().argmin())}
    print(f"parity traced vs production HardFlow: max |dx| = "
          f"{check_parity(model, x0[:2000], NormalizedConstraint(constraints[KIN_SHOWCASE], normalizer), args):.2e}",
          flush=True)

    results = {}
    for name, index in windows.items():
        constraint = constraints[index]
        wrapped = NormalizedConstraint(constraint, normalizer)
        truth = rejection_sample(constraint, problem, 2 * x0.shape[0], index, args, device)
        truth_u = normalizer.forward(truth)

        def side_fn(x1_hat_u, c=constraint):
            mass = target.invariant_mass(normalizer.inverse(x1_hat_u))
            gap = mass - c.mass_target
            return torch.where(gap.abs() <= c.epsilon, torch.zeros_like(gap), gap.sign())

        print(f"\n## kinematics6d window {index} ({name}): M* {constraint.mass_target:.2f} GeV, "
              f"eps {constraint.epsilon:.2f} GeV, mass {float(masses[index]) * 100:.2f}%")
        rows = {}
        truth_row = kinematic_summary(target, truth[:x0.shape[0]], constraint)
        truth_row.update(sr=100.0, in_support=float(target.in_support(truth).float().mean()) * 100)
        rows["exact"] = truth_row

        kept = base[constraint.is_feasible(base)]
        row = kinematic_summary(target, base, constraint)
        row.update(sr=constraint.success_rate(base),
                   in_support=float(target.in_support(base).float().mean()) * 100)
        row.update(distance_row(normalizer.forward(kept), truth_u, index))
        row["filtered"] = kinematic_summary(target, kept, constraint) if kept.shape[0] > 2 else {}
        rows["base (no projection), filtered to window"] = row

        for steps, active_from, damping in KIN_VARIANTS:
            run = traced_hardflow(model, x0, wrapped, steps, active_from, args.margin,
                                  args.projection_iters, damping, side_fn=side_fn)
            x = normalizer.inverse(run["x"])
            row = kinematic_summary(target, x, constraint)
            row.update(sr=constraint.success_rate(x),
                       in_support=float(target.in_support(x).float().mean()) * 100)
            row.update(distance_row(run["x"], truth_u, index))
            guide = (run["guide"] * std).mean(dim=0)
            drift = (run["drift"] * std).mean(dim=0)
            row.update(guide_T=float(guide[[0, 1, 3, 4]].mean()),
                       guide_L=float(guide[[2, 5]].mean()),
                       drift_T=float(drift[[0, 1, 3, 4]].mean()),
                       drift_L=float(drift[[2, 5]].mean()),
                       flips=float(wall_flips(run["side"]).mean()),
                       active=active_profile(run["active"]))
            rows[f"hardflow steps {steps} active from {active_from:g} damping {damping:g}"] = row

        print("| variant | SR % | in-support % | SWD / floor | MMD / floor | pT median | "
              "|eta|>2.5 % | corr(eta1,eta2) | phi R | below / above % | projection GeV T / L | "
              "drift GeV T / L | wall flips | active by t-quarter % |")
        print("|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|:---|---:|:---|")
        for key, r in rows.items():
            dist = (f"{r['swd'] / r['swd_floor']:.1f}x | {r['mmd'] / r['mmd_floor']:.0f}x"
                    if "swd" in r else "-- | --")
            anat = (f"{r['guide_T']:.1f} / {r['guide_L']:.1f} | {r['drift_T']:.1f} / "
                    f"{r['drift_L']:.1f} | {r['flips']:.2f} | "
                    + " / ".join(f"{a:.0f}" for a in r["active"])
                    if "guide_T" in r else "-- | -- | -- | --")
            print(f"| {key} | {r['sr']:.1f} | {r['in_support']:.1f} | {dist} | "
                  f"{r['pt_median']:.1f} | {r['eta_edge']:.1f} | {r['eta_corr']:+.2f} | "
                  f"{r['phi_resultant']:.3f} | {r['below']:.1f} / {r['above']:.1f} | {anat} |",
                  flush=True)
        filtered = rows["base (no projection), filtered to window"]["filtered"]
        if filtered:
            print(f"base filtered to window: pT median {filtered['pt_median']:.1f}, "
                  f"|eta|>2.5 {filtered['eta_edge']:.1f}%, corr(eta) {filtered['eta_corr']:+.2f}")
        results[name] = {"index": index, "rows": rows}

    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    torch.manual_seed(0)
    for problem in args.problems:
        result = run_bump(args, device) if problem == "bump2d" else run_kinematics(args, device)
        path = OUT_ROOT / problem / "hardflow_check.json"
        path.write_text(json.dumps(result, indent=1, default=float))
        print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
