# -*- coding: utf-8 -*-
"""Diagnostic: what HardFlow actually does on the two physics benchmarks.

bump2d -- why the signal is sometimes over-represented. Every HardFlow sample is paired with
the plain-Euler endpoint of the *same* start point, which splits the feasible samples into
those the base flow already put inside the polygon ("inside") and those the guidance had to
move in ("rescued"). The signal share of each group, and of the polygon population as a
whole, says whether the signal comes from the base flow or from the guidance.

kinematics6d -- whether the failure is the base model, the step count, or the guidance. A
sweep over guidance scale and steps on a few shells, the plain base flow filtered to the
shell as a no-guidance reference, and a variant whose guidance is preconditioned to be
isotropic in physical momentum rather than in the normalised frame.

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
from constrained_fm.src.metrics.distributional import compute_mmd, compute_swd
from constrained_fm.src.problems.base import NormalizedConstraint
from constrained_fm.src.problems.bump2d import BumpProblem
from constrained_fm.src.problems.kinematics6d import KinematicsProblem

OUT_ROOT = Path("constrained_fm/baselines/bench1k")
BUMP_SCALES = (10.0, 30.0, 100.0, 300.0)
KIN_SHOWCASE = 38
# (label, guidance scale, steps, preconditioned)
KIN_VARIANTS = (("hardflow", 1.0, 100, False), ("hardflow", 3.0, 100, False),
                ("hardflow", 10.0, 100, False), ("hardflow", 30.0, 100, False),
                ("hardflow", 100.0, 100, False), ("hardflow", 10.0, 400, False),
                ("hardflow", 100.0, 400, False), ("hardflow-iso", 10.0, 100, True),
                ("hardflow-iso", 100.0, 100, True), ("hardflow-iso", 1000.0, 100, True))
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


def traced_hardflow(model, x0: torch.Tensor, constraint, steps: int, scale: float,
                    margin: float, precondition: torch.Tensor | None = None,
                    side_fn=None, chunk_size: int = DEFAULT_CHUNK) -> dict[str, torch.Tensor]:
    """``sample_hardflow`` with per-sample bookkeeping; identical numerics when unpreconditioned.

    Returns the endpoints, the (steps, N) mask of steps whose hinge was active, the summed
    absolute per-coordinate displacement due to guidance and to the base drift, and, if
    ``side_fn`` is given, its (steps, N) value at the predicted endpoint.
    """
    parts: dict[str, list[torch.Tensor]] = {"x": [], "active": [], "guide": [], "drift": [],
                                            "side": []}
    dt = 1.0 / steps
    for start in range(0, x0.shape[0], chunk_size):
        x = x0[start:start + chunk_size]
        active = torch.zeros(steps, x.shape[0], dtype=torch.bool, device=x.device)
        side = torch.zeros(steps, x.shape[0], dtype=torch.int8, device=x.device)
        guide, drift = torch.zeros_like(x), torch.zeros_like(x)

        for i in range(steps):
            t = i * dt
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            with torch.enable_grad():
                x_leaf = x.detach().requires_grad_(True)
                v = model(x_leaf, t_batch)
                x1_hat = x_leaf + (1.0 - t) * v
                hinge = constraint.penalty(x1_hat, margin=margin)
                (grad,) = torch.autograd.grad(hinge.sum(), x_leaf)
            if precondition is not None:
                grad = grad * precondition
            v = v.detach()
            active[i] = hinge.detach() > 0
            if side_fn is not None:
                side[i] = side_fn(x1_hat.detach()).to(torch.int8)
            guide += (scale * grad * dt).abs()
            drift += (v * dt).abs()
            x = x_leaf.detach() + (v - scale * grad) * dt

        for key, value in (("x", x), ("active", active), ("guide", guide), ("drift", drift),
                           ("side", side)):
            parts[key].append(value)

    return {"x": torch.cat(parts["x"]), "active": torch.cat(parts["active"], dim=1),
            "guide": torch.cat(parts["guide"]), "drift": torch.cat(parts["drift"]),
            "side": torch.cat(parts["side"], dim=1)}


def active_profile(active: torch.Tensor) -> list[float]:
    """Percentage of samples with an active hinge, averaged over ``T_BUCKETS`` time bins."""
    return [float(chunk.float().mean()) * 100.0 for chunk in active.chunk(T_BUCKETS, dim=0)]


def wall_flips(side: torch.Tensor) -> torch.Tensor:
    """Per sample, how often the predicted endpoint jumps from one violated wall to the other."""
    last = torch.zeros_like(side[0])
    flips = torch.zeros(side.shape[1], device=side.device)
    for row in side:
        flips += ((row != 0) & (last != 0) & (row != last)).float()
        last = torch.where(row != 0, row, last)
    return flips


def check_parity(model, x0, wrapped, args) -> float:
    """Max deviation of the traced sampler from the production one on a small batch."""
    reference = sample_hardflow(model, x0, wrapped, steps=args.steps,
                                guidance_scale=args.guidance_scale, margin=args.margin)
    traced = traced_hardflow(model, x0, wrapped, args.steps, args.guidance_scale, args.margin)
    return float((reference - traced["x"]).abs().max())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def share(target, x: torch.Tensor) -> float:
    return signal_fraction(target, x) if x.shape[0] else float("nan")


# --- bump2d -------------------------------------------------------------------------------


def bump_polygon_row(model, constraint, x0, base_u, normalizer, target, scale: float,
                     args) -> dict:
    """HardFlow on one polygon, decomposed by where the base flow would have put each sample."""
    wrapped = NormalizedConstraint(constraint, normalizer)
    run = traced_hardflow(model, x0, wrapped, args.steps, scale, args.margin)
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

    # --- every eligible polygon at the production scale
    rows = []
    for index in eligible:
        row = bump_polygon_row(model, constraints[index], x0, base_u, normalizer, target,
                               args.guidance_scale, args)
        row.update(index=int(index), exact=float(exact[index]), depth=float(depth[index]),
                   mass=float(benchmark["mass"][index]), scored=float(scored_hf[index]))
        row["ratio"] = row["signal"] / row["exact"]
        rows.append(row)
    rows.sort(key=lambda r: r["ratio"])

    print(f"\n## bump2d: {len(rows)} eligible polygons, HardFlow scale {args.guidance_scale:g}, "
          f"{args.steps} steps\n")
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

    # --- the guidance-scale sweep on the illustrative polygons
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
        print("| scale | SR % | signal % | sig inside % | sig rescued % | inside share % "
              "| core share % | core from rescued % | rescued->core origin | ever active % "
              "| active by t-quarter % | median move of inside |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|:---|---:|")
        sweep[name] = {}
        for scale in BUMP_SCALES:
            r = bump_polygon_row(model, constraint, x0, base_u, normalizer, target, scale, args)
            sweep[name][scale] = r
            origin = ("--" if r["rescued_origin"] is None
                      else "(" + ", ".join(f"{v:.2f}" for v in r["rescued_origin"]) + ")")
            print(f"| {scale:g} | {r['sr']:.1f} | {r['signal']:.1f} | {r['signal_inside']:.1f} | "
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
    # Rescales the normalised-frame gradient so the physical step -std^2 * g_x becomes
    # -std_T^2 * g_x: isotropic in momentum, unchanged along the transverse axes.
    iso = (std[0] / std) ** 2
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
        rows["base (no guidance), filtered to window"] = row

        for label, scale, steps, precondition in KIN_VARIANTS:
            run = traced_hardflow(model, x0, wrapped, steps, scale, args.margin,
                                  precondition=iso if precondition else None, side_fn=side_fn)
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
            rows[f"{label} scale {scale:g} steps {steps}"] = row

        print("| variant | SR % | in-support % | SWD / floor | MMD / floor | pT median | "
              "|eta|>2.5 % | corr(eta1,eta2) | phi R | below / above % | guide GeV T / L | "
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
        filtered = rows["base (no guidance), filtered to window"]["filtered"]
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
