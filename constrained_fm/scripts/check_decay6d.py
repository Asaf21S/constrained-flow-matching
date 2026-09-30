# -*- coding: utf-8 -*-
"""Read-only gate for the decay6d target, its quadrature density and the box sampler.

1. Analytic moments vs the simulator.
2. Quadrature convergence: ``log p`` at the default node count vs 4x the nodes.
3. Split calibration: the PIT of the true ``u`` under ``p(u | x)`` must be uniform. The
   Jacobian ``8 u^3 (1-u)^3`` is u-dependent, so a wrong Jacobian shows up here.
4. Relative normalization: ``E_g[p / g] = 1`` with ``g`` the same model at 1.5x the noise.
5. Box filter: table ``P(B)`` vs Monte Carlo, acceptance rate, accepted-mass spread.

    python -m constrained_fm.scripts.check_decay6d
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from constrained_fm.src.consts import DECAY_BEAM_SIGMA, DECAY_MASS_SIGMA, KS_CRITICAL_95
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.problems.decay6d import (PARTICLE_DIM, DecayProblem, DecayTarget,
                                                 boxes_contain, sample_anchored_boxes,
                                                 trapezoid_nodes)

DEFAULT_OUTDIR = "constrained_fm/baselines/decay6d_is/check"
PIT_BINS = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="decay6d density and box-sampler gate.")
    parser.add_argument("--moment-samples", type=int, default=2_000_000)
    parser.add_argument("--quadrature-samples", type=int, default=2000)
    parser.add_argument("--pit-samples", type=int, default=20000)
    parser.add_argument("--norm-samples", type=int, default=2_000_000)
    parser.add_argument("--noise-inflation", type=float, default=1.5)
    parser.add_argument("--box-candidates", type=int, default=200_000)
    parser.add_argument("--mc-boxes", type=int, default=2000)
    parser.add_argument("--mc-pool", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser


def check_moments(target: DecayTarget, n: int, device) -> dict:
    x = target.sample(n, device)
    mean, std = target.mean_std(device, torch.float64)
    return {"max_abs_mean_error": (x.mean(0) - mean).abs().max().item(),
            "max_rel_std_error": ((x.std(0) - std) / std).abs().max().item(),
            "analytic_std": std.tolist()}


def check_quadrature(target: DecayTarget, n: int, device) -> dict:
    x = target.sample(n, device)
    base = target.log_prob(x)
    fine = target.log_prob(x, chunk_size=256, num_nodes=4 * target.quadrature_nodes)
    return {"max_abs_log_p_change_4x_nodes": (base - fine).abs().max().item(),
            "mean_log_p": base.mean().item()}


def check_split_pit(target: DecayTarget, n: int, device) -> dict:
    x, split = target.sample_with_split(n, device)
    nodes, log_w = trapezoid_nodes(target.split_lo, target.split_hi, target.quadrature_nodes,
                                   device)
    pit = []
    for xc, uc in zip(x.split(1024), split.split(1024)):
        post = torch.softmax(target.log_joint_split(xc, nodes) + log_w, dim=1)
        cdf = post.cumsum(dim=1)
        idx = torch.searchsorted(nodes, uc[:, None]).clamp(1, nodes.numel() - 1)
        lo, hi = nodes[idx - 1].squeeze(1), nodes[idx].squeeze(1)
        frac = (uc - lo) / (hi - lo)
        c_lo = cdf.gather(1, idx - 1).squeeze(1)
        c_hi = cdf.gather(1, idx).squeeze(1)
        pit.append(c_lo + frac * (c_hi - c_lo))
    pit = torch.cat(pit).sort().values
    uniform = torch.arange(1, n + 1, device=device, dtype=pit.dtype) / n
    ks = (pit - uniform).abs().max().item()
    counts = torch.histc(pit, bins=PIT_BINS, min=0.0, max=1.0)
    return {"ks_statistic": ks, "ks_95_critical": KS_CRITICAL_95 / math.sqrt(n),
            "pit_bin_counts": counts.tolist()}


def check_relative_normalization(target: DecayTarget, n: int, inflation: float, device) -> dict:
    wide = DecayTarget(mass_sigma=DECAY_MASS_SIGMA * inflation,
                       beam_sigma=DECAY_BEAM_SIGMA * inflation)
    x = wide.sample(n, device)
    ratio = torch.exp(target.log_prob(x) - wide.log_prob(x))
    return {"mean_p_over_g": ratio.mean().item(),
            "se": (ratio.std() / math.sqrt(n)).item(),
            "max_p_over_g": ratio.max().item()}


def check_box_filter(problem: DecayProblem, args, device) -> dict:
    target = problem.target()
    normalizer = problem.normalizer(torch.float64).to(device)
    table = problem.mass_table(device)

    p1 = normalizer.forward(target.sample(args.box_candidates, device))[:, :PARTICLE_DIM]
    centre, half = sample_anchored_boxes(p1, problem.half_width_range)
    table_mass = table.mass(centre - half, centre + half)
    keep = (table_mass >= problem.mass_range[0]) & (table_mass <= problem.mass_range[1])

    pool = normalizer.forward(target.sample(args.mc_pool, device))[:, :PARTICLE_DIM]
    lo, hi = (centre - half)[:args.mc_boxes], (centre + half)[:args.mc_boxes]
    mc = torch.stack([boxes_contain(pool, l, h).double().mean() for l, h in zip(lo, hi)])
    err = (table_mass[:args.mc_boxes] - mc).abs()
    log_mass = table_mass[keep].log10().cpu().numpy()
    hist, edges = np.histogram(log_mass, bins=10)
    return {"acceptance": keep.double().mean().item(),
            "table_vs_mc_abs_error_p50": err.median().item(),
            "table_vs_mc_abs_error_p99": err.quantile(0.99).item(),
            "table_vs_mc_rel_error_p50_mass_gt_1pct": (err / mc)[mc > 0.01].median().item(),
            "accepted_log10_mass_hist": hist.tolist(),
            "accepted_log10_mass_edges": edges.tolist()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    out = Path(args.outdir)
    run_id = pin_baseline_run(out, "decay6d_check", args)
    set_seed(args.seed)

    problem = DecayProblem()
    target = problem.target()
    report = {
        "run_id": run_id,
        "moments": check_moments(target, args.moment_samples, device),
        "quadrature": check_quadrature(target, args.quadrature_samples, device),
        "split_pit": check_split_pit(target, args.pit_samples, device),
        "relative_normalization": check_relative_normalization(
            target, args.norm_samples, args.noise_inflation, device),
        "box_filter": check_box_filter(problem, args, device),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
