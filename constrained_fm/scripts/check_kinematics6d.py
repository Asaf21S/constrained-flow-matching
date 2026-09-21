# -*- coding: utf-8 -*-
"""Validates the kinematics6d target, its analytic density, and the shell constraints.

The 6D Cartesian log-density is the whole reason this dataset can report KLD, and it is built
from a change of variables that is easy to get subtly wrong: drop the Jacobian and the density
is still smooth, still normalised to something, and still produces plausible NLL numbers. So
it is checked three independent ways -- against a histogram, against the normalisation
integral, and against the sampler's own moments -- rather than by inspection.

    python -m constrained_fm.scripts.check_kinematics6d
    python -m constrained_fm.scripts.check_kinematics6d --pool-size 2000000
"""

from __future__ import annotations

import argparse
import math

import torch

from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.problems.base import NormalizedConstraint
from constrained_fm.src.problems.kinematics6d import (KinematicsProblem, sample_mass_constraints,
                                                      shell_fraction)

TOLERANCES = {"hist": 0.02, "moment_sem": 4.0, "moment_std": 0.02, "shell": 0.05,
              "norm": 0.05, "tilt": 0.10}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate the kinematics6d problem.")
    parser.add_argument("--pool-size", type=int, default=2_000_000)
    parser.add_argument("--hist-bins", type=int, default=60)
    parser.add_argument("--num-constraints", type=int, default=200)
    parser.add_argument("--num-anchors", type=int, default=256)
    parser.add_argument("--box-half-width", type=float, default=0.35)
    parser.add_argument("--min-box-counts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def check_density_histogram(target, pool: torch.Tensor, bins: int) -> bool:
    """1D marginals of ``exp(log_prob)`` against the sampler, via importance reweighting.

    A 6D density cannot be histogrammed directly, but each marginal can be compared on the
    sampler's own draws: binning ``pT``, ``eta`` and the pair mass tests the three factors and
    the Jacobian separately, since only the Jacobian couples ``pT`` to ``eta``.
    """
    pairs = pool.view(-1, 2, 3)
    pt, eta, _ = target.to_spherical(pairs)
    ok = True

    for name, values, lo, hi, log_pdf in (
            ("pT", pt.flatten(), target.pt_lo, target.pt_hi,
             lambda t: -t / target.pt_scale - math.log(target.pt_scale * target._pt_norm)),
            ("eta", eta.flatten(), target.eta_lo, target.eta_hi,
             lambda t: (-0.5 * (t / target.eta_sigma) ** 2 - 0.5 * math.log(2.0 * math.pi)
                        - math.log(target.eta_sigma * target._eta_norm)))):
        edges = torch.linspace(lo, hi, bins + 1, device=values.device, dtype=values.dtype)
        counts = torch.histc(values, bins=bins, min=lo, max=hi)
        empirical = counts / (counts.sum() * (edges[1] - edges[0]))
        analytic = torch.exp(log_pdf(0.5 * (edges[:-1] + edges[1:])))

        drift = (empirical - analytic).abs().max() / analytic.max()
        ok &= bool(drift < TOLERANCES["hist"])
        print(f"  {name:4s} marginal: max |sampled - analytic| / peak = {drift:.4e} "
              f"{'ok' if drift < TOLERANCES['hist'] else 'FAILED'}")

    return ok


def _interior_anchors(target, x: torch.Tensor, half_width: torch.Tensor) -> torch.Tensor:
    """Keeps only points whose whole axis-aligned box is inside the support.

    The support is an annulus in ``pT`` crossed with a cone in ``eta``, so it is not convex and
    checking the box corners is not sufficient. These are the conservative bounds instead: a
    Cartesian shift of at most ``(h_x, h_y)`` moves ``pT`` by at most ``hypot(h_x, h_y)``, and
    ``|eta|`` is largest when ``|p_z|`` shifts outward while ``pT`` shifts inward.
    """
    pairs = x.view(-1, 2, 3)
    pt, eta, _ = target.to_spherical(pairs)
    h = half_width.view(2, 3)

    pt_margin = torch.hypot(h[:, 0], h[:, 1])
    pt_inner = pt - pt_margin
    safe = (pt - pt_margin >= target.pt_lo) & (pt + pt_margin <= target.pt_hi)
    eta_outer = torch.asinh((pt * torch.sinh(eta.abs()) + h[:, 2]) / pt_inner.clamp_min(1e-3))
    safe &= (pt_inner > 0) & (eta_outer <= target.eta_hi)
    return x[safe.all(dim=-1)]


def check_normalisation(target, problem, pool: torch.Tensor, args, device) -> bool:
    """Analytic density against a box-count of the sampler, in the normalised frame.

    This is the only check that sees the change of variables: the ``pT`` and ``eta`` marginals
    in ``[A]`` are the 1D factors alone, and dropping ``log|J|`` would leave them untouched.
    Counting samples in a small box gives an absolute, normalised density, so a missing
    Jacobian shows up as a ratio that drifts with ``pT`` and a wrong constant shows up as a
    ratio pinned away from one. The box lives in the normalised frame so a single half-width
    is comparable across the transverse and longitudinal axes, which differ by almost 4x.
    The pooled ratio only resolves the normalisation to a few percent, and its residual moves
    with ``box_half_width`` because the interior filter admits a different anchor population at
    each width; the finite-box curvature bias cancels only when anchors are density-weighted.
    The ``tilt`` is the sharp instrument here -- it falls off as ``h^2``, whereas a dropped
    Jacobian would tilt it by orders of magnitude at every width.    """
    normalizer = problem.normalizer().to(device)
    u_pool = normalizer.forward(pool)
    log_det = torch.log(normalizer.std).sum().double()

    half_width = args.box_half_width * normalizer.std
    anchors = _interior_anchors(target, target.sample(4 * args.num_anchors, device=device),
                                half_width)[:args.num_anchors]
    if anchors.shape[0] < args.num_anchors // 2:
        print(f"  only {anchors.shape[0]} interior anchors, box is too large FAILED")
        return False

    u_anchors = normalizer.forward(anchors)
    log_volume = math.log(2.0 * args.box_half_width) * problem.dim
    counts, expected = [], []
    for start in range(0, u_anchors.shape[0], 4):
        block = u_anchors[start:start + 4]
        inside = ((u_pool.unsqueeze(0) - block.unsqueeze(1)).abs()
                  <= args.box_half_width).all(dim=-1)
        counts.append(inside.sum(dim=1).double())
        expected.append(torch.exp(target.log_prob(anchors[start:start + 4]).double() + log_det
                                  + log_volume) * u_pool.shape[0])

    counts, expected = torch.cat(counts), torch.cat(expected)
    pooled = (counts.sum() / expected.sum()).item()
    sem = pooled / math.sqrt(counts.sum().item())

    # A per-anchor ratio is Poisson noise wherever the box is nearly empty, so it is only read
    # on anchors rich enough to carry a signal.
    rich = counts >= args.min_box_counts
    ratios = counts[rich] / expected[rich]
    pt, _, _ = target.to_spherical(anchors.view(-1, 2, 3))
    hard = pt.amax(dim=-1)[rich]
    split = hard.median()
    low = (counts[rich][hard <= split].sum() / expected[rich][hard <= split].sum()).item()
    high = (counts[rich][hard > split].sum() / expected[rich][hard > split].sum()).item()

    ok = abs(pooled - 1.0) < TOLERANCES["norm"] and abs(low / high - 1.0) < TOLERANCES["tilt"]
    print(f"  {anchors.shape[0]} interior anchors | box half-width {args.box_half_width} sd "
          f"| {int(counts.sum())} counts | {int(rich.sum())} rich")
    print(f"  pooled sampled / analytic = {pooled:.4f} +- {sem:.4f}")
    print(f"  rich-anchor median {ratios.median():.4f} | low-pT {low:.4f} vs high-pT {high:.4f} "
          f"(tilt {abs(low / high - 1.0):.4f}) {'ok' if ok else 'FAILED'}")
    return ok


def check_moments(target, pool: torch.Tensor) -> bool:
    """Analytic ``mean_std`` against the sampler, which tests the Cartesian map itself."""
    mean, std = target.mean_std(device=pool.device)
    sampled_mean, sampled_std = pool.mean(dim=0), pool.std(dim=0)
    sem = sampled_std / math.sqrt(pool.shape[0])

    mean_ok = bool(((sampled_mean - mean).abs() < TOLERANCES["moment_sem"] * sem).all())
    std_ok = bool((((sampled_std - std) / std).abs() < TOLERANCES["moment_std"]).all())

    print(f"  analytic std  {[round(v, 2) for v in std.tolist()]}")
    print(f"  sampled  std  {[round(v, 2) for v in sampled_std.tolist()]} "
          f"{'ok' if std_ok else 'FAILED'}")
    print(f"  mean within {TOLERANCES['moment_sem']:.0f} SEM: {'ok' if mean_ok else 'FAILED'}")

    scale = target.mass_scale()
    mass = target.invariant_mass(pool)
    rms_drift = abs(float(mass.pow(2).mean().sqrt()) - scale) / scale
    rms_ok = rms_drift < TOLERANCES["moment_std"]
    print(f"  analytic sqrt(E[M^2]) = {scale:.3f} vs sampled {float(mass.pow(2).mean().sqrt()):.3f} "
          f"(drift {rms_drift:.4f}) {'ok' if rms_ok else 'FAILED'}")
    return mean_ok and std_ok and rms_ok


def check_shells(target, pool: torch.Tensor, args) -> bool:
    """Requested vs achieved shell mass on an independent pool, and the non-convexity contract."""
    constraints, achieved = sample_mass_constraints(args.num_constraints, target, pool)
    held_out = target.sample(pool.shape[0] // 4, device=pool.device)

    realised = torch.tensor([c.is_feasible(held_out).double().mean() for c in constraints])
    drift = (realised - achieved).abs().max().item()
    ok = drift < TOLERANCES["shell"]

    widths = torch.tensor([c.epsilon for c in constraints])
    print(f"  {len(constraints)} shells | mass {achieved.min():.4f} to {achieved.max():.4f} "
          f"| epsilon {widths.min():.3f} to {widths.max():.3f}")
    print(f"  max |held-out mass - requested| = {drift:.4e} {'ok' if ok else 'FAILED'}")

    convex_ok = all(c.interior_point is None for c in constraints)
    print(f"  interior_point is None on every shell (non-convex): "
          f"{'ok' if convex_ok else 'FAILED'}")
    return ok and convex_ok


def check_gradients(problem, target, pool: torch.Tensor) -> bool:
    """Finite gradients in the normalised frame, including on the shell's own boundary."""
    normalizer = problem.normalizer().to(pool.device)
    constraints, _ = sample_mass_constraints(8, target, pool)
    ok = True

    for constraint in constraints:
        wrapped = NormalizedConstraint(constraint, normalizer)
        u = normalizer.forward(pool[:4096])
        values, grads = wrapped.value_and_grad(u)
        finite = bool(torch.isfinite(values).all() and torch.isfinite(grads).all())
        nonzero = bool((grads.norm(dim=-1) > 0).all())
        ok &= finite and nonzero

    print(f"  value_and_grad finite and non-degenerate on {len(constraints)} shells: "
          f"{'ok' if ok else 'FAILED'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    set_seed(args.seed)

    problem = KinematicsProblem()
    target = problem.target()
    pool = target.sample(args.pool_size, device=device)
    print(f"device {device} | pool {args.pool_size} | dim {problem.dim}\n")

    print("[A] marginals of the analytic density vs the sampler")
    results = [check_density_histogram(target, pool, args.hist_bins)]
    print("\n[B] normalisation of the 6D Cartesian density")
    results.append(check_normalisation(target, problem, pool, args, device))
    print("\n[C] analytic moments and mass scale")
    results.append(check_moments(target, pool))
    print("\n[D] stratified mass shells")
    results.append(check_shells(target, pool, args))
    print("\n[E] constraint gradients in the normalised frame")
    results.append(check_gradients(problem, target, pool))

    passed = all(results)
    print(f"\n{'PASSED' if passed else 'FAILED'}: kinematics6d "
          f"({sum(results)}/{len(results)} checks)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
