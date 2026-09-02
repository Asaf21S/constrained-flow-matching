# -*- coding: utf-8 -*-
"""Audits the exact-likelihood pipeline behind the reported NLL / KLD.

Three independent checks, each targeting one way the metric could lie:

  A. Jacobian completeness. ConstrainedFlowMatcher feeds SIREN(x_t, z) into the velocity
     field, so d(SIREN)/dx_t belongs to dv/dx_t. It used to be computed under
     torch.no_grad(), which silently zeroed that block in the exact-divergence trace while
     leaving the integrated field untouched. This panel scores the same points with the
     fixed model and with the SIREN feature re-detached, so the gap is the size of the bug.

  B. Normalization. If the trace matches the field, exp(log p) integrates to 1 over the
     plane. A missing Jacobian block breaks that identity, and the deficit is a direct,
     model-independent readout of the trace error in nats.

  C. Reference entropy. KLD = model NLL - truncated-GMM NLL, and the truncated density is
     p_gmm / mass. `mass` is a Monte Carlo pool fraction; this compares it against a fine
     deterministic quadrature of the same integral so the KLD shift log(mass_MC/mass_quad)
     is bounded rather than assumed.

A step-size ladder runs alongside A so solver discretization is separated from the
Jacobian error (midpoint is a fixed-step method, so atol/rtol never apply).

    python -m constrained_fm.scripts.audit_likelihood --run-id <run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from constrained_fm.src.datasets.gmm_target import compute_gmm_log_likelihood
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.experiment.registry import load_config, run_dir
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_siren,
                                                   resolve_device, set_seed)
from constrained_fm.src.metrics.eval_points import load_nll_eval_set
from constrained_fm.src.metrics.functa_fidelity import uniform_grid_points
from constrained_fm.src.metrics.likelihood import exact_log_likelihood

from constrained_fm.scripts.eval_fm import extract_validation_latents


class _DetachedSiren(nn.Module):
    """Reproduces the pre-fix behaviour: identical values, but no gradient path back to x."""

    def __init__(self, siren: nn.Module):
        super().__init__()
        self.inner = siren

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.inner(x, z)


class detached_siren_feature:
    """Context manager swapping the model's SIREN for the gradient-blocking wrapper."""

    def __init__(self, model):
        self.model = model
        self.original = None

    def __enter__(self):
        self.original = self.model.siren
        self.model.siren = _DetachedSiren(self.original)
        return self.model

    def __exit__(self, *exc):
        self.model.siren = self.original
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the NLL / KLD likelihood pipeline.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shapes", type=int, default=20, help="validation shapes to score")
    parser.add_argument("--nll-points", type=int, default=2000)
    parser.add_argument("--steps", type=float, nargs="+", default=[0.05, 0.02, 0.01, 0.005],
                        help="ODE step-size ladder; the first entry is the production setting")
    parser.add_argument("--norm-grid", type=int, default=160,
                        help="lattice side for the normalization integral (check B)")
    parser.add_argument("--mass-grid", type=int, default=3000,
                        help="lattice side for the mass quadrature (check C)")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--outdir", default="outputs/likelihood_audit")
    return parser


def score(model, x_valid, z, mass, step_size, num_points, device) -> dict[str, float]:
    """Reimplements constraint_nll without resampling, so variants share identical points."""
    log_p_model = exact_log_likelihood(model, x_valid, z=z, step_size=step_size, device=device)
    finite = torch.isfinite(log_p_model)
    if not bool(finite.any()):
        return {"nll": float("inf"), "kld": float("inf"), "dropped": float(len(x_valid))}

    nll = float(-log_p_model[finite].mean())
    log_p_true = compute_gmm_log_likelihood(x_valid, device=device) - float(np.log(mass))
    ideal = float(-log_p_true[finite].mean())
    return {"nll": nll, "kld": nll - ideal, "ideal_nll": ideal,
            "dropped": float((~finite).sum())}


def normalization_deficit(model, z, scale, grid_size, step_size, device) -> float:
    """log of the integral of exp(log p) over the plane. 0.0 iff the trace matches the field."""
    grid = uniform_grid_points(grid_size=grid_size, scale=scale, device=device)
    cell = (2.0 * scale / (grid_size - 1)) ** 2
    log_p = exact_log_likelihood(model, grid, z=z, step_size=step_size, device=device)
    log_p = torch.where(torch.isfinite(log_p), log_p, torch.full_like(log_p, -float("inf")))
    return float(torch.logsumexp(log_p, dim=0) + np.log(cell))


def quadrature_mass(C, degree, scale, grid_size, device) -> float:
    """Deterministic Riemann estimate of the GMM mass inside {P(x) <= 0}, in chunks."""
    axis = torch.linspace(-scale, scale, grid_size, device=device)
    cell = (2.0 * scale / (grid_size - 1)) ** 2
    total = 0.0
    for start in range(0, grid_size, 256):
        rows = axis[start:start + 256]
        gx, gy = torch.meshgrid(rows, axis, indexing="ij")
        pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)
        density = torch.exp(compute_gmm_log_likelihood(pts, device=device))
        inside = true_region_mask(C, pts, degree=degree, scale=scale)
        total += float((density * inside.float()).sum() * cell)
    return total


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.run_id)
    device = resolve_device()
    set_seed(args.seed)

    siren = load_siren(cfg, device)
    model = build_flow_matcher(cfg, siren, device)
    iteration = load_checkpoint(cfg, model, device)
    model.eval()
    print(f"run_id {cfg.run_id} | iteration {iteration} | device {device}")
    print(f"siren feature {cfg.fm.use_siren_feature} | production step size {cfg.evaluation.step_size}")

    if not cfg.fm.use_siren_feature:
        print("NOTE: this run has no SIREN feature, so check A is vacuous by construction.")

    val_set = get_validation_set(device=device)
    val_polys = val_set["polynomials"][:args.shapes].to(device)
    z_val, _ = extract_validation_latents(siren, cfg, val_polys, device)

    # Points and masses come from the frozen shared set, so this audit scores exactly what
    # the benchmark scores.
    nll_set = load_nll_eval_set(num_points=args.nll_points, degree=cfg.degree, scale=cfg.scale,
                                device=device)
    masses = nll_set["mass"]

    report: dict[str, object] = {"run_id": cfg.run_id, "iteration": iteration,
                                 "shapes": args.shapes, "steps": args.steps}

    # ---- C: is the reference entropy's normalization constant trustworthy? -------------
    print("\n[C] truncated-GMM normalization: Monte Carlo pool fraction vs grid quadrature")
    mass_mc, mass_q = [], []
    for i in range(args.shapes):
        m_mc = float(masses[i])
        m_q = quadrature_mass(val_polys[i], cfg.degree, cfg.scale, args.mass_grid, device)
        mass_mc.append(m_mc)
        mass_q.append(m_q)
    shift = np.log(np.array(mass_mc) / np.maximum(np.array(mass_q), 1e-12))
    print(f"    mass MC   : median {np.median(mass_mc):.4f}")
    print(f"    mass quad : median {np.median(mass_q):.4f}")
    print(f"    implied KLD shift log(mass_MC/mass_quad): "
          f"median {np.median(shift):+.5f} | max |.| {np.abs(shift).max():.5f} nats")
    report["mass_mc"] = mass_mc
    report["mass_quadrature"] = mass_q
    report["kld_shift_from_mass"] = shift.tolist()

    # ---- A: Jacobian completeness, across the step-size ladder -------------------------
    print("\n[A/B] per-shape NLL / KLD with the full Jacobian vs the detached SIREN feature")
    rows: list[dict] = []
    for i in range(args.shapes):
        C, z, mass = val_polys[i], z_val[i], float(masses[i])
        x_valid = nll_set["points"][i]

        entry: dict[str, object] = {"shape": i, "mass": mass, "n_points": int(x_valid.shape[0])}
        for step in args.steps:
            entry[f"fixed@{step}"] = score(model, x_valid, z, mass, step, args.nll_points, device)
            with detached_siren_feature(model):
                entry[f"detached@{step}"] = score(model, x_valid, z, mass, step,
                                                  args.nll_points, device)

        step0 = args.steps[0]
        entry["lognorm_fixed"] = normalization_deficit(model, z, cfg.scale, args.norm_grid,
                                                       step0, device)
        with detached_siren_feature(model):
            entry["lognorm_detached"] = normalization_deficit(model, z, cfg.scale,
                                                              args.norm_grid, step0, device)
        rows.append(entry)
        print(f"    shape {i:>3} | mass {mass:.3f} | "
              f"KLD fixed {entry[f'fixed@{step0}']['kld']:+.4f} "
              f"detached {entry[f'detached@{step0}']['kld']:+.4f} | "
              f"log Z fixed {entry['lognorm_fixed']:+.4f} "
              f"detached {entry['lognorm_detached']:+.4f}", flush=True)

    report["per_shape"] = rows

    def col(variant: str, step: float, key: str) -> np.ndarray:
        return np.array([r[f"{variant}@{step}"][key] for r in rows], dtype=float)

    print(f"\n{'step':>8} {'NLL fixed':>11} {'NLL detach':>11} {'KLD fixed':>11} "
          f"{'KLD detach':>11} {'delta KLD':>11}")
    for step in args.steps:
        kf, kd = col("fixed", step, "kld"), col("detached", step, "kld")
        print(f"{step:>8g} {np.nanmedian(col('fixed', step, 'nll')):>11.4f} "
              f"{np.nanmedian(col('detached', step, 'nll')):>11.4f} "
              f"{np.nanmedian(kf):>11.4f} {np.nanmedian(kd):>11.4f} "
              f"{np.nanmedian(kd - kf):>+11.4f}")

    fine, coarse = args.steps[-1], args.steps[0]
    drift = np.nanmedian(np.abs(col("fixed", coarse, "nll") - col("fixed", fine, "nll")))
    print(f"\n[step size] median |NLL({coarse}) - NLL({fine})| with the full Jacobian: {drift:.4f} nats")
    print(f"[B] median log Z (0 = exactly normalized): "
          f"fixed {np.median([r['lognorm_fixed'] for r in rows]):+.4f} | "
          f"detached {np.median([r['lognorm_detached'] for r in rows]):+.4f}")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.run_id}_likelihood_audit.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")
    print(f"(run directory for reference: {run_dir(cfg.run_id)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
