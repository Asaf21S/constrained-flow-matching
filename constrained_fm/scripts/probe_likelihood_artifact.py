# -*- coding: utf-8 -*-
"""Traces the faint parallel streaks visible in the model likelihood maps.

Three candidate causes, one panel each:

  A. the frozen SIREN's own field. Its first sine layer is NOT modulated by z, so
     sin(w0 * (W x + b)) lays down fixed parallel ridges whose orientation depends only on
     the frozen input weights. Those ripples enter the flow matcher through the pointwise
     SIREN(x_t, z) feature. If this is the cause, the residual SIREN(x, z) - tanh(P(x))
     shows stripes at the SAME positions for two different constraints.
  B. ODE discretization. Recomputing the likelihood at a 5x smaller step size removes the
     streaks if they are an artifact of the midpoint solver rather than of the field.
  C. transport separatrices. These would move with the constraint, since the truncated
     target - and therefore where the map has to tear - changes with it.

    python -m constrained_fm.scripts.probe_likelihood_artifact --run-id <run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.experiment.registry import load_config
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_siren,
                                                   resolve_device, set_seed)
from constrained_fm.src.geometry.polynomials import (compute_poly_features,
                                                     compute_poly_features_batched,
                                                     evaluate_poly, evaluate_poly_batched)
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.metrics.functa_fidelity import uniform_grid_points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain the streaks in the likelihood maps.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--grid", type=int, default=400, help="grid for the field panels")
    parser.add_argument("--likelihood-grid", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--outdir", default="outputs/artifact_probe")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.run_id)
    device = resolve_device()
    siren = load_siren(cfg, device)
    model = build_flow_matcher(cfg, siren, device)
    load_checkpoint(cfg, model, device)
    model.eval()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    polys = sample_valid_polynomials(2, degree=cfg.degree, scale=cfg.scale,
                                     min_area=cfg.pool.min_area, max_area=cfg.pool.max_area,
                                     device=device)
    X_raw = sample_query_points(2, cfg.extraction.points_per_shape, scale=cfg.scale,
                                gmm_fraction=cfg.extraction.query_gmm_fraction, device=device)
    x_pow, y_pow = compute_poly_features_batched(X_raw, degree=cfg.degree, scale=cfg.scale)
    Y = torch.tanh(evaluate_poly_batched(x_pow, y_pow, polys))
    z_batch, _ = extract_latents_batched(siren, X_raw / cfg.scale, Y, lr=cfg.extraction.lr,
                                         steps=cfg.extraction.steps)

    grid = uniform_grid_points(grid_size=args.grid, scale=cfg.scale, device=device)
    gx_pow, gy_pow = compute_poly_features(grid, degree=cfg.degree, scale=cfg.scale)
    extent = (-cfg.scale, cfg.scale, -cfg.scale, cfg.scale)

    # A: does the SIREN's decoding error carry fixed stripes, independent of the constraint?
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6))
    residuals = []
    for k in range(2):
        C = polys[k]
        with torch.no_grad():
            pred = siren(grid / cfg.scale, z_batch[k].view(-1)).squeeze(-1)
        true = torch.tanh(evaluate_poly(gx_pow, gy_pow,
                                        C.unsqueeze(0).expand(grid.shape[0], -1, -1)).squeeze(-1))
        res = (pred - true).reshape(args.grid, args.grid).cpu().numpy().T
        residuals.append(res)
        lim = float(np.percentile(np.abs(res), 99))
        im = axs[k].imshow(res, extent=extent, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        axs[k].set_title(f"SIREN(x, z) - tanh(P(x))   |   constraint {k + 1}")
        fig.colorbar(im, ax=axs[k])

    diff = residuals[0] - residuals[1]
    lim = float(np.percentile(np.abs(diff), 99))
    im = axs[2].imshow(diff, extent=extent, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[2].set_title("difference of the two residuals\n(stripes surviving here are z-dependent)")
    fig.colorbar(im, ax=axs[2])
    fig.tight_layout()
    fig.savefig(out / "siren_residual_stripes.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[A] wrote {out / 'siren_residual_stripes.png'}")

    corr = float(np.corrcoef(residuals[0].ravel(), residuals[1].ravel())[0, 1])
    print(f"[A] corr(residual_1, residual_2) = {corr:+.3f}  "
          f"(high => the stripe pattern does not depend on the constraint)")

    # B: do the streaks survive a 5x finer integration step?
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6))
    maps = {}
    for ax, step in zip(axs[:2], [cfg.evaluation.step_size, cfg.evaluation.step_size / 5]):
        lik = model.compute_likelihood_grid(z=z_batch[0], siren=siren, degree=cfg.degree,
                                            scale=cfg.scale, grid_size=args.likelihood_grid,
                                            step_size=step, device=device)
        maps[step] = np.asarray(lik)
        ax.imshow(maps[step], extent=extent, origin="lower", cmap="viridis")
        ax.set_title(f"likelihood | step size {step:g}")

    delta = maps[cfg.evaluation.step_size] - maps[cfg.evaluation.step_size / 5]
    lim = float(np.percentile(np.abs(delta), 99.5)) or 1e-12
    im = axs[2].imshow(delta, extent=extent, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[2].set_title("coarse - fine\n(streaks here => solver artifact)")
    fig.colorbar(im, ax=axs[2])
    fig.tight_layout()
    fig.savefig(out / "likelihood_step_size.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[B] wrote {out / 'likelihood_step_size.png'}")

    both = maps[cfg.evaluation.step_size], maps[cfg.evaluation.step_size / 5]
    finite = np.isfinite(both[0]) & np.isfinite(both[1]) & (both[0] > 0)
    rel = np.abs(both[0][finite] - both[1][finite]) / np.maximum(both[0][finite], 1e-12)
    print(f"[B] median relative change from 5x finer steps: {float(np.median(rel)):.4%}")

    spectral_report(cfg, siren, model, z_batch, args, device, maps[cfg.evaluation.step_size])
    return 0


def dominant_stripe(field: np.ndarray, scale: float, min_cycles: float = 3.0):
    """Wavelength and orientation of the strongest periodic component, ignoring smooth trend."""
    from scipy.ndimage import uniform_filter

    valid = np.isfinite(field) & (field != 0)
    work = np.where(valid, field, 0.0).astype(float)
    if work.std() == 0:
        return None

    high_pass = work - uniform_filter(work, size=9)
    high_pass *= np.hanning(high_pass.shape[0])[:, None] * np.hanning(high_pass.shape[1])[None, :]

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(high_pass)))
    n = field.shape[0]
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=2.0 * scale / (n - 1)))  # cycles per plot unit
    fx, fy = np.meshgrid(freqs, freqs, indexing="ij")
    radius = np.sqrt(fx ** 2 + fy ** 2)

    # Suppress the smooth trend and the Nyquist corner, keeping resolvable stripe frequencies.
    band = (radius > min_cycles / (2.0 * scale)) & (radius < 0.4 / (2.0 * scale / (n - 1)))
    spectrum = np.where(band, spectrum, 0.0)

    peak = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
    fx_p, fy_p = float(fx[peak]), float(fy[peak])
    wavelength = 1.0 / np.sqrt(fx_p ** 2 + fy_p ** 2)
    orientation = np.degrees(np.arctan2(fy_p, fx_p)) % 180.0
    power_ratio = float(spectrum[peak] / (spectrum.sum() + 1e-12))
    return wavelength, orientation, power_ratio, high_pass


def spectral_report(cfg, siren, model, z_batch, args, device, coarse_map) -> None:
    """Compares the streak period against the SIREN's own first-layer sine frequencies."""
    print("\n[C] spectral attribution")

    # The input layer is never modulated by z, so its frequencies are identical for every
    # constraint: lambda = 2*pi*scale / (w0 * ||W_row||) in plot units.
    W = siren.input_linear.weight.detach()
    norms = torch.linalg.norm(W, dim=1)
    lam = (2 * np.pi * cfg.scale / (siren.w0 * norms)).cpu().numpy()
    print(f"    SIREN first-layer stripe wavelengths (plot units): "
          f"p5 {np.percentile(lam, 5):.2f} | median {np.median(lam):.2f} | p95 {np.percentile(lam, 95):.2f}")

    lik = np.where(np.isfinite(coarse_map) & (coarse_map > 0), coarse_map, 0.0)
    found = dominant_stripe(lik, cfg.scale)
    if found:
        wl, ang, ratio, hp_a = found
        print(f"    likelihood streaks: wavelength {wl:.2f} | orientation {ang:.1f} deg | "
              f"peak power share {ratio:.4f}")

    # If the streaks come from the frozen input layer they sit at the same place for any z.
    second = model.compute_likelihood_grid(z=z_batch[1], siren=siren, degree=cfg.degree,
                                           scale=cfg.scale, grid_size=args.likelihood_grid,
                                           step_size=cfg.evaluation.step_size, device=device)
    second = np.asarray(second)
    lik2 = np.where(np.isfinite(second) & (second > 0), second, 0.0)
    found2 = dominant_stripe(lik2, cfg.scale)
    if found2:
        wl2, ang2, ratio2, hp_b = found2
        print(f"    second constraint  : wavelength {wl2:.2f} | orientation {ang2:.1f} deg | "
              f"peak power share {ratio2:.4f}")

        overlap = (lik > 0) & (lik2 > 0)
        if overlap.sum() > 100:
            corr = float(np.corrcoef(hp_a[overlap], hp_b[overlap])[0, 1])
            print(f"    corr of high-passed maps on shared support: {corr:+.3f}  "
                  f"(high => constraint-independent, i.e. baked into the frozen field)")


if __name__ == "__main__":
    raise SystemExit(main())
