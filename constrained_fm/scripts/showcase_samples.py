# -*- coding: utf-8 -*-
"""Renders showcase figures for constraints the model has never been scored on.

Draws fresh polynomials with a seed disjoint from the frozen validation benchmark, so these
panels are a qualitative demonstration rather than a re-run of the reported metrics. Each
shape gets a samples figure annotated with success rate, SWD, MMD and JSD, plus the exact
model likelihood on the same constraint.

    python -m constrained_fm.scripts.showcase_samples --run-id <run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch

from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.experiment.registry import load_config
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_siren,
                                                   resolve_device, set_seed)
from constrained_fm.src.geometry.polynomials import compute_poly_features_batched, evaluate_poly_batched
from constrained_fm.src.inference.evaluator import evaluate_single_configuration
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.metrics.functa_fidelity import region_iou
from constrained_fm.src.visualization import diagnostics as diag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render showcase figures on unseen polynomials.")
    parser.add_argument("--run-id", required=True, help="evaluated run whose checkpoint to use")
    parser.add_argument("--num-shapes", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260815,
                        help="must differ from the validation set seed to keep these shapes unseen")
    parser.add_argument("--outdir", default="constrained_fm/images/functa/showcase")
    parser.add_argument("--prefix", default="showcase")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.run_id)
    device = resolve_device()
    ev = cfg.evaluation

    siren = load_siren(cfg, device)
    model = build_flow_matcher(cfg, siren, device)
    iteration = load_checkpoint(cfg, model, device)
    model.eval()
    print(f"run_id {cfg.run_id} | iteration {iteration} | device {device}")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    polys = sample_valid_polynomials(args.num_shapes, degree=cfg.degree, scale=cfg.scale,
                                     min_area=cfg.pool.min_area, max_area=cfg.pool.max_area,
                                     device=device)

    # These must not be benchmark shapes, or the figures would just restate the metrics table.
    val_polys = get_validation_set(device=device)["polynomials"].to(device)
    overlap = (polys.unsqueeze(1) - val_polys.unsqueeze(0)).abs().amax(dim=(2, 3)).min()
    print(f"closest distance to any validation polynomial: {float(overlap):.4f}")
    assert float(overlap) > 1e-6, "sampled a polynomial from the validation set"

    gmm_pool, _ = get_points(ev.gmm_pool_size, device=device)

    X_raw = sample_query_points(polys.shape[0], cfg.extraction.points_per_shape, scale=cfg.scale,
                                gmm_fraction=cfg.extraction.query_gmm_fraction, device=device)
    x_pow, y_pow = compute_poly_features_batched(X_raw, degree=cfg.degree, scale=cfg.scale)
    Y = torch.tanh(evaluate_poly_batched(x_pow, y_pow, polys))
    z_batch, extraction_mse = extract_latents_batched(siren, X_raw / cfg.scale, Y,
                                                      lr=cfg.extraction.lr,
                                                      steps=cfg.extraction.steps)

    rows = []
    for i in range(polys.shape[0]):
        C, z = polys[i], z_batch[i]

        samples = model.sample(num_points=args.num_samples, z=z, step_size=ev.step_size,
                               return_intermediates=False, device=device)
        if samples.ndim == 3:
            samples = samples[-1]

        metrics = evaluate_single_configuration(samples, x_true_pool=gmm_pool, coeffs=C,
                                                degree=cfg.degree, scale=cfg.scale, device=device)
        iou = region_iou(siren, z, C, gmm_pool, degree=cfg.degree, scale=cfg.scale)
        mass = float((evaluate_poly_batched(*compute_poly_features_batched(
            gmm_pool.unsqueeze(0), degree=cfg.degree, scale=cfg.scale),
            C.unsqueeze(0)).squeeze() <= 0).float().mean())

        title = (f"Unseen constraint #{i + 1}   |   SR {metrics['success_rate']:.2f}%   "
                 f"SWD {metrics['swd']:.4f}   MMD {metrics['mmd']:.5f}   JSD {metrics['jsd']:.4f}\n"
                 f"constraint mass {mass:.3f}   |   decoded-region mass IoU {iou:.3f}")

        samples_path = out / f"{args.prefix}_{i + 1}_samples.png"
        diag.save_figure(
            diag.plot_final_samples(samples.cpu().numpy(), coeffs=C, title=title,
                                    degree=cfg.degree, scale=cfg.scale),
            samples_path)

        likelihood_path = out / f"{args.prefix}_{i + 1}_likelihood.png"
        likelihood = model.compute_likelihood_grid(z=z, siren=siren, degree=cfg.degree,
                                                   scale=cfg.scale, grid_size=ev.likelihood_grid,
                                                   step_size=ev.step_size, device=device)
        diag.save_figure(
            diag.plot_likelihood(likelihood, coeffs=C, degree=cfg.degree, scale=cfg.scale,
                                 grid_size=ev.likelihood_grid, device=device),
            likelihood_path)

        rows.append((i + 1, metrics, mass, iou, float(extraction_mse[i])))
        print(f"[{i + 1}/{polys.shape[0]}] {samples_path.name}, {likelihood_path.name}")

    print(f"\n{'shape':>6}{'SR':>9}{'SWD':>9}{'MMD':>10}{'JSD':>9}{'mass':>8}{'massIoU':>9}{'extrMSE':>10}")
    for idx, metrics, mass, iou, mse in rows:
        print(f"{idx:>6}{metrics['success_rate']:>9.2f}{metrics['swd']:>9.4f}"
              f"{metrics['mmd']:>10.5f}{metrics['jsd']:>9.4f}{mass:>8.3f}{iou:>9.3f}{mse:>10.5f}")

    print("\n| # | Success rate (%) | SWD | MMD | JSD | mass | mass IoU |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for idx, metrics, mass, iou, _ in rows:
        print(f"| {idx} | {metrics['success_rate']:.2f} | {metrics['swd']:.4f} | "
              f"{metrics['mmd']:.5f} | {metrics['jsd']:.4f} | {mass:.3f} | {iou:.3f} |")

    print(f"\nfigures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
