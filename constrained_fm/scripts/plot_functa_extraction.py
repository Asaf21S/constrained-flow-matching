# -*- coding: utf-8 -*-
"""Regenerates the Functa reconstruction figure: GT polynomial vs. decoded SIREN field.

Samples fresh polynomials, extracts their latents with the config's CAVIA inner loop, and
renders one row per shape into constrained_fm/images/functa/polynomial_functa.png.

    python -m constrained_fm.scripts.plot_functa_extraction --run-id <run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch

from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.experiment.registry import load_config
from constrained_fm.src.experiment.runtime import load_siren, resolve_device, set_seed
from constrained_fm.src.geometry.polynomials import compute_poly_features_batched, evaluate_poly_batched
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.visualization import diagnostics as diag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the Functa extraction figure.")
    parser.add_argument("--run-id", required=True, help="run whose SIREN / extraction config to use")
    parser.add_argument("--num-shapes", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=500,
                        help="rendering grid per axis; higher removes jaggedness in the zero level set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="constrained_fm/images/functa/polynomial_functa.png")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.run_id)
    device = resolve_device()
    siren = load_siren(cfg, device)
    print(f"run_id {cfg.run_id} | device {device}")

    set_seed(args.seed)
    polys = sample_valid_polynomials(args.num_shapes, degree=cfg.degree, scale=cfg.scale,
                                     min_area=cfg.pool.min_area, max_area=cfg.pool.max_area,
                                     device=device)

    X_raw = sample_query_points(args.num_shapes, cfg.extraction.points_per_shape, scale=cfg.scale,
                                gmm_fraction=cfg.extraction.query_gmm_fraction, device=device)
    x_pow, y_pow = compute_poly_features_batched(X_raw, degree=cfg.degree, scale=cfg.scale)
    Y = torch.tanh(evaluate_poly_batched(x_pow, y_pow, polys))
    z_batch, extraction_mse = extract_latents_batched(siren, X_raw / cfg.scale, Y,
                                                      lr=cfg.extraction.lr,
                                                      steps=cfg.extraction.steps)
    print(f"mean extraction MSE: {float(extraction_mse.mean()):.6f}")

    fig = diag.plot_functa_extraction(siren, polys, z_batch, degree=cfg.degree, scale=cfg.scale,
                                      resolution=args.resolution)
    path = diag.save_figure(fig, Path(args.out))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
