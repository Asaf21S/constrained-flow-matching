# -*- coding: utf-8 -*-
"""Regression check that CAVIA extraction is invariant to the chunk it is called with.

extract_latents_batched averages the inner loss over points and sums over shapes, so each
z_i takes the step of its own mean MSE and the result must not depend on how many shapes
share a call. Before that fix the loss was mean-reduced over (batch, points), which divided
the per-shape step by the chunk size: meta-training adapts at batch 16 while build_pool and
eval_fm extract in chunks of 128, an 8x smaller step for the same 15 steps, and CAVIA only
meta-learns an initialization that is optimal at the step size it trained with.

Boundary agreement is scored on held-out GMM samples, which is what the flow matcher
depends on. Every row should now agree to within sampling noise.

    python -m constrained_fm.scripts.probe_extraction_scale --config <config.yaml>
"""

from __future__ import annotations

import argparse

import torch

from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.experiment.config import ExperimentConfig
from constrained_fm.src.experiment.runtime import load_siren, resolve_device, set_seed
from constrained_fm.src.geometry.polynomials import compute_poly_features_batched, evaluate_poly_batched
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.metrics.functa_fidelity import region_iou_batched

META_TRAIN_BATCH = 16  # scripts/train_functa.py batch_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep the extraction chunk size / step scale.")
    parser.add_argument("--config", required=True, help="config YAML to probe")
    parser.add_argument("--num-polys", type=int, default=100)
    parser.add_argument("--iou-samples", type=int, default=20000)
    parser.add_argument("--chunks", type=int, nargs="+", default=[128, 64, 32, 16, 8, 4, 1])
    return parser


def extract(siren, cfg: ExperimentConfig, polys: torch.Tensor, device, chunk: int,
            lr: float) -> tuple[torch.Tensor, torch.Tensor]:
    z_chunks, mse_chunks = [], []
    for start in range(0, polys.shape[0], chunk):
        C = polys[start:start + chunk]
        X_raw = sample_query_points(C.shape[0], cfg.extraction.points_per_shape, scale=cfg.scale,
                                    gmm_fraction=cfg.extraction.query_gmm_fraction, device=device)
        x_pow, y_pow = compute_poly_features_batched(X_raw, degree=cfg.degree, scale=cfg.scale)
        Y = torch.tanh(evaluate_poly_batched(x_pow, y_pow, C))
        z, mse = extract_latents_batched(siren, X_raw / cfg.scale, Y, lr=lr,
                                         steps=cfg.extraction.steps)
        z_chunks.append(z)
        mse_chunks.append(mse)
    return torch.cat(z_chunks), torch.cat(mse_chunks)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    device = resolve_device()
    siren = load_siren(cfg, device)

    val_polys = get_validation_set(device=device)["polynomials"][:args.num_polys].to(device)
    gmm_pool, _ = get_points(100000, device=device)
    iou_points = gmm_pool[torch.randperm(gmm_pool.shape[0], device=device)[:args.iou_samples]]

    print(f"checkpoint {cfg.siren_path().name}")
    print(f"extraction {cfg.extraction.steps} steps @ lr {cfg.extraction.lr} | "
          f"{cfg.extraction.points_per_shape} pts | gmm fraction {cfg.extraction.query_gmm_fraction}")
    print(f"meta-training adapted with batch {META_TRAIN_BATCH}, so the matching chunk is "
          f"{META_TRAIN_BATCH}\n")

    header = f"{'chunk':>6} {'lr':>10} {'massIoU mean':>13} {'median':>8} {'p5':>8} {'extrMSE':>10}"
    print(header)
    print("-" * len(header))

    for chunk in args.chunks:
        set_seed(cfg.evaluation.seed)
        z, mse = extract(siren, cfg, val_polys, device, chunk, cfg.extraction.lr)
        iou = region_iou_batched(siren, z, val_polys, iou_points, degree=cfg.degree,
                                 scale=cfg.scale)
        ordered = iou.sort().values
        print(f"{chunk:>6} {cfg.extraction.lr:>10.4g} "
              f"{float(iou.mean()):>13.4f} {float(ordered[len(ordered) // 2]):>8.4f} "
              f"{float(ordered[max(0, int(0.05 * (len(ordered) - 1)))]):>8.4f} "
              f"{float(mse.mean()):>10.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
