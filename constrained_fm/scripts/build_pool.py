# -*- coding: utf-8 -*-
"""Stage 1: build the Functa conditioning pool for a config, keyed by its fingerprint.

The pool is the only expensive artifact shared across runs, so it is built once by a
dedicated job and reused by every training run whose SIREN and extraction settings match.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import torch

from constrained_fm.src.datasets.functa_conditioning import build_functa_pool, compute_pool_masses
from constrained_fm.src.experiment.config import ExperimentConfig
from constrained_fm.src.experiment.registry import write_json
from constrained_fm.src.experiment.runtime import (load_siren, proxy_features, resolve_device,
                                                   set_seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or reuse a Functa conditioning pool.")
    parser.add_argument("--config", required=True, help="path to the experiment config YAML")
    parser.add_argument("--smoke", action="store_true", help="shrink the pool for a fast pipeline test")
    parser.add_argument("--force", action="store_true", help="rebuild even if the cached pool exists")
    return parser


def ensure_pool(cfg: ExperimentConfig, device: torch.device, force: bool = False) -> torch.Tensor:
    """Builds the pool if absent and returns its path. Safe to call from the training job."""
    path = cfg.pool_path()
    if path.exists() and not force:
        print(f"pool already cached: {path}")
        return path

    set_seed(cfg.train.seed)
    siren = load_siren(cfg, device)
    proxy_x_pow, proxy_y_pow = proxy_features(cfg, device)

    print(f"building pool of {cfg.pool.size} polynomials -> {path}")
    started = time.time()
    pool = build_functa_pool(
        siren, proxy_x_pow=proxy_x_pow, proxy_y_pow=proxy_y_pow,
        pool_size=cfg.pool.size, degree=cfg.degree, scale=cfg.scale,
        points_per_shape=cfg.extraction.points_per_shape,
        extraction_steps=cfg.extraction.steps, extraction_lr=cfg.extraction.lr,
        min_area=cfg.pool.min_area, max_area=cfg.pool.max_area,
        chunk_size=cfg.pool.chunk_size,
        query_gmm_fraction=cfg.extraction.query_gmm_fraction, device=device,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(pool, tmp)
    tmp.replace(path)

    mass = compute_pool_masses({k: v.to(device) for k, v in pool.items()}, proxy_x_pow, proxy_y_pow)
    write_json(path.with_suffix(".json"), {
        **cfg.provenance(),
        "pool_size": cfg.pool.size,
        "extraction": dataclasses.asdict(cfg.extraction),
        "build_seconds": round(time.time() - started, 1),
        "mass_min": float(mass.min()),
        "mass_median": float(mass.median()),
        "mass_max": float(mass.max()),
    })

    print(f"saved pool in {time.time() - started:.0f}s | "
          f"valid mass min {mass.min():.3f} median {mass.median():.3f}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.smoke:
        cfg = cfg.smoke()

    device = resolve_device()
    print(f"device {device} | config {args.config} | run_id {cfg.run_id}")
    ensure_pool(cfg, device, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
