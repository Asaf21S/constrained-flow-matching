# -*- coding: utf-8 -*-
"""Stage 2: train the Functa-conditioned flow matcher for one config.

Writes a resumable checkpoint and the loss history into runs/<run_id>/, so evaluation,
plotting, and re-evaluation never require retraining.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import numpy as np
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from tqdm.auto import tqdm

from constrained_fm.src.datasets.functa_conditioning import compute_pool_masses, sample_from_functa_pool
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.experiment.config import ExperimentConfig
from constrained_fm.src.experiment.registry import LOSSES_NAME, init_run, write_state
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_pool,
                                                   load_siren, proxy_features, resolve_device,
                                                   save_checkpoint, set_seed)
from constrained_fm.scripts.build_pool import ensure_pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Functa-conditioned flow matcher.")
    parser.add_argument("--config", required=True, help="path to the experiment config YAML")
    parser.add_argument("--smoke", action="store_true", help="shrink every knob for a fast pipeline test")
    parser.add_argument("--resume", action="store_true", help="continue from the run's checkpoint")
    parser.add_argument("--build-pool", action="store_true",
                        help="build the conditioning pool in-job if it is missing")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve the config and artifacts, then exit without training")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.smoke:
        cfg = cfg.smoke()

    device = resolve_device()
    out_dir = init_run(cfg)
    print(f"config   {args.config}")
    print(f"run_id   {cfg.run_id}")
    print(f"run dir  {out_dir}")
    print(f"device   {device}")
    print(f"pool     {cfg.pool_path()}")

    if args.dry_run:
        print("dry run: config and paths resolved, nothing trained")
        return 0

    if args.build_pool:
        ensure_pool(cfg, device)

    set_seed(cfg.train.seed)
    siren = load_siren(cfg, device)
    pool = load_pool(cfg, device)
    proxy_x_pow, proxy_y_pow = proxy_features(cfg, device)
    pool_mass = compute_pool_masses(pool, proxy_x_pow, proxy_y_pow)
    print(f"pool     {pool['C'].shape[0]} shapes | valid mass min {pool_mass.min():.3f} "
          f"median {pool_mass.median():.3f}")

    model = build_flow_matcher(cfg, siren, device)
    prob_path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.iterations, eta_min=cfg.train.lr_min)

    losses_path = out_dir / LOSSES_NAME
    start_iteration = 0
    losses: list[float] = []
    if args.resume:
        start_iteration = load_checkpoint(cfg, model, device, optimizer, scheduler)
        if losses_path.exists():
            losses = np.load(losses_path).tolist()[:start_iteration]
        print(f"resumed at iteration {start_iteration}")

    trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"model    {trainable / 1e6:.2f}M trainable parameters")
    write_state(cfg.run_id, status="training", iteration=start_iteration,
                started_at=datetime.now().isoformat(timespec="seconds"))

    started = time.time()
    model.train()
    for iteration in tqdm(range(start_iteration, cfg.train.iterations), initial=start_iteration,
                          total=cfg.train.iterations, desc="training"):
        optimizer.zero_grad()

        x_1, _ = get_points(cfg.train.batch_size, device=device)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=device)
        path_sample = prob_path.sample(t=t, x_0=x_0, x_1=x_1)

        _, z, w = sample_from_functa_pool(
            path_sample.x_1, pool, degree=cfg.degree, scale=cfg.scale,
            mass_pos=pool_mass, weight_power=cfg.train.mass_weight_power,
            max_weight=cfg.train.max_weight,
        )

        pred_v = model(path_sample.x_t, path_sample.t, z)
        loss = (w * torch.pow(pred_v - path_sample.dx_t, 2).mean(dim=-1)).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (iteration + 1) % cfg.train.log_every == 0:
            print(f"| iter {iteration + 1:6d} | loss {loss.item():8.5f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

        if (iteration + 1) % cfg.train.checkpoint_every == 0:
            save_checkpoint(cfg, model, optimizer, scheduler, iteration + 1)
            np.save(losses_path, np.asarray(losses, dtype=np.float32))
            write_state(cfg.run_id, status="training", iteration=iteration + 1)

    save_checkpoint(cfg, model, optimizer, scheduler, cfg.train.iterations, include_optimizer=False)
    np.save(losses_path, np.asarray(losses, dtype=np.float32))
    write_state(cfg.run_id, status="trained", iteration=cfg.train.iterations,
                train_seconds=round(time.time() - started, 1),
                finished_at=datetime.now().isoformat(timespec="seconds"))

    tail_loss = float(np.mean(losses[-200:]))
    print(f"done in {(time.time() - started) / 60:.1f} min | final loss (last 200) {tail_loss:.5f}")
    print(f"RUN_ID={cfg.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
