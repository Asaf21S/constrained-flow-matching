# -*- coding: utf-8 -*-
"""Trains the Functa-conditioned flow matcher for bump2d: amortised polygon constraints.

One network covers every polygon, conditioned on the SIREN latent rather than on any explicit
description of the half-planes. Training runs in the normalised frame like the unconstrained
model, while the pool's polygons stay in physical units, so the containment test that pairs a
target sample with a constraint happens before normalisation.

    python -m constrained_fm.scripts.train_bump_functa
    python -m constrained_fm.scripts.train_bump_functa --iterations 2001 --no-siren-feature
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from tqdm import tqdm

from constrained_fm.src.consts import (BUMP_POOL_PATH, BUMP_QUERY_TARGET_FRACTION,
                                       BUMP_SIREN_CHECKPOINT, BUMP_SIREN_TAU)
from constrained_fm.src.datasets.bump_conditioning import (polygon_batch, polygon_values,
                                                           regression_targets,
                                                           sample_from_polygon_pool,
                                                           sample_query_points)
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.registry import pin_baseline_run, summarize
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.metrics.distributional import compute_mmd, compute_swd
from constrained_fm.src.models.constrained_functa import ConstrainedFlowMatcher
from constrained_fm.src.models.functa_siren import build_modulated_siren
from constrained_fm.src.problems.bump2d import BumpProblem

DEFAULT_OUTDIR = "constrained_fm/baselines/bump2d_functa"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the bump2d Functa-conditioned FM.")
    parser.add_argument("--siren", default=BUMP_SIREN_CHECKPOINT)
    parser.add_argument("--pool", default=BUMP_POOL_PATH)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    parser.add_argument("--iterations", type=int, default=15001)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--mass-weight-power", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--pool-rounds", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--time-emb-dim", type=int, default=128)
    parser.add_argument("--no-siren-feature", action="store_true",
                        help="drop the pointwise SIREN(x, z) input channel")
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--siren-hidden-dim", type=int, default=512)
    parser.add_argument("--siren-layers", type=int, default=4)
    parser.add_argument("--w0", type=float, default=30.0)

    parser.add_argument("--tau", type=float, default=BUMP_SIREN_TAU)
    parser.add_argument("--target-fraction", type=float, default=BUMP_QUERY_TARGET_FRACTION)
    parser.add_argument("--eval-shapes", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=4000)
    parser.add_argument("--eval-pool-size", type=int, default=200000)
    parser.add_argument("--mass-pool-size", type=int, default=20000)
    parser.add_argument("--points-per-shape", type=int, default=1000)
    parser.add_argument("--step-size", type=float, default=0.05)
    return parser


def load_siren(args, device: torch.device):
    siren = build_modulated_siren(latent_dim=args.latent_dim, hidden_dim=args.siren_hidden_dim,
                                  n_layers=args.siren_layers, w0=args.w0).to(device)
    siren.load_state_dict(torch.load(args.siren, map_location=device, weights_only=True))
    siren.eval()
    for p in siren.parameters():
        p.requires_grad_(False)
    return siren


def build_model(args, siren, problem, normalizer, device) -> ConstrainedFlowMatcher:
    """Wires the normalised training frame through to the SIREN's own ``[-1, 1]`` frame.

    The FM works in ``u = (x - mean) / std`` while the SIREN was meta-trained on
    ``2x/L - 1``, so the composed map is ``u * (2 std / L) + (2 mean / L - 1)``.
    """
    plane_scale = problem.domain / (2.0 * normalizer.std)
    coord_shift = 2.0 * normalizer.mean / problem.domain - 1.0
    return ConstrainedFlowMatcher(
        siren=None if args.no_siren_feature else siren,
        spatial_dim=problem.dim, latent_dim=args.latent_dim,
        time_emb_dim=args.time_emb_dim, hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks, plane_scale=plane_scale, coord_shift=coord_shift,
    ).to(device)


def train(args, model, problem, normalizer, pool, device) -> list[float]:
    target = problem.target()
    prob_path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=args.lr_min)

    losses: list[float] = []
    misses = 0
    for iteration in tqdm(range(args.iterations), desc="Training bump2d Functa FM"):
        optimizer.zero_grad(set_to_none=True)

        physical = target.sample(args.batch_size, device=device)
        z, w, hit = sample_from_polygon_pool(physical, pool, rounds=args.pool_rounds,
                                             weight_power=args.mass_weight_power,
                                             max_weight=args.max_weight)
        misses += int((~hit).sum())

        x_1 = normalizer.forward(physical)
        path_sample = prob_path.sample(t=torch.rand(x_1.shape[0], device=device),
                                       x_0=torch.randn_like(x_1), x_1=x_1)
        pred_v = model(path_sample.x_t, path_sample.t, z)

        error = torch.pow(pred_v - path_sample.dx_t, 2).mean(dim=-1)
        loss = (w * error * hit).sum() / hit.sum().clamp_min(1)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (iteration + 1) % args.log_every == 0:
            print(f"| iter {iteration + 1:6d} | loss {loss.item():.5f} "
                  f"| mean {np.mean(losses[-args.log_every:]):.5f} "
                  f"| unpaired {misses} | lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    return losses


def score(args, model, siren, problem, normalizer, device) -> tuple[dict, dict]:
    """Per-constraint feasibility and fidelity on freshly drawn, never-pooled polygons."""
    target = problem.target()
    mass_pool = target.sample(args.mass_pool_size, device=device)
    eval_pool = target.sample(args.eval_pool_size, device=device)

    shapes = polygon_batch(target, mass_pool, args.eval_shapes, domain=problem.domain,
                           min_mass=problem.min_mass, max_mass=problem.max_mass, device=device)
    points = sample_query_points(target, args.eval_shapes, args.points_per_shape,
                                 problem.domain, args.target_fraction, device)
    x_query, y_query = regression_targets(shapes, points, args.tau, problem.domain)
    z, _ = extract_latents_batched(siren, x_query, y_query)

    per_shape: dict[str, list[float]] = {k: [] for k in ("success_rate", "swd", "mmd")}
    for i in tqdm(range(args.eval_shapes), desc="Scoring"):
        samples = normalizer.inverse(
            model.sample(args.eval_samples, z=z[i], step_size=args.step_size, device=device))
        row = {k: shapes[k][i:i + 1] for k in ("normals", "offsets", "active")}

        feasible = polygon_values(samples.unsqueeze(0), **row).squeeze(0) <= 0
        truth = eval_pool[polygon_values(eval_pool.unsqueeze(0), **row).squeeze(0) <= 0]
        per_shape["success_rate"].append(feasible.float().mean().item() * 100.0)
        per_shape["swd"].append(compute_swd(normalizer.forward(samples),
                                            normalizer.forward(truth)))
        per_shape["mmd"].append(compute_mmd(normalizer.forward(samples),
                                            normalizer.forward(truth)))

    per_shape["mass"] = shapes["mass"].tolist()
    return per_shape, summarize(per_shape)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device()
    set_seed(args.seed)

    problem = BumpProblem()
    normalizer = problem.normalizer().to(device)
    siren = load_siren(args, device)
    model = build_model(args, siren, problem, normalizer, device)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"

    losses: list[float] = []
    if args.skip_train:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True),
                              strict=False)
    else:
        pool = {k: v.to(device) for k, v in torch.load(args.pool, map_location=device).items()}
        print(f"device {device} | pool {args.pool} | {pool['mass'].shape[0]} constraints "
              f"| siren feature {not args.no_siren_feature}")
        losses = train(args, model, problem, normalizer, pool, device)
        torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("siren.")},
                   ckpt_path)
        np.save(out / "losses.npy", np.array(losses))

    model.eval()
    per_shape, summary = score(args, model, siren, problem, normalizer, device)

    run_id = pin_baseline_run(out, "bump2d_functa", args, extra={"siren": args.siren})
    artifacts.write_manifest(out, run_id=run_id, method="bump2d_functa")
    (out / "metrics.json").write_text(json.dumps({
        "run_id": run_id,
        "problem": problem.name,
        "model": "ConstrainedFlowMatcher",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "siren": args.siren,
        "pool": args.pool,
        "frame": {"mean": normalizer.mean.tolist(), "std": normalizer.std.tolist()},
        "train": {"iterations": args.iterations, "batch_size": args.batch_size, "lr": args.lr,
                  "mass_weight_power": args.mass_weight_power, "seed": args.seed,
                  "use_siren_feature": not args.no_siren_feature},
        "eval": {"eval_shapes": args.eval_shapes, "eval_samples": args.eval_samples,
                 "step_size": args.step_size},
        "per_shape": per_shape,
        "summary": summary,
        "final_loss": float(np.mean(losses[-500:])) if losses else None,
    }, indent=2))

    print(f"\n### bump2d functa ({run_id})")
    for key in ("success_rate", "swd", "mmd"):
        print(f"  {key:14s} median {summary[f'{key}_median']:.4f} | "
              f"mean {summary[f'{key}_mean']:.4f} | p5 {summary[f'{key}_p5']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
