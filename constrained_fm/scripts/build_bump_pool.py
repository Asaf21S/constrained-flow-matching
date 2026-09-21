# -*- coding: utf-8 -*-
"""Extracts Functa latents for a pool of bump2d polygons, once, for reuse during training.

The FM sees a fresh constraint on every iteration; paying a 15-step CAVIA extraction there
would dominate the step. Precomputing a large pool turns that into a table lookup, at the
cost of the FM only ever seeing this many distinct constraints.

    python -m constrained_fm.scripts.build_bump_pool
    python -m constrained_fm.scripts.build_bump_pool --pool-size 5000 --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from constrained_fm.src.consts import (BUMP_POOL_PATH, BUMP_QUERY_TARGET_FRACTION,
                                       BUMP_SIREN_CHECKPOINT, BUMP_SIREN_TAU)
from constrained_fm.src.datasets.bump_conditioning import build_polygon_pool
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.models.functa_siren import build_modulated_siren
from constrained_fm.src.problems.bump2d import BumpProblem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the bump2d Functa conditioning pool.")
    parser.add_argument("--siren", default=BUMP_SIREN_CHECKPOINT)
    parser.add_argument("--out", default=BUMP_POOL_PATH)
    parser.add_argument("--pool-size", type=int, default=100000)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--points-per-shape", type=int, default=1000)
    parser.add_argument("--extraction-steps", type=int, default=15)
    parser.add_argument("--extraction-lr", type=float, default=6.25e-4)
    parser.add_argument("--tau", type=float, default=BUMP_SIREN_TAU)
    parser.add_argument("--target-fraction", type=float, default=BUMP_QUERY_TARGET_FRACTION)
    # Only resolves the stored mass and the acceptance filter; 20k points already pin a mass
    # in [0.02, 0.98] to about 0.004, and the filter runs once per chunk.
    parser.add_argument("--mass-pool-size", type=int, default=20000)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--w0", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="rebuild even if the pool exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists; pass --force to rebuild")
        return 0

    device = resolve_device()
    set_seed(args.seed)

    siren = build_modulated_siren(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
                                  n_layers=args.n_layers, w0=args.w0).to(device)
    siren.load_state_dict(torch.load(args.siren, map_location=device, weights_only=True))
    siren.eval()

    problem = BumpProblem()
    target = problem.target()
    mass_pool = target.sample(args.mass_pool_size, device=device)
    print(f"device {device} | siren {args.siren} | pool size {args.pool_size}")

    pool = build_polygon_pool(siren, target, mass_pool, pool_size=args.pool_size,
                             points_per_shape=args.points_per_shape,
                             extraction_steps=args.extraction_steps,
                             extraction_lr=args.extraction_lr, chunk_size=args.chunk_size,
                             tau=args.tau, domain=problem.domain,
                             target_fraction=args.target_fraction,
                             min_mass=problem.min_mass, max_mass=problem.max_mass,
                             device=device)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pool, out)
    mass = pool["mass"]
    meta = {"siren": args.siren, "pool_size": args.pool_size, "tau": args.tau,
            "target_fraction": args.target_fraction, "extraction_steps": args.extraction_steps,
            "extraction_lr": args.extraction_lr, "points_per_shape": args.points_per_shape,
            "seed": args.seed, "mass_mean": float(mass.mean()),
            "mass_p5": float(mass.quantile(0.05)), "mass_p95": float(mass.quantile(0.95))}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {out} | mass mean {mass.mean():.3f} "
          f"[{mass.min():.3f}, {mass.max():.3f}] | z {tuple(pool['z'].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
