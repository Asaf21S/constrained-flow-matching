# -*- coding: utf-8 -*-
"""Validates experiment configs and prints their derived identities.

Imports no torch, so it runs on the login node in milliseconds. Also serves as the
shell-scriptable accessor for run ids and pool paths used by scripts/run_sweep.sh.
"""

from __future__ import annotations

import argparse
import sys

from constrained_fm.src.experiment.config import ExperimentConfig
from constrained_fm.src.experiment.registry import run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate configs and print derived identities.")
    parser.add_argument("configs", nargs="+", help="paths to config YAML files")
    parser.add_argument("--smoke", action="store_true", help="apply the smoke-test shrink first")
    field = parser.add_mutually_exclusive_group()
    field.add_argument("--run-id", action="store_true", help="print only the run id")
    field.add_argument("--pool-key", action="store_true", help="print only the pool fingerprint")
    field.add_argument("--pool-path", action="store_true", help="print only the pool path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    terse = args.run_id or args.pool_key or args.pool_path

    failures = 0
    for path in args.configs:
        try:
            cfg = ExperimentConfig.from_yaml(path)
            if args.smoke:
                cfg = cfg.smoke()
        except Exception as exc:
            print(f"{path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue

        if args.run_id:
            print(cfg.run_id)
        elif args.pool_key:
            print(cfg.pool_fingerprint())
        elif args.pool_path:
            print(cfg.pool_path())
        else:
            pool = cfg.pool_path()
            run = run_dir(cfg.run_id)
            print(f"{path}")
            print(f"  run_id     {cfg.run_id}")
            print(f"  run dir    {run}  [{'exists' if run.exists() else 'new'}]")
            print(f"  siren      {cfg.siren_path().name}  sha {cfg.siren_digest()[:10]}")
            print(f"  pool       {pool.name}  [{'cached' if pool.exists() else 'MISSING'}]")
            print(f"  train      {cfg.train.iterations} iters | bs {cfg.train.batch_size} | "
                  f"lr {cfg.train.lr} | mass_power {cfg.train.mass_weight_power}")

    if failures and not terse:
        print(f"\n{failures} config(s) failed validation", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
