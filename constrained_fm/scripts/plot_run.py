# -*- coding: utf-8 -*-
"""Redraws a run's figures from its saved artifacts. No checkpoint, no ODE solve.

Use this for every figure tweak that goes into the paper: it reads only
``artifacts/*.npy``, ``metrics.json`` and ``losses.npy``, so iteration is seconds long and
the rendered panel is guaranteed to be the same tensor the reported metrics were computed
from. If an artifact is missing, re-run the evaluation stage for that run rather than
regenerating data here.

    python -m constrained_fm.scripts.plot_run --run-id baseline-129f59a4
    python -m constrained_fm.scripts.plot_run --root constrained_fm/baselines/poly_fm
    python -m constrained_fm.scripts.plot_run --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.config import REPO_ROOT
from constrained_fm.src.experiment.registry import RUNS_ROOT, run_dir
from constrained_fm.src.visualization.run_figures import render_run_figures

BASELINES_ROOT = REPO_ROOT / "constrained_fm" / "baselines"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-render figures from saved artifacts only.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", nargs="+", help="run ids under runs/")
    source.add_argument("--root", nargs="+", help="explicit artifact roots, e.g. a baseline dir")
    source.add_argument("--all", action="store_true", help="every run and baseline with artifacts")
    parser.add_argument("--out", help="figure directory; defaults to <root>/figures")
    return parser


def discover_roots() -> list[Path]:
    candidates = list(RUNS_ROOT.iterdir()) if RUNS_ROOT.exists() else []
    candidates += list(BASELINES_ROOT.iterdir()) if BASELINES_ROOT.exists() else []
    return [p for p in sorted(candidates) if p.is_dir() and artifacts.load_manifest(p)]


def resolve_roots(args) -> list[Path]:
    if args.all:
        return discover_roots()
    if args.run_id:
        return [run_dir(rid) for rid in args.run_id]
    return [Path(r) if Path(r).is_absolute() else REPO_ROOT / r for r in args.root]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = resolve_roots(args)
    if not roots:
        print("no artifact roots found; run the evaluation stage first")
        return 1

    failures = 0
    for root in roots:
        manifest = artifacts.load_manifest(root)
        try:
            paths = render_run_figures(root, out_dir=args.out)
        except FileNotFoundError as error:
            print(f"[skip] {root.name}: {error}")
            failures += 1
            continue
        print(f"[{root.name}] method={manifest.get('method', '?')} "
              f"commit={manifest.get('git_commit', '?')[:8]} -> {len(paths)} figures")
        for path in paths:
            print(f"    {path}")

    return 1 if failures == len(roots) else 0


if __name__ == "__main__":
    raise SystemExit(main())
