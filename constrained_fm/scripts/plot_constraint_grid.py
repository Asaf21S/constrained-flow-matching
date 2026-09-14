# -*- coding: utf-8 -*-
"""Renders grid(s) of validation-set constraint boundaries as a standalone paper figure.

    python -m constrained_fm.scripts.plot_constraint_grid --num-variants 1 --seed-start 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from constrained_fm.src.datasets.validation_v1k import resolve_validation_set
from constrained_fm.src.visualization.constraint_grid import (plot_constraint_grid,
                                                               render_constraint_grid_variants)

DEFAULT_OUT_DIR = "constrained_fm/images/thesis_pool/target_and_constraints"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="v1k", choices=["v1k", "legacy100"])
    parser.add_argument("--nrows", type=int, default=2)
    parser.add_argument("--ncols", type=int, default=5)
    parser.add_argument("--num-variants", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    val_set = resolve_validation_set(args.dataset)
    polys, masses = val_set["polynomials"], val_set["mass"]

    if args.num_variants == 1:
        out_path = Path(args.out_dir) / f"constraint_grid_seed{args.seed_start}"
        plot_constraint_grid(polys, masses, nrows=args.nrows, ncols=args.ncols,
                             seed=args.seed_start, save_path=out_path, show=False)
        print(f"Wrote {out_path}.png and {out_path}.pdf")
    else:
        paths = render_constraint_grid_variants(polys, masses, args.out_dir, args.num_variants,
                                                nrows=args.nrows, ncols=args.ncols,
                                                seed_start=args.seed_start)
        for p in paths:
            print(f"Wrote {p}.png and {p}.pdf")


if __name__ == "__main__":
    main()
