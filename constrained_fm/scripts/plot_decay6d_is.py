# -*- coding: utf-8 -*-
"""Draws the decay6d importance-sampling figures from saved artifacts; never runs a model.

    python -m constrained_fm.scripts.plot_decay6d_is
    python -m constrained_fm.scripts.plot_decay6d_is --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from constrained_fm.src.experiment import artifacts  # noqa: E402
from constrained_fm.src.visualization import decay6d_is as viz  # noqa: E402

EVAL_DIR = "constrained_fm/baselines/decay6d_is/eval"
SMOKE_EVAL_DIR = "constrained_fm/baselines/decay6d_is/smoke/eval"
FIGURE_DIR = "constrained_fm/images/thesis_pool/decay6d_is"
SMOKE_FIGURE_DIR = "constrained_fm/baselines/decay6d_is/smoke/figures"
OBSERVABLE_LABELS = {"p2_norm": r"$E[|\vec p_2|\,|\,\mathcal{B}]$",
                     "p2_z": r"$E[p_{2z}\,|\,\mathcal{B}]$",
                     "p2_tail": r"$P(|\vec p_2| > \tau_{\mathcal{B}}\,|\,\mathcal{B})$"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="decay6d IS figures from artifacts.")
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / name}.{{png,pdf}}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_dir = Path(args.eval_dir or (SMOKE_EVAL_DIR if args.smoke else EVAL_DIR))
    out = Path(args.figure_dir or (SMOKE_FIGURE_DIR if args.smoke else FIGURE_DIR))
    out.mkdir(parents=True, exist_ok=True)

    manifest = artifacts.load_manifest(eval_dir)
    if not manifest:
        raise FileNotFoundError(f"no artifact manifest under {eval_dir}; run the merge stage")
    boxes, estimators = manifest["boxes"], manifest["estimators"]
    weights, observables = manifest["weights"], manifest["observables"]
    bench_dir = Path(manifest["benchmark_dir"])
    arr = {k: artifacts.load_array(eval_dir, k) for k in
           ("estimates", "gt_mean", "gt_se", "n_values", "ess_frac", "max_weight")}

    for k, name in enumerate(observables):
        save(viz.plot_error_vs_n(arr["n_values"], arr["estimates"], arr["gt_mean"], arr["gt_se"],
                                 boxes, estimators, k, OBSERVABLE_LABELS[name]),
             out, f"error_vs_n_{name}")

    log_w = {kind: [artifacts.load_array(eval_dir, f"log_w_{kind}_box{b}")
                    for b in range(len(boxes))] for kind in weights}
    save(viz.plot_weight_histograms(log_w, boxes), out, "weight_histograms")
    save(viz.plot_ess(arr["n_values"], arr["ess_frac"], arr["max_weight"], boxes, weights),
         out, "ess_vs_n")

    for b, name in enumerate(boxes):
        fig = viz.plot_p2_marginals(artifacts.load_array(bench_dir, f"gt_p2_box{b}"),
                                    artifacts.load_array(eval_dir, f"q_p2_box{b}"),
                                    artifacts.load_array(eval_dir, f"q_inside_box{b}"),
                                    {kind: log_w[kind][b] for kind in weights}, name)
        save(fig, out, f"p2_marginals_box{b}_{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
