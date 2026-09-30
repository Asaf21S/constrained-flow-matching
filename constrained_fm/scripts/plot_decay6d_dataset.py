# -*- coding: utf-8 -*-
"""Draws the decay6d dataset and the evaluation boxes from fresh simulator draws.

    python -m constrained_fm.scripts.plot_decay6d_dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402

from constrained_fm.scripts.plot_decay6d_is import save  # noqa: E402
from constrained_fm.src.experiment.runtime import resolve_device  # noqa: E402
from constrained_fm.src.problems.decay6d import DecayProblem  # noqa: E402
from constrained_fm.src.visualization import decay6d_is as viz  # noqa: E402

BOXES = "constrained_fm/baselines/decay6d_is/benchmark/boxes.json"
FIGURE_DIR = "constrained_fm/images/thesis_pool/decay6d_is/dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="decay6d dataset + evaluation box figures.")
    parser.add_argument("--boxes", default=BOXES)
    parser.add_argument("--num-samples", type=int, default=4_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    boxes = json.loads(Path(args.boxes).read_text())["boxes"]

    device = resolve_device()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = DecayProblem().target().sample(args.num_samples, device, generator).float().cpu().numpy()

    save(viz.plot_dataset_p1_boxes(x, boxes), out, "dataset_p1_boxes")
    save(viz.plot_dataset_structure(x), out, "dataset_structure")
    save(viz.plot_dataset_p2_given_box(x, boxes), out, "dataset_p2_given_box")

    print(f"{'box':16s} {'P(B)':>7s}  centre / half-width (p1)  tau_B")
    for box in boxes:
        centre = ", ".join(f"{c:+.2f}" for c in box["centre"])
        half = ", ".join(f"{h:.3f}" for h in box["half_width"])
        print(f"{box['name']:16s} {100 * box['gt_mass']:6.2f}%  ({centre}) / ({half})  "
              f"{box['tail_threshold']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
