# -*- coding: utf-8 -*-
"""Builds the five fixed decay6d evaluation boxes and their simulator ground truth.

Each box has a fixed physical centre and aspect ratio; one scale ``s`` sets the normalized
half-widths ``s * aspect`` and is bisected on a simulator pool to hit the target ``P(B)``. The
builder refuses boxes outside the training distribution (half-widths or table mass outside the
ranges the box model saw). Ground truth for ``E[f(p2) | p1 in B]`` comes from rejection on a
large simulator stream, and up to ``--keep`` in-box ``p2`` per box are stored for the figures.

    python -m constrained_fm.scripts.build_decay6d_boxes
    python -m constrained_fm.scripts.build_decay6d_boxes --smoke
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from constrained_fm.src.consts import DECAY_EVAL_BOX_MASSES, DECAY_TAIL_QUANTILE
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device
from constrained_fm.src.problems.decay6d import (OBSERVABLE_NAMES, PARTICLE_DIM, BoxConstraint,
                                                 DecayProblem, boxes_contain, observables)

DEFAULT_OUTDIR = "constrained_fm/baselines/decay6d_is/benchmark"
SMOKE_OUTDIR = "constrained_fm/baselines/decay6d_is/smoke/benchmark"
BOXES_NAME = "boxes.json"
BISECT_STEPS = 60

# (name, physical centre of p1, per-axis aspect of the normalized half-widths)
BOX_SPECS = (
    ("small_offcentre", (0.45, 0.45, 0.35), (1.0, 1.0, 1.0)),
    ("thin_slab", (0.0, 0.0, 0.0), (1.0, 1.0, 0.25)),
    ("offcentre_cube", (-0.35, 0.25, -0.2), (1.0, 1.0, 1.0)),
    ("elongated", (0.3, 0.0, 0.0), (3.0, 1.0, 1.0)),
    ("bulk_cube", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed decay6d evaluation boxes + ground truth.")
    parser.add_argument("--pool", type=int, default=10_000_000,
                        help="simulator pool for tau and for bisecting each box scale")
    parser.add_argument("--gt-samples", type=int, default=1_000_000_000)
    parser.add_argument("--chunk", type=int, default=10_000_000)
    parser.add_argument("--keep", type=int, default=1_000_000,
                        help="in-box ground-truth p2 stored per box for the figures")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.smoke:
        args.pool, args.gt_samples, args.chunk, args.keep = 1_000_000, 4_000_000, 1_000_000, 20_000
    if args.outdir is None:
        args.outdir = SMOKE_OUTDIR if args.smoke else DEFAULT_OUTDIR
    return args


def pool_mass(p1_n: torch.Tensor, centre_n: torch.Tensor, half_n: torch.Tensor) -> float:
    return boxes_contain(p1_n, centre_n - half_n, centre_n + half_n).double().mean().item()


def bisect_scale(p1_n: torch.Tensor, centre_n: torch.Tensor, aspect: torch.Tensor, target: float,
                 half_range: tuple[float, float]) -> float:
    """Scale ``s`` with pool ``P(B) = target`` for half-widths ``s * aspect`` inside ``half_range``."""
    lo = math.log(half_range[0] / aspect.min().item())
    hi = math.log(half_range[1] / aspect.max().item())
    mass_lo = pool_mass(p1_n, centre_n, math.exp(lo) * aspect)
    mass_hi = pool_mass(p1_n, centre_n, math.exp(hi) * aspect)
    if not mass_lo <= target <= mass_hi:
        raise ValueError(f"target mass {target} unreachable: [{mass_lo:.4g}, {mass_hi:.4g}] "
                         f"over the training half-width range")
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if pool_mass(p1_n, centre_n, math.exp(mid) * aspect) < target:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(build_parser().parse_args(argv))
    device = resolve_device()
    out = Path(args.outdir)
    run_id = pin_baseline_run(out, "decay6d_boxes", args)

    problem = DecayProblem()
    target = problem.target()
    normalizer = problem.normalizer(torch.float64).to(device)
    std_1 = normalizer.std[:PARTICLE_DIM]
    mean_1 = normalizer.mean[:PARTICLE_DIM]
    table = problem.mass_table(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    pool = target.sample(args.pool, device, generator)
    p2_norm = pool[:, PARTICLE_DIM:].norm(dim=-1)
    p1_n = normalizer.forward(pool)[:, :PARTICLE_DIM]
    del pool

    boxes = []
    for (name, centre, aspect), mass in zip(BOX_SPECS, DECAY_EVAL_BOX_MASSES):
        centre = torch.tensor(centre, device=device, dtype=torch.float64)
        aspect = torch.tensor(aspect, device=device, dtype=torch.float64)
        centre_n = (centre - mean_1) / std_1
        scale = bisect_scale(p1_n, centre_n, aspect, mass, problem.half_width_range)
        half_n = scale * aspect
        table_mass = table.mass((centre_n - half_n)[None], (centre_n + half_n)[None]).item()
        if not problem.mass_range[0] <= table_mass <= problem.mass_range[1]:
            raise ValueError(f"{name}: table mass {table_mass:.4g} outside the training filter "
                             f"{problem.mass_range}")
        constraint = BoxConstraint.from_centre(centre.cpu(), (half_n * std_1).cpu())
        inside = boxes_contain(p1_n, centre_n - half_n, centre_n + half_n)
        tau = torch.quantile(p2_norm[inside].float(), DECAY_TAIL_QUANTILE).item()
        boxes.append({"name": name, "target_mass": mass, "scale": scale,
                      "aspect": aspect.tolist(), "pool_mass": pool_mass(p1_n, centre_n, half_n),
                      "table_mass": table_mass, "tail_threshold": tau,
                      "lo": constraint.lo.tolist(), "hi": constraint.hi.tolist(),
                      "centre": constraint.centre.tolist(),
                      "half_width": constraint.half_width.tolist(),
                      "conditioning": constraint.conditioning(normalizer.to("cpu")).tolist()})
    del p1_n, p2_norm

    num_f = len(OBSERVABLE_NAMES)
    count = torch.zeros(len(boxes), device=device, dtype=torch.float64)
    sums = torch.zeros(len(boxes), num_f, device=device, dtype=torch.float64)
    sq_sums = torch.zeros_like(sums)
    kept: list[list[torch.Tensor]] = [[] for _ in boxes]
    num_kept = [0] * len(boxes)
    lo = torch.tensor([b["lo"] for b in boxes], device=device, dtype=torch.float64)
    hi = torch.tensor([b["hi"] for b in boxes], device=device, dtype=torch.float64)

    drawn = 0
    while drawn < args.gt_samples:
        n = min(args.chunk, args.gt_samples - drawn)
        x = target.sample(n, device, generator)
        for b in range(len(boxes)):
            inside = boxes_contain(x[:, :PARTICLE_DIM], lo[b], hi[b])
            fb = observables(x[inside], boxes[b]["tail_threshold"])
            count[b] += fb.shape[0]
            sums[b] += fb.sum(0)
            sq_sums[b] += fb.pow(2).sum(0)
            if num_kept[b] < args.keep:
                take = x[inside, PARTICLE_DIM:][:args.keep - num_kept[b]]
                kept[b].append(take.cpu())
                num_kept[b] += take.shape[0]
        drawn += n

    mean = sums / count[:, None]
    var = (sq_sums / count[:, None] - mean.pow(2)).clamp_min(0.0)
    se = (var / count[:, None]).sqrt()
    for b, box in enumerate(boxes):
        mass = count[b].item() / drawn
        box["gt_mass"] = mass
        box["gt_mass_se"] = math.sqrt(mass * (1.0 - mass) / drawn)
        box["gt_count"] = int(count[b].item())
        box["gt"] = {name: {"mean": mean[b, k].item(), "se": se[b, k].item(),
                            "std": var[b, k].sqrt().item()}
                     for k, name in enumerate(OBSERVABLE_NAMES)}

    artifacts.save_arrays(out, **{f"gt_p2_box{b}": torch.cat(kept[b]).numpy()
                                  for b in range(len(boxes))})
    artifacts.write_manifest(out, run_id=run_id, boxes=[b["name"] for b in boxes])

    payload = {"run_id": run_id, "tail_quantile": DECAY_TAIL_QUANTILE,
               "observables": list(OBSERVABLE_NAMES), "gt_samples": drawn, "seed": args.seed,
               "normalizer_std": normalizer.std.tolist(), "boxes": boxes}
    (out / BOXES_NAME).write_text(json.dumps(payload, indent=2))

    print(f"### decay6d boxes ({run_id})")
    for box in boxes:
        gt = "  ".join(f"{k}={v['mean']:.5f}+-{v['se']:.1e}" for k, v in box["gt"].items())
        print(f"  {box['name']:16s} P(B)={box['gt_mass']:.5f} (target {box['target_mass']}, "
              f"table {box['table_mass']:.5f})  tau={box['tail_threshold']:.5f}  {gt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
