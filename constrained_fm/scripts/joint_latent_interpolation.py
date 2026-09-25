# -*- coding: utf-8 -*-
"""Linear interpolation in the joint SIREN's latent space, from a polygon to a polynomial.

For each pair, ``z(t) = (1 - t) z_polygon + t z_polynomial`` is decoded on a lattice at every
``t``. Each panel draws the decoded zero level set; the endpoint panels also draw the true
boundary. Decoded GMM mass and ``||z(t)||`` along each path are written to ``metrics.json``.

    interpolation/grid.{png,pdf}     every pair, one row each
    interpolation/pair<i>.{png,pdf}  one row

Arrays are written to ``<outdir>/artifacts/`` and ``--plot-only`` redraws from them.

    sbatch scripts/run_joint_interp.sh
    sbatch scripts/run_joint_interp.sh --plot-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from constrained_fm.scripts.plot_siren_encoder import decode_fields
from constrained_fm.src.datasets import joint_conditioning as jc
from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.config import REPO_ROOT
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.visualization import siren_encoder as se

SIREN_DIR = "constrained_fm/functa_dataset/joint_siren"
OUTDIR = "constrained_fm/baselines/joint_interpolation"
FIGURE_DIR = "constrained_fm/images/thesis_pool/joint_interpolation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--siren-dir", default=SIREN_DIR)
    parser.add_argument("--checkpoint", default="siren_best.pt")
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=9, help="interpolation times in [0, 1]")
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--iou-points", type=int, default=100000,
                        help="GMM draws backing endpoint IoU and decoded mass along each path")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--proxy-points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--style", default="strip", choices=sorted(se.STYLE_PRESETS))
    parser.add_argument("--colorbar", action="store_true")
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw from saved arrays; no checkpoint, no extraction")
    return parser


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def decoded_mass(siren, z_path: torch.Tensor, points: torch.Tensor, scale: float,
                 chunk_size: int) -> torch.Tensor:
    """Fraction of ``points`` with ``SIREN(x, z) <= 0`` for each latent; (T,) on CPU."""
    masses = []
    with torch.no_grad():
        for z in z_path:
            inside = torch.cat([siren(points[s:s + chunk_size] / scale, z).squeeze(-1) <= 0
                                for s in range(0, points.shape[0], chunk_size)])
            masses.append(inside.float().mean().cpu())
    return torch.stack(masses)


def render(interp: np.ndarray, true: np.ndarray, times: list[float], scale: float,
           args) -> list[Path]:
    num_pairs = interp.shape[0]
    style = se.get_style(args.style, show_ticks=False, show_colorbar=args.colorbar,
                         show_legend=args.legend, gt_label="ground-truth boundary")
    labels = [f"pair {i}" for i in range(num_pairs)]
    figure_dir = resolve_path(args.figure_dir) / "interpolation"

    fig = se.plot_interpolation_grid(interp, times, true[:num_pairs], true[num_pairs:],
                                     scale=scale, style=style, row_labels=labels)
    written = se.save_encoder_figure(fig, figure_dir / "grid", formats=args.formats,
                                     dpi=args.dpi)
    for i in range(num_pairs):
        fig = se.plot_interpolation_grid(interp[i:i + 1], times, true[i:i + 1],
                                         true[num_pairs + i:num_pairs + i + 1],
                                         scale=scale, style=style)
        written += se.save_encoder_figure(fig, figure_dir / f"pair{i}", formats=args.formats,
                                          dpi=args.dpi)
    return written


def replot(args) -> int:
    root = resolve_path(args.outdir)
    record = json.loads((root / "metrics.json").read_text())
    written = render(artifacts.load_array(root, "interp_fields"),
                     artifacts.load_array(root, "true_fields"),
                     [float(t) for t in artifacts.load_array(root, "interp_times")],
                     float(record["scale"]), args)
    print("\n".join(f"wrote {p}" for p in written))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plot_only:
        return replot(args)

    device = resolve_device()
    siren, meta = jc.load_joint_siren(resolve_path(args.siren_dir), args.checkpoint, device)
    tau, degree, scale = meta["tau"], meta["degree"], meta["scale"]
    num_pairs = args.num_pairs
    print(f"siren {meta['run_id']} ({args.checkpoint}) | tau {tau:.4f} | device {device} | "
          f"{num_pairs} pairs x {args.steps} steps")

    set_seed(args.seed)
    proxy = jc.proxy_set(args.proxy_points, degree, scale, device)
    sampled = [jc.sample_joint_shapes(num_pairs, proxy, fraction, False, degree, scale,
                                      meta["min_mass"], meta["max_mass"], device)
               for fraction in (1.0, 0.0)]
    shapes = {key: torch.cat([part[key] for part in sampled]) for key in jc.SHAPE_KEYS}

    x_raw = sample_query_points(2 * num_pairs, meta["points_per_shape"], scale=scale,
                                gmm_fraction=meta["query_gmm_fraction"], device=device)
    x, y = jc.regression_targets(shapes, x_raw, tau, degree, scale)
    z, extraction_mse = extract_latents_batched(siren, x, y, lr=meta["inner_lr"],
                                                steps=meta["inner_steps"])

    axis = torch.linspace(-scale, scale, args.resolution)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    lattice = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).to(device)
    with torch.no_grad():
        true = jc.constraint_values(shapes, lattice.unsqueeze(0).expand(2 * num_pairs, -1, -1),
                                    tau, degree, scale)
    true = true.view(-1, args.resolution, args.resolution).cpu().numpy().astype(np.float32)

    mass_points, _ = get_points(args.iou_points, device=device)
    iou = jc.mass_iou(siren, z, shapes, mass_points, tau, degree, scale)
    mass = jc.constraint_mass(shapes, mass_points, tau, degree, scale).cpu()

    times = torch.linspace(0.0, 1.0, args.steps, device=device)
    interp = np.empty((num_pairs, args.steps, args.resolution, args.resolution),
                      dtype=np.float32)
    path_mass = np.empty((num_pairs, args.steps), dtype=np.float32)
    path_norm = np.empty((num_pairs, args.steps), dtype=np.float32)
    for i in range(num_pairs):
        z_path = torch.lerp(z[i].unsqueeze(0), z[num_pairs + i].unsqueeze(0), times.view(-1, 1))
        interp[i] = decode_fields(siren, z_path, lattice, scale, args.resolution,
                                  args.chunk_size)
        path_mass[i] = decoded_mass(siren, z_path, mass_points, scale, args.chunk_size).numpy()
        path_norm[i] = z_path.norm(dim=-1).cpu().numpy()

    root = resolve_path(args.outdir)
    run_id = pin_baseline_run(root, "joint_interpolation", args,
                              extra={"siren_run_id": meta["run_id"], "tau": tau})
    artifacts.save_arrays(root, **{key: shapes[key] for key in jc.SHAPE_KEYS}, latents=z,
                          query_points=x_raw, true_fields=true, interp_fields=interp,
                          interp_times=times, path_mass=path_mass, path_z_norm=path_norm)
    artifacts.write_manifest(root, run_id=run_id, siren_run_id=meta["run_id"],
                             checkpoint=args.checkpoint, tau=tau)

    def endpoint(index: int) -> dict[str, float]:
        return {"mass": float(mass[index]), "mass_iou": float(iou[index]),
                "extraction_mse": float(extraction_mse[index]),
                "z_norm": float(z[index].norm())}

    pairs = [{"pair": i, "polygon": endpoint(i), "polynomial": endpoint(num_pairs + i),
              "path_mass": path_mass[i].tolist(), "path_z_norm": path_norm[i].tolist()}
             for i in range(num_pairs)]
    (root / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "siren_run_id": meta["run_id"], "checkpoint": args.checkpoint,
        "tau": tau, "scale": scale, "times": times.tolist(), "pairs": pairs}, indent=2))

    for item in pairs:
        gon, poly = item["polygon"], item["polynomial"]
        print(f"pair {item['pair']}: polygon IoU {gon['mass_iou']:.3f} mass {gon['mass']:.3f} "
              f"-> polynomial IoU {poly['mass_iou']:.3f} mass {poly['mass']:.3f} | decoded mass "
              + " ".join(f"{m:.2f}" for m in item["path_mass"]))

    written = render(interp, true, times.tolist(), scale, args)
    print("\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
