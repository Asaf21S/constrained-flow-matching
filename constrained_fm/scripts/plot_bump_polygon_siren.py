# -*- coding: utf-8 -*-
"""Decode the bump2d polygon SIREN on benchmark polygons and compare zero contours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from constrained_fm.src.consts import (BUMP_QUERY_TARGET_FRACTION, BUMP_SIREN_CHECKPOINT,
                                       BUMP_SIREN_TAU)
from constrained_fm.src.datasets.benchmark_1k import constraints_from, load_benchmark_1k
from constrained_fm.src.datasets.bump_conditioning import polygon_values, sample_query_points
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.experiment.config import REPO_ROOT
from constrained_fm.src.experiment.registry import pin_baseline_run
from constrained_fm.src.experiment.runtime import resolve_device, set_seed
from constrained_fm.src.geometry.polygons import polygon_sdf
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.models.functa_siren import build_modulated_siren
from constrained_fm.src.problems.bump2d import BumpProblem
from constrained_fm.src.visualization.bumphunt import half_plane_vertices
from constrained_fm.src.visualization import siren_encoder as se

OUTDIR = "constrained_fm/baselines/bump_polygon_siren"
FIGURE_DIR = "constrained_fm/images/thesis_pool/bump_polygon_siren"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--num-shapes", type=int, default=12)
    result.add_argument("--resolution", type=int, default=600)
    result.add_argument("--mass-points", type=int, default=100000)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--outdir", default=OUTDIR)
    result.add_argument("--figure-dir", default=FIGURE_DIR)
    result.add_argument("--plot-only", action="store_true")
    return result


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def select_indices(mass: torch.Tensor, count: int) -> list[int]:
    if count > mass.numel():
        raise ValueError(f"requested {count} polygons from a benchmark of {mass.numel()}")
    order = torch.argsort(mass)
    positions = torch.linspace(0, len(order) - 1, count).round().long()
    return order[positions].tolist()


def padded_shapes(constraints, device: torch.device) -> dict[str, torch.Tensor]:
    max_faces = max(item.normals.shape[0] for item in constraints)
    count = len(constraints)
    normals = torch.zeros(count, max_faces, 2, device=device)
    offsets = torch.zeros(count, max_faces, device=device)
    active = torch.zeros(count, max_faces, dtype=torch.bool, device=device)
    for i, constraint in enumerate(constraints):
        faces = constraint.normals.shape[0]
        normals[i, :faces] = constraint.normals
        offsets[i, :faces] = constraint.offsets
        active[i, :faces] = True
    return {"normals": normals, "offsets": offsets, "active": active}


def decode(siren, latents: torch.Tensor, axis: torch.Tensor,
           resolution: int) -> np.ndarray:
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    fields = []
    with torch.no_grad():
        for z in latents:
            fields.append(siren(points * (2.0 / BumpProblem().domain), z).squeeze(-1)
                          .reshape(resolution, resolution).cpu().numpy())
    return np.asarray(fields, dtype=np.float32)


def render(pred: np.ndarray, truth: np.ndarray, labels: list[str], args) -> list[Path]:
    style = se.get_style("paper", clip_to_unit=True, show_legend=True,
                         show_colorbar=True, colorbar_label=r"SIREN field $f_\theta(x,z)$",
                         gt_label="GT polygon", pred_label="SIREN contour")
    fig = se.plot_boundary_grid(pred, truth, 3, 4, scale=5.0, style=style)
    for ax, label in zip(fig.axes[:len(labels)], labels):
        ax.set_title(label, fontsize=14, family="serif", pad=4)
    return se.save_encoder_figure(fig, resolve_path(args.figure_dir) / "convex_mass_grid")


def replot(args) -> int:
    root = resolve_path(args.outdir)
    record = json.loads((root / "metrics.json").read_text())
    written = render(artifacts.load_array(root, "pred_fields"),
                     artifacts.load_array(root, "true_fields"),
                     record["panel_labels"], args)
    print("\n".join(f"wrote {path}" for path in written))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.num_shapes != 12:
        raise ValueError("the contour grid is fixed at 3x4; --num-shapes must be 12")
    if args.plot_only:
        return replot(args)

    device = resolve_device()
    set_seed(args.seed)
    problem = BumpProblem()
    benchmark = load_benchmark_1k("bump2d", device=device)
    indices = select_indices(benchmark["mass"], args.num_shapes)
    all_constraints = constraints_from(benchmark, problem, device=device)
    constraints = [all_constraints[i] for i in indices]
    shapes = padded_shapes(constraints, device)

    meta_path = REPO_ROOT / BUMP_SIREN_CHECKPOINT.replace(".pt", ".json")
    meta = json.loads(meta_path.read_text())
    siren = build_modulated_siren(latent_dim=meta["latent_dim"],
                                  hidden_dim=meta["hidden_dim"],
                                  n_layers=meta["n_layers"], w0=meta["w0"]).to(device)
    checkpoint = REPO_ROOT / BUMP_SIREN_CHECKPOINT
    siren.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    siren.eval()
    for parameter in siren.parameters():
        parameter.requires_grad_(False)

    target = problem.target()
    query = sample_query_points(target, args.num_shapes, 1000, domain=problem.domain,
                                target_fraction=BUMP_QUERY_TARGET_FRACTION, device=device)
    x_scaled = 2.0 * query / problem.domain - 1.0
    values = polygon_values(query, **shapes)
    targets = torch.tanh(values / BUMP_SIREN_TAU)
    latents, extraction_mse = extract_latents_batched(
        siren, x_scaled, targets, lr=meta["inner_lr"], steps=meta["inner_steps"])

    axis = torch.linspace(-problem.domain / 2, problem.domain / 2, args.resolution, device=device)
    pred = decode(siren, latents, axis, args.resolution)
    vertices = [half_plane_vertices(c.normals.cpu().numpy(), c.offsets.cpu().numpy())
                for c in constraints]
    vertices = [torch.as_tensor(v, device=device, dtype=torch.float32) - problem.domain / 2
                for v in vertices]
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([xx, yy], dim=-1)
    truth = np.stack([polygon_sdf(points, v).cpu().numpy() for v in vertices])

    mass_points = target.sample(args.mass_points, device=device)
    records, labels = [], []
    for i, (constraint, vertex) in enumerate(zip(constraints, vertices)):
        gt_inside = constraint.value(mass_points) <= 0
        pred_inside = siren(2.0 * mass_points / problem.domain - 1.0,
                            latents[i]).squeeze(-1) <= 0
        iou = float((gt_inside & pred_inside).sum() /
                    (gt_inside | pred_inside).sum().clamp_min(1))
        mass = float(benchmark["mass"][indices[i]])
        labels.append(f"mass={mass:.2f}  K={vertex.shape[0]}")
        records.append({"benchmark_index": indices[i], "mass": mass,
                        "num_vertices": int(vertex.shape[0]),
                        "extraction_mse": float(extraction_mse[i]), "mass_iou": iou})
        print(f"index {indices[i]:4d} | mass {mass:.3f} | vertices {vertex.shape[0]:2d} "
              f"| mass-IoU {iou:.4f} | CAVIA MSE {float(extraction_mse[i]):.3e}")

    root = resolve_path(args.outdir)
    run_id = pin_baseline_run(root, "bump_polygon_siren", args, extra={
        "siren_checkpoint": BUMP_SIREN_CHECKPOINT,
        "benchmark_digest": benchmark["digest"], "checkpoint_meta": meta})
    vertex_counts = np.asarray([v.shape[0] for v in vertices], dtype=np.int32)
    padded_vertices = np.full((len(vertices), vertex_counts.max(), 2), np.nan,
                              dtype=np.float32)
    for i, vertex in enumerate(vertices):
        padded_vertices[i, :vertex.shape[0]] = vertex.cpu().numpy()
    artifacts.save_arrays(root, latents=latents.cpu().numpy(), pred_fields=pred,
                          true_fields=truth, benchmark_indices=np.asarray(indices),
                          polygon_vertices=padded_vertices, polygon_vertex_counts=vertex_counts)
    artifacts.write_manifest(root, run_id=run_id, benchmark_digest=benchmark["digest"])
    (root / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "benchmark_digest": benchmark["digest"],
        "siren_checkpoint": BUMP_SIREN_CHECKPOINT, "resolution": args.resolution,
        "panel_labels": labels, "per_shape": records}, indent=2))
    written = render(pred, truth, labels, args)
    print("\n".join(f"wrote {path}" for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())