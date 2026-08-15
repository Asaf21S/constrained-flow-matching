# -*- coding: utf-8 -*-
"""Stage 3: evaluate a trained run against the static validation set.

Runs against a frozen checkpoint, so metrics and diagnostic figures can be regenerated
in minutes without retraining. Writes runs/<run_id>/metrics.json and figures/.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.datasets.validation import get_validation_set
from constrained_fm.src.experiment.config import ExperimentConfig
from constrained_fm.src.experiment.registry import (FIGURES_DIR, LOSSES_NAME, METRICS_NAME,
                                                    correlation, load_config, readme_table,
                                                    run_dir, summarize, write_json, write_state)
from constrained_fm.src.experiment.runtime import (build_flow_matcher, load_checkpoint, load_siren,
                                                   resolve_device, set_seed)
from constrained_fm.src.geometry.polynomials import compute_poly_features_batched, evaluate_poly_batched
from constrained_fm.src.inference.evaluator import (evaluate_validation_set_metrics,
                                                    run_evaluation_inference)
from constrained_fm.src.inference.latent_extractor import extract_latents_batched
from constrained_fm.src.metrics.functa_fidelity import constraint_masses, region_iou_batched
from constrained_fm.src.visualization import diagnostics as diag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained run from its checkpoint.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="path to the experiment config YAML")
    source.add_argument("--run-id", help="run id of an existing run directory")
    parser.add_argument("--smoke", action="store_true", help="shrink every knob (with --config only)")
    parser.add_argument("--no-figures", action="store_true", help="write metrics.json only")
    return parser


def extract_validation_latents(siren, cfg: ExperimentConfig, val_polys: torch.Tensor,
                               device: torch.device,
                               chunk_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Fresh CAVIA extraction for the validation polynomials, matching the training budget."""
    z_chunks, mse_chunks = [], []
    for start in range(0, val_polys.shape[0], chunk_size):
        C_chunk = val_polys[start:start + chunk_size]
        X_raw = sample_query_points(C_chunk.shape[0], cfg.extraction.points_per_shape,
                                    scale=cfg.scale,
                                    gmm_fraction=cfg.extraction.query_gmm_fraction, device=device)
        x_pow, y_pow = compute_poly_features_batched(X_raw, degree=cfg.degree, scale=cfg.scale)
        Y = torch.tanh(evaluate_poly_batched(x_pow, y_pow, C_chunk))
        z_chunk, mse_chunk = extract_latents_batched(siren, X_raw / cfg.scale, Y,
                                                     lr=cfg.extraction.lr, steps=cfg.extraction.steps)
        z_chunks.append(z_chunk)
        mse_chunks.append(mse_chunk)

    return torch.cat(z_chunks, dim=0), torch.cat(mse_chunks, dim=0)


def render_figures(cfg: ExperimentConfig, siren, model, val_polys: torch.Tensor,
                   z_val: torch.Tensor, val_samples: np.ndarray,
                   per_shape: dict[str, list[float]], device: torch.device) -> None:
    out = run_dir(cfg.run_id) / FIGURES_DIR
    ev = cfg.evaluation
    success = np.asarray(per_shape["success_rate"])
    order = np.argsort(success)

    losses_path = run_dir(cfg.run_id) / LOSSES_NAME
    if losses_path.exists():
        diag.save_figure(diag.plot_loss_curve(np.load(losses_path)), out / "loss_curve.png")

    diag.save_figure(
        diag.plot_success_vs_fidelity(success, per_shape["mass"], per_shape["mass_iou"]),
        out / "success_vs_fidelity.png")

    worst = order[:min(ev.num_worst_plots, len(order))].tolist()
    titles = [f"shape {i} | SR {success[i]:.1f}%\n"
              f"mass IoU {per_shape['mass_iou'][i]:.2f} | mass {per_shape['mass'][i]:.2f}"
              for i in worst]
    diag.save_figure(
        diag.plot_believed_vs_true(siren, [val_samples[i] for i in worst], [z_val[i] for i in worst],
                                   [val_polys[i] for i in worst], titles,
                                   grid_size=ev.iou_grid_size, degree=cfg.degree, scale=cfg.scale,
                                   device=device),
        out / "worst_believed_vs_true.png")

    # Median-success shape: representative of typical behaviour rather than of the tail.
    typical = int(order[len(order) // 2])
    trajectory, time_grid = model.sample(num_points=ev.num_vis_samples, z=z_val[typical],
                                         step_size=ev.step_size, return_intermediates=True,
                                         device=device)
    diag.save_figure(
        diag.plot_sample_trajectory(trajectory.cpu().numpy(), time_grid.cpu().numpy(),
                                    coeffs=val_polys[typical], degree=cfg.degree, scale=cfg.scale,
                                    device=device),
        out / "typical_trajectory.png")
    diag.save_figure(
        diag.plot_final_samples(trajectory[-1].cpu().numpy(), coeffs=val_polys[typical],
                                title=f"shape {typical} | SR {success[typical]:.2f}% | "
                                      f"SWD {per_shape['swd'][typical]:.4f}",
                                degree=cfg.degree, scale=cfg.scale),
        out / "typical_samples.png")

    # Covers best/typical/worst regions of the score distribution in one view.
    q_idxs = sorted(set([
        int(order[0]),
        int(order[len(order) // 4]),
        int(order[len(order) // 2]),
        int(order[(3 * len(order)) // 4]),
        int(order[-1]),
    ]))
    gallery_samples = []
    gallery_coeffs = []
    gallery_titles = []
    for idx in q_idxs:
        s = model.sample(num_points=ev.num_vis_samples, z=z_val[idx], step_size=ev.step_size,
                         return_intermediates=False, device=device)
        if isinstance(s, torch.Tensor) and s.ndim == 3:
            s = s[-1]
        gallery_samples.append(s.detach().cpu().numpy())
        gallery_coeffs.append(val_polys[idx])
        gallery_titles.append(
            f"shape {idx} | SR {success[idx]:.2f}%\n"
            f"SWD {per_shape['swd'][idx]:.4f} | JSD {per_shape['jsd'][idx]:.4f}")

    diag.save_figure(
        diag.plot_final_samples_gallery(gallery_samples, gallery_coeffs, gallery_titles,
                                        degree=cfg.degree, scale=cfg.scale),
        out / "final_samples_gallery.png")

    if ev.likelihood_grid > 0:
        likelihood = model.compute_likelihood_grid(
            z=z_val[typical], siren=siren, degree=cfg.degree, scale=cfg.scale,
            grid_size=ev.likelihood_grid, step_size=ev.step_size, device=device)
        diag.save_figure(
            diag.plot_likelihood(likelihood, coeffs=val_polys[typical], degree=cfg.degree,
                                 scale=cfg.scale, grid_size=ev.likelihood_grid, device=device),
            out / "typical_likelihood.png")

    print(f"figures written to {out}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.run_id:
        cfg = load_config(args.run_id)
    else:
        cfg = ExperimentConfig.from_yaml(args.config)
        if args.smoke:
            cfg = cfg.smoke()

    device = resolve_device()
    ev = cfg.evaluation
    print(f"run_id  {cfg.run_id}")
    print(f"device  {device}")

    set_seed(ev.seed)
    siren = load_siren(cfg, device)
    model = build_flow_matcher(cfg, siren, device)
    iteration = load_checkpoint(cfg, model, device)
    model.eval()
    print(f"loaded checkpoint at iteration {iteration}")

    gmm_true_pool, _ = get_points(ev.gmm_pool_size, device=device)
    val_set = get_validation_set(device=device)
    val_polys = val_set["polynomials"][:ev.num_polys].to(device)
    val_x0 = val_set["x0"][:ev.num_x0].to(device)

    z_val, extraction_mse = extract_validation_latents(siren, cfg, val_polys, device)
    val_mass = constraint_masses(val_polys, gmm_true_pool, degree=cfg.degree, scale=cfg.scale)

    # Mass-weighted: level-set disagreement away from the data cannot move the metrics.
    subset = torch.randperm(gmm_true_pool.shape[0], device=device)[:ev.iou_mass_samples]
    val_iou_mass = region_iou_batched(siren, z_val, val_polys, gmm_true_pool[subset],
                                      degree=cfg.degree, scale=cfg.scale)
    print(f"valid GMM mass : min {val_mass.min():.3f} | median {val_mass.median():.3f}")
    print(f"mass IoU       : mean {val_iou_mass.mean():.4f} | min {val_iou_mass.min():.4f}")

    val_samples = run_evaluation_inference(model, val_x0, z=z_val, step_size=ev.step_size,
                                           device=device)
    metrics = evaluate_validation_set_metrics(val_samples, x_true_pool=gmm_true_pool,
                                              coeffs=val_polys, degree=cfg.degree,
                                              scale=cfg.scale, device=device)

    per_shape = {
        "success_rate": [float(v) for v in metrics["success_rate"]],
        "swd": [float(v) for v in metrics["swd"]],
        "mmd": [float(v) for v in metrics["mmd"]],
        "jsd": [float(v) for v in metrics["jsd"]],
        "mass": val_mass.cpu().tolist(),
        "mass_iou": val_iou_mass.cpu().tolist(),
        "extraction_mse": extraction_mse.cpu().tolist(),
    }

    summary = summarize(per_shape)
    summary["corr_success_mass"] = correlation(per_shape["success_rate"], per_shape["mass"])
    summary["corr_success_mass_iou"] = correlation(per_shape["success_rate"], per_shape["mass_iou"])

    write_json(run_dir(cfg.run_id) / METRICS_NAME, {
        "run_id": cfg.run_id,
        "iteration": iteration,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_config": dataclasses.asdict(ev),
        "per_shape": per_shape,
        "summary": summary,
    })
    write_state(cfg.run_id, status="evaluated",
                evaluated_at=datetime.now().isoformat(timespec="seconds"),
                finished_at=datetime.now().isoformat(timespec="seconds"))

    print()
    print(readme_table(summary))
    print()
    print(f"corr(success_rate, mass)     = {summary['corr_success_mass']:+.3f}")
    print(f"corr(success_rate, mass_IoU) = {summary['corr_success_mass_iou']:+.3f}")

    print(f"\n{'rank':>4} {'SR':>7} {'mass':>7} {'massIoU':>8} {'swd':>8} {'jsd':>8}")
    for rank, i in enumerate(np.argsort(per_shape["success_rate"])[:10]):
        print(f"{rank:>4} {per_shape['success_rate'][i]:7.2f} {per_shape['mass'][i]:7.3f} "
              f"{per_shape['mass_iou'][i]:8.3f} {per_shape['swd'][i]:8.4f} "
              f"{per_shape['jsd'][i]:8.4f}")

    if not args.no_figures:
        render_figures(cfg, siren, model, val_polys, z_val, val_samples, per_shape, device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
