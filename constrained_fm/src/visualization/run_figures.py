# -*- coding: utf-8 -*-
"""Redraws a run's figure set from its saved artifacts alone.

Nothing here builds a network, loads a checkpoint, or integrates an ODE: every panel is a
function of the arrays written by the evaluation stage plus ``metrics.json``. Editing a
title, a colormap, or a panel layout is therefore a seconds-long, GPU-free operation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from constrained_fm.src.consts import PLANE_SCALE, POLYNOMIAL_DEGREE
from constrained_fm.src.experiment import artifacts
from constrained_fm.src.visualization import diagnostics as diag

FIGURES_DIR = "figures"


def _coeffs(polynomials: np.ndarray, index: int) -> torch.Tensor:
    """Constraint coefficients as a CPU tensor; the overlay helpers read .device off them."""
    return torch.from_numpy(np.asarray(polynomials[index], dtype=np.float32))


def _read_metrics(root: Path) -> dict:
    path = root / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def render_run_figures(root: str | Path, out_dir: str | Path | None = None,
                       degree: int = POLYNOMIAL_DEGREE,
                       scale: float = PLANE_SCALE) -> list[Path]:
    """Writes every figure the saved artifacts support and returns the paths written."""
    root = Path(root)
    manifest = artifacts.load_manifest(root)
    if not manifest:
        raise FileNotFoundError(
            f"no plotting artifacts under {root}. Re-run the evaluation stage for this run.")

    degree = int(manifest.get("degree", degree))
    scale = float(manifest.get("scale", scale))
    out = Path(out_dir) if out_dir is not None else root / FIGURES_DIR
    metrics = _read_metrics(root)
    per_shape = metrics.get("per_shape", {})
    written: list[Path] = []

    losses_path = root / "losses.npy"
    if losses_path.exists():
        written.append(diag.save_figure(diag.plot_loss_curve(np.load(losses_path)),
                                        out / "loss_curve.png"))

    if {"success_rate", "mass", "mass_iou"} <= set(per_shape):
        written.append(diag.save_figure(
            diag.plot_success_vs_fidelity(per_shape["success_rate"], per_shape["mass"],
                                          per_shape["mass_iou"]),
            out / "success_vs_fidelity.png"))

    polynomials = artifacts.load_array(root, "polynomials")
    samples = artifacts.load_array(root, "samples")

    if artifacts.has_array(root, "believed_fields"):
        fields = artifacts.load_array(root, "believed_fields")
        ids = artifacts.load_array(root, "believed_ids").tolist()
        titles = [f"shape {i} | SR {per_shape['success_rate'][i]:.1f}\n"
                  f"mass IoU {per_shape['mass_iou'][i]:.2f} | mass {per_shape['mass'][i]:.2f}"
                  for i in ids]
        written.append(diag.save_figure(
            diag.plot_believed_vs_true([samples[i] for i in ids], list(fields),
                                       [_coeffs(polynomials, i) for i in ids], titles,
                                       degree=degree, scale=scale),
            out / "worst_believed_vs_true.png"))

    typical = manifest.get("typical_id")
    if artifacts.has_array(root, "trajectory") and typical is not None:
        written.append(diag.save_figure(
            diag.plot_sample_trajectory(artifacts.load_array(root, "trajectory"),
                                        artifacts.load_array(root, "trajectory_time"),
                                        coeffs=_coeffs(polynomials, typical),
                                        degree=degree, scale=scale),
            out / "typical_trajectory.png"))

    if artifacts.has_array(root, "gallery_samples"):
        gallery = artifacts.load_array(root, "gallery_samples")
        ids = artifacts.load_array(root, "gallery_ids").tolist()
        titles = [f"shape {i} | SR {per_shape['success_rate'][i]:.2f}%\n"
                  f"SWD {per_shape['swd'][i]:.4f} | JSD {per_shape['jsd'][i]:.4f}" for i in ids]

        if typical is not None and typical in ids:
            position = ids.index(typical)
            written.append(diag.save_figure(
                diag.plot_final_samples(gallery[position], coeffs=_coeffs(polynomials, typical),
                                        title=titles[position].replace("\n", " | "),
                                        degree=degree, scale=scale),
                out / "typical_samples.png"))

        written.append(diag.save_figure(
            diag.plot_final_samples_gallery(list(gallery),
                                            [_coeffs(polynomials, i) for i in ids], titles,
                                            degree=degree, scale=scale),
            out / "final_samples_gallery.png"))

    if artifacts.has_array(root, "likelihood") and typical is not None:
        likelihood = artifacts.load_array(root, "likelihood")
        written.append(diag.save_figure(
            diag.plot_likelihood(likelihood, coeffs=_coeffs(polynomials, typical), degree=degree,
                                 scale=scale, grid_size=likelihood.shape[0]),
            out / "typical_likelihood.png"))

    return written


__all__ = ["render_run_figures"]
