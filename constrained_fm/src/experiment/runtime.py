# -*- coding: utf-8 -*-
"""Torch-side bootstrap shared by the pool / train / eval entry points."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from constrained_fm.src.datasets.gmm_target import get_points
from constrained_fm.src.experiment.config import ExperimentConfig
from constrained_fm.src.experiment.registry import CHECKPOINT_NAME, run_dir
from constrained_fm.src.geometry.polynomials import compute_poly_features
from constrained_fm.src.models.constrained_functa import ConstrainedFlowMatcher
from constrained_fm.src.models.functa_siren import ModulatedSIREN, build_modulated_siren


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_siren(cfg: ExperimentConfig, device: torch.device) -> ModulatedSIREN:
    """Builds the SIREN from the config's architecture block and loads its frozen weights."""
    siren = build_modulated_siren(
        latent_dim=cfg.siren.latent_dim, hidden_dim=cfg.siren.hidden_dim,
        n_layers=cfg.siren.n_layers, w0=cfg.siren.w0,
    ).to(device)
    state = torch.load(cfg.siren_path(), map_location=device, weights_only=True)
    siren.load_state_dict(state)
    siren.eval()
    for p in siren.parameters():
        p.requires_grad_(False)
    return siren


def build_flow_matcher(cfg: ExperimentConfig, siren: ModulatedSIREN,
                       device: torch.device) -> ConstrainedFlowMatcher:
    return ConstrainedFlowMatcher(
        siren=siren if cfg.fm.use_siren_feature else None,
        latent_dim=cfg.siren.latent_dim,
        time_emb_dim=cfg.fm.time_emb_dim,
        hidden_dim=cfg.fm.hidden_dim,
        num_blocks=cfg.fm.num_blocks,
        plane_scale=cfg.scale,
    ).to(device)


def proxy_features(cfg: ExperimentConfig, device: torch.device,
                   num_points: int = 10000) -> tuple[torch.Tensor, torch.Tensor]:
    """GMM-sampled proxy features backing area-ratio rejection sampling and mass estimates."""
    proxy_x, _ = get_points(batch_size=num_points, device=device)
    return compute_poly_features(proxy_x.to(device), degree=cfg.degree, scale=cfg.scale)


def load_pool(cfg: ExperimentConfig, device: torch.device) -> dict[str, torch.Tensor]:
    path = cfg.pool_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Functa pool missing: {path}\n"
            f"Build it first: sbatch scripts/run_pool.sh <config.yaml>")
    pool = torch.load(path, map_location=device)
    return {k: v.to(device) for k, v in pool.items()}


def save_checkpoint(cfg: ExperimentConfig, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Any, iteration: int, include_optimizer: bool = True) -> Path:
    """Writes atomically so an interrupted job never leaves a truncated checkpoint behind.

    Adam moments triple the file size and are only needed to resume an interrupted run,
    so the final save drops them.
    """
    path = run_dir(cfg.run_id) / CHECKPOINT_NAME
    tmp = path.with_suffix(".pt.tmp")
    payload = {
        "iteration": iteration,
        "model": {k: v for k, v in model.state_dict().items() if not k.startswith("siren.")},
        "run_id": cfg.run_id,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(cfg: ExperimentConfig, model: torch.nn.Module, device: torch.device,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler: Any | None = None) -> int:
    """Restores trainable weights (the frozen SIREN is rebuilt, never stored). Returns the iteration."""
    path = run_dir(cfg.run_id) / CHECKPOINT_NAME
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint for run '{cfg.run_id}': {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    unexpected = [k for k in unexpected if not k.startswith("siren.")]
    missing = [k for k in missing if not k.startswith("siren.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing={missing}, unexpected={unexpected}")

    if optimizer is not None:
        if "optimizer" not in ckpt:
            raise RuntimeError(f"run '{cfg.run_id}' finished training; its checkpoint carries no "
                               f"optimizer state to resume from")
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["iteration"])


__all__ = ["resolve_device", "set_seed", "load_siren", "build_flow_matcher", "proxy_features",
           "load_pool", "save_checkpoint", "load_checkpoint"]
