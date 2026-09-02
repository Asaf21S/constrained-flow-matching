# -*- coding: utf-8 -*-
"""Plot-ready artifact store shared by every method's evaluation loop.

Each evaluation writes the arrays its figures consume -- generated samples, constraint
coefficients, latents, decoded fields, trajectories, likelihood grids -- next to the
metrics they were scored with, so a figure can be redrawn from disk without a model,
a checkpoint, or an ODE solve.

Layout, relative to a run root (``runs/<run_id>/`` or ``constrained_fm/baselines/<name>/``)::

    artifacts/
        manifest.json   run id, git commit, and the shape/dtype of every array below
        <name>.npy      one file per saved array
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from constrained_fm.src.experiment.config import git_commit

ARTIFACTS_DIR = "artifacts"
MANIFEST_NAME = "manifest.json"


def artifacts_dir(root: str | Path, create: bool = False) -> Path:
    path = Path(root) / ARTIFACTS_DIR
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _to_numpy(value: Any) -> np.ndarray:
    """Accepts torch tensors without importing torch, so this module stays dependency-light."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def save_arrays(root: str | Path, **arrays: Any) -> Path:
    """Writes one .npy per keyword and records its shape and dtype in the manifest."""
    out = artifacts_dir(root, create=True)
    entries = load_manifest(root).get("arrays", {})

    for name, value in arrays.items():
        array = _to_numpy(value)
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        np.save(out / f"{name}.npy", array)
        entries[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}

    write_manifest(root, arrays=entries)
    return out


def load_array(root: str | Path, name: str) -> np.ndarray:
    path = artifacts_dir(root) / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"artifact '{name}' missing under {path.parent}. Re-run the evaluation stage "
            f"for this run to regenerate its plotting artifacts.")
    return np.load(path)


def has_array(root: str | Path, name: str) -> bool:
    return (artifacts_dir(root) / f"{name}.npy").exists()


def load_arrays(root: str | Path, *names: str) -> dict[str, np.ndarray]:
    """Loads the named arrays, silently skipping the ones that were never written."""
    return {name: load_array(root, name) for name in names if has_array(root, name)}


def write_manifest(root: str | Path, **fields: Any) -> Path:
    """Merges fields into the manifest, refreshing the provenance stamp on every write."""
    path = artifacts_dir(root, create=True) / MANIFEST_NAME
    manifest = load_manifest(root)
    manifest.update(fields)
    manifest["written_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["git_commit"] = git_commit()
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_manifest(root: str | Path) -> dict[str, Any]:
    path = artifacts_dir(root) / MANIFEST_NAME
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


__all__ = ["ARTIFACTS_DIR", "MANIFEST_NAME", "artifacts_dir", "save_arrays", "load_array",
           "has_array", "load_arrays", "write_manifest", "load_manifest"]
