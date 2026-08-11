# -*- coding: utf-8 -*-
"""Typed, hashable experiment configuration for the Functa-conditioned pipeline.

Importable without torch or numpy so configs can be validated on the login node.
"""

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SirenConfig:
    """The frozen Functa encoder. Architecture fields must match the checkpoint."""

    checkpoint: str = "constrained_fm/functa_dataset/siren_best.pt"
    latent_dim: int = 512
    hidden_dim: int = 512
    n_layers: int = 4
    w0: float = 30.0


@dataclass(frozen=True)
class ExtractionConfig:
    """CAVIA inner loop used for every latent extraction: pool, training, evaluation."""

    points_per_shape: int = 1000
    steps: int = 15
    lr: float = 1e-2
    # Must match the value the SIREN was meta-trained with; drift here silently degrades fidelity.
    query_gmm_fraction: float = 0.0


@dataclass(frozen=True)
class PoolConfig:
    """Precomputed (C, z_pos, z_neg) pool that removes SIREN calls from the training loop."""

    size: int = 100000
    chunk_size: int = 128
    min_area: float = 0.05
    max_area: float = 0.95


@dataclass(frozen=True)
class FMConfig:
    """ConstrainedFlowMatcher architecture.

    use_siren_feature toggles only the auxiliary pointwise input SIREN(x_t, z); z itself is
    always concatenated into the input and always modulates the AdaGN blocks.
    """

    hidden_dim: int = 1024
    num_blocks: int = 4
    time_emb_dim: int = 128
    use_siren_feature: bool = True


@dataclass(frozen=True)
class TrainConfig:
    iterations: int = 15001
    batch_size: int = 1024
    lr: float = 1e-3
    lr_min: float = 1e-5
    # Per-example loss weight mass^(-power); 0.0 leaves exposure proportional to valid mass.
    mass_weight_power: float = 0.0
    max_weight: float = 20.0
    seed: int = 0
    log_every: int = 500
    checkpoint_every: int = 2500


@dataclass(frozen=True)
class EvalConfig:
    """Excluded from the run fingerprint: re-evaluating never forks a new run."""

    num_polys: int = 100
    num_x0: int = 10000
    step_size: float = 0.05
    gmm_pool_size: int = 100000
    iou_grid_size: int = 200
    iou_mass_samples: int = 20000
    likelihood_grid: int = 200
    num_vis_samples: int = 50000
    num_worst_plots: int = 4
    seed: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "baseline"
    description: str = ""
    degree: int = POLYNOMIAL_DEGREE
    scale: float = PLANE_SCALE
    siren: SirenConfig = field(default_factory=SirenConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # --- construction -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return _build(cls, data or {}, path="")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(_load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def smoke(self) -> "ExperimentConfig":
        """Shrinks every expensive knob so one pass exercises the whole pipeline in minutes."""
        return replace(
            self,
            name=f"{self.name}-smoke",
            pool=replace(self.pool, size=2048),
            train=replace(self.train, iterations=201, log_every=50, checkpoint_every=200),
            evaluation=replace(self.evaluation, num_polys=8, num_x0=2000, gmm_pool_size=20000,
                               iou_mass_samples=5000, likelihood_grid=50, num_vis_samples=5000),
        )

    # --- identity ---------------------------------------------------------

    def siren_path(self) -> Path:
        path = Path(self.siren.checkpoint)
        return path if path.is_absolute() else REPO_ROOT / path

    def siren_digest(self) -> str:
        return file_digest(self.siren_path())

    def pool_fingerprint(self) -> str:
        """Identifies a pool by everything that changes its contents, SIREN weights included."""
        return _sha({
            "degree": self.degree,
            "scale": self.scale,
            "siren": dataclasses.asdict(self.siren),
            "siren_digest": self.siren_digest(),
            "extraction": dataclasses.asdict(self.extraction),
            "pool": dataclasses.asdict(self.pool),
        })

    def pool_path(self) -> Path:
        return (REPO_ROOT / "constrained_fm" / "functa_dataset" / "pools" /
                f"pool_{self.pool.size}_{self.pool_fingerprint()[:10]}.pt")

    def fingerprint(self) -> str:
        """Covers everything that changes the trained checkpoint; evaluation knobs do not."""
        return _sha({
            "degree": self.degree,
            "scale": self.scale,
            "siren": dataclasses.asdict(self.siren),
            "siren_digest": self.siren_digest(),
            "extraction": dataclasses.asdict(self.extraction),
            "pool": dataclasses.asdict(self.pool),
            "fm": dataclasses.asdict(self.fm),
            "train": dataclasses.asdict(self.train),
        })

    @property
    def run_id(self) -> str:
        return f"{self.name}-{self.fingerprint()[:8]}"

    def provenance(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "description": self.description,
            "fingerprint": self.fingerprint(),
            "siren_checkpoint": str(self.siren_path()),
            "siren_digest": self.siren_digest(),
            "pool_path": str(self.pool_path()),
            "pool_fingerprint": self.pool_fingerprint(),
            "git_commit": git_commit(),
        }


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_yaml(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Loads a config, deep-merging over the base named by an optional `extends:` key."""
    resolved = _resolve(path).resolve()
    _seen = _seen or set()
    if resolved in _seen:
        raise ValueError(f"circular 'extends' chain at {resolved}")
    _seen.add(resolved)

    with open(resolved, "r") as f:
        data = yaml.safe_load(f) or {}

    base_ref = data.pop("extends", None)
    if base_ref is None:
        return data

    base_path = _resolve(base_ref)
    if not base_path.exists():
        base_path = resolved.parent / base_ref
    return _deep_merge(_load_yaml(base_path, _seen), data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build(cls, data: dict[str, Any], path: str):
    """Recursively instantiates nested dataclasses, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping at '{path or 'root'}', got {type(data).__name__}")

    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} at '{path or 'root'}'; "
                         f"valid keys: {sorted(fields)}")

    kwargs = {}
    for key, value in data.items():
        field_type = fields[key].type
        prefix = f"{path}.{key}" if path else key
        if dataclasses.is_dataclass(field_type):
            kwargs[key] = _build(field_type, value, path=prefix)
        elif value is not None and not isinstance(value, field_type):
            # YAML happily yields int where float is meant; anything else is a real typo.
            if field_type is float and isinstance(value, int) and not isinstance(value, bool):
                kwargs[key] = float(value)
            else:
                raise TypeError(f"'{prefix}' expects {field_type.__name__}, "
                                f"got {type(value).__name__}")
        else:
            kwargs[key] = value

    return cls(**kwargs)


__all__ = ["ExperimentConfig", "SirenConfig", "ExtractionConfig", "PoolConfig",
           "FMConfig", "TrainConfig", "EvalConfig", "REPO_ROOT", "file_digest", "git_commit"]
