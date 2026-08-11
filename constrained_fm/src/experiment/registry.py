# -*- coding: utf-8 -*-
"""Filesystem registry for training runs: layout, provenance, and cross-run comparison.

Pure stdlib (plus yaml) so it can be used from the login node without torch or numpy.

Layout::

    runs/<run_id>/
        config.yaml      resolved configuration
        provenance.json  siren digest, pool fingerprint, git commit
        state.json       training status and last iteration
        ckpt.pt          model + optimizer + scheduler, resumable
        losses.npy       per-iteration training loss
        metrics.json     per-shape evaluation arrays and summary
        figures/*.png    diagnostic figures
"""

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from constrained_fm.src.experiment.config import REPO_ROOT, ExperimentConfig

RUNS_ROOT = REPO_ROOT / "runs"

CONFIG_NAME = "config.yaml"
PROVENANCE_NAME = "provenance.json"
STATE_NAME = "state.json"
CHECKPOINT_NAME = "ckpt.pt"
LOSSES_NAME = "losses.npy"
METRICS_NAME = "metrics.json"
FIGURES_DIR = "figures"


def run_dir(run_id: str, create: bool = False) -> Path:
    path = RUNS_ROOT / run_id
    if create:
        (path / FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def init_run(cfg: ExperimentConfig) -> Path:
    """Creates the run directory and pins the resolved config and provenance to disk."""
    path = run_dir(cfg.run_id, create=True)
    cfg.save_yaml(path / CONFIG_NAME)
    write_json(path / PROVENANCE_NAME, cfg.provenance())
    return path


def load_config(run_id: str) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(run_dir(run_id) / CONFIG_NAME)


def load_metrics(run_id: str) -> dict[str, Any] | None:
    return read_json(run_dir(run_id) / METRICS_NAME)


def load_state(run_id: str) -> dict[str, Any] | None:
    return read_json(run_dir(run_id) / STATE_NAME)


def write_state(run_id: str, **fields: Any) -> None:
    path = run_dir(run_id) / STATE_NAME
    state = read_json(path) or {}
    state.update(fields)
    write_json(path, state)


def list_runs() -> list[dict[str, Any]]:
    """One record per run directory, newest last, safe to call on an empty registry."""
    if not RUNS_ROOT.exists():
        return []

    records = []
    for path in sorted(RUNS_ROOT.iterdir()):
        if not (path / CONFIG_NAME).exists():
            continue
        state = read_json(path / STATE_NAME) or {}
        metrics = read_json(path / METRICS_NAME) or {}
        records.append({
            "run_id": path.name,
            "status": state.get("status", "unknown"),
            "iteration": state.get("iteration"),
            "finished_at": state.get("finished_at", ""),
            "has_checkpoint": (path / CHECKPOINT_NAME).exists(),
            "summary": metrics.get("summary", {}),
            "description": (read_json(path / PROVENANCE_NAME) or {}).get("description", ""),
        })
    return records


def latest_run(name_prefix: str | None = None) -> str | None:
    candidates = [r for r in list_runs()
                  if r["summary"] and (name_prefix is None or r["run_id"].startswith(name_prefix))]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["finished_at"])["run_id"]


# --- statistics ---------------------------------------------------------------


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile over finite values; q in [0, 100]."""
    data = sorted(v for v in values if math.isfinite(v))
    if not data:
        return float("nan")
    pos = (len(data) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return data[lo]
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def summarize(per_shape: dict[str, list[float]]) -> dict[str, float]:
    """Median / mean / tail percentile per metric, ignoring non-finite entries."""
    summary: dict[str, float] = {}
    for key, values in per_shape.items():
        clean = [float(v) for v in values if math.isfinite(float(v))]
        if not clean:
            continue
        summary[f"{key}_median"] = statistics.median(clean)
        summary[f"{key}_mean"] = statistics.fmean(clean)
        # Success rate fails low, discrepancies fail high.
        summary[f"{key}_p5"] = percentile(clean, 5.0)
        summary[f"{key}_p95"] = percentile(clean, 95.0)
    return summary


def correlation(a: Iterable[float], b: Iterable[float]) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return float("nan")


# --- reporting ----------------------------------------------------------------


METRIC_LABELS = [
    ("success_rate", "Success Rate (%)", "p5", "Higher is better"),
    ("swd", "Sliced Wasserstein (SWD)", "p95", "Lower is better"),
    ("mmd", "Mean Discrepancy (MMD)", "p95", "Lower is better"),
    ("jsd", "Jensen-Shannon (JSD)", "p95", "Lower is better"),
]


def readme_table(summary: dict[str, float]) -> str:
    """README-ready markdown table, matching the format already used in the project docs."""
    lines = ["| Metric | Median / Value | Mean | Worst 5% | Target |",
             "| :--- | :--- | :--- | :--- | :--- |"]
    for key, label, tail, target in METRIC_LABELS:
        if f"{key}_median" not in summary:
            continue
        digits = 2 if key == "success_rate" else 4
        lines.append(
            f"| **{label}** | {summary[f'{key}_median']:.{digits}f} | "
            f"{summary[f'{key}_mean']:.{digits}f} | "
            f"{summary[f'{key}_{tail}']:.{digits}f} | *{target}* |")
    return "\n".join(lines)


def comparison_table(run_ids: list[str] | None = None) -> str:
    """Markdown table of one row per run, for ranking a sweep at a glance."""
    records = list_runs()
    if run_ids is not None:
        wanted = set(run_ids)
        records = [r for r in records if r["run_id"] in wanted]
    records = [r for r in records if r["summary"]]

    header = ("| run_id | SR median | SR worst5% | SWD median | JSD median | mass IoU mean |\n"
              "| :--- | ---: | ---: | ---: | ---: | ---: |")
    rows = []
    for r in sorted(records, key=lambda x: -x["summary"].get("success_rate_median", 0.0)):
        s = r["summary"]
        rows.append(
            f"| {r['run_id']} | {s.get('success_rate_median', float('nan')):.2f} | "
            f"{s.get('success_rate_p5', float('nan')):.2f} | "
            f"{s.get('swd_median', float('nan')):.4f} | "
            f"{s.get('jsd_median', float('nan')):.4f} | "
            f"{s.get('mass_iou_mean', float('nan')):.3f} |")
    return "\n".join([header, *rows]) if rows else "no evaluated runs yet"


__all__ = ["RUNS_ROOT", "run_dir", "init_run", "load_config", "load_metrics", "load_state",
           "write_state", "list_runs", "latest_run", "summarize", "correlation", "percentile",
           "readme_table", "comparison_table", "read_json", "write_json",
           "CONFIG_NAME", "PROVENANCE_NAME", "STATE_NAME", "CHECKPOINT_NAME",
           "LOSSES_NAME", "METRICS_NAME", "FIGURES_DIR"]
