# -*- coding: utf-8 -*-
"""Reads the merged shared-budget sweep: per-method, per-budget rows on one constraint subset.

A budget too small to put points inside a constraint leaves both methods nothing to learn
from, so every comparison is restricted to the constraints with at least ``min_inside``
inside points at the smallest budget. Budgets are nested prefixes, so those constraints
have at least that many at every budget and the subset is the same for every row.

Pure stdlib, so the table runs on the login node.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

DEFAULT_METRICS = "constrained_fm/baselines/shared_budget_v1k/metrics.json"
FUNCTA = "functa"
FEWSHOT = "fewshot_ft"
METHODS = (FUNCTA, FEWSHOT)
LABELS = {FUNCTA: "Functa (ours)", FEWSHOT: "Few-Shot FT"}
DEFAULT_MIN_INSIDE = 5

_METHOD_NAME = re.compile(rf"^({'|'.join(METHODS)})_N(\d+)$")


def load(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python3 -m constrained_fm.scripts.merge_val1k "
            f"--outdir {path.parent}` first")
    return json.loads(path.read_text())


def budgets(payload: dict) -> dict[str, dict[int, dict]]:
    """method -> {N: merged entry}; every method must cover the same budgets."""
    by_method: dict[str, dict[int, dict]] = {method: {} for method in METHODS}
    for name, merged in payload["methods"].items():
        match = _METHOD_NAME.match(name)
        if match:
            by_method[match.group(1)][int(match.group(2))] = merged

    grids = {method: sorted(rows) for method, rows in by_method.items()}
    if len({tuple(grid) for grid in grids.values()}) != 1 or not grids[FUNCTA]:
        raise RuntimeError(f"methods cover different budgets: {grids}")
    return by_method


def eligible(by_method: dict[str, dict[int, dict]], min_inside: int) -> list[int]:
    """Constraints with at least ``min_inside`` inside points at the smallest budget."""
    n_min = min(by_method[FUNCTA])
    counts = by_method[FUNCTA][n_min]["per_shape"]["n_inside"]
    if counts != by_method[FEWSHOT][n_min]["per_shape"]["n_inside"]:
        raise RuntimeError("the two methods were not given the same points")
    return [index for index, count in enumerate(counts)
            if math.isfinite(count) and count >= min_inside]


def column(merged: dict, key: str, indices: list[int]) -> list[float]:
    values = merged["per_shape"].get(key)
    if values is None:
        return [float("nan")] * len(indices)
    return [float(values[index]) for index in indices]


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else float("nan")
