# -*- coding: utf-8 -*-
"""Stitches the per-shard query-budget scores into one metrics.json.

Pure stdlib, so it runs on the login node without the container. Refuses to merge a budget
whose shards do not cover every constraint exactly once, since a silently truncated sweep
would still produce a plausible-looking bar chart.

    python3 -m constrained_fm.scripts.merge_query_budget
    python3 -m constrained_fm.scripts.merge_query_budget --allow-partial
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from constrained_fm.src.experiment.registry import summarize

DEFAULT_OUTDIR = "constrained_fm/baselines/query_budget"
METRIC_KEYS = ("mass_iou", "extraction_mse", "success_rate", "swd", "mmd", "jsd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge query-budget ablation shards.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--expected", type=int, default=None,
                        help="constraint count to require (default: inferred from the shards)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="merge anyway and leave the gaps as NaN")
    return parser


def load_shards(shard_dir: Path) -> dict[int, list[dict]]:
    if not shard_dir.exists():
        raise FileNotFoundError(
            f"no shards at {shard_dir} -- run sbatch scripts/run_query_budget_eval.sh")

    by_budget: dict[int, list[dict]] = {}
    for path in sorted(shard_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        by_budget.setdefault(int(payload["num_points"]), []).append(payload)
    if not by_budget:
        raise FileNotFoundError(f"{shard_dir} holds no shard files")
    return by_budget


def merge_budget(shards: list[dict], expected: int, allow_partial: bool) -> dict:
    """Scatters every shard's rows into one array per metric, indexed by constraint."""
    per_shape = {key: [float("nan")] * expected for key in (*METRIC_KEYS, "mass")}
    seen: dict[int, str] = {}

    for shard in shards:
        label = f"[{shard['start_idx']}, {shard['end_idx']})"
        for position, index in enumerate(shard["indices"]):
            if index in seen:
                raise RuntimeError(f"constraint {index} appears in both {seen[index]} and {label}")
            seen[index] = label
            for key, values in shard["per_shape"].items():
                if key in per_shape:
                    per_shape[key][index] = float(values[position])

    missing = [i for i in range(expected) if i not in seen]
    if missing and not allow_partial:
        raise RuntimeError(
            f"shards cover {len(seen)}/{expected} constraints; {len(missing)} missing "
            f"(first few: {missing[:10]}). Re-run the failed array tasks, or pass "
            f"--allow-partial to merge the gaps as NaN.")

    per_shape = {key: values for key, values in per_shape.items()
                 if any(math.isfinite(v) for v in values)}
    reference = shards[0]
    return {
        "num_points": reference["num_points"],
        "run_id": reference["run_id"],
        "eval": reference["eval"],
        "num_constraints": expected,
        "num_scored": len(seen),
        "missing": missing,
        "per_shape": per_shape,
        "summary": summarize(per_shape),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.outdir)
    by_budget = load_shards(out / "shards")

    expected = args.expected
    if expected is None:
        expected = max(shard["end_idx"] for shards in by_budget.values() for shard in shards)

    digests = {shard["poly_digest"] for shards in by_budget.values() for shard in shards}
    if len(digests) > 1:
        raise RuntimeError(f"shards were scored against different validation sets: {digests}")

    budgets = {str(n): merge_budget(shards, expected, args.allow_partial)
               for n, shards in sorted(by_budget.items())}
    n_values = sorted(int(n) for n in budgets)

    mass = next((b["per_shape"]["mass"] for b in budgets.values() if "mass" in b["per_shape"]), None)
    payload = {
        "validation_set": next(iter(by_budget.values()))[0]["validation_set"],
        "poly_digest": digests.pop(),
        "num_constraints": expected,
        "n_values": n_values,
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "mass": mass,
        "budgets": budgets,
    }

    path = out / "metrics.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    print(f"{'N':>6}  {'scored':>10}  {'IoU mean':>9}  {'MSE mean':>10}  {'SR mean':>8}  "
          f"{'SWD mean':>9}")
    for n in n_values:
        merged = budgets[str(n)]
        summary = merged["summary"]
        nan = float("nan")
        print(f"{n:>6}  {merged['num_scored']:>5}/{expected:<4}  "
              f"{summary.get('mass_iou_mean', nan):>9.4f}  "
              f"{summary.get('extraction_mse_mean', nan):>10.2e}  "
              f"{summary.get('success_rate_mean', nan):>8.2f}  "
              f"{summary.get('swd_mean', nan):>9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
