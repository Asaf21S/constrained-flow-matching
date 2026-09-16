# -*- coding: utf-8 -*-
"""The few-shot budget sweep as a standalone LaTeX table.

One row per shot budget N, one column per metric. Both the 100-constraint benchmark and the
1000-constraint validation set carry a sweep; they are reported as separate blocks rather
than interleaved, since a row from one is not comparable with a row from the other.

Pure stdlib, so it runs on the login node without the container.

    python3 -m constrained_fm.scripts.table_few_shot
    python3 -m constrained_fm.scripts.table_few_shot --source both
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from constrained_fm.scripts.table_val1k import format_cell

DEFAULT_SUMMARY = "constrained_fm/baselines/few_shot/summary.json"
DEFAULT_VAL1K = "constrained_fm/baselines/val1k/metrics.json"
DEFAULT_TABLE = "constrained_fm/tables/few_shot.tex"

# The headline budget kept the bare method name; later budgets carry an _N<points> suffix.
FEWSHOT_METHOD = re.compile(r"^fewshot(?:_N(\d+))?$")

# (metric key, column header, decimals, higher_is_better)
METRICS = (
    ("success_rate", r"Acceptance Rate (\%)", 2, True),
    ("swd", "SWD", 4, False),
    ("mmd", "MMD", 5, False),
    ("kld", "KLD", 4, False),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit the few-shot budget sweep as LaTeX.")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY,
                        help="few-shot sweep summary holding per_n_median")
    parser.add_argument("--val1k", default=DEFAULT_VAL1K, help="merged v1k metrics")
    parser.add_argument("--source", choices=("auto", "v1k", "100set", "both"), default="auto",
                        help="auto prefers the v1k sweep and falls back to the 100-set one")
    parser.add_argument("--out", default=DEFAULT_TABLE, help="path of the .tex file to write")
    parser.add_argument("--label", default="tab:few-shot")
    return parser


def sweep_100set(path: Path) -> dict | None:
    """Per-budget medians from the 100-constraint sweep."""
    if not path.exists():
        return None
    per_n = json.loads(path.read_text())["per_n_median"]
    rows = {int(n): {key: float(values.get(key, float("nan"))) for key, _, _, _ in METRICS}
            for n, values in per_n.items()}
    count = max(int(values.get("count", 0)) for values in per_n.values())
    return {"rows": rows, "title": f"{count}-polynomial benchmark"}


def sweep_v1k(path: Path) -> dict | None:
    """Every few-shot budget merged into the v1k metrics, keyed by N."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    rows: dict[int, dict[str, float]] = {}
    for method, merged in payload.get("methods", {}).items():
        match = FEWSHOT_METHOD.match(method)
        if match is None:
            continue
        n_points = merged.get("eval", {}).get("n_points") or match.group(1)
        if n_points is None:
            continue
        summary = merged["summary"]
        rows[int(n_points)] = {key: float(summary.get(f"{key}_median", float("nan")))
                               for key, _, _, _ in METRICS}
    if not rows:
        return None
    return {"rows": rows,
            "title": f"{int(payload['num_constraints'])}-polynomial validation set"}


def select_blocks(args: argparse.Namespace) -> list[dict]:
    v1k = sweep_v1k(Path(args.val1k))
    hundred = sweep_100set(Path(args.summary))

    if args.source == "v1k":
        chosen = [v1k]
    elif args.source == "100set":
        chosen = [hundred]
    elif args.source == "both":
        chosen = [hundred, v1k]
    else:
        chosen = [v1k] if v1k is not None and len(v1k["rows"]) > 1 else [hundred]

    blocks = [block for block in chosen if block is not None]
    if not blocks:
        raise FileNotFoundError(f"no few-shot results at {args.summary} or {args.val1k}")
    return blocks


def best_values(rows: dict[int, dict[str, float]]) -> dict[str, float]:
    """Bolding compares budgets within one benchmark, never across benchmarks."""
    best: dict[str, float] = {}
    if len(rows) < 2:
        return best
    for key, _, _, higher_better in METRICS:
        finite = [values[key] for values in rows.values() if math.isfinite(values[key])]
        if finite:
            best[key] = max(finite) if higher_better else min(finite)
    return best


def format_row(label: str, values: dict[str, float], best: dict[str, float]) -> str:
    cells = []
    for key, _, decimals, _ in METRICS:
        value = float(values.get(key, float("nan")))
        target = best.get(key)
        is_best = (target is not None and math.isfinite(value)
                   and math.isclose(value, target))
        cells.append(format_cell(value, decimals, is_best))
    return f"{label} & " + " & ".join(cells) + r" \\"


def build_table(blocks: list[dict], label: str) -> str:
    headers = " & ".join(header for _, header, _, _ in METRICS)
    span = len(METRICS) + 1

    lines = [
        r"% Generated by constrained_fm.scripts.table_few_shot -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{r" + "r" * len(METRICS) + "}",
        r"\toprule",
        rf"$N$ & {headers} \\",
    ]

    for block in blocks:
        lines.append(r"\midrule")
        if len(blocks) > 1:
            lines.append(rf"\multicolumn{{{span}}}{{l}}{{\emph{{{block['title']}}}}} \\")
        best = best_values(block["rows"])
        lines += [format_row(f"{n}", block["rows"][n], best) for n in sorted(block["rows"])]

    described = f"the {blocks[0]['title']}" if len(blocks) == 1 else "each benchmark"
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Few-shot unconstrained baseline as a function of the number of valid "
        r"training samples $N$. For each constraint, $N$ points satisfying $P(x) \le 0$ are "
        r"rejection-sampled and an unconditional flow matcher is trained from scratch on "
        r"them, so every row is one independently trained model per constraint of "
        rf"{described}. Values are medians. Training length is chosen by early stopping "
        r"against 10{,}000 held-out constraint-satisfying points, far more data than the "
        r"model is allowed to train on; this removes training length as a confound but makes "
        r"these numbers an optimistic upper bound on the baseline.}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    blocks = select_blocks(args)

    table = build_table(blocks, args.label)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table)

    for block in blocks:
        print(f"{block['title']}: budgets {sorted(block['rows'])}")
    print(f"wrote {out}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
