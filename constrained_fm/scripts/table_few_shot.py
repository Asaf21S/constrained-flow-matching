# -*- coding: utf-8 -*-
"""The few-shot budget sweep as a standalone LaTeX table.

One row per shot budget N, one column per metric. Both the 100-constraint benchmark and the
1000-constraint validation set carry a sweep; they are reported as separate blocks rather
than interleaved, since a row from one is not comparable with a row from the other.

The v1k sweep exists twice, trained from scratch and fine-tuned from the base_fm checkpoint;
``--fewshot-source`` picks one, and each writes its own .tex file by default.

Pure stdlib, so it runs on the login node without the container.

    python3 -m constrained_fm.scripts.table_few_shot
    python3 -m constrained_fm.scripts.table_few_shot --fewshot-source scratch --source both
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from constrained_fm.scripts import fewshot_source
from constrained_fm.scripts.table_val1k import format_cell

DEFAULT_SUMMARY = "constrained_fm/baselines/few_shot/summary.json"
DEFAULT_VAL1K = "constrained_fm/baselines/val1k/metrics.json"
DEFAULT_TABLES = {fewshot_source.SCRATCH: "constrained_fm/tables/few_shot.tex",
                  fewshot_source.FINETUNED: "constrained_fm/tables/few_shot_finetuned.tex"}

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
    parser.add_argument("--out", default=None,
                        help="path of the .tex file to write (default depends on the run)")
    parser.add_argument("--label", default=None,
                        help="LaTeX label (default tab:few-shot, or tab:few-shot-ft)")
    fewshot_source.add_arguments(parser, budget=False)
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


def sweep_v1k(path: Path, source: str, finetuned_path: str) -> dict | None:
    """Every budget of one v1k few-shot run, keyed by N."""
    if not path.exists():
        return None
    by_n, payload = fewshot_source.budgets(source, json.loads(path.read_text()), finetuned_path)
    if not by_n:
        return None
    rows = {n: {key: float(merged["summary"].get(f"{key}_median", float("nan")))
                for key, _, _, _ in METRICS}
            for n, merged in by_n.items()}
    return {"rows": rows, "eval": next(iter(by_n.values())).get("eval", {}),
            "title": f"{int(payload['num_constraints'])}-polynomial validation set"}


def select_blocks(args: argparse.Namespace) -> list[dict]:
    if args.fewshot_source == fewshot_source.FINETUNED and args.source in ("100set", "both"):
        raise ValueError("the 100-constraint sweep was only trained from scratch; pass "
                         f"--fewshot-source {fewshot_source.SCRATCH} to include it")
    v1k = sweep_v1k(Path(args.val1k), args.fewshot_source, args.fewshot_metrics)
    hundred = sweep_100set(Path(args.summary))

    if args.source == "v1k" or args.fewshot_source == fewshot_source.FINETUNED:
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


def latex_scientific(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    power = rf"10^{{{int(exponent)}}}"
    return power if mantissa == "1" else rf"{mantissa} \times {power}"


def training_sentence(source: str, blocks: list[dict]) -> str:
    if source == fewshot_source.SCRATCH:
        return (r"an unconditional flow matcher is trained from scratch on them, so every row "
                r"is one independently trained model per constraint")
    lr = blocks[0].get("eval", {}).get("lr")
    optimiser = rf" with learning rate ${latex_scientific(lr)}$" if lr is not None else ""
    return (r"the unconditional flow matcher that ECI and HardFlow sample from is fine-tuned "
            rf"on them{optimiser}, restarting from the same base weights each time, so every "
            r"row is one fine-tuned model per constraint")


def build_table(blocks: list[dict], label: str, source: str) -> str:
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
    # The base model competes for best checkpoint, so fine-tuning cannot end up worse than it.
    kept = ("" if source == fewshot_source.SCRATCH else
            r" The base model itself is kept whenever no fine-tuning step improves on it.")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Few-shot unconstrained baseline as a function of the number of valid "
        r"training samples $N$. For each constraint, $N$ points satisfying $P(x) \le 0$ are "
        rf"rejection-sampled and {training_sentence(source, blocks)} of {described}. "
        r"Values are medians. "
        r"Training length is chosen by early stopping against 10{,}000 held-out "
        r"constraint-satisfying points, far more data than the model is allowed to train on; "
        r"this removes training length as a confound but makes these numbers an optimistic "
        rf"upper bound on the baseline.{kept}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    blocks = select_blocks(args)

    finetuned = args.fewshot_source == fewshot_source.FINETUNED
    label = args.label or ("tab:few-shot-ft" if finetuned else "tab:few-shot")
    table = build_table(blocks, label, args.fewshot_source)
    out = Path(args.out or DEFAULT_TABLES[args.fewshot_source])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table)

    for block in blocks:
        print(f"{block['title']}: budgets {sorted(block['rows'])}")
    print(f"wrote {out}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
