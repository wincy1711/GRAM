#!/usr/bin/env python3
"""Summarise one or more training runs into a comparison table.

    python scripts/summarize.py runs/gc6_demo_gram runs/gc6_demo_deterministic
    python scripts/summarize.py runs/* --metric coverage --history
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLUMNS = [
    ("exact_match", "exact"),
    ("constraint_accuracy", "valid"),
    ("coverage", "coverage"),
    ("conflict_edges", "conflicts"),
    ("sample_diversity", "diversity"),
    ("gen_validity", "gen_valid"),
    ("mean_halt_step", "halt"),
]


def read_run(directory: Path) -> Optional[Dict]:
    path = directory / "metrics.jsonl"
    if not path.exists():
        return None
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    evals = [r for r in records if r.get("split") == "eval"]
    if not evals:
        return None
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return {
        "name": directory.name,
        "guidance": config.get("model", {}).get("guidance", "?"),
        "params": config.get("model", {}),
        "evals": evals,
        "best": max(evals, key=lambda r: r.get("exact_match", 0.0)),
        "last": evals[-1],
    }


def format_table(runs: List[Dict], use_best: bool) -> str:
    key = "best" if use_best else "last"
    present = [c for c in COLUMNS if any(c[0] in r[key] for r in runs)]
    header = f"{'run':<28}{'guidance':<17}{'epoch':>7}" + "".join(
        f"{label:>12}" for _, label in present
    )
    lines = [header, "-" * len(header)]
    for run in runs:
        row = run[key]
        line = f"{run['name']:<28}{run['guidance']:<17}{row.get('epoch', 0):>7}"
        for field, _ in present:
            value = row.get(field)
            line += f"{value:>12.4f}" if isinstance(value, (int, float)) else f"{'-':>12}"
        lines.append(line)
    return "\n".join(lines)


def format_history(runs: List[Dict], metric: str) -> str:
    lines = [f"\n{metric} over training:"]
    for run in runs:
        points = " ".join(
            f"{r['epoch']}:{r[metric]:.3f}" for r in run["evals"] if metric in r
        )
        lines.append(f"  {run['name']:<28}{points}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--best", action="store_true",
                        help="report the best epoch instead of the last")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--metric", default="coverage")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runs = [r for r in (read_run(Path(p)) for p in args.runs) if r is not None]
    if not runs:
        raise SystemExit("no runs with evaluation records found")

    if args.json:
        print(json.dumps(
            {r["name"]: (r["best"] if args.best else r["last"]) for r in runs},
            indent=2, default=float,
        ))
        return

    print(format_table(runs, args.best))
    if args.history:
        print(format_history(runs, args.metric))


if __name__ == "__main__":
    main()
