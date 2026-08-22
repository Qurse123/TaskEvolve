"""Emit a per-arm provenance table from ``experiments/results.csv``.

$0, read-only. Groups run rows into arms by ``(agent_model, harness_version,
split)`` and reports, per arm: task count, N (repeats), pass_rate mean, and the
run-date range (from ``generated_at``). Seeds are NOT in results.csv (they live in
the per-task JSONs), so they are intentionally absent here — see
``docs/paper/analysis/provenance.md`` §4 G1.

    python -m scripts.provenance                       # markdown table to stdout
    python -m scripts.provenance --csv experiments/results.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

DEFAULT_CSV = Path("experiments/results.csv")


class ArmRow(NamedTuple):
    agent_model: str
    harness_version: str
    split: str
    domain: str
    num_tasks: str
    n_repeats: int
    pass_rate_mean: Optional[float]
    date_first: str
    date_last: str


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(csv_path: Path = DEFAULT_CSV) -> List[ArmRow]:
    """Group results.csv rows into arms and return one summary row per arm."""
    groups: Dict[tuple, list] = defaultdict(list)
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["agent_model"], row["harness_version"], row["split"])
            groups[key].append(row)

    out: List[ArmRow] = []
    for (agent_model, harness, split), rows in groups.items():
        dates = sorted(r["generated_at"] for r in rows)
        rates = [r for r in (_to_float(x["pass_rate"]) for x in rows) if r is not None]
        out.append(
            ArmRow(
                agent_model=agent_model,
                harness_version=harness,
                split=split,
                domain=rows[0].get("domain", ""),
                num_tasks=rows[0].get("num_tasks", ""),
                n_repeats=len(rows),
                pass_rate_mean=(statistics.mean(rates) if rates else None),
                date_first=dates[0][:10],
                date_last=dates[-1][:10],
            )
        )
    out.sort(key=lambda a: (a.date_first, a.split, a.agent_model))
    return out


def render_markdown(arms: List[ArmRow]) -> str:
    header = (
        "| agent_model | harness | split | domain | tasks | N | pass_rate (mean) | dates |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for a in arms:
        pr = f"{a.pass_rate_mean:.3f}" if a.pass_rate_mean is not None else "-"
        dates = a.date_first if a.date_first == a.date_last else f"{a.date_first}..{a.date_last}"
        lines.append(
            f"| {a.agent_model} | {a.harness_version} | {a.split} | {a.domain} "
            f"| {a.num_tasks} | {a.n_repeats} | {pr} | {dates} |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)
    print(render_markdown(summarize(args.csv)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
