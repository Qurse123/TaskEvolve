"""Shared loader for the cached held-out test statistics.

Rows come from scripts/build_test_task_stats.py. Colours and order match
scripts/plot_paper_fig1_frontier.SYSTEMS so a reader carries one legend
through the whole paper.
"""
from __future__ import annotations

import csv
from pathlib import Path

from scripts import paper_style as style

STATS = Path("experiments/analysis/test_task_stats.csv")
OCHRE = "#B58A47"

# key -> (label, colour)
SYSTEMS = [
    ("opus_static", "Opus 4.8, static harness", style.WARM_GRAY),
    ("opus_iterator", "Opus 4.8 plus agent iterator", OCHRE),
    ("inkling_base", "Inkling-Small base", style.STEEL_BLUE),
    ("inkling_tuned", "Inkling-Small fine-tuned", style.TERRACOTTA),
]

DOMAINS = [("test_retail", "Retail"), ("test_airline", "Airline"), ("test_telecom", "Telecom")]

NUMERIC = {"reward": float, "passed": int, "agent_cost_usd": float, "user_cost_usd": float,
           "input_tokens": int, "output_tokens": int, "assistant_messages": int,
           "tool_calls": int, "handoff": int}


def load(path: Path = STATS) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} is missing; run: uv run python -m scripts.build_test_task_stats")
    rows = []
    for row in csv.DictReader(path.open()):
        for field, cast in NUMERIC.items():
            row[field] = cast(row[field])
        rows.append(row)
    return rows


def select(rows, split, system):
    return [r for r in rows if r["split"] == split and r["system"] == system]
