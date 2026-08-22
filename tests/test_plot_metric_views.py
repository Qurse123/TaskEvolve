"""$0 unit tests for scripts.plot_metric_views — the headline-metric math.

Covers the cost_per_successful_task formula and its undefined-at-zero-success
handling (the blow-up the metric_justification note flags), plus arm labelling
and CSV aggregation. No matplotlib rendering, no I/O beyond a temp CSV.

Run:  python -m pytest tests/test_plot_metric_views.py -q
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from scripts.plot_metric_views import (
    _arm_label,
    cost_per_successful_task,
    load,
)


def test_formula_matches_definition() -> None:
    # cost/(num_tasks*pass_rate) == cost_per_task / success_rate
    got = cost_per_successful_task(total_cost=10.0, num_tasks=20, pass_rate=0.5)
    assert math.isclose(got, 10.0 / (20 * 0.5))  # == 1.0
    assert math.isclose(got, (10.0 / 20) / 0.5)


def test_matches_ledger_row() -> None:
    # gpt-4.1 retail: total ~ 0.0497*35 over the run; check a clean synthetic case
    # matching the published $0.0619 shape: cost/task 0.0497, pass 0.806.
    cpt, pass_rate = 0.0497, 0.806
    total = cpt * 35
    got = cost_per_successful_task(total, 35, pass_rate)
    assert math.isclose(got, cpt / pass_rate, rel_tol=1e-9)
    assert math.isclose(got, 0.06166, rel_tol=1e-3)


def test_undefined_at_zero_success() -> None:
    # pass_rate == 0 -> undefined (no successful task), returns None, never inf/raise.
    assert cost_per_successful_task(5.0, 10, 0.0) is None


def test_undefined_at_nonpositive_tasks() -> None:
    assert cost_per_successful_task(5.0, 0, 0.9) is None


def test_low_success_explodes_but_defined() -> None:
    # banking-style: pass_rate 0.10 -> a huge but finite number (the blow-up).
    got = cost_per_successful_task(total_cost=1.0, num_tasks=10, pass_rate=0.10)
    assert got == 1.0  # 1.0 / (10*0.1); large relative to cost/task 0.1
    assert got == 10 * (1.0 / 10)  # 10x the cost-per-task at 10% success


def test_arm_label_routing_harness() -> None:
    # v0.2 = the accepted whole-agent swap: base Opus, effectively Sonnet.
    assert _arm_label("anthropic/claude-opus-4-8", "v0.2") == "Opus->Sonnet5 (v0.2)"
    assert _arm_label("openai/thinkingmachines/Inkling", "v0.1") == "Inkling (v0.1)"


def _write_csv(path: Path, rows: list[dict]) -> Path:
    fields = ["split", "domain", "agent_model", "harness_version",
              "num_tasks", "pass_rate", "total_cost_usd"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def test_load_filters_retail_validation_and_averages(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "results.csv", [
        {"split": "validation", "domain": "retail", "agent_model": "gpt-4.1",
         "harness_version": "v0.1", "num_tasks": "10", "pass_rate": "0.8",
         "total_cost_usd": "1.0"},
        {"split": "validation", "domain": "retail", "agent_model": "gpt-4.1",
         "harness_version": "v0.1", "num_tasks": "10", "pass_rate": "0.6",
         "total_cost_usd": "1.0"},
        # excluded: wrong split
        {"split": "proxy", "domain": "retail", "agent_model": "gpt-4.1",
         "harness_version": "v0.1", "num_tasks": "10", "pass_rate": "0.5",
         "total_cost_usd": "9.0"},
        # excluded: wrong domain
        {"split": "validation", "domain": "airline", "agent_model": "gpt-4.1",
         "harness_version": "v0.1", "num_tasks": "10", "pass_rate": "0.5",
         "total_cost_usd": "9.0"},
    ])
    arms = load(csv_path)
    assert set(arms) == {"gpt-4.1 (v0.1)"}
    a = arms["gpt-4.1 (v0.1)"]
    assert math.isclose(a["pass"], 0.7)   # (0.8+0.6)/2
    assert math.isclose(a["cpt"], 0.1)    # both 1.0/10
    # cst averaged over the two runs' per-run cost_per_successful_task
    assert math.isclose(a["cst"], (0.1 / 0.8 + 0.1 / 0.6) / 2)


def test_load_drops_zero_success_from_cst(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "results.csv", [
        {"split": "validation", "domain": "retail", "agent_model": "x/m",
         "harness_version": "v0.1", "num_tasks": "10", "pass_rate": "0.0",
         "total_cost_usd": "1.0"},
    ])
    arms = load(csv_path)
    assert arms["m (v0.1)"]["cst"] is None  # only run had zero success -> no cst
