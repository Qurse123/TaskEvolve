"""Unit tests for the iterator's baseline loader — the current-best proxy
distribution (mean ± std) the comparator scores candidates against.

Run from the repo root:
    python -m pytest tests/test_baseline.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from iterator_agent.baseline import Distribution, load_distribution
from results.logger import SUMMARY_FIELDS


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SUMMARY_FIELDS})


def _row(run_id: str, *, split: str, harness_version: str, pass_rate, cps) -> dict:
    return {
        "run_id": run_id,
        "split": split,
        "domain": "retail",
        "agent_model": "gpt-4.1",
        "harness_version": harness_version,
        "num_tasks": 12,
        "num_passed": int(round(float(pass_rate) * 12)),
        "pass_rate": pass_rate,
        "total_cost_usd": 1.0,
        "cost_per_successful_task": cps,
        "mean_reward": pass_rate,
        "generated_at": "2026-06-12T19:50:30+00:00",
    }


def test_load_distribution_computes_pass_rate_mean_std(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        [
            _row("proxy_1", split="proxy", harness_version="v0.1", pass_rate=0.5, cps=0.08),
            _row("proxy_2", split="proxy", harness_version="v0.1", pass_rate=0.6, cps=0.09),
            _row("proxy_3", split="proxy", harness_version="v0.1", pass_rate=0.7, cps=0.10),
        ],
    )

    dist = load_distribution("proxy", "v0.1", csv_path=csv_path)

    assert isinstance(dist, Distribution)
    assert dist.n == 3
    assert dist.pass_rate_mean == pytest.approx(0.6)
    assert dist.pass_rate_std == pytest.approx(0.1)
    assert dist.cost_per_successful_task_mean == pytest.approx(0.09)
    assert dist.cost_per_successful_task_std == pytest.approx(0.01)


def test_filters_by_split_and_harness_version(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        [
            _row("proxy_1", split="proxy", harness_version="v0.1", pass_rate=0.5, cps=0.08),
            _row("proxy_2", split="proxy", harness_version="v0.1", pass_rate=0.7, cps=0.10),
            # Different split and different harness — must be ignored.
            _row("val_1", split="validation", harness_version="v0.1", pass_rate=0.9, cps=0.05),
            _row("proxy_b", split="proxy", harness_version="v0.2", pass_rate=0.1, cps=0.30),
        ],
    )

    dist = load_distribution("proxy", "v0.1", csv_path=csv_path)

    assert dist.n == 2
    assert dist.pass_rate_mean == pytest.approx(0.6)
    assert dist.run_ids == ("proxy_1", "proxy_2")


def test_single_row_has_zero_std(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        [_row("proxy_1", split="proxy", harness_version="v0.1", pass_rate=0.6, cps=0.09)],
    )

    dist = load_distribution("proxy", "v0.1", csv_path=csv_path)

    assert dist.n == 1
    assert dist.pass_rate_std == 0.0
    assert dist.cost_per_successful_task_std == 0.0


def test_missing_cost_rows_are_ignored_for_cost_stats(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        [
            _row("proxy_1", split="proxy", harness_version="v0.1", pass_rate=0.5, cps=0.08),
            _row("proxy_2", split="proxy", harness_version="v0.1", pass_rate=0.7, cps=""),
        ],
    )

    dist = load_distribution("proxy", "v0.1", csv_path=csv_path)

    # Both rows count for pass-rate; only the one with a cost counts for cost.
    assert dist.n == 2
    assert dist.pass_rate_mean == pytest.approx(0.6)
    assert dist.cost_per_successful_task_mean == pytest.approx(0.08)
    assert dist.cost_per_successful_task_std == 0.0


def test_all_costs_missing_yields_none_cost(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        [
            _row("proxy_1", split="proxy", harness_version="v0.1", pass_rate=0.5, cps=""),
            _row("proxy_2", split="proxy", harness_version="v0.1", pass_rate=0.7, cps=""),
        ],
    )

    dist = load_distribution("proxy", "v0.1", csv_path=csv_path)

    assert dist.cost_per_successful_task_mean is None
    assert dist.cost_per_successful_task_std is None


def test_no_matching_rows_raises(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        [_row("proxy_1", split="proxy", harness_version="v0.1", pass_rate=0.5, cps=0.08)],
    )

    with pytest.raises(ValueError):
        load_distribution("proxy", "v9.9", csv_path=csv_path)
