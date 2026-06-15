"""Unit tests for scripts.plot_results pure helpers + render smoke tests.

Run from the repo root so the flat package layout is importable:
    python -m pytest tests/test_plot_results.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.plot_results import (
    RunPoint,
    _mean_std,
    _padded_range,
    _row_to_point,
    choose_mode,
    load_points,
    plot_frontier,
    plot_per_metric,
)


def _write_csv(path: Path, rows: list[dict]) -> Path:
    """Write rows to a results-style CSV and return the path."""
    fields = [
        "split", "harness_version", "num_tasks", "pass_rate",
        "total_cost_usd", "cost_per_successful_task",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(split="proxy", pass_rate="0.5", cost="0.6", cps="0.1", num_tasks="12") -> dict:
    return {
        "split": split, "harness_version": "v0.1", "num_tasks": num_tasks,
        "pass_rate": pass_rate, "total_cost_usd": cost, "cost_per_successful_task": cps,
    }


# --- _row_to_point -----------------------------------------------------------

def test_row_to_point_computes_cost_per_task():
    point = _row_to_point(_row(cost="0.6", num_tasks="12"))
    assert point is not None
    assert point.cost_per_task == pytest.approx(0.05)
    assert point.pass_rate == pytest.approx(0.5)
    assert point.arm == "proxy / v0.1"


def test_row_to_point_returns_none_when_num_tasks_zero():
    assert _row_to_point(_row(num_tasks="0")) is None


def test_row_to_point_returns_none_when_cost_missing():
    assert _row_to_point(_row(cost="")) is None


# --- _mean_std ---------------------------------------------------------------

def test_mean_std_returns_std_for_multiple_values():
    mean, std = _mean_std([0.5, 0.6667, 0.5833, 0.5, 0.6667])
    assert mean == pytest.approx(0.58334, abs=1e-4)
    assert std == pytest.approx(0.0833, abs=1e-3)


def test_mean_std_std_is_none_for_single_value():
    mean, std = _mean_std([0.42])
    assert mean == pytest.approx(0.42)
    assert std is None


# --- _padded_range -----------------------------------------------------------

def test_padded_range_pads_by_fraction_of_range():
    low, high = _padded_range([0.0, 1.0], floor=0.01)
    assert low == pytest.approx(-0.1)
    assert high == pytest.approx(1.1)


def test_padded_range_uses_floor_when_degenerate():
    low, high = _padded_range([0.05, 0.05], floor=0.005)
    assert low == pytest.approx(0.045)
    assert high == pytest.approx(0.055)


def test_padded_range_respects_clamps():
    # range 0.9, margin 0.09 -> raw bounds -0.04 / 1.04, both clamped.
    low, high = _padded_range([0.05, 0.95], floor=0.01, clamp_low=0.0, clamp_high=1.0)
    assert low == 0.0
    assert high == 1.0


# --- choose_mode -------------------------------------------------------------

def _points(*arms: str) -> list[RunPoint]:
    return [RunPoint(arm=a, cost_per_task=0.05, pass_rate=0.5, cost_per_successful_task=0.1)
            for a in arms]


def test_choose_mode_single_arm_defaults_to_per_metric():
    assert choose_mode(_points("a", "a"), frontier=False, per_metric=False) == "per_metric"


def test_choose_mode_multiple_arms_defaults_to_frontier():
    assert choose_mode(_points("a", "b"), frontier=False, per_metric=False) == "frontier"


def test_choose_mode_explicit_flags_win():
    assert choose_mode(_points("a"), frontier=True, per_metric=False) == "frontier"
    assert choose_mode(_points("a", "b"), frontier=False, per_metric=True) == "per_metric"


def test_choose_mode_rejects_both_flags():
    with pytest.raises(ValueError):
        choose_mode(_points("a"), frontier=True, per_metric=True)


# --- load_points -------------------------------------------------------------

def test_load_points_filters_by_split(tmp_path):
    csv_path = _write_csv(tmp_path / "results.csv", [_row(split="proxy"), _row(split="smoke")])
    points = load_points(csv_path, split="proxy")
    assert len(points) == 1
    assert points[0].arm == "proxy / v0.1"


def test_load_points_raises_when_csv_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_points(tmp_path / "nope.csv")


# --- render smoke tests ------------------------------------------------------

def test_plot_per_metric_writes_file(tmp_path):
    points = [_row_to_point(_row(pass_rate=str(p))) for p in (0.5, 0.6667, 0.5833)]
    out = plot_per_metric([p for p in points if p], out_path=tmp_path / "pm.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_frontier_writes_file(tmp_path):
    points = [_row_to_point(_row(pass_rate=str(p), cost=str(c)))
              for p, c in ((0.5, 0.6), (0.6667, 0.5))]
    out = plot_frontier([p for p in points if p], out_path=tmp_path / "fr.png")
    assert out.exists() and out.stat().st_size > 0
