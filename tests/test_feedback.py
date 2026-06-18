"""Unit tests for the iterator's feedback reader — turns the most recent proxy
run folder into a structured failure summary for the editor (proxy-only).

Run from the repo root:
    python -m pytest tests/test_feedback.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmark.adapter import EvalResult
from iterator_agent.feedback import (
    FeedbackSummary,
    build_feedback,
    find_latest_run_dir,
    summarize_run,
)
from results.logger import finalize_run, log_task, start_run


def _result(task_id, *, passed, reward, cost, term="user_stop", seed=1, split="proxy"):
    return EvalResult(
        task_id=task_id,
        domain="retail",
        split=split,
        agent_model="gpt-4.1",
        reward=reward,
        passed=passed,
        agent_cost=cost,
        termination_reason=term,
        seed=seed,
        timestamp="2026-01-01T00:00:00+00:00",
    )


def _make_run(logs_root: Path, split: str, when: datetime, results) -> str:
    run = start_run(split, logs_root=logs_root, now=when)
    for result in results:
        log_task(run, result)
    finalize_run(run, results, results_csv=logs_root / "results.csv")
    return run.run_id


def _at(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=timezone.utc)


def test_find_latest_run_dir_picks_newest_proxy(tmp_path):
    _make_run(tmp_path, "proxy", _at(1), [_result("t1", passed=True, reward=1.0, cost=0.1)])
    _make_run(tmp_path, "proxy", _at(2), [_result("t1", passed=True, reward=1.0, cost=0.1)])
    # A later validation run must never be chosen when asking for proxy.
    _make_run(tmp_path, "validation", _at(3), [_result("t1", passed=True, reward=1.0, cost=0.1, split="validation")])

    latest = find_latest_run_dir("proxy", logs_root=tmp_path)

    assert latest.name == "proxy_20260102_000000"


def test_find_latest_run_dir_raises_when_no_matching_split(tmp_path):
    _make_run(tmp_path, "validation", _at(1), [_result("t1", passed=True, reward=1.0, cost=0.1, split="validation")])

    with pytest.raises(ValueError):
        find_latest_run_dir("proxy", logs_root=tmp_path)


def test_summarize_run_counts_and_collects_failures(tmp_path):
    run_id = _make_run(
        tmp_path,
        "proxy",
        _at(1),
        [
            _result("pass_1", passed=True, reward=1.0, cost=0.10),
            _result("pass_2", passed=True, reward=0.8, cost=0.12),
            _result("fail_1", passed=False, reward=0.0, cost=0.20, term="max_steps"),
        ],
    )
    run_dir = tmp_path / run_id

    summary = summarize_run(run_dir)

    assert isinstance(summary, FeedbackSummary)
    assert summary.num_tasks == 3
    assert summary.num_passed == 2
    assert summary.num_failed == 1
    assert [t.task_id for t in summary.failed_tasks] == ["fail_1"]
    assert ("max_steps", 1) in summary.termination_reason_counts


def test_build_feedback_uses_latest_proxy_run(tmp_path):
    _make_run(tmp_path, "proxy", _at(1), [_result("old", passed=False, reward=0.0, cost=0.2, term="max_errors")])
    _make_run(tmp_path, "proxy", _at(2), [_result("new", passed=False, reward=0.0, cost=0.2, term="max_steps")])

    summary = build_feedback("proxy", logs_root=tmp_path)

    assert summary.run_id == "proxy_20260102_000000"
    assert [t.task_id for t in summary.failed_tasks] == ["new"]


def test_summary_ignores_run_summary_json(tmp_path):
    run_id = _make_run(
        tmp_path,
        "proxy",
        _at(1),
        [_result("t1", passed=True, reward=1.0, cost=0.1)],
    )
    run_dir = tmp_path / run_id

    summary = summarize_run(run_dir)

    # run_summary.json exists in the folder but must not be parsed as a task.
    assert (run_dir / "run_summary.json").exists()
    assert summary.num_tasks == 1
