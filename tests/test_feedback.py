"""Unit tests for the iterator's feedback reader — turns the most recent proxy
run folder into a structured failure summary for the editor (proxy-only).

Run from the repo root:
    python -m pytest tests/test_feedback.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmark.adapter import EvalResult
from iterator_agent.feedback import (
    FeedbackSummary,
    build_feedback,
    find_latest_run_dir,
    find_latest_run_dir_for_version,
    summarize_run,
)
from settings import config
from results.logger import finalize_run, log_task, start_run


def _result(task_id, *, passed, reward, cost, term="user_stop", seed=1, split="proxy",
            turns=5, tool_calls=2):
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
        turn_count=turns,
        tool_call_count=tool_calls,
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


def test_summary_is_cost_centric_across_all_tasks(tmp_path):
    run_id = _make_run(
        tmp_path,
        "proxy",
        _at(1),
        [
            _result("pass_1", passed=True, reward=1.0, cost=0.10, turns=4, tool_calls=1),
            _result("pass_2", passed=True, reward=1.0, cost=0.12, turns=6, tool_calls=3),
            _result("fail_1", passed=False, reward=0.0, cost=0.20, term="max_steps"),
        ],
    )
    summary = summarize_run(tmp_path / run_id)

    # The cost table covers ALL tasks (not just failures), with per-task cost levers.
    assert {t.task_id for t in summary.tasks} == {"pass_1", "pass_2", "fail_1"}
    pass_1 = next(t for t in summary.tasks if t.task_id == "pass_1")
    assert (pass_1.turn_count, pass_1.tool_call_count) == (4, 1)
    # Objective signal: total cost and cost-per-successful-task.
    assert summary.total_cost_usd == pytest.approx(0.42)
    assert summary.cost_per_successful_task == pytest.approx(0.21)  # 0.42 / 2 passed


def test_expensive_digests_render_costliest_transcripts(tmp_path):
    run_id = _make_run(
        tmp_path,
        "proxy",
        _at(1),
        [
            _result("cheap", passed=True, reward=1.0, cost=0.05),
            _result("pricey", passed=True, reward=1.0, cost=0.50),
        ],
    )
    run_dir = tmp_path / run_id
    # Persist a transcript for the costliest task (as run_eval would).
    (run_dir / "task_pricey_messages.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": None,
                     "tool_calls": [{"name": "get_order_details"}]},
                    {"role": "tool", "content": "order #123 found"},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_run(run_dir)

    assert len(summary.expensive_digests) == 1
    task_id, digest = summary.expensive_digests[0]
    assert task_id == "pricey"  # the costliest task is digested first
    assert "get_order_details" in digest
    assert "order #123 found" in digest


def _write_transcript(run_dir: Path, task_id: str, tool_names) -> None:
    (run_dir / f"task_{task_id}_messages.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": None,
                     "tool_calls": [{"name": name}]}
                    for name in tool_names
                ]
            }
        ),
        encoding="utf-8",
    )


def test_summary_reports_aggregate_cost_levers(tmp_path):
    run_id = _make_run(
        tmp_path, "proxy", _at(1),
        [
            _result("t1", passed=True, reward=1.0, cost=0.1, turns=4, tool_calls=2),
            _result("t2", passed=False, reward=0.0, cost=0.2, turns=8, tool_calls=6, term="max_steps"),
        ],
    )
    run_dir = tmp_path / run_id
    _write_transcript(run_dir, "t1", ["get_order_details"])
    _write_transcript(run_dir, "t2", ["get_order_details", "get_user_details", "get_order_details"])

    summary = summarize_run(run_dir)

    # Aggregate turn/tool means across all tasks.
    assert summary.mean_turns == pytest.approx(6.0)
    assert summary.mean_tool_calls == pytest.approx(4.0)
    # Tool histogram: most-used first, ties broken by name.
    assert summary.tool_usage_counts == (("get_order_details", 3), ("get_user_details", 1))


def test_expensive_digests_include_representative_failures(tmp_path):
    # One pricey PASS and one cheaper FAIL: the failing task must still get a digest.
    run_id = _make_run(
        tmp_path, "proxy", _at(1),
        [
            _result("pricey_pass", passed=True, reward=1.0, cost=0.50),
            _result("cheap_fail", passed=False, reward=0.0, cost=0.05, term="max_steps"),
        ],
    )
    run_dir = tmp_path / run_id
    _write_transcript(run_dir, "pricey_pass", ["get_order_details"])
    _write_transcript(run_dir, "cheap_fail", ["get_user_details"])

    summary = summarize_run(run_dir)

    digested = {task_id for task_id, _ in summary.expensive_digests}
    assert "cheap_fail" in digested  # failures are surfaced even when not costliest


def test_load_run_tasks_ignores_messages_files(tmp_path):
    run_id = _make_run(
        tmp_path, "proxy", _at(1),
        [_result("t1", passed=True, reward=1.0, cost=0.1)],
    )
    run_dir = tmp_path / run_id
    # A transcript file sits beside the verdict; it must NOT be parsed as a task.
    (run_dir / "task_t1_messages.json").write_text(
        json.dumps({"messages": []}), encoding="utf-8"
    )

    summary = summarize_run(run_dir)

    assert summary.num_tasks == 1


def test_find_latest_run_dir_for_version_picks_by_version(tmp_path, monkeypatch):
    # An older v0.1 run and a newer v0.2 run; asking for v0.1 must skip the newer one.
    monkeypatch.setattr(config, "HARNESS_VERSION", "v0.1")
    old = _make_run(tmp_path, "proxy", _at(1), [_result("t", passed=True, reward=1.0, cost=0.1)])
    monkeypatch.setattr(config, "HARNESS_VERSION", "v0.2")
    _make_run(tmp_path, "proxy", _at(2), [_result("t", passed=True, reward=1.0, cost=0.1)])

    run_dir = find_latest_run_dir_for_version(
        "proxy", "v0.1", logs_root=tmp_path, csv_path=tmp_path / "results.csv"
    )

    assert run_dir.name == old


def test_build_feedback_diagnoses_current_best_not_latest(tmp_path, monkeypatch):
    # The current-best is v0.1; a later v0.2 candidate run must NOT be the feedback.
    monkeypatch.setattr(config, "HARNESS_VERSION", "v0.1")
    best_run = _make_run(tmp_path, "proxy", _at(1), [_result("best_task", passed=True, reward=1.0, cost=0.1)])
    monkeypatch.setattr(config, "HARNESS_VERSION", "v0.2")
    _make_run(tmp_path, "proxy", _at(2), [_result("candidate_task", passed=False, reward=0.0, cost=0.2)])

    summary = build_feedback(
        "proxy", harness_version="v0.1", logs_root=tmp_path, csv_path=tmp_path / "results.csv"
    )

    assert summary.run_id == best_run
    assert {t.task_id for t in summary.tasks} == {"best_task"}


def test_build_feedback_falls_back_when_version_has_no_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HARNESS_VERSION", "v0.1")
    _make_run(tmp_path, "proxy", _at(1), [_result("t", passed=True, reward=1.0, cost=0.1)])

    # No run exists for v9.9 -> fall back to the latest proxy run rather than crash.
    summary = build_feedback(
        "proxy", harness_version="v9.9", logs_root=tmp_path, csv_path=tmp_path / "results.csv"
    )

    assert summary.num_tasks == 1
