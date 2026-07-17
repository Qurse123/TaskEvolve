"""Unit tests for run_train_eval's per-task failure hardening.

Run from the repo root:
    python -m pytest tests/test_run_train_eval.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark.adapter import EvalResult
from scripts import run_train_eval as rte


class RateLimitError(Exception):
    """Stand-in whose class name matches a TRANSIENT_ERROR_MARKER."""


def _ok_result(task_id: str = "t1") -> EvalResult:
    return EvalResult(
        task_id=task_id, domain="retail", split="validation", agent_model="gpt-4.1",
        reward=1.0, passed=True, agent_cost=0.05, termination_reason="done",
        seed=2001, turn_count=4, tool_call_count=2,
        timestamp="2026-06-15T00:00:00+00:00",
    )


# --- _is_transient -----------------------------------------------------------

def test_is_transient_true_for_ratelimit():
    assert rte._is_transient(RateLimitError("429"))


def test_is_transient_false_for_plain_error():
    assert not rte._is_transient(ValueError("bad input"))


# --- _run_task_with_retries --------------------------------------------------

def test_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_run_eval(task_id, *, split, domain, seed, transcript_path=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("429")
        return _ok_result(task_id)

    monkeypatch.setattr(rte, "run_eval", fake_run_eval)
    monkeypatch.setattr(rte.time, "sleep", lambda _s: None)
    result = rte._run_task_with_retries("t1", split="validation", domain="retail", seed=2001)
    assert result.passed
    assert calls["n"] == 3


def test_reraises_non_transient_immediately(monkeypatch):
    calls = {"n": 0}

    def fake_run_eval(*_a, **_k):
        calls["n"] += 1
        raise ValueError("not retryable")

    monkeypatch.setattr(rte, "run_eval", fake_run_eval)
    monkeypatch.setattr(rte.time, "sleep", lambda _s: None)
    with pytest.raises(ValueError):
        rte._run_task_with_retries("t1", split="validation", domain="retail", seed=2001)
    assert calls["n"] == 1  # no retries for a non-transient error


def test_exhausts_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def fake_run_eval(*_a, **_k):
        calls["n"] += 1
        raise RateLimitError("429")

    monkeypatch.setattr(rte, "run_eval", fake_run_eval)
    monkeypatch.setattr(rte.time, "sleep", lambda _s: None)
    with pytest.raises(RateLimitError):
        rte._run_task_with_retries("t1", split="validation", domain="retail", seed=2001)
    assert calls["n"] == rte.MAX_TASK_RETRIES + 1  # initial try + N retries


# --- _failed_result ----------------------------------------------------------

def test_failed_result_counts_as_failure():
    r = rte._failed_result("t1", split="validation", domain="retail", seed=2001,
                           error=RateLimitError("429"))
    assert r.passed is False
    assert r.reward == 0.0
    assert r.agent_cost is None
    assert r.turn_count == 0
    assert r.tool_call_count == 0
    assert "RateLimitError" in r.termination_reason


# --- _run_one_repeat ---------------------------------------------------------

def test_run_one_repeat_records_failure_instead_of_aborting(monkeypatch, tmp_path):
    seen_transcripts: list = []

    def fake_run_eval(task_id, *, split, domain, seed, transcript_path=None):
        seen_transcripts.append(transcript_path)
        if task_id == "bad":
            raise RateLimitError("429")
        return _ok_result(task_id)

    logged: list = []
    monkeypatch.setattr(rte, "run_eval", fake_run_eval)
    monkeypatch.setattr(rte.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        rte, "start_run", lambda split: SimpleNamespace(run_id="run_test", run_dir=tmp_path)
    )
    monkeypatch.setattr(rte, "log_task", lambda run, result: logged.append(result))
    monkeypatch.setattr(rte, "finalize_run", lambda run, results: None)

    metrics = rte._run_one_repeat("validation", ["good", "bad"], domain="retail", seed=2001)

    assert len(logged) == 2  # both tasks logged; the run did not abort
    assert metrics.pass_rate == pytest.approx(0.5)  # one pass, one recorded failure
    # Each task's transcript path is threaded through to run_eval, beside its verdict.
    # ("bad" repeats because it is retried; assert the distinct set of paths.)
    assert set(seen_transcripts) == {
        tmp_path / "task_good_messages.json",
        tmp_path / "task_bad_messages.json",
    }


def test_run_one_repeat_counts_harness_errors(monkeypatch, tmp_path):
    # A permanently failing task is recorded with a harness_error termination;
    # the repeat's metrics must surface that count so the acceptance rule can
    # treat the run as invalid (never accept what cannot be verified).
    def fake_run_eval(task_id, *, split, domain, seed, transcript_path=None):
        if task_id == "bad":
            raise ValueError("broken harness edit")
        return _ok_result(task_id)

    monkeypatch.setattr(rte, "run_eval", fake_run_eval)
    monkeypatch.setattr(rte.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        rte, "start_run", lambda split: SimpleNamespace(run_id="run_test", run_dir=tmp_path)
    )
    monkeypatch.setattr(rte, "log_task", lambda run, result: None)
    monkeypatch.setattr(rte, "finalize_run", lambda run, results: None)

    metrics = rte._run_one_repeat("validation", ["good", "bad"], domain="retail", seed=2001)

    assert metrics.harness_error_count == 1


def test_write_metrics_json_round_trips(tmp_path):
    # The subprocess eval path reads this file to rebuild per-repeat metrics.
    metrics = [
        rte.RepeatMetrics(
            run_id="r1", seed=3101, pass_rate=0.583,
            cost_per_successful_task=0.083, cost_per_task=0.048,
            harness_error_count=0,
        ),
        rte.RepeatMetrics(
            run_id="r2", seed=3102, pass_rate=0.5,
            cost_per_successful_task=None, cost_per_task=None,
            harness_error_count=2,
        ),
    ]
    out = tmp_path / "metrics.json"

    rte.write_metrics_json(metrics, out)

    import json
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows[0] == {
        "run_id": "r1", "seed": 3101, "pass_rate": 0.583,
        "cost_per_successful_task": 0.083, "cost_per_task": 0.048,
        "harness_error_count": 0,
    }
    assert rows[1]["cost_per_task"] is None
    assert rows[1]["harness_error_count"] == 2


def test_cli_metrics_json_flag(monkeypatch, tmp_path):
    # --metrics-json writes the repeat metrics where the flag points.
    out = tmp_path / "m.json"
    recorded = {}

    def fake_run_repeats(split, *, seed_start, repeats, domain=None):
        recorded["args"] = (split, seed_start, repeats)
        return [
            rte.RepeatMetrics(
                run_id="r1", seed=seed_start, pass_rate=1.0,
                cost_per_successful_task=0.01, cost_per_task=0.01,
                harness_error_count=0,
            )
        ]

    monkeypatch.setattr(rte, "run_repeats", fake_run_repeats)

    rte.main([
        "--split", "proxy", "--seed-start", "3101", "--repeats", "1",
        "--metrics-json", str(out),
    ])

    import json
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert recorded["args"] == ("proxy", 3101, 1)
    assert rows[0]["run_id"] == "r1"
