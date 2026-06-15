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
        seed=2001, timestamp="2026-06-15T00:00:00+00:00",
    )


# --- _is_transient -----------------------------------------------------------

def test_is_transient_true_for_ratelimit():
    assert rte._is_transient(RateLimitError("429"))


def test_is_transient_false_for_plain_error():
    assert not rte._is_transient(ValueError("bad input"))


# --- _run_task_with_retries --------------------------------------------------

def test_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_run_eval(task_id, *, split, domain, seed):
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
    assert "RateLimitError" in r.termination_reason


# --- _run_one_repeat ---------------------------------------------------------

def test_run_one_repeat_records_failure_instead_of_aborting(monkeypatch):
    def fake_run_eval(task_id, *, split, domain, seed):
        if task_id == "bad":
            raise RateLimitError("429")
        return _ok_result(task_id)

    logged: list = []
    monkeypatch.setattr(rte, "run_eval", fake_run_eval)
    monkeypatch.setattr(rte.time, "sleep", lambda _s: None)
    monkeypatch.setattr(rte, "start_run", lambda split: SimpleNamespace(run_id="run_test"))
    monkeypatch.setattr(rte, "log_task", lambda run, result: logged.append(result))
    monkeypatch.setattr(rte, "finalize_run", lambda run, results: None)

    metrics = rte._run_one_repeat("validation", ["good", "bad"], domain="retail", seed=2001)

    assert len(logged) == 2  # both tasks logged; the run did not abort
    assert metrics.pass_rate == pytest.approx(0.5)  # one pass, one recorded failure
