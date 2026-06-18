"""Tests for iterator_agent.run_iteration — the one-iteration orchestrator.

All LLM / eval / feedback calls are injected, so the full propose -> guard ->
apply -> proxy x2 -> accept/revert -> log cycle runs at $0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iterator_agent.acceptance import Guardrails, RunMetrics
from iterator_agent.baseline import Distribution
from iterator_agent.edit_guard import load_policy
from iterator_agent.feedback import FeedbackSummary
from iterator_agent.run_iteration import run_iteration
from settings import config

ALLOWED_TARGET = "target_agent/prompts/system_prompt.j2"
ORIGINAL_CONTENT = "ORIGINAL PROMPT\n"


def _feedback() -> FeedbackSummary:
    return FeedbackSummary(
        run_id="proxy_20260618_120000",
        split="proxy",
        num_tasks=12,
        num_passed=7,
        num_failed=5,
        failed_tasks=(),
        termination_reason_counts=(("wrong_tool", 3),),
    )


def _best(cost: float = 0.083, pass_rate: float = 0.583) -> Distribution:
    return Distribution(
        split="proxy",
        harness_version="v0.1",
        n=5,
        pass_rate_mean=pass_rate,
        pass_rate_std=0.083,
        cost_per_successful_task_mean=cost,
        cost_per_successful_task_std=0.016,
        run_ids=("proxy_a", "proxy_b"),
    )


def _guardrails() -> Guardrails:
    return Guardrails(
        task_success_floor_frac_of_best=0.95,
        max_cost_per_successful_task_usd=None,
        max_invalid_action_rate=None,
    )


def _proposal_json(target: str = ALLOWED_TARGET) -> str:
    return json.dumps(
        {
            "target_file": target,
            "new_content": "IMPROVED PROMPT\n",
            "change_summary": "Tighten tool-selection guidance.",
            "reason_for_change": "3 of 5 failures called the wrong tool first.",
        }
    )


def _fake_complete(proposal_json: str):
    """A two-call editor stub: diagnose, then propose."""
    calls = {"n": 0}

    def complete(prompt: str) -> str:
        calls["n"] += 1
        return "diagnosis text" if calls["n"] == 1 else proposal_json

    return complete


def _eval_counter(*results: RunMetrics):
    """Return (eval_fn, calls) yielding the given RunMetrics in order."""
    calls = {"n": 0}

    def eval_fn(seed: int) -> RunMetrics:
        idx = calls["n"]
        calls["n"] += 1
        return results[idx]

    return eval_fn, calls


@pytest.fixture(autouse=True)
def _restore_harness_version():
    original = config.HARNESS_VERSION
    yield
    config.HARNESS_VERSION = original


def _repo(tmp_path: Path) -> Path:
    target = tmp_path / ALLOWED_TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ORIGINAL_CONTENT, encoding="utf-8")
    return tmp_path


def _run(tmp_path: Path, *, complete, eval_fn, **overrides):
    return run_iteration(
        iteration_id=overrides.pop("iteration_id", "iter_0001"),
        seeds=(1001, 1002),
        repo_root=_repo(tmp_path),
        policy=load_policy(),
        guardrails=_guardrails(),
        best=_best(),
        feedback=_feedback(),
        complete=complete,
        eval_seed=eval_fn,
        current_version="v0.1",
        iterations_root=tmp_path / "iterations",
        experiments_dir=tmp_path / "experiments",
        **overrides,
    )


def test_accepts_when_both_runs_improve(tmp_path: Path) -> None:
    # Arrange: both proxy runs beat the best cost, success holds.
    eval_fn, calls = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069),
    )

    # Act
    result = _run(tmp_path, complete=_fake_complete(_proposal_json()), eval_fn=eval_fn)

    # Assert
    assert result.accepted is True
    assert calls["n"] == 2  # both seeds run
    assert result.harness_version == "v0.2"  # bumped
    assert (tmp_path / ALLOWED_TARGET).read_text() == "IMPROVED PROMPT\n"  # kept
    assert (tmp_path / "experiments" / "accepted_changes.md").exists()


def test_accept_writes_iteration_record(tmp_path: Path) -> None:
    # Arrange
    eval_fn, _ = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069),
    )

    # Act
    result = _run(tmp_path, complete=_fake_complete(_proposal_json()), eval_fn=eval_fn)
    record = json.loads(result.record_path.read_text(encoding="utf-8"))

    # Assert
    assert record["iteration_id"] == "iter_0001"
    assert record["harness_version"] == "v0.2"
    assert record["changed_file"] == ALLOWED_TARGET
    assert record["proxy_seeds"] == [1001, 1002]
    assert record["accepted_or_rejected"] == "accepted"
    assert record["cost_per_successful_task_before"] == 0.083


def test_rejects_and_reverts_when_first_run_no_improvement(tmp_path: Path) -> None:
    # Arrange: first run costs more than best -> reject without running seed B.
    eval_fn, calls = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.090),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069),
    )

    # Act
    result = _run(tmp_path, complete=_fake_complete(_proposal_json()), eval_fn=eval_fn)

    # Assert
    assert result.accepted is False
    assert calls["n"] == 1  # short-circuit: seed B never runs
    assert result.harness_version == "v0.1"  # restored
    assert (tmp_path / ALLOWED_TARGET).read_text() == ORIGINAL_CONTENT  # reverted
    assert (tmp_path / "experiments" / "rejected_changes.md").exists()
    assert not (tmp_path / "experiments" / "accepted_changes.md").exists()


def test_rejects_when_second_run_fails(tmp_path: Path) -> None:
    # Arrange: A improves, B does not.
    eval_fn, calls = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.090),
    )

    # Act
    result = _run(tmp_path, complete=_fake_complete(_proposal_json()), eval_fn=eval_fn)

    # Assert
    assert result.accepted is False
    assert calls["n"] == 2
    assert (tmp_path / ALLOWED_TARGET).read_text() == ORIGINAL_CONTENT  # reverted


def test_rejects_forbidden_target_without_running_eval(tmp_path: Path) -> None:
    # Arrange: editor proposes a forbidden file.
    eval_fn, calls = _eval_counter(
        RunMetrics(pass_rate=0.9, cost_per_successful_task=0.01),
    )
    forbidden = _proposal_json(target="benchmark/adapter.py")

    # Act
    result = _run(tmp_path, complete=_fake_complete(forbidden), eval_fn=eval_fn)

    # Assert
    assert result.accepted is False
    assert calls["n"] == 0  # guard blocks before any eval
    assert not (tmp_path / "benchmark" / "adapter.py").exists()
    assert (tmp_path / "experiments" / "rejected_changes.md").exists()


def test_accept_invokes_commit_hook(tmp_path: Path) -> None:
    # Arrange
    eval_fn, _ = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069),
    )
    committed = []

    # Act
    _run(
        tmp_path,
        complete=_fake_complete(_proposal_json()),
        eval_fn=eval_fn,
        commit=lambda res: committed.append(res.iteration_id),
    )

    # Assert
    assert committed == ["iter_0001"]


def test_reject_does_not_invoke_commit_hook(tmp_path: Path) -> None:
    # Arrange
    eval_fn, _ = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.090),
    )
    committed = []

    # Act
    _run(
        tmp_path,
        complete=_fake_complete(_proposal_json()),
        eval_fn=eval_fn,
        commit=lambda res: committed.append(res.iteration_id),
    )

    # Assert
    assert committed == []
