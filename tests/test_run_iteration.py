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
from iterator_agent.preflight import PreflightResult
from iterator_agent.run_iteration import proposal_hash, run_iteration
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
        cost_per_task_mean=cost,
        cost_per_task_std=0.016,
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


def _pass_preflight(_root, _target):
    return PreflightResult(True, "ok")


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
        current_version=overrides.pop("current_version", "v0.1"),
        iterations_root=tmp_path / "iterations",
        experiments_dir=tmp_path / "experiments",
        preflight=overrides.pop("preflight", _pass_preflight),
        **overrides,
    )


def test_accepts_when_both_runs_improve(tmp_path: Path) -> None:
    # Arrange: both proxy runs beat the best cost, success holds.
    eval_fn, calls = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071, cost_per_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069, cost_per_task=0.069),
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
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071, cost_per_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069, cost_per_task=0.069),
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
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.090, cost_per_task=0.090),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069, cost_per_task=0.069),
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
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071, cost_per_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.090, cost_per_task=0.090),
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
        RunMetrics(pass_rate=0.9, cost_per_successful_task=0.01, cost_per_task=0.01),
    )
    forbidden = _proposal_json(target="benchmark/adapter.py")

    # Act
    result = _run(tmp_path, complete=_fake_complete(forbidden), eval_fn=eval_fn)

    # Assert
    assert result.accepted is False
    assert calls["n"] == 0  # guard blocks before any eval
    assert not (tmp_path / "benchmark" / "adapter.py").exists()
    assert (tmp_path / "experiments" / "rejected_changes.md").exists()


def test_malformed_harness_version_fails_before_edit_or_eval(tmp_path: Path) -> None:
    eval_fn, calls = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071, cost_per_task=0.071),
    )

    with pytest.raises(ValueError, match="vMAJOR.MINOR"):
        _run(
            tmp_path,
            complete=_fake_complete(_proposal_json()),
            eval_fn=eval_fn,
            current_version="draft",
        )

    assert calls["n"] == 0
    assert (tmp_path / ALLOWED_TARGET).read_text() == ORIGINAL_CONTENT


def test_accept_invokes_commit_hook(tmp_path: Path) -> None:
    # Arrange
    eval_fn, _ = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071, cost_per_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069, cost_per_task=0.069),
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


class _CostingComplete:
    """A two-call editor stub that also reports accumulated search cost."""

    def __init__(self, proposal_json: str, cost_per_call: float = 0.01) -> None:
        self._json = proposal_json
        self._cost = cost_per_call
        self.total_cost_usd = 0.0
        self._n = 0

    def __call__(self, prompt: str) -> str:
        self._n += 1
        self.total_cost_usd += self._cost
        return "diagnosis text" if self._n == 1 else self._json


def test_records_editor_search_cost(tmp_path: Path) -> None:
    # Arrange
    eval_fn, _ = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071, cost_per_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069, cost_per_task=0.069),
    )
    complete = _CostingComplete(_proposal_json(), cost_per_call=0.015)

    # Act
    result = _run(tmp_path, complete=complete, eval_fn=eval_fn)
    record = json.loads(result.record_path.read_text(encoding="utf-8"))

    # Assert: two editor calls x $0.015 = $0.03 of iterator search cost.
    assert result.search_cost_usd == pytest.approx(0.03)
    assert record["iterator_search_cost_usd"] == pytest.approx(0.03)


def test_search_cost_zero_when_completion_untracked(tmp_path: Path) -> None:
    # Arrange: a plain function complete (no total_cost_usd attribute).
    eval_fn, _ = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071, cost_per_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069, cost_per_task=0.069),
    )

    # Act
    result = _run(tmp_path, complete=_fake_complete(_proposal_json()), eval_fn=eval_fn)

    # Assert
    assert result.search_cost_usd == 0.0


def test_reject_does_not_invoke_commit_hook(tmp_path: Path) -> None:
    # Arrange
    eval_fn, _ = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.090, cost_per_task=0.090),
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


def _recording_complete(proposal_json: str, prompts: list):
    """Editor stub that records every prompt it is shown."""
    calls = {"n": 0}

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        calls["n"] += 1
        return "diagnosis text" if calls["n"] == 1 else proposal_json

    return complete


def test_history_is_shown_to_the_editor(tmp_path: Path) -> None:
    from iterator_agent.iteration_log import IterationRecord

    prompts: list = []
    prior = IterationRecord(
        iteration_id="iter_0000", timestamp="t", harness_version="v0.1",
        changed_surface="harness", changed_file="target_agent/harness.py",
        change_summary="PRIOR_REJECTED_IDEA", reason_for_change="x",
        proxy_seeds=(1, 2), proxy_task_success_before_mean=0.5,
        proxy_task_success_before_std=0.0, proxy_task_success_after_mean=0.4,
        proxy_task_success_after_std=0.0, cost_per_successful_task_before=0.08,
        cost_per_successful_task_after=0.09, accepted_or_rejected="rejected",
        reason_accepted_or_rejected="hurt success",
    )
    eval_fn, _ = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071, cost_per_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069, cost_per_task=0.069),
    )

    _run(
        tmp_path,
        complete=_recording_complete(_proposal_json(), prompts),
        eval_fn=eval_fn,
        history=[prior],
    )

    # The diagnose prompt (first editor call) carried the prior rejected change.
    assert "PRIOR_REJECTED_IDEA" in prompts[0]


def test_change_diff_written_for_accepted_iteration(tmp_path: Path) -> None:
    eval_fn, _ = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071, cost_per_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069, cost_per_task=0.069),
    )

    _run(tmp_path, complete=_fake_complete(_proposal_json()), eval_fn=eval_fn)

    diff = (tmp_path / "iterations" / "iter_0001" / "change.diff").read_text()
    assert "IMPROVED PROMPT" in diff  # the new content
    assert "ORIGINAL PROMPT" in diff  # and the prior content it replaced


def test_change_diff_written_even_when_rejected(tmp_path: Path) -> None:
    # A rejected edit is reverted from the tree, but its diff must still be captured.
    eval_fn, _ = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.090, cost_per_task=0.090),
    )

    _run(tmp_path, complete=_fake_complete(_proposal_json()), eval_fn=eval_fn)

    diff = (tmp_path / "iterations" / "iter_0001" / "change.diff").read_text()
    assert "IMPROVED PROMPT" in diff
    # ...while the working tree was rolled back.
    assert (tmp_path / ALLOWED_TARGET).read_text() == ORIGINAL_CONTENT


def test_record_captures_editor_and_agent_models(tmp_path: Path) -> None:
    eval_fn, _ = _eval_counter(
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.071, cost_per_task=0.071),
        RunMetrics(pass_rate=0.667, cost_per_successful_task=0.069, cost_per_task=0.069),
    )

    result = _run(tmp_path, complete=_fake_complete(_proposal_json()), eval_fn=eval_fn)
    record = json.loads(result.record_path.read_text(encoding="utf-8"))

    assert record["editor_model"] == (config.ITERATOR_MODEL or "")
    assert record["agent_model"] == (config.AGENT_MODEL or "")


def test_preflight_failure_rejects_and_reverts_before_eval(tmp_path: Path) -> None:
    # AutoPK port: a structurally broken edit is rejected at $0 — the eval must
    # never run, the file must be reverted, and the version must be restored.
    def eval_fn(seed: int) -> RunMetrics:
        raise AssertionError("eval must not run when preflight fails")

    def failing_preflight(_root, target):
        return PreflightResult(False, f"harness build_messages crashed (edit: {target})")

    result = _run(
        tmp_path,
        complete=_fake_complete(_proposal_json()),
        eval_fn=eval_fn,
        preflight=failing_preflight,
    )

    assert result.accepted is False
    assert "preflight" in result.decision.reason.lower()
    assert result.harness_version == "v0.1"
    assert config.HARNESS_VERSION == "v0.1"
    assert (tmp_path / ALLOWED_TARGET).read_text() == ORIGINAL_CONTENT  # reverted
    assert (tmp_path / "experiments" / "rejected_changes.md").exists()
    # The attempted edit is still auditable.
    diff = (tmp_path / "iterations" / "iter_0001" / "change.diff").read_text()
    assert "IMPROVED PROMPT" in diff


def test_duplicate_rejected_proposal_short_circuits(tmp_path: Path) -> None:
    # AutoPK port (rejected-ticket memory / bounce-loop guard): an edit that is
    # content-identical to a previously rejected one is refused without applying
    # the edit, running preflight, or spending any eval.
    def eval_fn(seed: int) -> RunMetrics:
        raise AssertionError("eval must not run for a duplicate proposal")

    def preflight(_root, _target):
        raise AssertionError("preflight must not run for a duplicate proposal")

    import json as _json
    prior = _json.loads(_proposal_json())
    from iterator_agent.researcher import ProposedEdit
    prior_hash = proposal_hash(
        ProposedEdit(
            target_file=prior["target_file"],
            new_content=prior["new_content"],
            change_summary="different summary text",  # hash covers content, not prose
            reason_for_change="different reason",
        )
    )

    result = _run(
        tmp_path,
        complete=_fake_complete(_proposal_json()),
        eval_fn=eval_fn,
        preflight=preflight,
        rejected_hashes=frozenset({prior_hash}),
    )

    assert result.accepted is False
    assert "duplicate" in result.decision.reason.lower()
    assert (tmp_path / ALLOWED_TARGET).read_text() == ORIGINAL_CONTENT  # never applied
    assert (tmp_path / "experiments" / "rejected_changes.md").exists()


def test_default_eval_seed_runs_fresh_process(monkeypatch, tmp_path: Path) -> None:
    # The candidate edit may touch .py surfaces; an in-process eval would run
    # stale imported modules. The default eval must therefore spawn a fresh
    # interpreter, carry the candidate HARNESS_VERSION in its environment, and
    # rebuild RunMetrics (incl. harness_error_count) from the metrics JSON.
    from types import SimpleNamespace

    from iterator_agent import run_iteration as ri

    captured = {}

    def fake_run(cmd, env=None, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = env
        mpath = Path(cmd[cmd.index("--metrics-json") + 1])
        mpath.write_text(json.dumps([{
            "run_id": "r1", "seed": 3101, "pass_rate": 0.6,
            "cost_per_successful_task": 0.08, "cost_per_task": 0.05,
            "harness_error_count": 1,
        }]), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ri.subprocess, "run", fake_run)
    config.HARNESS_VERSION = "v0.7"

    metrics = ri._default_eval_seed("proxy")(3101)

    assert metrics.pass_rate == 0.6
    assert metrics.cost_per_task == 0.05
    assert metrics.harness_error_count == 1
    assert captured["env"]["HARNESS_VERSION"] == "v0.7"
    assert "scripts.run_train_eval" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--seed-start") + 1] == "3101"


def test_default_eval_seed_raises_on_subprocess_failure(monkeypatch) -> None:
    from types import SimpleNamespace

    from iterator_agent import run_iteration as ri

    monkeypatch.setattr(
        ri.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=3)
    )

    with pytest.raises(RuntimeError, match="exit 3"):
        ri._default_eval_seed("proxy")(3101)
