"""Tests for scripts.run_iterator — the fixed-budget optimization driver.

The driver loops iterator_agent.run_iteration under a pre-declared budget
(iteration count and/or search-cost cap), updating the current-best proxy
distribution after each accepted change. All LLM / eval calls are injected, so
the whole multi-iteration run executes at $0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iterator_agent.acceptance import Guardrails, RunMetrics
from iterator_agent.baseline import Distribution
from iterator_agent.edit_guard import load_policy
from scripts.run_iterator import run_iterator
from settings import config

ALLOWED_TARGET = "target_agent/prompts/system_prompt.j2"


@pytest.fixture(autouse=True)
def _restore_harness_version():
    original = config.HARNESS_VERSION
    yield
    config.HARNESS_VERSION = original


def _best() -> Distribution:
    return Distribution(
        split="proxy", harness_version="v0.1", n=5,
        pass_rate_mean=0.583, pass_rate_std=0.083,
        cost_per_successful_task_mean=0.083, cost_per_successful_task_std=0.016,
        run_ids=(),
        cost_per_task_mean=0.083, cost_per_task_std=0.016,
    )


def _guardrails() -> Guardrails:
    return Guardrails(
        task_success_floor_frac_of_best=0.95,
        max_cost_per_successful_task_usd=None,
        max_invalid_action_rate=None,
    )


def _proposal_json() -> str:
    return json.dumps(
        {
            "target_file": ALLOWED_TARGET,
            "new_content": "IMPROVED PROMPT\n",
            "change_summary": "Tighten guidance.",
            "reason_for_change": "wrong tool first.",
        }
    )


def _ticket(index: int = 1):
    """A hypothesis ticket pinned to the one editable file the tests rewrite."""
    from iterator_agent.hypothesis import Ticket

    return Ticket(
        ticket_id=f"t{index}",
        hypothesis="trim verbose prompt",
        target_surface=ALLOWED_TARGET,
        rationale="input tokens",
        expected_effect="lower cost",
        priority=index,
    )


def _fake_backlog(*, cost: float = 0.0, size: int = 1, history_sink=None):
    """Backlog provider stub: returns `size` tickets + `cost`; records history seen."""

    def provide(current_version: str, history):
        if history_sink is not None:
            history_sink.append(list(history))
        return tuple(_ticket(i + 1) for i in range(size)), cost

    return provide


def _fake_complete():
    """Editor stub: with a ticket the editor makes one (propose) call — return JSON."""

    def complete(prompt: str) -> str:
        return _proposal_json()

    return complete


class _CostingComplete:
    """Editor stub that also accumulates search cost across all its calls."""

    def __init__(self, cost_per_call: float) -> None:
        self._cost = cost_per_call
        self.total_cost_usd = 0.0
        self._n = 0

    def __call__(self, prompt: str) -> str:
        self._n += 1
        self.total_cost_usd += self._cost
        return _proposal_json()


def _eval_seq(costs, pass_rate: float = 0.667):
    """Eval stub yielding the given costs in order; records the seeds it saw."""
    seq = list(costs)
    seen = []

    def eval_fn(seed: int) -> RunMetrics:
        seen.append(seed)
        cost = seq.pop(0)
        return RunMetrics(
            pass_rate=pass_rate, cost_per_successful_task=cost, cost_per_task=cost
        )

    return eval_fn, seen


def _seed_proxy_log(logs_root: Path) -> Path:
    """Write a minimal proxy run folder so build_feedback() has something to read."""
    run_dir = logs_root / "proxy_20260618_000000"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_001.json").write_text(
        json.dumps(
            {
                "task_id": "t1", "reward": 0.0, "passed": False,
                "termination_reason": "max_steps", "cost_usd": 0.1, "seed": 1,
            }
        ),
        encoding="utf-8",
    )
    return logs_root


def _repo(tmp_path: Path) -> Path:
    target = tmp_path / ALLOWED_TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ORIGINAL\n", encoding="utf-8")
    return tmp_path


def _run_iterator(tmp_path: Path, *, complete, eval_fn, **kwargs):
    return run_iterator(
        max_iterations=kwargs.pop("max_iterations", 3),
        seed_start=kwargs.pop("seed_start", 1001),
        split="proxy",
        repo_root=_repo(tmp_path),
        policy=load_policy(),
        guardrails=_guardrails(),
        initial_best=_best(),
        initial_version="v0.1",
        complete=complete,
        eval_seed=eval_fn,
        generate_backlog=kwargs.pop("generate_backlog", _fake_backlog()),
        iterations_root=tmp_path / "iterations",
        experiments_dir=tmp_path / "experiments",
        logs_root=_seed_proxy_log(tmp_path / "logs"),
        **kwargs,
    )


def test_stops_at_max_iterations(tmp_path: Path) -> None:
    # Arrange: every run costs more than best -> always reject.
    eval_fn, seen = _eval_seq([0.09, 0.09, 0.09])

    # Act
    results = _run_iterator(tmp_path, complete=_fake_complete(), eval_fn=eval_fn, max_iterations=3)

    # Assert: 3 iterations, all rejected; version never bumped; seed B skipped each time.
    assert len(results) == 3
    assert all(not r.accepted for r in results)
    assert results[-1].harness_version == "v0.1"
    assert seen == [1001, 1003, 1005]


def test_accept_updates_current_best_for_next_iteration(tmp_path: Path) -> None:
    # Arrange: iter 1 beats 0.083, iter 2 must then beat the new best (0.07).
    eval_fn, seen = _eval_seq([0.07, 0.07, 0.06, 0.06])

    # Act
    results = _run_iterator(tmp_path, complete=_fake_complete(), eval_fn=eval_fn, max_iterations=2)

    # Assert
    assert [r.accepted for r in results] == [True, True]
    assert results[0].harness_version == "v0.2"
    assert results[1].harness_version == "v0.3"
    # Iteration 2 was compared against iteration 1's accepted distribution, not Arm A.
    assert results[1].record.cost_per_successful_task_before == pytest.approx(0.07)
    assert seen == [1001, 1002, 1003, 1004]


def test_reject_keeps_previous_best(tmp_path: Path) -> None:
    # Arrange: iter 1 accepts (0.07), iter 2 fails to beat 0.07, iter 3 must still beat 0.07.
    eval_fn, _ = _eval_seq([0.07, 0.07, 0.09, 0.065, 0.065])

    # Act
    results = _run_iterator(tmp_path, complete=_fake_complete(), eval_fn=eval_fn, max_iterations=3)

    # Assert: accept, reject, accept; the rejected iter did not move the baseline.
    assert [r.accepted for r in results] == [True, False, True]
    assert results[2].record.cost_per_successful_task_before == pytest.approx(0.07)
    assert results[2].harness_version == "v0.3"


def test_search_cost_budget_stops_loop_early(tmp_path: Path) -> None:
    # Arrange: $0.02 search cost per iteration (one editor call x $0.02); cap at $0.03.
    eval_fn, _ = _eval_seq([0.09, 0.09, 0.09, 0.09])
    complete = _CostingComplete(cost_per_call=0.02)

    # Act
    results = _run_iterator(
        tmp_path, complete=complete, eval_fn=eval_fn,
        max_iterations=4, max_search_cost_usd=0.03,
    )

    # Assert: stops after the spend crosses the cap, before exhausting max_iterations.
    assert len(results) == 2
    assert sum(r.search_cost_usd for r in results) == pytest.approx(0.04)


def test_history_accumulates_and_reaches_backlog(tmp_path: Path) -> None:
    # Both iterations reject; iter 2's backlog generation must see iter 1's change.
    histories: list = []
    eval_fn, _ = _eval_seq([0.09, 0.09])  # > best 0.083 -> reject each iteration

    _run_iterator(
        tmp_path, complete=_fake_complete(), eval_fn=eval_fn, max_iterations=2,
        generate_backlog=_fake_backlog(history_sink=histories),
    )

    # Iteration 1 had no prior history; iteration 2's backlog was given iter 1's record.
    assert len(histories) == 2
    assert histories[0] == []
    assert histories[1] and histories[1][0].change_summary == "Tighten guidance."


def test_main_rejects_nonpositive_iterations(tmp_path: Path) -> None:
    from scripts.run_iterator import main

    with pytest.raises(SystemExit):
        main(["--max-iterations", "0", "--seed-start", "1001"])


def _fake_clock(step: float):
    """Monotonic-clock stub advancing by `step` seconds each call (first call = 0)."""
    state = {"v": -step}

    def clock() -> float:
        state["v"] += step
        return state["v"]

    return clock


def test_stops_at_max_minutes_deadline(tmp_path: Path) -> None:
    # Arrange: clock advances 50s/check; a 2-minute (120s) budget crosses on the
    # 3rd deadline check, so only 2 iterations run despite a higher iteration cap.
    eval_fn, _ = _eval_seq([0.09, 0.09, 0.09, 0.09, 0.09])  # always reject

    # Act
    results = _run_iterator(
        tmp_path, complete=_fake_complete(), eval_fn=eval_fn,
        max_iterations=5, max_minutes=2.0, clock=_fake_clock(50),
    )

    # Assert: the wall-clock deadline stopped the loop before the iteration cap.
    assert len(results) == 2


def test_iteration_cap_still_bounds_when_time_budget_generous(tmp_path: Path) -> None:
    # Arrange: a huge time budget never trips; the iteration cap is the binding bound.
    eval_fn, _ = _eval_seq([0.09, 0.09, 0.09])

    # Act
    results = _run_iterator(
        tmp_path, complete=_fake_complete(), eval_fn=eval_fn,
        max_iterations=3, max_minutes=10_000.0, clock=_fake_clock(1),
    )

    # Assert
    assert len(results) == 3
