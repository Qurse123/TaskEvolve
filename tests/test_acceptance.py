"""Unit tests for the iterator's deterministic acceptance rule (experiment.md §20-21).

A candidate is accepted iff BOTH proxy runs improve the objective
(cost_per_successful_task below the current-best mean) AND clear every guardrail
(success floor + optional cost / invalid-action ceilings).

Run from the repo root:
    python -m pytest tests/test_acceptance.py
"""

from __future__ import annotations

import pytest

from iterator_agent.acceptance import (
    Guardrails,
    RunMetrics,
    check_run,
    evaluate_candidate,
    load_guardrails,
    load_protocol,
)
from iterator_agent.baseline import Distribution

# Current-best proxy distribution: mean pass_rate 0.6, mean cost/success 0.09.
BEST = Distribution(
    split="proxy",
    harness_version="v0.1",
    n=2,
    pass_rate_mean=0.6,
    pass_rate_std=0.05,
    cost_per_successful_task_mean=0.09,
    cost_per_successful_task_std=0.01,
    run_ids=("proxy_a", "proxy_b"),
)

# floor = 0.95 * 0.6 = 0.57; no cost/invalid ceilings.
GUARDRAILS = Guardrails(
    task_success_floor_frac_of_best=0.95,
    max_cost_per_successful_task_usd=None,
    max_invalid_action_rate=None,
)


def test_load_guardrails_reads_real_yaml():
    g = load_guardrails()

    assert g.task_success_floor_frac_of_best == 0.95
    assert g.max_cost_per_successful_task_usd is None
    assert g.max_invalid_action_rate is None


def test_run_that_lowers_cost_and_holds_success_passes():
    run = RunMetrics(pass_rate=0.6, cost_per_successful_task=0.08)

    check = check_run(run, BEST, GUARDRAILS)

    assert check.passed is True


def test_run_below_success_floor_fails():
    run = RunMetrics(pass_rate=0.5, cost_per_successful_task=0.08)  # 0.5 < 0.57 floor

    check = check_run(run, BEST, GUARDRAILS)

    assert check.passed is False
    assert "success" in check.reason.lower()


def test_run_without_cost_improvement_fails():
    run = RunMetrics(pass_rate=0.6, cost_per_successful_task=0.10)  # >= 0.09 best

    check = check_run(run, BEST, GUARDRAILS)

    assert check.passed is False
    assert "cost" in check.reason.lower() or "improve" in check.reason.lower()


def test_cost_ceiling_guardrail_blocks_even_when_improved():
    run = RunMetrics(pass_rate=0.6, cost_per_successful_task=0.088)  # < 0.09 best, > 0.085 ceiling
    guardrails = Guardrails(
        task_success_floor_frac_of_best=0.95,
        max_cost_per_successful_task_usd=0.085,
        max_invalid_action_rate=None,
    )

    check = check_run(run, BEST, guardrails)

    assert check.passed is False
    assert "ceiling" in check.reason.lower() or "cost" in check.reason.lower()


def test_both_runs_passing_accepts():
    runs = [
        RunMetrics(pass_rate=0.60, cost_per_successful_task=0.080),
        RunMetrics(pass_rate=0.62, cost_per_successful_task=0.082),
    ]

    decision = evaluate_candidate(runs, BEST, GUARDRAILS)

    assert decision.accepted is True


def test_one_failing_run_rejects():
    runs = [
        RunMetrics(pass_rate=0.60, cost_per_successful_task=0.080),
        RunMetrics(pass_rate=0.50, cost_per_successful_task=0.082),  # below floor
    ]

    decision = evaluate_candidate(runs, BEST, GUARDRAILS)

    assert decision.accepted is False


def test_single_run_cannot_be_accepted():
    runs = [RunMetrics(pass_rate=0.60, cost_per_successful_task=0.080)]

    decision = evaluate_candidate(runs, BEST, GUARDRAILS)

    assert decision.accepted is False
    assert "two" in decision.reason.lower() or "2" in decision.reason


def test_load_protocol_reads_real_yaml():
    assert load_protocol() == 2


def test_load_protocol_rejects_below_hard_minimum(tmp_path):
    policy = tmp_path / "allowed_edits.yaml"
    policy.write_text("protocol:\n  proxy_runs_per_candidate: 1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_protocol(path=policy)


def test_evaluate_respects_required_runs(tmp_path):
    # Two passing runs are not enough when the protocol demands three.
    runs = [
        RunMetrics(pass_rate=0.60, cost_per_successful_task=0.080),
        RunMetrics(pass_rate=0.62, cost_per_successful_task=0.082),
    ]

    decision = evaluate_candidate(runs, BEST, GUARDRAILS, required_runs=3)

    assert decision.accepted is False
    assert "3" in decision.reason
