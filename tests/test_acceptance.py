"""Unit tests for the iterator's deterministic acceptance rule (experiment.md §20-21).

A candidate is accepted iff BOTH proxy runs improve the objective
(cost_per_successful_task below the current-best mean) AND clear every guardrail
(success floor + optional cost / invalid-action ceilings).

Run from the repo root:
    python -m pytest tests/test_acceptance.py
"""

from __future__ import annotations

from iterator_agent.acceptance import (
    Guardrails,
    RunMetrics,
    check_run,
    evaluate_candidate,
    load_guardrails,
)
from iterator_agent.baseline import Distribution

# Current-best proxy distribution: mean pass_rate 0.6; per-task cost (the
# objective) mirrors cost/success at 0.09 so scenario numbers stay readable.
BEST = Distribution(
    split="proxy",
    harness_version="v0.1",
    n=2,
    pass_rate_mean=0.6,
    pass_rate_std=0.05,
    cost_per_successful_task_mean=0.09,
    cost_per_successful_task_std=0.01,
    run_ids=("proxy_a", "proxy_b"),
    cost_per_task_mean=0.09,
    cost_per_task_std=0.01,
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
    assert g.accept_margin_sigma == 1.0


def test_margin_rejects_improvement_within_one_sigma_of_noise():
    # best mean 0.09, std 0.01 -> with a 1σ margin the threshold is 0.08.
    guardrails = Guardrails(
        task_success_floor_frac_of_best=0.95,
        max_cost_per_successful_task_usd=None,
        max_invalid_action_rate=None,
        accept_margin_sigma=1.0,
    )
    # 0.085 beats the mean but sits inside the noise band -> rejected.
    within_noise = RunMetrics(pass_rate=0.6, cost_per_successful_task=0.085, cost_per_task=0.085)
    assert check_run(within_noise, BEST, guardrails).passed is False

    # 0.079 clears mean - 1σ -> accepted.
    beyond_noise = RunMetrics(pass_rate=0.6, cost_per_successful_task=0.079, cost_per_task=0.079)
    assert check_run(beyond_noise, BEST, guardrails).passed is True


def test_zero_margin_keeps_legacy_below_mean_rule():
    # Default margin 0.0 accepts anything strictly below the mean (legacy behavior).
    legacy = Guardrails(
        task_success_floor_frac_of_best=0.95,
        max_cost_per_successful_task_usd=None,
        max_invalid_action_rate=None,
    )
    run = RunMetrics(pass_rate=0.6, cost_per_successful_task=0.085, cost_per_task=0.085)  # < 0.09 mean
    assert check_run(run, BEST, legacy).passed is True


def test_run_that_lowers_cost_and_holds_success_passes():
    run = RunMetrics(pass_rate=0.6, cost_per_successful_task=0.08, cost_per_task=0.08)

    check = check_run(run, BEST, GUARDRAILS)

    assert check.passed is True


def test_run_below_success_floor_fails():
    run = RunMetrics(pass_rate=0.5, cost_per_successful_task=0.08, cost_per_task=0.08)  # 0.5 < 0.57 floor

    check = check_run(run, BEST, GUARDRAILS)

    assert check.passed is False
    assert "success" in check.reason.lower()


def test_run_without_cost_improvement_fails():
    run = RunMetrics(pass_rate=0.6, cost_per_successful_task=0.10, cost_per_task=0.10)  # >= 0.09 best

    check = check_run(run, BEST, GUARDRAILS)

    assert check.passed is False
    assert "cost" in check.reason.lower() or "improve" in check.reason.lower()


def test_cost_ceiling_guardrail_blocks_even_when_improved():
    run = RunMetrics(
        pass_rate=0.6, cost_per_successful_task=0.088
    )  # < 0.09 best, > 0.085 ceiling
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
        RunMetrics(pass_rate=0.60, cost_per_successful_task=0.080, cost_per_task=0.080),
        RunMetrics(pass_rate=0.62, cost_per_successful_task=0.082, cost_per_task=0.082),
    ]

    decision = evaluate_candidate(runs, BEST, GUARDRAILS)

    assert decision.accepted is True


def test_one_failing_run_rejects():
    runs = [
        RunMetrics(pass_rate=0.60, cost_per_successful_task=0.080, cost_per_task=0.080),
        RunMetrics(pass_rate=0.50, cost_per_successful_task=0.082, cost_per_task=0.082),  # below floor
    ]

    decision = evaluate_candidate(runs, BEST, GUARDRAILS)

    assert decision.accepted is False


def test_single_run_cannot_be_accepted():
    runs = [RunMetrics(pass_rate=0.60, cost_per_successful_task=0.080, cost_per_task=0.080)]

    decision = evaluate_candidate(runs, BEST, GUARDRAILS)

    assert decision.accepted is False
    assert "two" in decision.reason.lower() or "2" in decision.reason


def test_run_with_harness_errors_is_rejected_even_if_cost_improves():
    # AutoPK port: never accept what cannot be verified. A run where tasks
    # crashed in the harness (harness_error terminations) is an invalid sample —
    # its "cost improvement" is an artifact of tasks doing no work.
    run = RunMetrics(
        pass_rate=0.60,
        cost_per_successful_task=0.010,
        cost_per_task=0.010,
        harness_error_count=3,
    )

    check = check_run(run, BEST, GUARDRAILS)

    assert check.passed is False
    assert "harness" in check.reason.lower()


def test_run_without_harness_errors_unaffected_by_new_field():
    run = RunMetrics(
        pass_rate=0.60,
        cost_per_successful_task=0.080,
        cost_per_task=0.080,
        harness_error_count=0,
    )

    check = check_run(run, BEST, GUARDRAILS)

    assert check.passed is True
