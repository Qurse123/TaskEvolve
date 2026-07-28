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

# absolute success floor 0.55; no cost/invalid ceilings.
GUARDRAILS = Guardrails(
    absolute_success_floor=0.55,
    max_cost_per_successful_task_usd=None,
    max_invalid_action_rate=None,
)


def test_load_guardrails_reads_real_yaml():
    g = load_guardrails()

    assert g.absolute_success_floor == 0.5
    assert g.max_cost_per_successful_task_usd is None
    assert g.max_invalid_action_rate is None
    assert g.accept_margin_sigma == 1.0


def test_margin_rejects_improvement_within_one_sigma_of_noise():
    # best mean 0.09, std 0.01 -> with a 1σ margin the threshold is 0.08.
    guardrails = Guardrails(
        absolute_success_floor=0.55,
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
        absolute_success_floor=0.55,
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
    run = RunMetrics(pass_rate=0.5, cost_per_successful_task=0.08, cost_per_task=0.08)  # 0.5 < 0.55 floor

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
        absolute_success_floor=0.55,
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


# --- Near-miss seed extension (§20 M3 amendment) --------------------------------
# When a run clears every guardrail and beats the current-best MEAN but misses the
# noise margin, the orchestrator may spend extra seeds; with n >= 3 runs the accept
# test becomes: every run below the best mean, all guardrails clear, and the MEAN
# cost across runs below the same mu - k*sigma threshold. Same bar, more power.

MARGIN_GUARDRAILS = Guardrails(
    absolute_success_floor=0.55,
    max_cost_per_successful_task_usd=None,
    max_invalid_action_rate=None,
    accept_margin_sigma=1.0,  # threshold = 0.09 - 0.01 = 0.08
)


def _run_at(cost: float, pass_rate: float = 0.60) -> RunMetrics:
    return RunMetrics(
        pass_rate=pass_rate, cost_per_successful_task=cost, cost_per_task=cost
    )


def test_near_miss_is_true_between_threshold_and_mean():
    from iterator_agent.acceptance import near_miss

    assert near_miss(_run_at(0.085), BEST, MARGIN_GUARDRAILS) is True   # < mean, >= threshold
    assert near_miss(_run_at(0.075), BEST, MARGIN_GUARDRAILS) is True   # beats threshold too
    assert near_miss(_run_at(0.095), BEST, MARGIN_GUARDRAILS) is False  # above best mean
    assert near_miss(_run_at(0.085, pass_rate=0.40), BEST, MARGIN_GUARDRAILS) is False  # floor


def test_three_run_mean_below_threshold_accepts():
    runs = [_run_at(0.078), _run_at(0.085), _run_at(0.074)]  # mean 0.079 < 0.08

    decision = evaluate_candidate(runs, BEST, MARGIN_GUARDRAILS)

    assert decision.accepted is True
    assert "mean" in decision.reason.lower()


def test_three_run_mean_at_threshold_rejects():
    runs = [_run_at(0.078), _run_at(0.085), _run_at(0.077)]  # mean 0.080 == threshold

    decision = evaluate_candidate(runs, BEST, MARGIN_GUARDRAILS)

    assert decision.accepted is False


def test_extension_run_at_best_mean_rejects_even_if_mean_clears():
    runs = [_run_at(0.060), _run_at(0.060), _run_at(0.090)]  # 0.090 == best mean

    decision = evaluate_candidate(runs, BEST, MARGIN_GUARDRAILS)

    assert decision.accepted is False


def test_two_run_path_still_requires_both_to_beat_margin():
    # Near-miss on seed B with only 2 runs must NOT accept (mean rule needs n>=3).
    runs = [_run_at(0.078), _run_at(0.085)]  # mean 0.0815 would beat 0.09... but n=2

    decision = evaluate_candidate(runs, BEST, MARGIN_GUARDRAILS)

    assert decision.accepted is False
