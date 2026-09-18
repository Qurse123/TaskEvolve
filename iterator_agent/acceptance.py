"""Acceptance rule — the iterator's deterministic comparator (experiment.md §20-21).

No LLM. A candidate harness change is accepted iff BOTH of its proxy runs
**improve** the optimization objective (mean ``cost_per_successful_task`` below
the current-best mean by the noise margin) AND **clear every guardrail** (the
optional cost / invalid-action ceilings declared in ``allowed_edits.yaml``).
Otherwise the change is reverted.

Objective = **cost per successful task** (cost ÷ success). This single scalar
*is* the cost-vs-precision tradeoff the study optimizes: lowering it means the
agent got cheaper, more successful, or struck a favorable trade. A change that
tanks success raises the metric and self-rejects; a genuine cost cut, whether
token savings or a cheaper model that holds up, lowers it and is accepted.

There is **no task-success floor**, because success already sits inside the
objective: a candidate that halves cost and halves success leaves the number
unchanged. Earlier rules did impose one, first a relative 0.95×best floor and
later a low absolute floor, and both discarded measurements before the objective
was computed. The only remaining guardrails are the two ceilings above, which
are declared and currently unset.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import yaml

from iterator_agent.baseline import Distribution
from iterator_agent.edit_guard import DEFAULT_POLICY_PATH

REQUIRED_PROXY_RUNS = 2
# Near-miss extension (experiment.md §20 M3 amendment): a candidate whose extra
# runs all beat the current-best mean may spend up to this many total proxy runs;
# with n >= 3 the accept test is the MEAN cost beating the same mu - k*sigma
# threshold (same bar, halved standard error — not a loosening).
MAX_PROXY_RUNS = 4


@dataclass(frozen=True)
class RunMetrics:
    """One proxy run's headline metrics (a single seed)."""

    pass_rate: float
    cost_per_successful_task: Optional[float]
    invalid_action_rate: Optional[float] = None
    # Mean cost per task (total_cost / num_tasks). Its fixed denominator gives it
    # ~4x less variance than cost per successful task, so it is carried for
    # reporting and for the multi-split aggregate. It is NOT the accept objective:
    # _threshold() tests cost_per_successful_task against the baseline.
    cost_per_task: Optional[float] = None
    # Tasks that crashed inside the harness (termination_reason "harness_error").
    # Any crash invalidates the run as an acceptance sample: crashed tasks do no
    # work, so their "cost savings" are an artifact (experiment.md §20 rule 5).
    harness_error_count: int = 0


@dataclass(frozen=True)
class Guardrails:
    """Acceptance guardrails declared in allowed_edits.yaml (frozen for the run)."""

    # Cost ceiling: a candidate whose cost per successful task exceeds this is
    # rejected however much it improves on the baseline. Declared as null in
    # allowed_edits.yaml, so it is currently not enforced.
    max_cost_per_successful_task_usd: Optional[float]
    max_invalid_action_rate: Optional[float]
    # Noise floor: a run must beat the current-best cost mean by at least this many
    # standard deviations of the best distribution (experiment.md §20). 0.0 = strict
    # below-mean (legacy). >0 absorbs a single lucky 2-seed accept. Default 0.0 so
    # hand-built Guardrails (tests) keep the legacy rule.
    accept_margin_sigma: float = 0.0


@dataclass(frozen=True)
class RunCheck:
    """Whether one run improved within guardrails, with a human-readable reason."""

    passed: bool
    reason: str


@dataclass(frozen=True)
class AcceptanceDecision:
    """The final keep/revert verdict for a candidate, plus per-run detail."""

    accepted: bool
    reason: str
    run_checks: Tuple[RunCheck, ...]


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def load_guardrails(path: Union[str, Path] = DEFAULT_POLICY_PATH) -> Guardrails:
    """Load the acceptance guardrails block from the edit-policy YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    block = data.get("guardrails") or {}
    return Guardrails(
        max_cost_per_successful_task_usd=_opt_float(
            block.get("max_cost_per_successful_task_usd")
        ),
        max_invalid_action_rate=_opt_float(block.get("max_invalid_action_rate")),
        accept_margin_sigma=float(block.get("accept_margin_sigma") or 0.0),
    )


def _check_guardrails(
    run: RunMetrics, best: Distribution, guardrails: Guardrails
) -> RunCheck:
    """Validity + guardrail checks alone (no objective-improvement judgment)."""
    # Validity: a run with harness crashes is not a usable sample at all —
    # never accept what cannot be verified (experiment.md §20 rejection rule 5).
    if run.harness_error_count > 0:
        return RunCheck(
            False,
            f"invalid run: {run.harness_error_count} task(s) crashed in the harness "
            "(harness_error) — the change breaks the harness",
        )

    # No task-success floor. Success is already inside the objective: cost per
    # successful task divides by successes, so a candidate that halves cost and
    # halves success does not improve the number. A floor would only discard
    # measurements before the objective is computed, which is what killed 8 of 17
    # candidates in the previous run.

    # Guardrail: cost-per-successful-task ceiling (if set and reported).
    ceiling = guardrails.max_cost_per_successful_task_usd
    if (
        ceiling is not None
        and run.cost_per_successful_task is not None
        and run.cost_per_successful_task > ceiling
    ):
        return RunCheck(
            False,
            f"crossed cost ceiling: {run.cost_per_successful_task:.6f} > {ceiling:.6f}",
        )

    # Guardrail: invalid-action ceiling (if set and reported).
    max_invalid = guardrails.max_invalid_action_rate
    if (
        max_invalid is not None
        and run.invalid_action_rate is not None
        and run.invalid_action_rate > max_invalid
    ):
        return RunCheck(
            False,
            f"crossed invalid-action ceiling: {run.invalid_action_rate:.4f} > {max_invalid:.4f}",
        )

    return RunCheck(True, "all guardrails clear")


def _threshold(best: Distribution, guardrails: Guardrails) -> Optional[float]:
    """The accept threshold μ - kσ on cost per successful task (None without a baseline)."""
    if best.cost_per_successful_task_mean is None:
        return None
    sigma = best.cost_per_successful_task_std or 0.0
    return best.cost_per_successful_task_mean - guardrails.accept_margin_sigma * sigma


def check_run(run: RunMetrics, best: Distribution, guardrails: Guardrails) -> RunCheck:
    """Decide whether one proxy run improves the objective within all guardrails."""
    guard = _check_guardrails(run, best, guardrails)
    if not guard.passed:
        return guard

    # Improvement on the objective: cost per successful task strictly lower. This
    # scalar folds cost and precision together (cost ÷ success), so a change that
    # trades success for cost only wins if the trade is net-favorable.
    threshold = _threshold(best, guardrails)
    if threshold is None:
        return RunCheck(
            False, "cannot judge improvement: best has no cost-per-successful-task baseline"
        )
    if run.cost_per_successful_task is None:
        return RunCheck(
            False, "no cost improvement: run reported no cost per successful task"
        )
    # Improvement must clear the current-best mean by a noise margin (μ - kσ), so a
    # single lucky 2-seed draw within the benchmark's seed noise cannot be accepted.
    if not (run.cost_per_successful_task < threshold):
        sigma = best.cost_per_successful_task_std or 0.0
        return RunCheck(
            False,
            f"no cost improvement beyond noise: cost/successful-task "
            f"{run.cost_per_successful_task:.6f} >= threshold {threshold:.6f} "
            f"(best mean {best.cost_per_successful_task_mean:.6f} "
            f"- {guardrails.accept_margin_sigma:g}σ·{sigma:.6f})",
        )

    return RunCheck(True, "improved cost per successful task within all guardrails")


def near_miss(run: RunMetrics, best: Distribution, guardrails: Guardrails) -> bool:
    """True when a run clears every guardrail and beats the current-best MEAN cost
    (whether or not it clears the noise margin) — the pattern worth extra seeds."""
    if not _check_guardrails(run, best, guardrails).passed:
        return False
    if best.cost_per_successful_task_mean is None or run.cost_per_successful_task is None:
        return False
    return run.cost_per_successful_task < best.cost_per_successful_task_mean


def evaluate_candidate(
    runs: Sequence[RunMetrics],
    best: Distribution,
    guardrails: Guardrails,
) -> AcceptanceDecision:
    """The deterministic accept rule over a candidate's proxy runs.

    n == 2 (the double-run fast path): accept iff BOTH runs individually beat the
    μ - kσ threshold within guardrails (unchanged M2 rule).

    n >= 3 (near-miss extension, §20 M3 amendment): accept iff every run clears
    every guardrail, every run's cost beats the current-best MEAN, and the MEAN
    cost across runs beats the same μ - kσ threshold. Same bar, more power.
    """
    checks = tuple(check_run(run, best, guardrails) for run in runs)

    if len(runs) < REQUIRED_PROXY_RUNS:
        return AcceptanceDecision(
            False,
            f"rejected: the double-run rule needs {REQUIRED_PROXY_RUNS} proxy runs, got {len(runs)}",
            checks,
        )

    failed = [c for c in checks if not c.passed]
    if not failed:
        return AcceptanceDecision(
            True,
            f"accepted: all {len(checks)} proxy runs improved cost per successful task within guardrails",
            checks,
        )

    if len(runs) >= 3:
        threshold = _threshold(best, guardrails)
        costs = [r.cost_per_successful_task for r in runs]
        if (
            threshold is not None
            and all(near_miss(r, best, guardrails) for r in runs)
            and all(c is not None for c in costs)
            and statistics.mean(costs) < threshold  # type: ignore[arg-type]
        ):
            return AcceptanceDecision(
                True,
                f"accepted (near-miss extension): mean cost/successful-task over {len(runs)} runs "
                f"{statistics.mean(costs):.6f} < threshold {threshold:.6f}, "  # type: ignore[arg-type]
                "every run below the current-best mean within guardrails",
                checks,
            )

    return AcceptanceDecision(
        False,
        f"rejected: {len(failed)} of {len(checks)} proxy run(s) failed — {failed[0].reason}",
        checks,
    )
