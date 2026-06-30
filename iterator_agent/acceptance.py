"""Acceptance rule — the iterator's deterministic comparator (experiment.md §20-21).

No LLM. A candidate harness change is accepted iff BOTH of its proxy runs
**improve** the objective (``cost_per_successful_task`` strictly below the
current-best mean) AND **clear every guardrail** (success floor + optional cost /
invalid-action ceilings declared in ``allowed_edits.yaml``). Otherwise the change
is reverted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import yaml

from iterator_agent.baseline import Distribution
from iterator_agent.edit_guard import DEFAULT_POLICY_PATH

REQUIRED_PROXY_RUNS = 2


@dataclass(frozen=True)
class RunMetrics:
    """One proxy run's headline metrics (a single seed)."""

    pass_rate: float
    cost_per_successful_task: Optional[float]
    invalid_action_rate: Optional[float] = None


@dataclass(frozen=True)
class Guardrails:
    """Acceptance guardrails declared in allowed_edits.yaml (frozen for the run)."""

    task_success_floor_frac_of_best: float
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
    floor = block.get("task_success_floor_frac_of_best")
    if floor is None:
        raise ValueError(
            f"guardrails.task_success_floor_frac_of_best missing from {path}"
        )
    return Guardrails(
        task_success_floor_frac_of_best=float(floor),
        max_cost_per_successful_task_usd=_opt_float(
            block.get("max_cost_per_successful_task_usd")
        ),
        max_invalid_action_rate=_opt_float(block.get("max_invalid_action_rate")),
        accept_margin_sigma=float(block.get("accept_margin_sigma") or 0.0),
    )


def check_run(run: RunMetrics, best: Distribution, guardrails: Guardrails) -> RunCheck:
    """Decide whether one proxy run improves the objective within all guardrails."""
    # 1. Improvement on the objective: cost per successful task strictly lower.
    if best.cost_per_successful_task_mean is None:
        return RunCheck(False, "cannot judge improvement: best has no cost baseline")
    if run.cost_per_successful_task is None:
        return RunCheck(
            False, "no cost improvement: run reported no cost per successful task"
        )
    # Improvement must clear the current-best mean by a noise margin (μ - kσ), so a
    # single lucky 2-seed draw within the benchmark's seed noise cannot be accepted.
    sigma = best.cost_per_successful_task_std or 0.0
    margin = guardrails.accept_margin_sigma * sigma
    threshold = best.cost_per_successful_task_mean - margin
    if not (run.cost_per_successful_task < threshold):
        return RunCheck(
            False,
            f"no cost improvement beyond noise: {run.cost_per_successful_task:.6f} "
            f">= threshold {threshold:.6f} "
            f"(best mean {best.cost_per_successful_task_mean:.6f} "
            f"- {guardrails.accept_margin_sigma:g}σ·{sigma:.6f})",
        )

    # 2. Guardrail: task-success floor (a fraction of the current-best mean).
    floor = guardrails.task_success_floor_frac_of_best * best.pass_rate_mean
    if run.pass_rate < floor:
        return RunCheck(
            False,
            f"crossed success guardrail: pass_rate {run.pass_rate:.4f} < floor {floor:.4f}",
        )

    # 3. Guardrail: cost-per-successful-task ceiling (if set).
    ceiling = guardrails.max_cost_per_successful_task_usd
    if ceiling is not None and run.cost_per_successful_task > ceiling:
        return RunCheck(
            False,
            f"crossed cost ceiling: {run.cost_per_successful_task:.6f} > {ceiling:.6f}",
        )

    # 4. Guardrail: invalid-action ceiling (if set and reported).
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

    return RunCheck(True, "improved cost per successful task within all guardrails")


def evaluate_candidate(
    runs: Sequence[RunMetrics],
    best: Distribution,
    guardrails: Guardrails,
) -> AcceptanceDecision:
    """Apply the double-run rule: accept iff exactly two proxy runs are present and pass."""
    checks = tuple(check_run(run, best, guardrails) for run in runs)

    if len(runs) != REQUIRED_PROXY_RUNS:
        return AcceptanceDecision(
            False,
            f"rejected: the double-run rule needs {REQUIRED_PROXY_RUNS} proxy runs, got {len(runs)}",
            checks,
        )

    failed = [c for c in checks if not c.passed]
    if failed:
        return AcceptanceDecision(
            False,
            f"rejected: {len(failed)} of {len(checks)} proxy run(s) failed — {failed[0].reason}",
            checks,
        )

    return AcceptanceDecision(
        True,
        "accepted: both proxy runs improved cost per successful task within guardrails",
        checks,
    )
