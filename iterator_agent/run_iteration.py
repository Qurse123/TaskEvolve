"""Orchestrator — one full iterator iteration (experiment.md §21).

Drives the deterministic spine for a single propose -> test -> keep/revert cycle:

    1-2. load the current-best proxy distribution + the latest proxy feedback
    3.   editor proposes exactly one change (the only LLM step)
    4-5. edit_guard checks the target is on the allowed surface (else reject)
    6.   apply the change to the working tree
    7.   run proxy eval at seed A
    8.   if seed A improves, run proxy eval at seed B (short-circuit otherwise)
    9-10. acceptance rule: keep iff BOTH runs improve within guardrails, else revert
    11-13. write the iteration note + append to the accepted/rejected changelog

Only the editor calls a model; everything else is deterministic. All external
calls (the LLM, the eval-runner, the feedback build, the commit) are injected so
the whole loop runs at $0 in tests. The fixed-budget loop over this function is
the driver's job (``scripts/run_iterator.py``).
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

from iterator_agent.acceptance import (
    AcceptanceDecision,
    Guardrails,
    RunMetrics,
    check_run,
    evaluate_candidate,
    load_guardrails,
)
from iterator_agent.baseline import Distribution, load_distribution
from iterator_agent.edit_guard import EditPolicy, evaluate, load_policy
from iterator_agent.feedback import FeedbackSummary, build_feedback
from iterator_agent.iteration_log import (
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_ITERATIONS_ROOT,
    IterationRecord,
    append_changelog,
    write_record,
)
from iterator_agent.researcher import (
    CompletionFn,
    CostTrackingCompletion,
    ProposedEdit,
    run_editor,
)
from results.logger import DEFAULT_LOGS_ROOT
from settings import config

# Per-seed eval: a seed in, that proxy run's headline metrics out.
EvalSeedFn = Callable[[int], RunMetrics]

_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)$")


@dataclass(frozen=True)
class IterationResult:
    """The outcome of one iteration — the decision plus everything it produced."""

    iteration_id: str
    accepted: bool
    decision: AcceptanceDecision
    record: IterationRecord
    proposed_edit: ProposedEdit
    harness_version: str  # version after this iteration (bumped iff accepted)
    search_cost_usd: float  # editor LLM cost for this iteration (experiment.md §15.3)
    record_path: Path
    changelog_path: Path


def run_iteration(
    *,
    iteration_id: str,
    seeds: Tuple[int, int],
    split: str = "proxy",
    repo_root: Union[str, Path] = ".",
    policy: Optional[EditPolicy] = None,
    guardrails: Optional[Guardrails] = None,
    best: Optional[Distribution] = None,
    feedback: Optional[FeedbackSummary] = None,
    complete: Optional[CompletionFn] = None,
    eval_seed: Optional[EvalSeedFn] = None,
    current_version: Optional[str] = None,
    iterations_root: Union[str, Path] = DEFAULT_ITERATIONS_ROOT,
    experiments_dir: Union[str, Path] = DEFAULT_EXPERIMENTS_DIR,
    logs_root: Union[str, Path] = DEFAULT_LOGS_ROOT,
    commit: Optional[Callable[[IterationResult], None]] = None,
) -> IterationResult:
    """Run one propose -> guard -> proxy x2 -> accept/revert cycle (experiment.md §21).

    Args:
        iteration_id: Stable id for this iteration (the driver assigns it).
        seeds: The two proxy seeds for the double-run (seed B only runs if A improves).
        split: Eval split — always ``"proxy"`` during optimization (validation stays blind).
        repo_root: Working tree the edit is applied to.
        policy/guardrails/best/feedback: Inputs; loaded from defaults when omitted.
        complete: Injected LLM call for the editor (default: configured model).
        eval_seed: Injected per-seed eval (default: one proxy repeat via run_train_eval).
        current_version: Current-best harness version (default: ``config.HARNESS_VERSION``).
        commit: Optional hook called with the result on accept (e.g. a git commit).
    """
    repo_root = Path(repo_root)
    policy = policy if policy is not None else load_policy()
    guardrails = guardrails if guardrails is not None else load_guardrails()
    current_version = current_version or config.HARNESS_VERSION
    if best is None:
        best = load_distribution(split, current_version)
    if feedback is None:
        feedback = build_feedback(split, logs_root=logs_root)

    # 3. Editor proposes exactly one change (the only LLM step). A cost-tracking
    # wrapper measures the iterator's search cost (experiment.md §15.3); an injected
    # `complete` keeps it as-is (tracked iff it exposes `total_cost_usd`).
    editor_complete = complete if complete is not None else CostTrackingCompletion()
    cost_before = float(getattr(editor_complete, "total_cost_usd", 0.0))
    proposal = run_editor(
        feedback, policy=policy, complete=editor_complete, repo_root=repo_root
    )
    # Per-iteration delta, so a tracker reused across iterations (the driver) still
    # attributes only this iteration's editor cost.
    search_cost = float(getattr(editor_complete, "total_cost_usd", 0.0)) - cost_before

    # 4-5. Allowed Change Check — a forbidden target is rejected before any eval.
    guard = evaluate(proposal.target_file, policy)
    if not guard.allowed:
        decision = AcceptanceDecision(False, f"rejected: {guard.reason}", ())
        return _persist(
            iteration_id=iteration_id,
            accepted=False,
            decision=decision,
            proposal=proposal,
            runs=(),
            seeds=(),
            best=best,
            harness_version=current_version,
            search_cost=search_cost,
            iterations_root=iterations_root,
            experiments_dir=experiments_dir,
            commit=commit,
        )

    # 6. Apply the change; tag candidate evals with the bumped version.
    candidate_version = _bump_version(current_version)
    original = _apply_edit(repo_root, proposal.target_file, proposal.new_content)
    eval_fn = eval_seed if eval_seed is not None else _default_eval_seed(split)
    config.HARNESS_VERSION = candidate_version

    runs, seeds_used, decision = _run_double(seeds, eval_fn, best, guardrails)

    # 9-10. Keep on accept (version stays bumped) or revert (restore file + version).
    if decision.accepted:
        result_version = candidate_version
    else:
        _revert_edit(repo_root, proposal.target_file, original)
        config.HARNESS_VERSION = current_version
        result_version = current_version

    return _persist(
        iteration_id=iteration_id,
        accepted=decision.accepted,
        decision=decision,
        proposal=proposal,
        runs=runs,
        seeds=seeds_used,
        best=best,
        harness_version=result_version,
        search_cost=search_cost,
        iterations_root=iterations_root,
        experiments_dir=experiments_dir,
        commit=commit,
    )


def _run_double(
    seeds: Tuple[int, int],
    eval_fn: EvalSeedFn,
    best: Distribution,
    guardrails: Guardrails,
) -> Tuple[Tuple[RunMetrics, ...], Tuple[int, ...], AcceptanceDecision]:
    """Run seed A, then seed B only if A improves; return runs, seeds, and the verdict."""
    run_a = eval_fn(seeds[0])
    check_a = check_run(run_a, best, guardrails)
    if not check_a.passed:
        # 8. Short-circuit: don't spend seed B when the first run already fails.
        decision = AcceptanceDecision(
            False,
            f"rejected: first proxy run did not improve — {check_a.reason}",
            (check_a,),
        )
        return (run_a,), (seeds[0],), decision

    run_b = eval_fn(seeds[1])
    runs = (run_a, run_b)
    decision = evaluate_candidate(runs, best, guardrails)
    return runs, (seeds[0], seeds[1]), decision


def _persist(
    *,
    iteration_id: str,
    accepted: bool,
    decision: AcceptanceDecision,
    proposal: ProposedEdit,
    runs: Sequence[RunMetrics],
    seeds: Tuple[int, ...],
    best: Distribution,
    harness_version: str,
    search_cost: float,
    iterations_root: Union[str, Path],
    experiments_dir: Union[str, Path],
    commit: Optional[Callable[[IterationResult], None]],
) -> IterationResult:
    """Build the §22 record, write it + the changelog, fire the commit hook on accept."""
    after_mean, after_std, after_cost = _after_stats(runs)
    record = IterationRecord(
        iteration_id=iteration_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        harness_version=harness_version,
        changed_surface=Path(proposal.target_file).stem,
        changed_file=proposal.target_file,
        change_summary=proposal.change_summary,
        reason_for_change=proposal.reason_for_change,
        proxy_seeds=seeds,
        proxy_task_success_before_mean=best.pass_rate_mean,
        proxy_task_success_before_std=best.pass_rate_std,
        proxy_task_success_after_mean=after_mean,
        proxy_task_success_after_std=after_std,
        cost_per_successful_task_before=best.cost_per_successful_task_mean,
        cost_per_successful_task_after=after_cost,
        accepted_or_rejected="accepted" if accepted else "rejected",
        reason_accepted_or_rejected=decision.reason,
        iterator_search_cost_usd=search_cost,
    )
    record_path = write_record(record, logs_root=iterations_root)
    changelog_path = append_changelog(record, experiments_dir=experiments_dir)

    result = IterationResult(
        iteration_id=iteration_id,
        accepted=accepted,
        decision=decision,
        record=record,
        proposed_edit=proposal,
        harness_version=harness_version,
        search_cost_usd=search_cost,
        record_path=record_path,
        changelog_path=changelog_path,
    )
    if accepted and commit is not None:
        commit(result)
    return result


def _after_stats(runs: Sequence[RunMetrics]) -> Tuple[float, float, Optional[float]]:
    """Mean ± std of the candidate runs' pass rate, and their mean cost-per-success."""
    if not runs:
        return 0.0, 0.0, None
    pass_rates = [r.pass_rate for r in runs]
    mean = statistics.mean(pass_rates)
    std = statistics.stdev(pass_rates) if len(pass_rates) > 1 else 0.0
    costs = [
        r.cost_per_successful_task
        for r in runs
        if r.cost_per_successful_task is not None
    ]
    cost = statistics.mean(costs) if costs else None
    return mean, std, cost


def _apply_edit(repo_root: Path, target: str, new_content: str) -> Optional[str]:
    """Write ``new_content`` to ``target``; return the prior content (None if new)."""
    path = repo_root / target
    original = path.read_text(encoding="utf-8") if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
    return original


def _revert_edit(repo_root: Path, target: str, original: Optional[str]) -> None:
    """Restore ``target`` to its prior state (delete it if it did not exist before)."""
    path = repo_root / target
    if original is None:
        if path.exists():
            path.unlink()
    else:
        path.write_text(original, encoding="utf-8")


def _bump_version(version: str) -> str:
    """Increment a ``vMAJOR.MINOR`` label's minor component (``v0.1`` -> ``v0.2``)."""
    match = _VERSION_RE.match(version)
    if not match:
        raise ValueError(f"harness version must match vMAJOR.MINOR, got {version!r}")
    major, minor = int(match.group(1)), int(match.group(2))
    return f"v{major}.{minor + 1}"


def _default_eval_seed(split: str) -> EvalSeedFn:
    """Default eval: one proxy repeat at the seed via the M1 runner (writes results.csv)."""
    from scripts.run_train_eval import run_repeats

    def eval_fn(seed: int) -> RunMetrics:
        metrics = run_repeats(split, seed_start=seed, repeats=1)[0]
        return RunMetrics(
            pass_rate=metrics.pass_rate,
            cost_per_successful_task=metrics.cost_per_successful_task,
        )

    return eval_fn
