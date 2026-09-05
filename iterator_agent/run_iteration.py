"""Orchestrator — one full iterator iteration (experiment.md §21).

Drives the deterministic spine for a single propose -> test -> keep/revert cycle:

    1-2. load the current-best proxy distribution + the latest proxy feedback
    3.   editor proposes exactly one change (the only LLM step)
    4-5. edit_guard checks the target is on the allowed surface (else reject)
    6.   apply the change to the working tree
    7.   run proxy eval at seed A
    8.   measure the candidate on both seeds (no short-circuit)
    9-10. acceptance rule: keep iff BOTH runs improve within guardrails, else revert
    11-13. write the iteration note + append to the accepted/rejected changelog

Only the editor calls a model; everything else is deterministic. All external
calls (the LLM, the eval-runner, the feedback build, the commit) are injected so
the whole loop runs at $0 in tests. The fixed-budget loop over this function is
the driver's job (``scripts/run_iterator.py``).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Callable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from iterator_agent.acceptance import (
    MAX_PROXY_RUNS,
    AcceptanceDecision,
    Guardrails,
    RunMetrics,
    check_run,
    evaluate_candidate,
    load_guardrails,
    near_miss,
)
from iterator_agent.baseline import Distribution, load_distribution
from iterator_agent.edit_guard import EditPolicy, evaluate, load_policy
from iterator_agent.feedback import FeedbackSummary, build_feedback
from iterator_agent.iteration_log import (
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_ITERATIONS_ROOT,
    IterationRecord,
    append_changelog,
    write_artifact,
    write_record,
)
from iterator_agent.preflight import PreflightResult, run_preflight
from iterator_agent.researcher import (
    CompletionFn,
    CostTrackingCompletion,
    ProposedEdit,
    run_editor,
)
from results.logger import DEFAULT_LOGS_ROOT
from settings import config

if TYPE_CHECKING:
    from iterator_agent.hypothesis import Ticket

# Per-seed eval: a seed in, that proxy run's headline metrics out.
EvalSeedFn = Callable[[int], RunMetrics]

# Candidate preflight: (repo_root, target_file) -> structural verdict at $0.
PreflightFn = Callable[[Path, str], PreflightResult]

_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)$")


def proposal_hash(proposal: ProposedEdit) -> str:
    """Content identity of an edit: what file it touches and what it writes there.

    Prose fields (summary/reason) are excluded — a re-worded pitch for the same
    bytes is still the same change (AutoPK's rejected-ticket memory)."""
    digest = hashlib.sha256()
    digest.update(proposal.target_file.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(proposal.new_content.encode("utf-8"))
    return digest.hexdigest()


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
    seeds: Tuple[int, ...],
    split: str = "proxy",
    repo_root: Union[str, Path] = ".",
    policy: Optional[EditPolicy] = None,
    guardrails: Optional[Guardrails] = None,
    best: Optional[Distribution] = None,
    feedback: Optional[FeedbackSummary] = None,
    complete: Optional[CompletionFn] = None,
    eval_seed: Optional[EvalSeedFn] = None,
    current_version: Optional[str] = None,
    history: Sequence[IterationRecord] = (),
    iterations_root: Union[str, Path] = DEFAULT_ITERATIONS_ROOT,
    experiments_dir: Union[str, Path] = DEFAULT_EXPERIMENTS_DIR,
    logs_root: Union[str, Path] = DEFAULT_LOGS_ROOT,
    commit: Optional[Callable[[IterationResult], None]] = None,
    ticket: Optional["Ticket"] = None,
    preflight: Optional[PreflightFn] = None,
    rejected_hashes: Optional[AbstractSet[str]] = None,
) -> IterationResult:
    """Run one propose -> guard -> preflight -> proxy x2 -> accept/revert cycle (experiment.md §21).

    Args:
        iteration_id: Stable id for this iteration (the driver assigns it).
        seeds: The two proxy seeds for the double-run (seed B only runs if A improves).
        split: Eval split — always ``"proxy"`` during optimization (validation stays blind).
        repo_root: Working tree the edit is applied to.
        policy/guardrails/best/feedback: Inputs; loaded from defaults when omitted.
        complete: Injected LLM call for the editor (default: configured model).
        eval_seed: Injected per-seed eval (default: one proxy repeat via run_train_eval).
        current_version: Current-best harness version (default: ``config.HARNESS_VERSION``).
        commit: Retained for API compatibility; never called, since every
            candidate is reverted and there is nothing to commit.
        preflight: Injected structural check run after the edit is applied but
            before any eval spend (default: the real subprocess preflight).
        rejected_hashes: Content hashes (``proposal_hash``) of previously rejected
            edits; a duplicate proposal is refused at $0 (AutoPK bounce guard).
    """
    repo_root = Path(repo_root)
    policy = policy if policy is not None else load_policy()
    guardrails = guardrails if guardrails is not None else load_guardrails()
    current_version = current_version or config.HARNESS_VERSION
    if best is None:
        best = load_distribution(split, current_version)
    if feedback is None and ticket is None:
        # Only the free-form (no-ticket) editor path reads feedback for its diagnose
        # step; when a ticket drives the iteration the ticket IS the diagnosis, so we
        # skip the (otherwise unused) run-folder read. Diagnose the CURRENT-BEST
        # harness's latest run — see feedback.build_feedback.
        feedback = build_feedback(
            split, harness_version=current_version, logs_root=logs_root
        )

    # Provenance: the models in play this iteration (experiment.md §22).
    editor_model = config.ITERATOR_MODEL or ""
    agent_model = config.AGENT_MODEL or ""

    # 3. Editor proposes exactly one change (the only LLM step). A cost-tracking
    # wrapper measures the iterator's search cost (experiment.md §15.3); an injected
    # `complete` keeps it as-is (tracked iff it exposes `total_cost_usd`). `history`
    # (prior accepted+rejected records) gives the editor memory of what it has tried.
    editor_complete = complete if complete is not None else CostTrackingCompletion()
    cost_before = float(getattr(editor_complete, "total_cost_usd", 0.0))
    proposal = run_editor(
        feedback,
        policy=policy,
        complete=editor_complete,
        repo_root=repo_root,
        history=history,
        ticket=ticket,
    )
    # Per-iteration delta, so a tracker reused across iterations (the driver) still
    # attributes only this iteration's editor cost.
    search_cost = float(getattr(editor_complete, "total_cost_usd", 0.0)) - cost_before

    def _reject_before_eval(reason: str, original: Optional[str]) -> IterationResult:
        decision = AcceptanceDecision(False, reason, ())
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
            original=original,
            editor_model=editor_model,
            agent_model=agent_model,
            iterations_root=iterations_root,
            experiments_dir=experiments_dir,
            commit=commit,
            ticket=ticket,
        )

    # 4-5. Allowed Change Check — a forbidden target is rejected before any eval.
    guard = evaluate(proposal.target_file, policy)
    if not guard.allowed:
        return _reject_before_eval(f"rejected: {guard.reason}", original=None)

    # 5b. Rejected-change memory (AutoPK bounce guard): an edit content-identical
    # to one already rejected this run is refused without applying or evaluating.
    if rejected_hashes and proposal_hash(proposal) in rejected_hashes:
        return _reject_before_eval(
            "rejected: duplicate of a previously rejected change "
            f"(same content for {proposal.target_file})",
            original=None,
        )

    # 6. Apply the change; tag candidate evals with the bumped version.
    candidate_version = _bump_version(current_version)
    original = _apply_edit(repo_root, proposal.target_file, proposal.new_content)

    # 6b. Preflight (AutoPK "validate before eval"): exercise the edited tree in
    # a fresh interpreter at $0; a structurally broken candidate never reaches
    # the paid proxy eval (experiment.md §20 rule 5, §21 step 5).
    preflight_fn = preflight if preflight is not None else run_preflight
    verdict = preflight_fn(repo_root, proposal.target_file)
    if not verdict.passed:
        _revert_edit(repo_root, proposal.target_file, original)
        return _reject_before_eval(
            f"rejected: preflight failed — {verdict.reason}", original=original
        )

    eval_fn = eval_seed if eval_seed is not None else _default_eval_seed(split)
    config.HARNESS_VERSION = candidate_version

    runs, seeds_used, decision = _run_double(seeds, eval_fn, best, guardrails)

    # 9-10. Keep on accept (version stays bumped) or revert (restore file + version).
    # Always revert. Every candidate is measured from the same v0.1 starting
    # point so the rows are directly comparable, and the winner is chosen from
    # the full table after the run rather than greedily mid-run. Keeping an edit
    # here would make each later candidate a measurement of a different system.
    # `decision.accepted` is retained as an advisory flag in the record: it says
    # whether this candidate beat the baseline, not whether it was kept.
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
        original=original,
        editor_model=editor_model,
        agent_model=agent_model,
        iterations_root=iterations_root,
        experiments_dir=experiments_dir,
        commit=commit,
        ticket=ticket,
    )


def _run_double(
    seeds: Tuple[int, ...],
    eval_fn: EvalSeedFn,
    best: Distribution,
    guardrails: Guardrails,
) -> Tuple[Tuple[RunMetrics, ...], Tuple[int, ...], AcceptanceDecision]:
    """Measure the candidate on both seeds. No short-circuit, no early exit.

    Every candidate gets a complete two-seed measurement so it can be ranked
    against every other candidate afterwards. Skipping seed B when seed A looked
    bad discarded measurements we had already paid for and made rows
    incomparable: one candidate judged on one seed, another on two.

    The returned decision is advisory only. The caller reverts the edit either
    way; the winner is chosen from the full table once the run is over.
    """
    runs = [eval_fn(seeds[0]), eval_fn(seeds[1])]
    used = [seeds[0], seeds[1]]
    decision = evaluate_candidate(tuple(runs), best, guardrails)
    return tuple(runs), tuple(used), decision


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
    original: Optional[str],
    editor_model: str,
    agent_model: str,
    iterations_root: Union[str, Path],
    experiments_dir: Union[str, Path],
    commit: Optional[Callable[[IterationResult], None]],
    ticket: Optional["Ticket"] = None,
) -> IterationResult:
    """Build the §22 record, write it + the changelog + the change diff, commit on accept."""
    after_mean, after_std, after_cost, after_per_task = _after_stats(runs)
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
        editor_model=editor_model,
        agent_model=agent_model,
        cost_per_task_before=best.cost_per_task_mean,
        cost_per_task_after=after_per_task,
        ticket_id=ticket.ticket_id if ticket is not None else "",
        hypothesis=ticket.hypothesis if ticket is not None else "",
    )
    record_path = write_record(record, logs_root=iterations_root)
    changelog_path = append_changelog(record, experiments_dir=experiments_dir)
    # Persist the exact edit (accepted AND rejected) so the change survives a revert
    # and the user can audit what the iterator tried on each file (user: track changes).
    write_artifact(
        iteration_id,
        "change.diff",
        _unified_diff(original, proposal.new_content, proposal.target_file),
        logs_root=iterations_root,
    )

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
    # The commit hook is disabled: every candidate is reverted, so there is
    # nothing on disk to commit. `accepted` now only records that the candidate
    # beat the baseline, which the post-run ranking uses to pick a winner.
    if False and commit is not None:
        commit(result)
    return result


def _after_stats(
    runs: Sequence[RunMetrics],
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """Mean ± std pass rate, mean cost-per-success, and mean per-task cost (objective)."""
    if not runs:
        return 0.0, 0.0, None, None
    pass_rates = [r.pass_rate for r in runs]
    mean = statistics.mean(pass_rates)
    std = statistics.stdev(pass_rates) if len(pass_rates) > 1 else 0.0
    costs = [
        r.cost_per_successful_task
        for r in runs
        if r.cost_per_successful_task is not None
    ]
    cost = statistics.mean(costs) if costs else None
    per_task = [r.cost_per_task for r in runs if r.cost_per_task is not None]
    per_task_cost = statistics.mean(per_task) if per_task else None
    return mean, std, cost, per_task_cost


def _unified_diff(original: Optional[str], new_content: str, target: str) -> str:
    """Unified diff of a file's prior content → proposed content (for change tracking)."""
    before = (original or "").splitlines(keepends=True)
    after = new_content.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(before, after, fromfile=f"a/{target}", tofile=f"b/{target}")
    )


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


def _split_size(split: str) -> int:
    """Task count for a split manifest, used to weight multi-domain aggregation."""
    data = json.loads(
        (Path("benchmark/splits") / f"{split}.json").read_text(encoding="utf-8")
    )
    ids = data if isinstance(data, list) else data.get("task_ids", data)
    return len(ids)


def _aggregate(parts: Sequence[Tuple[RunMetrics, int]]) -> RunMetrics:
    """Fold per-domain results into one sample, weighted by task count.

    Success is total passed over total tasks, so a domain contributes in
    proportion to its size instead of every domain counting equally. Cost is
    reconstructed as cost_per_task * tasks, because the per-repeat payload
    carries ratios rather than totals. Harness errors sum: a crash anywhere
    invalidates the whole sample, exactly as it does for a single split.
    """
    total_tasks = sum(n for _, n in parts)
    total_passed = sum(m.pass_rate * n for m, n in parts)
    costed = [(m, n) for m, n in parts if m.cost_per_task is not None]
    total_cost = sum(m.cost_per_task * n for m, n in costed)
    return RunMetrics(
        pass_rate=total_passed / total_tasks,
        cost_per_successful_task=(
            total_cost / total_passed if costed and total_passed else None
        ),
        cost_per_task=(total_cost / total_tasks) if costed else None,
        harness_error_count=sum(m.harness_error_count for m, _ in parts),
    )


def _default_eval_seed(targets) -> EvalSeedFn:
    """Default eval: one repeat per target at the seed, in FRESH interpreters.

    ``targets`` is a single split name, or a sequence of (split, domain) pairs.
    Multiple targets exist because TAU2 evaluates one domain per invocation, so
    covering retail, airline and telecom means three subprocesses whose results
    fold into a single sample via ``_aggregate``. Optimizing against one domain
    lets a candidate look good on evidence that does not represent deployment.

    A fresh interpreter per run matters: the candidate edit may touch
    ``harness.py`` or ``model_routing.py``, and an in-process eval would silently
    run the stale, already-imported modules instead of the candidate. The
    subprocess also carries the candidate ``HARNESS_VERSION`` so its
    ``results.csv`` rows attribute to the candidate.
    """
    pairs = [(targets, None)] if isinstance(targets, str) else list(targets)

    def eval_fn(seed: int) -> RunMetrics:
        parts = []
        for split, domain in pairs:
            with tempfile.TemporaryDirectory() as tmp:
                metrics_path = Path(tmp) / "metrics.json"
                cmd = [
                    sys.executable, "-m", "scripts.run_train_eval",
                    "--split", split,
                    "--seed-start", str(seed),
                    "--repeats", "1",
                    "--metrics-json", str(metrics_path),
                ]
                if domain:
                    cmd += ["--domain", domain]
                proc = subprocess.run(
                    cmd,
                    env={**os.environ, "HARNESS_VERSION": config.HARNESS_VERSION},
                )
                if proc.returncode != 0 or not metrics_path.exists():
                    raise RuntimeError(
                        f"eval subprocess failed (exit {proc.returncode}) "
                        f"for split={split} seed={seed}"
                    )
                row = json.loads(metrics_path.read_text(encoding="utf-8"))[0]
            parts.append((
                RunMetrics(
                    pass_rate=float(row["pass_rate"]),
                    cost_per_successful_task=row["cost_per_successful_task"],
                    cost_per_task=row["cost_per_task"],
                    harness_error_count=int(row.get("harness_error_count", 0)),
                ),
                _split_size(split),
            ))
        return parts[0][0] if len(parts) == 1 else _aggregate(parts)

    return eval_fn
