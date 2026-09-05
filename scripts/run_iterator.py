"""Arm B optimization driver — loop the iterator under a fixed budget (experiment.md §17.9, §21).

A *ticket* is the pre-declared budget: a maximum iteration count (always) and an
optional iterator search-cost cap. The driver loads the Arm A proxy distribution
as the initial current-best, then repeatedly runs one iteration
(``iterator_agent.run_iteration``). After each **accepted** change the accepted
candidate's proxy distribution becomes the new current-best (greedy hill-climb);
a rejected iteration leaves the baseline untouched. The loop is **proxy-only** —
validation stays blind until the post-hoc event.

Run (proxy, e.g. 10 iterations from seed 3001):
    python -m scripts.run_iterator --max-iterations 10 --seed-start 3001
    python -m scripts.run_iterator --max-iterations 10 --seed-start 3001 --max-search-cost-usd 5.0
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from iterator_agent.acceptance import MAX_PROXY_RUNS, Guardrails
from iterator_agent.baseline import Distribution, load_distribution
from iterator_agent.edit_guard import EditPolicy
from iterator_agent.feedback import build_feedback
from iterator_agent.hypothesis import TICKETS_PER_BACKLOG, Ticket, generate_tickets
from iterator_agent.iteration_log import (
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_ITERATIONS_ROOT,
    IterationRecord,
)
from iterator_agent.researcher import CompletionFn, CostTrackingCompletion
from iterator_agent.run_iteration import (
    _default_eval_seed,
    EvalSeedFn,
    IterationResult,
    PreflightFn,
    proposal_hash,
    run_iteration,
)
from results.logger import DEFAULT_LOGS_ROOT
from settings import config

# A backlog provider: given (current_version, history) -> (ranked tickets, search cost).
# Injectable so the driver runs at $0 in tests; the default calls the real LLM.
BacklogFn = Callable[
    [str, "Sequence[IterationRecord]"], "Tuple[Sequence[Ticket], float]"
]

logger = logging.getLogger(__name__)

# Seeds reserved per iteration: the double-run pair plus up to two near-miss
# extension seeds (§20 M3 amendment). Unused seeds are simply skipped, keeping
# every iteration's seed block disjoint and reproducible.
SEEDS_PER_ITERATION = MAX_PROXY_RUNS


def run_iterator(
    *,
    max_iterations: int,
    seed_start: int,
    split: str = "proxy",
    agent_model: Optional[str] = None,
    max_search_cost_usd: Optional[float] = None,
    max_minutes: Optional[float] = None,
    clock: Callable[[], float] = time.monotonic,
    repo_root: Union[str, Path] = ".",
    policy: Optional[EditPolicy] = None,
    guardrails: Optional[Guardrails] = None,
    initial_best: Optional[Distribution] = None,
    initial_version: Optional[str] = None,
    complete: Optional[CompletionFn] = None,
    eval_seed: Optional[EvalSeedFn] = None,
    preflight: Optional[PreflightFn] = None,
    generate_backlog: Optional[BacklogFn] = None,
    tickets_per_backlog: int = TICKETS_PER_BACKLOG,
    commit: Optional[Callable[[IterationResult], None]] = None,
    iterations_root: Union[str, Path] = DEFAULT_ITERATIONS_ROOT,
    experiments_dir: Union[str, Path] = DEFAULT_EXPERIMENTS_DIR,
    logs_root: Union[str, Path] = DEFAULT_LOGS_ROOT,
) -> List[IterationResult]:
    """Drive iterations until the iteration count or search-cost budget is exhausted.

    Returns the per-iteration results in order. The current-best distribution and
    harness version advance only on accepted iterations.
    """
    current_version = initial_version or config.HARNESS_VERSION
    # Pin the campaign to one agent model so the shared results.csv / logs (which
    # now hold gpt-4.1, Inkling and Opus runs) don't blend into the baseline.
    agent_model = agent_model or config.AGENT_MODEL
    best = (
        initial_best
        if initial_best is not None
        else load_distribution(split, current_version, agent_model=agent_model)
    )
    provide_backlog = generate_backlog or _default_backlog_fn(
        split=split,
        policy=policy,
        repo_root=repo_root,
        logs_root=logs_root,
        n=tickets_per_backlog,
        agent_model=agent_model,
    )

    results: List[IterationResult] = []
    total_search_cost = 0.0
    # A ranked hypothesis backlog worked one ticket per iteration; refilled when empty
    # and cleared after every accept so the next batch is drawn from the new landscape
    # (hill-climb). Off-surface / unusable batches fall back to the free-form editor.
    backlog: List[Ticket] = []
    # Rejected-change memory (AutoPK bounce guard): content hashes of every edit
    # rejected this run, so a re-proposed identical change is refused at $0.
    rejected_hashes: set[str] = set()
    # Wall-clock budget (experiment.md §17.9): the primary bound when set; the
    # iteration count and search-cost caps remain as reproducible ceilings, so the
    # loop stops at whichever bound trips first.
    deadline = (clock() + max_minutes * 60.0) if max_minutes is not None else None

    for index in range(max_iterations):
        if max_search_cost_usd is not None and total_search_cost >= max_search_cost_usd:
            logger.info(
                "search-cost budget $%.4f reached after %d iteration(s); stopping.",
                max_search_cost_usd, index,
            )
            break
        if deadline is not None and clock() >= deadline:
            logger.info(
                "time budget %.1f min reached after %d iteration(s); stopping.",
                max_minutes, index,
            )
            break

        seed_a = seed_start + SEEDS_PER_ITERATION * index
        history = [r.record for r in results]

        backlog_cost = 0.0
        if not backlog:
            try:
                tickets, backlog_cost = provide_backlog(current_version, history)
                backlog = list(tickets)
            except ValueError as exc:
                logger.warning(
                    "backlog generation produced no usable tickets (%s); "
                    "falling back to the free-form editor this iteration.", exc,
                )
        ticket = backlog.pop(0) if backlog else None

        result = run_iteration(
            iteration_id=f"iter_{index + 1:04d}",
            seeds=tuple(seed_a + j for j in range(SEEDS_PER_ITERATION)),
            split=split,
            repo_root=repo_root,
            policy=policy,
            guardrails=guardrails,
            best=best,
            current_version=current_version,
            history=history,
            complete=complete,
            eval_seed=eval_seed,
            preflight=preflight,
            rejected_hashes=frozenset(rejected_hashes),
            commit=commit,
            iterations_root=iterations_root,
            experiments_dir=experiments_dir,
            logs_root=logs_root,
            ticket=ticket,
        )
        results.append(result)
        total_search_cost += result.search_cost_usd + backlog_cost
        if not result.accepted:
            rejected_hashes.add(proposal_hash(result.proposed_edit))

        # No promotion. `best` stays the unmodified-harness baseline for the whole
        # run, so every candidate is measured against the same reference and the
        # rows are comparable. The winner is picked from the full table afterwards
        # by scripts/rank_candidates.py, not greedily here.

        logger.info(
            "%s: %s baseline | %s | search $%.4f (total $%.4f)",
            result.iteration_id,
            "beats" if result.accepted else "below",
            result.proposed_edit.target_file,
            result.search_cost_usd,
            total_search_cost,
        )

    _log_summary(results, total_search_cost, current_version)
    return results


def _distribution_from_result(
    result: IterationResult, split: str, version: str, *, noise_std: float
) -> Distribution:
    """The accepted candidate's proxy double-run becomes the next current-best.

    ``noise_std`` carries forward the prior best's cost-per-successful-task std (the
    objective's seed-noise scale), so the acceptance margin (μ - kσ) stays meaningful
    across the hill-climb rather than collapsing to a 2-seed std of ~0.
    """
    record = result.record
    return Distribution(
        split=split,
        harness_version=version,
        n=len(record.proxy_seeds),
        pass_rate_mean=record.proxy_task_success_after_mean,
        pass_rate_std=record.proxy_task_success_after_std,
        cost_per_successful_task_mean=record.cost_per_successful_task_after,
        cost_per_successful_task_std=noise_std,
        run_ids=(),
        cost_per_task_mean=record.cost_per_task_after,
        cost_per_task_std=0.0,
    )


def _default_backlog_fn(
    *,
    split: str,
    policy: Optional[EditPolicy],
    repo_root: Union[str, Path],
    logs_root: Union[str, Path],
    n: int,
    agent_model: Optional[str] = None,
) -> BacklogFn:
    """Real backlog provider: summarize the current-best proxy run, then ask the LLM.

    Uses its own :class:`CostTrackingCompletion` so the backlog-generation spend is
    measured and returned separately from the editor's per-iteration search cost.
    """

    def provide(
        current_version: str, history: Sequence[IterationRecord]
    ) -> Tuple[Sequence[Ticket], float]:
        tracker = CostTrackingCompletion()
        feedback = build_feedback(
            split, harness_version=current_version, agent_model=agent_model,
            logs_root=logs_root,
        )
        tickets = generate_tickets(
            feedback,
            policy=policy,
            complete=tracker,
            repo_root=repo_root,
            history=history,
            n=n,
        )
        return tickets, tracker.total_cost_usd

    return provide


def _log_summary(
    results: List[IterationResult], total_search_cost: float, final_version: str
) -> None:
    accepted = sum(1 for r in results if r.accepted)
    logger.info("=" * 60)
    logger.info("iterator finished: %d iteration(s), %d accepted", len(results), accepted)
    logger.info("final harness version:   %s", final_version)
    logger.info("total iterator search cost: $%.4f", total_search_cost)
    logger.info("=" * 60)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Arm B iterator on proxy under a fixed optimization budget."
    )
    parser.add_argument(
        "--max-iterations", type=int, required=True,
        help="Iteration-count budget (pre-declared before the run).",
    )
    parser.add_argument(
        "--seed-start", type=int, required=True,
        help="First proxy seed; iteration i reserves the block seed_start+4i..+3 "
        "(double-run pair + up to two near-miss extension seeds).",
    )
    parser.add_argument(
        "--max-search-cost-usd", type=float, default=None,
        help="Optional iterator search-cost cap (USD); stops once total spend reaches it.",
    )
    parser.add_argument(
        "--max-minutes", type=float, default=None,
        help="Optional wall-clock budget (minutes); stops once elapsed time reaches it. "
        "--max-iterations still applies as a reproducible ceiling.",
    )
    parser.add_argument(
        "--eval-targets",
        default=None,
        help=(
            "Comma-separated split:domain pairs to evaluate each candidate on, "
            "e.g. 'proxy:retail,eval_airline:airline,eval_telecom:telecom'. TAU2 "
            "runs one domain per invocation, so each pair becomes its own "
            "subprocess and the results fold into one weighted sample. Omit to "
            "evaluate on --split alone."
        ),
    )
    args = parser.parse_args(argv)

    if args.max_iterations < 1:
        parser.error("--max-iterations must be >= 1")
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error("--max-minutes must be > 0")

    eval_seed = None
    if args.eval_targets:
        pairs = []
        for chunk in args.eval_targets.split(","):
            split, _, domain = chunk.strip().partition(":")
            if not split:
                parser.error(f"malformed --eval-targets entry: {chunk!r}")
            pairs.append((split, domain or None))
        eval_seed = _default_eval_seed(pairs)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_iterator(
        max_iterations=args.max_iterations,
        seed_start=args.seed_start,
        max_search_cost_usd=args.max_search_cost_usd,
        max_minutes=args.max_minutes,
        eval_seed=eval_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
