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
from pathlib import Path
from typing import Callable, List, Optional, Union

from iterator_agent.acceptance import Guardrails
from iterator_agent.baseline import Distribution, load_distribution
from iterator_agent.edit_guard import EditPolicy
from iterator_agent.iteration_log import DEFAULT_EXPERIMENTS_DIR, DEFAULT_ITERATIONS_ROOT
from iterator_agent.researcher import CompletionFn
from iterator_agent.run_iteration import EvalSeedFn, IterationResult, run_iteration
from results.logger import DEFAULT_LOGS_ROOT
from settings import config

logger = logging.getLogger(__name__)

# Two logged proxy seeds per iteration (the double-run rule).
SEEDS_PER_ITERATION = 2


def run_iterator(
    *,
    max_iterations: int,
    seed_start: int,
    split: str = "proxy",
    max_search_cost_usd: Optional[float] = None,
    repo_root: Union[str, Path] = ".",
    policy: Optional[EditPolicy] = None,
    guardrails: Optional[Guardrails] = None,
    initial_best: Optional[Distribution] = None,
    initial_version: Optional[str] = None,
    complete: Optional[CompletionFn] = None,
    eval_seed: Optional[EvalSeedFn] = None,
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
    best = (
        initial_best
        if initial_best is not None
        else load_distribution(split, current_version)
    )

    results: List[IterationResult] = []
    total_search_cost = 0.0

    for index in range(max_iterations):
        if max_search_cost_usd is not None and total_search_cost >= max_search_cost_usd:
            logger.info(
                "search-cost budget $%.4f reached after %d iteration(s); stopping.",
                max_search_cost_usd, index,
            )
            break

        seed_a = seed_start + SEEDS_PER_ITERATION * index
        history = [r.record for r in results]
        result = run_iteration(
            iteration_id=f"iter_{index + 1:04d}",
            seeds=(seed_a, seed_a + 1),
            split=split,
            repo_root=repo_root,
            policy=policy,
            guardrails=guardrails,
            best=best,
            current_version=current_version,
            history=history,
            complete=complete,
            eval_seed=eval_seed,
            commit=commit,
            iterations_root=iterations_root,
            experiments_dir=experiments_dir,
            logs_root=logs_root,
        )
        results.append(result)
        total_search_cost += result.search_cost_usd

        if result.accepted:
            noise_std = best.cost_per_successful_task_std or 0.0
            current_version = result.harness_version
            best = _distribution_from_result(
                result, split, current_version, noise_std=noise_std
            )

        logger.info(
            "%s: %s -> %s | search $%.4f (total $%.4f)",
            result.iteration_id,
            "ACCEPT" if result.accepted else "reject",
            current_version,
            result.search_cost_usd,
            total_search_cost,
        )

    _log_summary(results, total_search_cost, current_version)
    return results


def _distribution_from_result(
    result: IterationResult, split: str, version: str, *, noise_std: float
) -> Distribution:
    """The accepted candidate's proxy double-run becomes the next current-best.

    ``noise_std`` carries forward the prior best's cost std as a stable estimate of
    the benchmark's seed noise, so the acceptance margin (μ - kσ) stays meaningful
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
    )


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
        help="First proxy seed; iteration i uses seeds seed_start+2i and +1.",
    )
    parser.add_argument(
        "--max-search-cost-usd", type=float, default=None,
        help="Optional iterator search-cost cap (USD); stops once total spend reaches it.",
    )
    args = parser.parse_args(argv)

    if args.max_iterations < 1:
        parser.error("--max-iterations must be >= 1")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_iterator(
        max_iterations=args.max_iterations,
        seed_start=args.seed_start,
        max_search_cost_usd=args.max_search_cost_usd,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
