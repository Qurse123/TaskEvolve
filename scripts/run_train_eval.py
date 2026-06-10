"""Run the proxy or validation split N times to measure a baseline distribution.

The benchmark is stochastic (LLM user-sim + provider non-determinism even at
temperature 0), so a single pass is one noisy sample. Each ``--repeats`` is an
independent run with its own seed, its own ``run_id``, and its own row in
``experiments/results.csv`` — so the Arm A baseline is reported as **mean ± std**
across repeats, not a single point (systems_design.md §"Milestone 1 repeat
protocol").

Milestone 1 fixed seed schedule:
    proxy       (iterator's training signal):
        python -m scripts.run_train_eval --split proxy --repeats 5 --seed-start 1001
    validation  (blind post-hoc overfitting check — run once, after optimization):
        python -m scripts.run_train_eval --split validation --repeats 5 --seed-start 2001

Seeds for a run are ``seed_start, seed_start+1, ..., seed_start+repeats-1``.
"""

from __future__ import annotations

import argparse
import logging
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence

from benchmark.adapter import EvalResult, run_eval
from benchmark.splits import load_split
from results.logger import finalize_run, log_task, start_run
from settings import config
from target_agent.traces.langfuse_setup import flush_tracing, init_tracing

logger = logging.getLogger(__name__)

TRAIN_SPLITS = ("proxy", "validation")


@dataclass(frozen=True)
class RepeatMetrics:
    """Headline numbers for one repeat (the authoritative row is in results.csv)."""

    run_id: str
    seed: int
    pass_rate: float
    cost_per_successful_task: Optional[float]


def run_repeats(
    split: str,
    *,
    seed_start: int,
    repeats: int,
    domain: Optional[str] = None,
) -> List[RepeatMetrics]:
    """Run ``split`` ``repeats`` times with consecutive seeds; return per-repeat metrics."""
    domain = domain or config.DEFAULT_DOMAIN
    task_ids = load_split(split)
    logger.info(
        "%s baseline: %d task(s) x %d repeat(s), seeds %d..%d, domain=%s",
        split, len(task_ids), repeats, seed_start, seed_start + repeats - 1, domain,
    )

    init_tracing()
    metrics: List[RepeatMetrics] = []
    try:
        for offset in range(repeats):
            seed = seed_start + offset
            metrics.append(_run_one_repeat(split, task_ids, domain=domain, seed=seed))
    finally:
        flush_tracing()

    _log_distribution(split, metrics)
    return metrics


def _run_one_repeat(
    split: str, task_ids: Sequence[str], *, domain: str, seed: int
) -> RepeatMetrics:
    """Run every task once at ``seed``, log the run, return its headline metrics."""
    run = start_run(split)
    results: List[EvalResult] = []
    for task_id in task_ids:
        result = run_eval(task_id, split=split, domain=domain, seed=seed)
        log_task(run, result)
        results.append(result)
    finalize_run(run, results)

    pass_rate = sum(1 for r in results if r.passed) / len(results) if results else 0.0
    cost_per_success = _cost_per_successful_task(results)
    logger.info(
        "  run %s (seed=%d): pass_rate=%.3f cost_per_success=%s",
        run.run_id, seed, pass_rate, _fmt(cost_per_success),
    )
    return RepeatMetrics(run.run_id, seed, pass_rate, cost_per_success)


def _cost_per_successful_task(results: Sequence[EvalResult]) -> Optional[float]:
    """Total agent cost divided by passed-task count; None if unknown or zero passes."""
    costs = [r.agent_cost for r in results if r.agent_cost is not None]
    num_passed = sum(1 for r in results if r.passed)
    if not costs or not num_passed:
        return None
    return sum(costs) / num_passed


def _log_distribution(split: str, metrics: Sequence[RepeatMetrics]) -> None:
    """Print the mean ± std baseline across repeats (the Arm A headline)."""
    if not metrics:
        logger.warning("No repeats completed for %s.", split)
        return
    pass_rates = [m.pass_rate for m in metrics]
    costs = [m.cost_per_successful_task for m in metrics if m.cost_per_successful_task is not None]

    logger.info("=" * 60)
    logger.info("%s baseline over %d repeat(s):", split, len(metrics))
    logger.info("  pass_rate:                %s", _mean_std(pass_rates))
    logger.info("  cost_per_successful_task: %s", _mean_std(costs))
    logger.info("  per-run rows preserved in experiments/results.csv")
    logger.info("=" * 60)


def _mean_std(values: Sequence[float]) -> str:
    """Format a sample as ``mean ± std (n=N)``; std is 0 for a single sample."""
    if not values:
        return "n/a (no cost data)"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f} (n={len(values)})"


def _fmt(value: Optional[float]) -> str:
    return f"{value:.6f}" if value is not None else "n/a"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a proxy/validation baseline as a mean ± std distribution."
    )
    parser.add_argument("--split", required=True, choices=TRAIN_SPLITS)
    parser.add_argument(
        "--seed-start",
        type=int,
        required=True,
        help="First seed; repeats use seed_start..seed_start+repeats-1 (M1: proxy 1001, validation 2001).",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats (M1 baseline: 5).")
    parser.add_argument("--domain", default=None, help="TAU2 domain (default: settings.config.DEFAULT_DOMAIN).")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_repeats(
        args.split,
        seed_start=args.seed_start,
        repeats=args.repeats,
        domain=args.domain,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
