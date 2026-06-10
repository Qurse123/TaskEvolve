"""Smoke test: run the 3 mock-domain tasks end to end to verify wiring.

This is the zero-cost gate before spending real money on the retail splits. It
exercises the full pipeline exactly as the real runners will:

    init_tracing -> start_run -> run_eval (per task) -> log_task -> finalize_run -> flush_tracing

The smoke split uses TAU2's trivial mock domain, so LLM cost is negligible. A
green run proves the agent injection, orchestrator loop, evaluator, results
logger, and Langfuse callback are all correctly wired before we touch retail.

Run from the repo root:
    python -m scripts.run_smoke
"""

from __future__ import annotations

import logging
from typing import List

from benchmark.adapter import EvalResult, run_eval
from benchmark.splits import load_split
from results.logger import finalize_run, log_task, start_run
from target_agent.traces.langfuse_setup import flush_tracing, init_tracing

logger = logging.getLogger(__name__)

SMOKE_SPLIT = "smoke"
SMOKE_DOMAIN = "mock"  # smoke task IDs come from TAU2's trivial mock domain


def run_smoke() -> int:
    """Run every smoke task, log results, print a summary. Returns the passed count."""
    task_ids = load_split(SMOKE_SPLIT)
    logger.info("Smoke split: %d mock task(s) -> %s", len(task_ids), task_ids)

    init_tracing()
    run = start_run(SMOKE_SPLIT)
    results: List[EvalResult] = []
    try:
        for task_id in task_ids:
            result = run_eval(task_id, split=SMOKE_SPLIT, domain=SMOKE_DOMAIN)
            log_task(run, result)
            results.append(result)
            logger.info(
                "  %s: reward=%.2f passed=%s cost=%s",
                result.task_id,
                result.reward,
                result.passed,
                result.agent_cost,
            )
        summary_path = finalize_run(run, results)
    finally:
        # Always flush traces, even if a task raised mid-run.
        flush_tracing()

    num_passed = sum(1 for r in results if r.passed)
    logger.info("Smoke complete: %d/%d passed", num_passed, len(results))
    logger.info("Logs:    %s", run.run_dir)
    logger.info("Summary: %s", summary_path)
    return num_passed


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
