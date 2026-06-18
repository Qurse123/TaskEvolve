"""Per-run structured logging for TaskEvolve evaluations.

Writes one folder per eval run under ``experiments/logs/``:

    experiments/logs/<run_id>/task_<task_id>.json   verdict per task
    experiments/logs/<run_id>/run_summary.json       aggregate metrics
    experiments/results.csv                           one row per run

``run_id`` is ``<split>_<YYYYMMDD>_<HHMMSS>``. Records are verdict-only (reward,
pass/fail, cost, identity); richer per-turn detail is not duplicated here (it can
be recovered from TAU2's SimulationRun transcript if persisted locally).

The caller owns the task loop and the list of results; this module holds no
mutable run state. Typical use:

    run = start_run("proxy")
    results = []
    for task_id in task_ids:
        result = run_eval(task_id, split="proxy")
        log_task(run, result)
        results.append(result)
    finalize_run(run, results)
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from settings import config

if TYPE_CHECKING:
    from benchmark.adapter import EvalResult

logger = logging.getLogger(__name__)

DEFAULT_LOGS_ROOT = Path("experiments/logs")
DEFAULT_RESULTS_CSV = Path("experiments/results.csv")
RUN_SUMMARY_FILENAME = "run_summary.json"
RUN_ID_TIME_FORMAT = "%Y%m%d_%H%M%S"

# Column order for results.csv; also the key set of each run summary.
SUMMARY_FIELDS = (
    "run_id",
    "split",
    "domain",
    "agent_model",
    "harness_version",
    "num_tasks",
    "num_passed",
    "pass_rate",
    "total_cost_usd",
    "cost_per_successful_task",
    "mean_reward",
    "generated_at",
)

# Filesystem-safe replacement for characters TAU2 task IDs may contain.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class RunLog:
    """Handle to one eval run's log folder. Immutable; created by start_run()."""

    run_id: str
    run_dir: Path
    split: str


def start_run(
    split: str,
    *,
    logs_root: Path = DEFAULT_LOGS_ROOT,
    now: Optional[datetime] = None,
) -> RunLog:
    """Create the log folder for a new run and return its handle.

    Args:
        split: Split label ("smoke"/"proxy"/"validation") — prefixes the run_id.
        logs_root: Parent directory for run folders.
        now: Timestamp source for the run_id; defaults to UTC now (injectable
            for tests).
    """
    if not split:
        raise ValueError("split must be a non-empty string")
    moment = now or datetime.now(timezone.utc)
    run_id = f"{split}_{moment.strftime(RUN_ID_TIME_FORMAT)}"
    run_dir = logs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Started run %s -> %s", run_id, run_dir)
    return RunLog(run_id=run_id, run_dir=run_dir, split=split)


def log_task(run: RunLog, result: "EvalResult") -> Path:
    """Write the verdict JSON for a single task. Returns the file path."""
    record = _task_record(result, run.run_id)
    path = run.run_dir / f"task_{_safe(result.task_id)}.json"
    _write_json(path, record)
    logger.info(
        "Logged task %s (reward=%.2f) -> %s", result.task_id, result.reward, path
    )
    return path


def finalize_run(
    run: RunLog,
    results: Sequence["EvalResult"],
    *,
    results_csv: Path = DEFAULT_RESULTS_CSV,
) -> Path:
    """Write run_summary.json and append the run's row to results.csv.

    Returns the path to run_summary.json.
    """
    summary = _summary_record(run, results)
    summary_path = run.run_dir / RUN_SUMMARY_FILENAME
    _write_json(summary_path, summary)
    _append_csv_row(results_csv, summary)
    logger.info(
        "Finalized run %s: %d/%d passed, summary -> %s",
        run.run_id,
        summary["num_passed"],
        summary["num_tasks"],
        summary_path,
    )
    return summary_path


def _task_record(result: "EvalResult", run_id: str) -> dict:
    """Map an EvalResult to the verdict-only task record."""
    return {
        "task_id": result.task_id,
        "domain": result.domain,
        "split": result.split,
        "harness_version": config.HARNESS_VERSION,
        "agent_model": result.agent_model,
        "reward": result.reward,
        "passed": result.passed,
        "cost_usd": result.agent_cost,
        "termination_reason": result.termination_reason,
        "seed": result.seed,
        "timestamp": result.timestamp,
        "run_id": run_id,
    }


def _summary_record(run: RunLog, results: Sequence["EvalResult"]) -> dict:
    """Aggregate a run's results into the summary / CSV row.

    Cost fields are ``None`` when no result reported a cost; cost-per-success is
    ``None`` when cost is unknown or nothing passed (avoids divide-by-zero).
    """
    num_tasks = len(results)
    num_passed = sum(1 for r in results if r.passed)
    costs = [r.agent_cost for r in results if r.agent_cost is not None]
    total_cost = sum(costs) if costs else None
    mean_reward = (sum(r.reward for r in results) / num_tasks) if num_tasks else 0.0
    pass_rate = (num_passed / num_tasks) if num_tasks else 0.0
    cost_per_success = (
        total_cost / num_passed if (total_cost is not None and num_passed) else None
    )
    first = results[0] if results else None
    return {
        "run_id": run.run_id,
        "split": run.split,
        "domain": first.domain if first else None,
        "agent_model": first.agent_model if first else None,
        "harness_version": config.HARNESS_VERSION,
        "num_tasks": num_tasks,
        "num_passed": num_passed,
        "pass_rate": round(pass_rate, 4),
        "total_cost_usd": round(total_cost, 6) if total_cost is not None else None,
        "cost_per_successful_task": (
            round(cost_per_success, 6) if cost_per_success is not None else None
        ),
        "mean_reward": round(mean_reward, 4),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _safe(task_id: str) -> str:
    """Return a filesystem-safe form of a task ID for use in a filename."""
    return _UNSAFE_FILENAME_CHARS.sub("_", task_id).strip("_") or "task"


def _write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=False), encoding="utf-8")


def _append_csv_row(results_csv: Path, summary: dict) -> None:
    """Append one run summary as a CSV row, writing the header for a new file."""
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({key: summary.get(key) for key in SUMMARY_FIELDS})
