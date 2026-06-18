"""Feedback reader — the editor's input (proxy-only).

Reads the most recent **proxy** run folder under ``experiments/logs/`` and
summarizes its failures (per-task reward, termination reason, cost) into a
structured :class:`FeedbackSummary` the editor (``researcher.py``) reasons over.

This module only ever reads the split it is asked for. The orchestrator always
asks for ``"proxy"``, so the iterator never sees validation or test logs during
optimization (CLAUDE.md "Hard Constraints" #2). Per-LLM-call telemetry lives in
Langfuse and is correlated by ``task_id`` + ``run_id``, not duplicated here.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from results.logger import DEFAULT_LOGS_ROOT


@dataclass(frozen=True)
class TaskRecord:
    """One task's verdict, parsed from its ``task_*.json`` log."""

    task_id: str
    reward: float
    passed: bool
    cost_usd: Optional[float]
    termination_reason: str
    seed: Optional[int]


@dataclass(frozen=True)
class FeedbackSummary:
    """Structured failure summary for one run — the editor's context."""

    run_id: str
    split: str
    num_tasks: int
    num_passed: int
    num_failed: int
    failed_tasks: Tuple[TaskRecord, ...]
    termination_reason_counts: Tuple[Tuple[str, int], ...]


def find_latest_run_dir(
    split: str, *, logs_root: Union[str, Path] = DEFAULT_LOGS_ROOT
) -> Path:
    """Return the newest ``<split>_*`` run folder under ``logs_root``.

    Folder names embed a ``YYYYMMDD_HHMMSS`` stamp, so the lexically greatest
    name is the most recent. Only folders for the requested split are considered,
    so asking for ``"proxy"`` can never return a validation/test run.

    Raises:
        ValueError: If no run folder for ``split`` exists.
    """
    root = Path(logs_root)
    prefix = f"{split}_"
    candidates = (
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith(prefix)]
        if root.exists()
        else []
    )
    if not candidates:
        raise ValueError(f"no run folder for split={split!r} under {root}")
    return max(candidates, key=lambda d: d.name)


def _load_task(path: Path) -> TaskRecord:
    record = json.loads(path.read_text(encoding="utf-8"))
    return TaskRecord(
        task_id=record["task_id"],
        reward=float(record.get("reward", 0.0)),
        passed=bool(record.get("passed", False)),
        cost_usd=record.get("cost_usd"),
        termination_reason=str(record.get("termination_reason", "")),
        seed=record.get("seed"),
    )


def load_run_tasks(run_dir: Union[str, Path]) -> List[TaskRecord]:
    """Parse every ``task_*.json`` in ``run_dir`` (ignores ``run_summary.json``)."""
    run_dir = Path(run_dir)
    return [_load_task(p) for p in sorted(run_dir.glob("task_*.json"))]


def summarize_run(run_dir: Union[str, Path]) -> FeedbackSummary:
    """Aggregate a run folder's task verdicts into a :class:`FeedbackSummary`."""
    run_dir = Path(run_dir)
    tasks = load_run_tasks(run_dir)
    failed = tuple(t for t in tasks if not t.passed)
    reason_counts = Counter(t.termination_reason for t in failed)
    # Most common first; ties broken by reason for a stable, testable order.
    ordered = tuple(sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    split = run_dir.name.split("_", 1)[0]
    return FeedbackSummary(
        run_id=run_dir.name,
        split=split,
        num_tasks=len(tasks),
        num_passed=len(tasks) - len(failed),
        num_failed=len(failed),
        failed_tasks=failed,
        termination_reason_counts=ordered,
    )


def build_feedback(
    split: str = "proxy", *, logs_root: Union[str, Path] = DEFAULT_LOGS_ROOT
) -> FeedbackSummary:
    """Summarize the most recent run for ``split`` (defaults to proxy)."""
    return summarize_run(find_latest_run_dir(split, logs_root=logs_root))
