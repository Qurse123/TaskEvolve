"""Feedback reader — the editor's input (proxy-only).

Reads the most recent **proxy** run folder under ``experiments/logs/`` and
summarizes its failures (per-task reward, termination reason, cost) into a
structured :class:`FeedbackSummary` the editor (``researcher.py``) reasons over.

This module only ever reads the split it is asked for. The orchestrator always
asks for ``"proxy"``, so the iterator never sees validation or test logs during
optimization (CLAUDE.md "Hard Constraints" #2).
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from results.logger import DEFAULT_LOGS_ROOT, DEFAULT_RESULTS_CSV, _safe

# How many of the costliest tasks get a transcript digest in the feedback, plus a few
# representative FAILED tasks (success-floor risk lives there), and how far each message
# is truncated — all bound the editor's (iterator search) cost.
NUM_EXPENSIVE_DIGESTS = 3
NUM_FAILED_DIGESTS = 2
DIGEST_MAX_CHARS = 200


@dataclass(frozen=True)
class TaskRecord:
    """One task's verdict, parsed from its ``task_*.json`` log."""

    task_id: str
    reward: float
    passed: bool
    cost_usd: Optional[float]
    termination_reason: str
    seed: Optional[int]
    # Per-task cost levers (experiment.md §22); default 0 for older logs / harness errors.
    turn_count: int = 0
    tool_call_count: int = 0


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
    # Cost-centric evidence; default-empty so call sites that only set the failure
    # fields still construct cleanly (summarize_run always populates these).
    tasks: Tuple[TaskRecord, ...] = ()
    total_cost_usd: Optional[float] = None
    cost_per_successful_task: Optional[float] = None
    expensive_digests: Tuple[Tuple[str, str], ...] = ()
    # Aggregate cost levers across the run — where the search should look for waste.
    # Default-safe so hand-built summaries (tests) that omit them still construct.
    tool_usage_counts: Tuple[Tuple[str, int], ...] = ()
    mean_turns: float = 0.0
    mean_tool_calls: float = 0.0


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


def find_latest_run_dir_for_version(
    split: str,
    harness_version: str,
    *,
    logs_root: Union[str, Path] = DEFAULT_LOGS_ROOT,
    csv_path: Union[str, Path] = DEFAULT_RESULTS_CSV,
) -> Path:
    """Return the newest run folder for ``(split, harness_version)``.

    The editor must diagnose the *current-best* harness, not whichever run wrote a
    folder last (which, mid-loop, is the previous rejected candidate). ``results.csv``
    tags every run with its ``harness_version`` and ``run_id`` (== the run-dir name),
    so we pick the lexically greatest matching ``run_id`` (names embed a sortable
    ``YYYYMMDD_HHMMSS`` stamp).

    Raises:
        ValueError: If the CSV is missing or no row matches ``(split, version)``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise ValueError(f"results CSV not found: {csv_path}")
    run_ids = [
        row.get("run_id") or ""
        for row in csv.DictReader(csv_path.open(newline="", encoding="utf-8"))
        if row.get("split") == split and row.get("harness_version") == harness_version
    ]
    run_ids = [rid for rid in run_ids if rid]
    if not run_ids:
        raise ValueError(
            f"no run in {csv_path} for split={split!r} harness_version={harness_version!r}"
        )
    return Path(logs_root) / max(run_ids)


def _load_task(path: Path) -> TaskRecord:
    record = json.loads(path.read_text(encoding="utf-8"))
    return TaskRecord(
        task_id=record["task_id"],
        reward=float(record.get("reward", 0.0)),
        passed=bool(record.get("passed", False)),
        cost_usd=record.get("cost_usd"),
        turn_count=int(record.get("turn_count", 0) or 0),
        tool_call_count=int(record.get("tool_call_count", 0) or 0),
        termination_reason=str(record.get("termination_reason", "")),
        seed=record.get("seed"),
    )


def load_run_tasks(run_dir: Union[str, Path]) -> List[TaskRecord]:
    """Parse every verdict ``task_*.json`` in ``run_dir``.

    Skips the paired ``task_*_messages.json`` transcripts (raw evidence, not a
    verdict) and ``run_summary.json``.
    """
    run_dir = Path(run_dir)
    return [
        _load_task(p)
        for p in sorted(run_dir.glob("task_*.json"))
        if not p.name.endswith("_messages.json")
    ]


def _render_message(message: dict) -> str:
    """One compact transcript line: role + tool name(s) or truncated content."""
    role = message.get("role", "?")
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        names = ", ".join(str(tc.get("name", "?")) for tc in tool_calls)
        return f"{role} -> tool_call: {names}"
    content = (message.get("content") or "").strip().replace("\n", " ")
    if len(content) > DIGEST_MAX_CHARS:
        content = content[:DIGEST_MAX_CHARS] + "…"
    return f"{role}: {content}"


def _load_transcript_messages(run_dir: Path, task_id: str) -> List[dict]:
    """Return a task's persisted transcript messages ([] when none was written)."""
    path = run_dir / f"task_{_safe(task_id)}_messages.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("messages") or []


def _load_transcript_digest(run_dir: Path, task_id: str) -> Optional[str]:
    """Render a compact turn-by-turn digest of a task's transcript, if persisted."""
    messages = _load_transcript_messages(run_dir, task_id)
    if not messages:
        return None
    return "\n".join(_render_message(m) for m in messages)


def _tool_usage_counts(
    run_dir: Path, tasks: List[TaskRecord]
) -> Tuple[Tuple[str, int], ...]:
    """Count tool-call names across every task transcript (which tools dominate cost)."""
    counter: "Counter[str]" = Counter()
    for task in tasks:
        for message in _load_transcript_messages(run_dir, task.task_id):
            for call in message.get("tool_calls") or []:
                counter[str(call.get("name", "?"))] += 1
    # Most-used first; ties broken by name for a stable, testable order.
    return tuple(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _expensive_digests(
    run_dir: Path, tasks: List[TaskRecord]
) -> Tuple[Tuple[str, str], ...]:
    """Digests for the costliest tasks plus a few representative failures.

    Cost waste is not only in the priciest tasks — failures drive success-floor risk,
    so the costliest ``NUM_FAILED_DIGESTS`` failing tasks not already shown are appended.
    """
    with_cost = sorted(
        (t for t in tasks if t.cost_usd is not None),
        key=lambda t: t.cost_usd or 0.0,
        reverse=True,
    )
    digests: List[Tuple[str, str]] = []
    chosen: set = set()
    for task in with_cost[:NUM_EXPENSIVE_DIGESTS]:
        digest = _load_transcript_digest(run_dir, task.task_id)
        if digest:
            digests.append((task.task_id, digest))
            chosen.add(task.task_id)

    failing = sorted(
        (t for t in tasks if not t.passed and t.task_id not in chosen),
        key=lambda t: t.cost_usd or 0.0,
        reverse=True,
    )
    for task in failing[:NUM_FAILED_DIGESTS]:
        digest = _load_transcript_digest(run_dir, task.task_id)
        if digest:
            digests.append((task.task_id, digest))
    return tuple(digests)


def summarize_run(run_dir: Union[str, Path]) -> FeedbackSummary:
    """Aggregate a run folder into a cost-centric :class:`FeedbackSummary`.

    Covers all tasks (passing tasks waste cost too), reports total cost and
    cost-per-successful-task (the objective), and attaches transcript digests for
    the costliest tasks so the editor can target concrete waste.
    """
    run_dir = Path(run_dir)
    tasks = load_run_tasks(run_dir)
    failed = tuple(t for t in tasks if not t.passed)
    num_passed = len(tasks) - len(failed)

    reason_counts = Counter(t.termination_reason for t in failed)
    # Most common first; ties broken by reason for a stable, testable order.
    ordered = tuple(sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    costs = [t.cost_usd for t in tasks if t.cost_usd is not None]
    total_cost = sum(costs) if costs else None
    cost_per_success = (
        total_cost / num_passed if (total_cost is not None and num_passed) else None
    )
    mean_turns = statistics.mean([t.turn_count for t in tasks]) if tasks else 0.0
    mean_tool_calls = (
        statistics.mean([t.tool_call_count for t in tasks]) if tasks else 0.0
    )

    split = run_dir.name.split("_", 1)[0]
    return FeedbackSummary(
        run_id=run_dir.name,
        split=split,
        num_tasks=len(tasks),
        num_passed=num_passed,
        num_failed=len(failed),
        tasks=tuple(tasks),
        failed_tasks=failed,
        termination_reason_counts=ordered,
        total_cost_usd=round(total_cost, 6) if total_cost is not None else None,
        cost_per_successful_task=(
            round(cost_per_success, 6) if cost_per_success is not None else None
        ),
        expensive_digests=_expensive_digests(run_dir, tasks),
        tool_usage_counts=_tool_usage_counts(run_dir, tasks),
        mean_turns=mean_turns,
        mean_tool_calls=mean_tool_calls,
    )


def build_feedback(
    split: str = "proxy",
    *,
    harness_version: Optional[str] = None,
    logs_root: Union[str, Path] = DEFAULT_LOGS_ROOT,
    csv_path: Union[str, Path] = DEFAULT_RESULTS_CSV,
) -> FeedbackSummary:
    """Summarize the run the editor should diagnose (defaults to proxy).

    When ``harness_version`` is given, summarize the newest run for *that* version —
    the current-best harness — so the editor never reasons over a reverted candidate.
    Falls back to the newest run of any version when the version has no run yet (e.g.
    the first iteration off the Arm A baseline that produced no version-tagged row).
    """
    if harness_version is not None:
        try:
            run_dir = find_latest_run_dir_for_version(
                split, harness_version, logs_root=logs_root, csv_path=csv_path
            )
        except ValueError:
            run_dir = find_latest_run_dir(split, logs_root=logs_root)
    else:
        run_dir = find_latest_run_dir(split, logs_root=logs_root)
    return summarize_run(run_dir)
