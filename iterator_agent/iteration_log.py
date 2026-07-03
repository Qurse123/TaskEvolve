"""Per-iteration record sink — the iterator's local-debug surface (no LLM).

The orchestrator (``run_iteration.py``) assembles an :class:`IterationRecord`
holding the ``experiment.md §22`` field subset relevant to Arm B, then persists
it through this module. Two outputs per iteration:

    experiments/iterations/<iteration_id>/iteration.json   the structured record
    experiments/iterations/<iteration_id>/*                ad-hoc debug artifacts
    experiments/accepted_changes.md | rejected_changes.md  human-readable changelog

The ``iteration.json`` files (keyed by ``iteration_id``) are the source for the
per-iteration trajectory plot. This module only persists — it makes no decisions
and calls no model (the local files replace the removed Langfuse UI).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

DEFAULT_ITERATIONS_ROOT = Path("experiments/iterations")
DEFAULT_EXPERIMENTS_DIR = Path("experiments")
RECORD_FILENAME = "iteration.json"

ACCEPTED = "accepted"
REJECTED = "rejected"
_CHANGELOG_FILES = {
    ACCEPTED: "accepted_changes.md",
    REJECTED: "rejected_changes.md",
}


@dataclass(frozen=True)
class IterationRecord:
    """One iteration's structured note (the ``experiment.md §22`` Arm B subset).

    Fields not applicable to Arm B (open-weight model, specialization method,
    validation event — which stays blind during optimization) are intentionally
    omitted; they belong to later milestones and the post-hoc validation report.
    """

    iteration_id: str
    timestamp: str
    harness_version: str
    changed_surface: str
    changed_file: str
    change_summary: str
    reason_for_change: str
    proxy_seeds: Tuple[int, ...]
    proxy_task_success_before_mean: float
    proxy_task_success_before_std: float
    proxy_task_success_after_mean: float
    proxy_task_success_after_std: float
    cost_per_successful_task_before: Optional[float]
    cost_per_successful_task_after: Optional[float]
    accepted_or_rejected: str
    reason_accepted_or_rejected: str
    iterator_search_cost_usd: Optional[float] = None
    notes: str = ""
    # Provenance: which models were in play this iteration (editor = the LLM that
    # proposed the edit; agent = the task model the proxy eval ran). Default-safe.
    editor_model: str = ""
    agent_model: str = ""
    # Optimization objective (mean cost per task) before/after — the low-variance
    # signal the acceptance rule scores; cost_per_successful_task above stays the
    # reported headline.
    cost_per_task_before: Optional[float] = None
    cost_per_task_after: Optional[float] = None
    # Which hypothesis-backlog ticket drove this iteration (empty for the free-form
    # diagnose path). Surfaced on the trajectory plot so the climb is auditable.
    ticket_id: str = ""
    hypothesis: str = ""


def iteration_dir(
    iteration_id: str, *, logs_root: Union[str, Path] = DEFAULT_ITERATIONS_ROOT
) -> Path:
    """Return (creating if needed) the per-iteration folder for ``iteration_id``."""
    path = Path(logs_root) / iteration_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_record(
    record: IterationRecord, *, logs_root: Union[str, Path] = DEFAULT_ITERATIONS_ROOT
) -> Path:
    """Write ``record`` as ``<iteration_id>/iteration.json``; return its path."""
    folder = iteration_dir(record.iteration_id, logs_root=logs_root)
    path = folder / RECORD_FILENAME
    path.write_text(
        json.dumps(asdict(record), indent=2, sort_keys=False), encoding="utf-8"
    )
    return path


def write_artifact(
    iteration_id: str,
    filename: str,
    content: str,
    *,
    logs_root: Union[str, Path] = DEFAULT_ITERATIONS_ROOT,
) -> Path:
    """Write an ad-hoc debug artifact (diagnosis, prompt, proposal...) into the folder."""
    folder = iteration_dir(iteration_id, logs_root=logs_root)
    path = folder / filename
    path.write_text(content, encoding="utf-8")
    return path


def append_changelog(
    record: IterationRecord,
    *,
    experiments_dir: Union[str, Path] = DEFAULT_EXPERIMENTS_DIR,
) -> Path:
    """Append a one-line markdown bullet to accepted_ or rejected_changes.md.

    Raises:
        ValueError: If ``accepted_or_rejected`` is neither "accepted" nor "rejected".
    """
    filename = _CHANGELOG_FILES.get(record.accepted_or_rejected)
    if filename is None:
        raise ValueError(
            f"accepted_or_rejected must be {ACCEPTED!r} or {REJECTED!r}, "
            f"got {record.accepted_or_rejected!r}"
        )
    path = Path(experiments_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_changelog_line(record))
    return path


def _changelog_line(record: IterationRecord) -> str:
    """Render one changelog bullet for ``record``."""
    return (
        f"- **{record.iteration_id}** ({record.harness_version}) "
        f"`{record.changed_file}` — {record.change_summary} "
        f"_{record.reason_accepted_or_rejected}_\n"
    )
