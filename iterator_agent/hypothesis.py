"""Hypothesis-ticket backlog — the iterator's exploration planner (experiment.md §8.5).

Instead of one greedy proposal per iteration, the iterator reads the current-best
proxy evidence and emits a **ranked backlog of testable hypotheses** ("tickets"), each
naming exactly one allowed edit surface and the cost effect it expects. The driver then
works one ticket per iteration (edit -> proxy double-run -> accept/revert) and
regenerates the backlog as the landscape changes (hill-climb).

This is still one LLM call (injected ``complete``, so tests run at $0) and still
proposes changes only to the allowed surface — ``edit_guard`` remains the authoritative
gate in the orchestrator. Tickets whose ``target_surface`` is off-surface are dropped
here as an early, cheap filter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple, Union

import jinja2

from iterator_agent.edit_guard import EditPolicy, load_policy
from iterator_agent.feedback import FeedbackSummary
from iterator_agent.researcher import (
    CompletionFn,
    _default_complete,
    _read_allowed_files,
    _strip_fences,
)

if TYPE_CHECKING:
    from iterator_agent.iteration_log import IterationRecord

# Default backlog size — how many hypotheses to request per generation. Bounded to keep
# each ticket distinct and the iterator's search cost predictable.
TICKETS_PER_BACKLOG = 5

_REQUIRED_KEYS = ("hypothesis", "target_surface", "rationale", "expected_effect")

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass(frozen=True)
class Ticket:
    """One testable hypothesis: a lever, why, and the cost effect it should have."""

    ticket_id: str
    hypothesis: str
    target_surface: str
    rationale: str
    expected_effect: str
    priority: int  # 1 = highest expected impact; assigned from the model's ranking


def render_generate(
    feedback: FeedbackSummary,
    allowed_files: Sequence[Tuple[str, str]],
    *,
    history: Sequence["IterationRecord"] = (),
    n: int = TICKETS_PER_BACKLOG,
) -> str:
    """Render the backlog-generation prompt from feedback + the editable files."""
    return (
        _JINJA.get_template("generate_tickets.j2")
        .render(s=feedback, allowed_files=list(allowed_files), history=history, n=n)
        .strip()
    )


def parse_tickets(raw: str, *, allowed_surfaces: Iterable[str]) -> Tuple[Ticket, ...]:
    """Parse the model's JSON array into ranked Tickets, dropping off-surface ones.

    Order is preserved as the model's ranking: ``ticket_id`` and ``priority`` are
    assigned from the surviving order (t1 = highest priority).

    Raises:
        ValueError: If the text is not a JSON array, or no ticket targets an allowed
            surface (nothing workable to hand the loop).
    """
    allowed = set(allowed_surfaces)
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"ticket response is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("ticket response JSON must be an array of tickets")

    tickets: List[Ticket] = []
    for item in data:
        if not isinstance(item, dict) or any(k not in item for k in _REQUIRED_KEYS):
            continue
        target = str(item["target_surface"])
        if target not in allowed:
            continue  # off-surface hypothesis — edit_guard would reject it anyway
        index = len(tickets) + 1
        tickets.append(
            Ticket(
                ticket_id=f"t{index}",
                hypothesis=str(item["hypothesis"]),
                target_surface=target,
                rationale=str(item["rationale"]),
                expected_effect=str(item["expected_effect"]),
                priority=index,
            )
        )

    if not tickets:
        raise ValueError("no usable tickets: none targeted an allowed edit surface")
    return tuple(tickets)


def generate_tickets(
    feedback: FeedbackSummary,
    *,
    policy: Optional[EditPolicy] = None,
    complete: Optional[CompletionFn] = None,
    repo_root: Union[str, Path] = Path("."),
    history: Sequence["IterationRecord"] = (),
    n: int = TICKETS_PER_BACKLOG,
) -> Tuple[Ticket, ...]:
    """Generate a ranked backlog of hypothesis tickets from proxy feedback.

    ``complete`` is injected so the loop is testable at $0; the default wraps the
    configured iterator model. Only the allowed edit surface is offered to the model,
    and off-surface tickets are filtered out on parse.
    """
    policy = policy if policy is not None else load_policy()
    complete = complete if complete is not None else _default_complete
    allowed_files = _read_allowed_files(policy, Path(repo_root))
    raw = complete(render_generate(feedback, allowed_files, history=history, n=n))
    return parse_tickets(raw, allowed_surfaces=policy.allowed_paths)
