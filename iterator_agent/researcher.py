"""Editor — the iterator's only LLM role (experiment.md §8.5).

Two steps, each its own Jinja template:
  1. ``diagnose.j2``     — turn a :class:`~iterator_agent.feedback.FeedbackSummary`
                           into a failure analysis.
  2. ``propose_edit.j2`` — turn that analysis into exactly ONE concrete change to
                           ONE allowed file (full-file replacement) as JSON.

The LLM call is injected (``complete``) so the loop is testable at zero cost and
the model is swappable. The editor only *proposes*; ``edit_guard`` is the
authoritative path check, applied by the orchestrator before any write.
"""

from __future__ import annotations
import litellm
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple, Union

import jinja2

from iterator_agent.edit_guard import EditPolicy, load_policy
from iterator_agent.feedback import FeedbackSummary
from settings import config

if TYPE_CHECKING:
    from iterator_agent.hypothesis import Ticket
    from iterator_agent.iteration_log import IterationRecord

# A pluggable text-completion call: prompt in, model response out.
CompletionFn = Callable[[str], str]

_REQUIRED_KEYS = ("target_file", "new_content", "change_summary", "reason_for_change")

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass(frozen=True)
class ProposedEdit:
    """One concrete change to one allowed file (full-file replacement)."""

    target_file: str
    new_content: str
    change_summary: str
    reason_for_change: str


def render_diagnose(
    feedback: FeedbackSummary, history: Sequence["IterationRecord"] = ()
) -> str:
    """Render the failure-analysis prompt from a feedback summary + prior changes.

    ``history`` is every earlier iteration's record (accepted + rejected) this run,
    so the editor can avoid re-proposing rejected ideas and build on accepted ones.
    """
    return _JINJA.get_template("diagnose.j2").render(s=feedback, history=history).strip()


def render_propose(diagnosis: str, allowed_files: List[Tuple[str, str]]) -> str:
    """Render the propose-edit prompt from a diagnosis + the editable files."""
    return (
        _JINJA.get_template("propose_edit.j2")
        .render(diagnosis=diagnosis, allowed_files=allowed_files)
        .strip()
    )


def _strip_fences(raw: str) -> str:
    """Remove a leading ```/```json fence and trailing ``` if present."""
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]  # drop the opening fence line
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_proposal(raw: str) -> ProposedEdit:
    """Parse the editor's JSON response into a :class:`ProposedEdit`.

    Raises:
        ValueError: If the text is not valid JSON or a required key is missing.
    """
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"editor response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("editor response JSON must be an object")
    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"editor proposal missing keys: {', '.join(missing)}")
    return ProposedEdit(
        target_file=str(data["target_file"]),
        new_content=str(data["new_content"]),
        change_summary=str(data["change_summary"]),
        reason_for_change=str(data["reason_for_change"]),
    )


def _read_allowed_files(policy: EditPolicy, repo_root: Path) -> List[Tuple[str, str]]:
    """Read the current contents of every allowed file (missing → empty string)."""
    files: List[Tuple[str, str]] = []
    for rel in sorted(policy.allowed_paths):
        path = repo_root / rel
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        files.append((rel, content))
    return files


# A raw completion returns the provider's response object (LiteLLM-shaped):
# ``.choices[0].message.content`` plus ``._hidden_params["response_cost"]``.
RawCompletionFn = Callable[[str], Any]


def _litellm_raw(prompt: str) -> Any:
    """Call LiteLLM with the configured iterator model; return the raw response."""
    model = config.ITERATOR_MODEL
    if not model:
        raise RuntimeError(
            "ITERATOR_MODEL is not set. Define it in .env (see .env.example), "
            "e.g. ITERATOR_MODEL=claude-opus-4-8"
        )
    return litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        # The editor model is swappable via .env; reasoning models (o-series)
        # reject temperature — drop_params strips what the model can't take.
        drop_params=True,
    )


def _content_of(response: Any) -> str:
    """Extract the assistant text from a LiteLLM-shaped response."""
    return response.choices[0].message.content or ""  # type: ignore[union-attr]


def _response_cost(response: Any) -> float:
    """Read LiteLLM's per-call ``response_cost`` (USD); 0.0 when unavailable."""
    hidden = getattr(response, "_hidden_params", None)
    cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
    return float(cost) if cost else 0.0


class CostTrackingCompletion:
    """A ``str -> str`` completion that sums LiteLLM ``response_cost`` across calls.

    This is how the iterator's **search cost** is measured (experiment.md §15.3):
    the orchestrator uses one instance for an iteration's editor calls, then reads
    :attr:`total_cost_usd`. The raw call is injectable so tests run at $0.
    """

    def __init__(self, raw: Optional[RawCompletionFn] = None) -> None:
        self._raw = raw if raw is not None else _litellm_raw
        self.total_cost_usd = 0.0

    def __call__(self, prompt: str) -> str:
        response = self._raw(prompt)
        self.total_cost_usd += _response_cost(response)
        return _content_of(response)


def _default_complete(prompt: str) -> str:
    """Default LLM call: LiteLLM with the configured iterator model (no cost tracking)."""
    return _content_of(_litellm_raw(prompt))


def _ticket_focus(ticket: "Ticket") -> str:
    """Render a hypothesis ticket as the propose step's focus block (no LLM call).

    When the backlog supplies a ticket, the ticket *is* the diagnosis, so the separate
    diagnose LLM call is skipped and this text takes its place at the top of the propose
    prompt — pinning the edit to the ticket's single target surface.
    """
    return (
        f"Focus on this hypothesis (ticket {ticket.ticket_id}): {ticket.hypothesis}\n"
        f"Rationale: {ticket.rationale}\n"
        f"Expected effect: {ticket.expected_effect}\n"
        f"You MUST edit exactly this file: {ticket.target_surface}"
    )


def run_editor(
    feedback: Optional[FeedbackSummary],
    *,
    policy: Optional[EditPolicy] = None,
    complete: Optional[CompletionFn] = None,
    repo_root: Union[str, Path] = Path("."),
    history: Sequence["IterationRecord"] = (),
    ticket: Optional["Ticket"] = None,
) -> ProposedEdit:
    """Propose exactly one edit to one allowed file.

    With a ``ticket`` (from the hypothesis backlog), the ticket is the diagnosis: the
    diagnose LLM call is skipped and the propose step is pinned to the ticket's single
    target surface. Without a ticket, fall back to the free-form diagnose -> propose
    path. ``history`` (prior accepted + rejected records) is shown so the editor does
    not repeat rejected ideas and can build on accepted ones.
    """
    policy = policy if policy is not None else load_policy()
    complete = complete if complete is not None else _default_complete
    all_files = _read_allowed_files(policy, Path(repo_root))

    if ticket is not None:
        diagnosis = _ticket_focus(ticket)
        allowed_files = [(p, c) for (p, c) in all_files if p == ticket.target_surface]
    else:
        if feedback is None:
            raise ValueError("run_editor requires feedback when no ticket is given")
        diagnosis = complete(render_diagnose(feedback, history))
        allowed_files = all_files

    raw = complete(render_propose(diagnosis, allowed_files))
    return parse_proposal(raw)
