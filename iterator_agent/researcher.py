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
from typing import Callable, List, Optional, Tuple, Union

import jinja2

from iterator_agent.edit_guard import EditPolicy, load_policy
from iterator_agent.feedback import FeedbackSummary
from settings import config

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


def render_diagnose(feedback: FeedbackSummary) -> str:
    """Render the failure-analysis prompt from a feedback summary."""
    return _JINJA.get_template("diagnose.j2").render(s=feedback).strip()


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


def _default_complete(prompt: str) -> str:
    """Default LLM call: LiteLLM with the configured iterator model."""
    model = config.ITERATOR_MODEL
    if not model:
        raise RuntimeError(
            "ITERATOR_MODEL is not set. Define it in .env (see .env.example), "
            "e.g. ITERATOR_MODEL=claude-opus-4-8"
        )


    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""  # type: ignore[union-attr]


def run_editor(
    feedback: FeedbackSummary,
    *,
    policy: Optional[EditPolicy] = None,
    complete: Optional[CompletionFn] = None,
    repo_root: Union[str, Path] = Path("."),
) -> ProposedEdit:
    """Diagnose the failures, then propose exactly one edit to one allowed file."""
    policy = policy if policy is not None else load_policy()
    complete = complete if complete is not None else _default_complete

    diagnosis = complete(render_diagnose(feedback))
    allowed_files = _read_allowed_files(policy, Path(repo_root))
    raw = complete(render_propose(diagnosis, allowed_files))
    return parse_proposal(raw)
