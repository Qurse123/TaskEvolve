"""Unit tests for the iterator's editor (the only LLM role).

The LLM call is injected so these tests run with no API cost: a fake `complete`
returns the diagnosis text, then the proposal JSON.

Run from the repo root:
    python -m pytest tests/test_researcher.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from iterator_agent.edit_guard import EditPolicy
from iterator_agent.feedback import FeedbackSummary, TaskRecord
from iterator_agent.researcher import (
    CostTrackingCompletion,
    ProposedEdit,
    parse_proposal,
    render_diagnose,
    render_propose,
    run_editor,
)

_FAIL_7 = TaskRecord("retail_7", reward=0.0, passed=False, cost_usd=0.2, termination_reason="max_steps", seed=1, turn_count=30, tool_call_count=12)
_FAIL_9 = TaskRecord("retail_9", reward=0.0, passed=False, cost_usd=0.3, termination_reason="max_steps", seed=1, turn_count=40, tool_call_count=15)
_PASS_3 = TaskRecord("retail_3", reward=1.0, passed=True, cost_usd=0.05, termination_reason="user_stop", seed=1, turn_count=8, tool_call_count=3)

FEEDBACK = FeedbackSummary(
    run_id="proxy_20260101_000000",
    split="proxy",
    num_tasks=3,
    num_passed=1,
    num_failed=2,
    failed_tasks=(_FAIL_7, _FAIL_9),
    termination_reason_counts=(("max_steps", 2),),
    tasks=(_PASS_3, _FAIL_7, _FAIL_9),
    total_cost_usd=0.55,
    cost_per_successful_task=0.55,
    expensive_digests=(("retail_9", "assistant -> tool_call: get_order_details\ntool: not found"),),
)

_VALID_JSON = (
    '{"target_file": "target_agent/prompts/system_prompt.j2", '
    '"new_content": "NEW PROMPT", '
    '"change_summary": "tighten step budget", '
    '"reason_for_change": "tasks hit max_steps"}'
)


class _FakeLLM:
    """Returns queued responses in order; records the prompts it was given."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_parse_proposal_parses_clean_json():
    edit = parse_proposal(_VALID_JSON)

    assert isinstance(edit, ProposedEdit)
    assert edit.target_file == "target_agent/prompts/system_prompt.j2"
    assert edit.new_content == "NEW PROMPT"
    assert edit.change_summary == "tighten step budget"
    assert edit.reason_for_change == "tasks hit max_steps"


def test_parse_proposal_strips_markdown_fences():
    fenced = "```json\n" + _VALID_JSON + "\n```"

    edit = parse_proposal(fenced)

    assert edit.target_file == "target_agent/prompts/system_prompt.j2"


def test_parse_proposal_raises_on_missing_key():
    incomplete = '{"target_file": "target_agent/harness.py", "new_content": "x"}'

    with pytest.raises(ValueError):
        parse_proposal(incomplete)


def test_parse_proposal_raises_on_non_json():
    with pytest.raises(ValueError):
        parse_proposal("not json at all")


def test_render_diagnose_includes_run_and_failures():
    text = render_diagnose(FEEDBACK)

    assert "proxy_20260101_000000" in text
    assert "retail_7" in text
    assert "max_steps" in text


def test_render_diagnose_includes_cost_table_and_digest():
    text = render_diagnose(FEEDBACK)

    # Cost-centric framing + per-task cost levers across all tasks (incl. the pass).
    assert "cost per successful task" in text
    assert "retail_3" in text
    assert "turns=30" in text
    # The costliest task's transcript digest is shown to the editor.
    assert "get_order_details" in text


def test_render_propose_includes_diagnosis_and_allowed_file():
    allowed_files = [("target_agent/harness.py", "OLD HARNESS CODE")]

    text = render_propose("DIAGNOSIS HERE", allowed_files)

    assert "DIAGNOSIS HERE" in text
    assert "target_agent/harness.py" in text
    assert "OLD HARNESS CODE" in text
    assert "json" in text.lower()


def test_run_editor_returns_proposed_edit(tmp_path):
    # A temp repo with one editable file the editor can rewrite.
    target = tmp_path / "target_agent" / "prompts" / "system_prompt.j2"
    target.parent.mkdir(parents=True)
    target.write_text("OLD PROMPT", encoding="utf-8")
    policy = EditPolicy(
        allowed_paths=frozenset({"target_agent/prompts/system_prompt.j2"}),
        forbidden_prefixes=(),
    )
    fake = _FakeLLM(["DIAGNOSIS", _VALID_JSON])

    edit = run_editor(FEEDBACK, policy=policy, complete=fake, repo_root=tmp_path)

    assert edit.new_content == "NEW PROMPT"
    # Two LLM calls: diagnose then propose; the propose prompt saw the old file.
    assert len(fake.prompts) == 2
    assert "OLD PROMPT" in fake.prompts[1]


def _fake_response(content: str, cost):
    """A LiteLLM-shaped response object: message content + hidden cost param."""
    hidden = {"response_cost": cost} if cost is not None else {}
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], _hidden_params=hidden)


def test_cost_tracking_completion_accumulates_response_cost():
    raw = iter([_fake_response("diagnosis", 0.01), _fake_response("proposal", 0.02)])
    complete = CostTrackingCompletion(raw=lambda prompt: next(raw))

    first = complete("p1")
    second = complete("p2")

    assert (first, second) == ("diagnosis", "proposal")
    assert complete.total_cost_usd == pytest.approx(0.03)


def test_cost_tracking_completion_treats_missing_cost_as_zero():
    complete = CostTrackingCompletion(raw=lambda prompt: _fake_response("text", None))

    assert complete("p") == "text"
    assert complete.total_cost_usd == 0.0
