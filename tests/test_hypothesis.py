"""Tests for iterator_agent.hypothesis — the hypothesis-ticket backlog generator.

The generator reads a proxy FeedbackSummary + the editable-file contents and emits a
ranked backlog of testable Tickets, each naming exactly one allowed edit surface. The
LLM call is injected, so every test runs at $0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iterator_agent.edit_guard import load_policy
from iterator_agent.feedback import FeedbackSummary, TaskRecord
from iterator_agent.hypothesis import (
    Ticket,
    generate_tickets,
    parse_tickets,
    render_generate,
)

ALLOWED = "target_agent/prompts/system_prompt.j2"
ALLOWED_2 = "target_agent/model_routing.py"
FORBIDDEN = "benchmark/adapter.py"


def _feedback() -> FeedbackSummary:
    tasks = (
        TaskRecord("t1", 0.0, False, 0.20, "max_steps", 1, turn_count=18, tool_call_count=9),
        TaskRecord("t2", 1.0, True, 0.05, "user_stop", 2, turn_count=6, tool_call_count=2),
    )
    return FeedbackSummary(
        run_id="proxy_20260618_000000",
        split="proxy",
        num_tasks=2,
        num_passed=1,
        num_failed=1,
        failed_tasks=(tasks[0],),
        termination_reason_counts=(("max_steps", 1),),
        tasks=tasks,
        total_cost_usd=0.25,
        cost_per_successful_task=0.25,
    )


def _tickets_json(surfaces) -> str:
    return json.dumps(
        [
            {
                "hypothesis": f"Trim lever {i}",
                "target_surface": surface,
                "rationale": "verbose context inflates input tokens",
                "expected_effect": "lower mean cost per task, success held",
            }
            for i, surface in enumerate(surfaces)
        ]
    )


def _repo(tmp_path: Path) -> Path:
    target = tmp_path / ALLOWED
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("SYSTEM PROMPT\n", encoding="utf-8")
    return tmp_path


def test_parse_tickets_ranks_and_ids(tmp_path: Path) -> None:
    policy = load_policy()
    raw = _tickets_json([ALLOWED, ALLOWED_2])

    tickets = parse_tickets(raw, allowed_surfaces=policy.allowed_paths)

    assert [t.ticket_id for t in tickets] == ["t1", "t2"]
    assert [t.priority for t in tickets] == [1, 2]
    assert tickets[0].target_surface == ALLOWED
    assert all(isinstance(t, Ticket) for t in tickets)


def test_parse_tickets_drops_forbidden_surface(tmp_path: Path) -> None:
    policy = load_policy()
    raw = _tickets_json([FORBIDDEN, ALLOWED])

    tickets = parse_tickets(raw, allowed_surfaces=policy.allowed_paths)

    # The forbidden-surface ticket is dropped; the valid one survives and is re-ranked.
    assert [t.target_surface for t in tickets] == [ALLOWED]
    assert tickets[0].ticket_id == "t1"


def test_parse_tickets_raises_on_malformed_json() -> None:
    with pytest.raises(ValueError):
        parse_tickets("not json", allowed_surfaces={ALLOWED})


def test_parse_tickets_raises_when_no_valid_tickets() -> None:
    raw = _tickets_json([FORBIDDEN])
    with pytest.raises(ValueError):
        parse_tickets(raw, allowed_surfaces={ALLOWED})


def test_generate_tickets_uses_injected_completion(tmp_path: Path) -> None:
    policy = load_policy()
    seen = {"prompt": None}

    def complete(prompt: str) -> str:
        seen["prompt"] = prompt
        return _tickets_json([ALLOWED, ALLOWED_2])

    tickets = generate_tickets(
        _feedback(), policy=policy, complete=complete, repo_root=_repo(tmp_path), n=5
    )

    assert len(tickets) == 2
    # The prompt embedded the current editable-file contents (grounds the hypotheses).
    assert "SYSTEM PROMPT" in seen["prompt"]


def test_render_generate_lists_allowed_surfaces() -> None:
    prompt = render_generate(_feedback(), [(ALLOWED, "SYSTEM PROMPT\n")], history=(), n=5)
    assert ALLOWED in prompt
    assert "max_steps" in prompt  # feedback evidence reaches the prompt
