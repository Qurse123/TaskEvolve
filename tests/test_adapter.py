"""Tests for benchmark.adapter cost-lever extraction + transcript persistence.

These cover only the pure helpers (no TAU2 environment or API calls), so they run
at $0 against a fake SimulationRun.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from benchmark.adapter import _normalize, _persist_transcript


def _msg(tool_calls=None):
    return SimpleNamespace(tool_calls=tool_calls)


def _sim(messages, *, reward=1.0):
    return SimpleNamespace(
        task_id="t1",
        reward_info=SimpleNamespace(reward=reward),
        termination_reason="user_stop",
        agent_cost=0.05,
        seed=1,
        messages=messages,
    )


def test_normalize_counts_turns_and_tool_calls() -> None:
    # Arrange: 3 messages, 3 tool calls across two of them.
    sim = _sim([_msg(["a", "b"]), _msg(None), _msg(["c"])])

    # Act
    result = _normalize(sim, domain="retail", split="proxy", agent_model="gpt-4.1")  # type: ignore[arg-type]

    # Assert
    assert result.turn_count == 3
    assert result.tool_call_count == 3
    assert result.passed is True


def test_normalize_handles_no_messages() -> None:
    # Arrange
    sim = _sim(None, reward=0.0)

    # Act
    result = _normalize(sim, domain="retail", split="proxy", agent_model="gpt-4.1")  # type: ignore[arg-type]

    # Assert
    assert result.turn_count == 0
    assert result.tool_call_count == 0
    assert result.passed is False


def test_persist_transcript_writes_model_dump_json(tmp_path: Path) -> None:
    # Arrange: a fake sim that serializes via Pydantic v2's model_dump_json.
    sim = SimpleNamespace(model_dump_json=lambda indent=2: '{"messages": []}')
    path = tmp_path / "task_t1_messages.json"

    # Act
    _persist_transcript(sim, path)  # type: ignore[arg-type]

    # Assert
    assert path.read_text(encoding="utf-8") == '{"messages": []}'
