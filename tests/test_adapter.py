"""Tests for benchmark.adapter cost-lever extraction + transcript persistence.

These cover only the pure helpers (no TAU2 environment or API calls), so they run
at $0 against a fake SimulationRun.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from benchmark.adapter import _agent_llm_args, _normalize, _persist_transcript
from settings import config


def _msg(tool_calls=None):
    return SimpleNamespace(tool_calls=tool_calls)


def test_agent_llm_args_default_keeps_temperature_no_endpoint(monkeypatch) -> None:
    # Closed arms A/B: no open-weight endpoint, temperature kept (TAU2 default).
    monkeypatch.setattr(config, "AGENT_API_BASE", None)
    monkeypatch.setattr(config, "AGENT_NO_TEMPERATURE", False)
    args = _agent_llm_args(None)
    assert "api_base" not in args
    assert "temperature" in args


def test_agent_llm_args_adds_openai_compatible_endpoint(monkeypatch) -> None:
    # Open-weight arms (Inkling via Together): base URL + key threaded through.
    monkeypatch.setattr(config, "AGENT_API_BASE", "https://api.together.xyz/v1")
    monkeypatch.setattr(config, "AGENT_API_KEY", "sk-test")
    monkeypatch.setattr(config, "AGENT_NO_TEMPERATURE", False)
    args = _agent_llm_args(None)
    assert args["api_base"] == "https://api.together.xyz/v1"
    assert args["api_key"] == "sk-test"


def test_agent_llm_args_drops_temperature_when_flagged(monkeypatch) -> None:
    # Frontier reference (Opus 4.8) deprecates `temperature`; the flag strips it.
    monkeypatch.setattr(config, "AGENT_API_BASE", None)
    monkeypatch.setattr(config, "AGENT_NO_TEMPERATURE", True)
    args = _agent_llm_args({"temperature": 0.0, "max_tokens": 10})
    assert "temperature" not in args
    assert args["max_tokens"] == 10


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
