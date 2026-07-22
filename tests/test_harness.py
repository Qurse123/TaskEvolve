"""Tests pinning the static Arm A v0.1 harness used by the open-weight arms (C/D).

Arm C/D run the *unchanged* Arm A v0.1 harness (experiment.md §13.3/§19.1): the
system prompt (domain policy) is sent on every turn, and model routing is the
identity. This differs from Arm B's v0.2 (which drops the system prompt after the
first turn). These tests lock the v0.1 contract so the open-weight arms measure
the model, not a harness variant.

Run from the repo root:
    python -m pytest tests/test_harness.py
"""

from __future__ import annotations

from target_agent.harness import build_messages, filter_tools
from target_agent.model_routing import get_model


def test_build_messages_includes_system_on_nonempty_history() -> None:
    # v0.1: the system prompt/policy is present on EVERY turn, not just the first.
    system = ["SYSTEM_POLICY"]
    history = ["user_1", "assistant_1", "user_2"]
    assert build_messages(system, history) == ["SYSTEM_POLICY", "user_1", "assistant_1", "user_2"]


def test_build_messages_first_turn() -> None:
    assert build_messages(["SYSTEM_POLICY"], []) == ["SYSTEM_POLICY"]


def test_filter_tools_is_passthrough() -> None:
    tools = [object(), object()]
    assert filter_tools(tools) == tools


def test_get_model_is_identity() -> None:
    # v0.1: no cheap-model routing; the configured (open-weight) model is used as-is.
    assert get_model("together_ai/thinkingmachines/Inkling") == "together_ai/thinkingmachines/Inkling"
    assert get_model("gpt-4.1") == "gpt-4.1"
