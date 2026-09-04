"""Context-building helpers for TaskEvolveAgent.

Assembles the message list and the tool list handed to the model on each turn.
"""

from __future__ import annotations

from typing import Sequence

from tau2.data_model.message import Message, SystemMessage
from tau2.environment.tool import Tool


def build_messages(
    system_messages: list[SystemMessage],
    history: Sequence[Message],
) -> list[Message]:
    """Return the full message list to pass to generate().

    Static Arm A v0.1 harness: pass-through — system + history with no
    compression, so the system prompt/policy is present on every turn. This is
    the frozen baseline the open-weight arms (C/D) run against; the Arm B iterator
    variant (v0.2, drop-system-after-turn-1) lives on its own branch.
    """
    return [*system_messages, *history]


def filter_tools(tools: list[Tool]) -> list[Tool]:
    """Return the tool list to expose to the LLM.

    M1: pass-through — all tools exposed unchanged.
    Iterator can narrow the set or rewrite descriptions here.
    """
    return list(tools)
