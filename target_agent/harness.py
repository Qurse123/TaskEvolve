"""Context-building helpers for TaskEvolveAgent.

The iterator (Milestone 2+) may modify this file to add:
- History compression when context grows large
- Tool list filtering / description rewriting
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

    M1: pass-through — system + history with no compression.
    Iterator can add sliding-window or summary compression here.
    """
    return [*system_messages, *history]


def filter_tools(tools: list[Tool]) -> list[Tool]:
    """Return the tool list to expose to the LLM.

    M1: pass-through — all tools exposed unchanged.
    Iterator can narrow the set or rewrite descriptions here.
    """
    return list(tools)
