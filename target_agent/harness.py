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

    M2 tweak (ticket t4): only include the verbose system messages on the very
    first generation call. After the first turn, the model already has those
    instructions in context, so re-sending them wastes tokens.
    """
    # If this is the first turn (no prior user/assistant messages), include
    # the system prompts; otherwise omit them to save tokens.
    if len(history) == 0:
        return [*system_messages]
    return list(history)


def filter_tools(tools: list[Tool]) -> list[Tool]:
    """Return the tool list to expose to the LLM.

    M1: pass-through — all tools exposed unchanged.
    Iterator can narrow the set or rewrite descriptions here.
    """
    return list(tools)
