"""Model selection for TaskEvolveAgent.

The iterator (Milestone 2+) may modify this file to add:
- Cheap/strong routing based on turn count or confidence
- Per-task-type model selection
"""

from __future__ import annotations


def get_model(configured_model: str, history=None) -> str:
    """Return the model to use for the current generation call.

    Routes to gpt-4.1-mini if there is no conversation history (i.e., first turn),
    otherwise uses the configured model. This sends simple, tool-free, single-turn tasks
    to the cheaper model by default.
    """
    if history is not None and len(history) == 0:
        return "gpt-4.1-mini"
    return configured_model
