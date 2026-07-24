"""Model selection for TaskEvolveAgent.

The iterator (Milestone 2+) may modify this file to add:
- Cheap/strong routing based on turn count or confidence
- Per-task-type model selection
"""

from __future__ import annotations


def get_model(configured_model: str, history=None) -> str:
    """Return the model to use for the current generation call.

    ``history`` is the conversation so far (the agent passes ``state.messages``),
    so a routing policy can branch on context — e.g. a cheap model for simple
    early turns, a strong model for complex later ones. This baseline is the
    identity: it ignores ``history`` and always returns the configured model.
    The iterator (Exp 2) may rewrite this into a cost-aware routing policy over
    the allowed model pool.
    """
    return configured_model
