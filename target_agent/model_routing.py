"""Model selection for TaskEvolveAgent.

The iterator (Milestone 2+) may modify this file to add:
- Cheap/strong routing based on turn count or confidence
- Per-task-type model selection
"""

from __future__ import annotations


def get_model(configured_model: str) -> str:
    """Return the model to use for the current generation call.

    M1: pass-through — always uses the model set at agent init time.
    Iterator can add routing logic here (e.g. gpt-4.1-mini for simple turns).
    """
    return configured_model
