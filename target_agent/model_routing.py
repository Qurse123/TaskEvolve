"""Model selection for TaskEvolveAgent.

The iterator (Milestone 2+) may modify this file to add:
- Cheap/strong routing based on turn count or confidence
- Per-task-type model selection
"""

from __future__ import annotations


def get_model(configured_model: str) -> str:
    """Return the model to use for the current generation call.

    Static Arm A v0.1 harness: pass-through — always uses the model set at agent
    init time (the configured open-weight or closed model). No cheap-model
    routing. The Arm B iterator routing variant lives on its own branch.
    """
    return configured_model
