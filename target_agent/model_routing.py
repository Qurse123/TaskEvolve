"""Model selection for TaskEvolveAgent.

Selects the model used for each generation call.
"""

from __future__ import annotations


def get_model(configured_model: str, history=None) -> str:
    """Return the model to use for the current generation call.

    v0.1 (static harness for Arms A/C/D): identity routing — always use the
    configured AGENT_MODEL. Do not override the model here; the model axis is
    controlled entirely by AGENT_MODEL so open-weight arms actually run their
    own model.
    """
    return configured_model
