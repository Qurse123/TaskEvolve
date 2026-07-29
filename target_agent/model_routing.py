"""Model selection for TaskEvolveAgent.

The iterator (Milestone 2+) may modify this file to add:
- Cheap/strong routing based on turn count or confidence
- Per-task-type model selection
"""

from __future__ import annotations


def get_model(configured_model: str, history=None) -> str:
    """Return the model to use for the current generation call.

    Routing policy: the retail tasks are dominated by mechanical tool
    sequences (get_order_details, exchange/return flows) with explicit user
    confirmations, so the reasoning demands are low. We return a single mid
    model (sonnet-5) for the WHOLE agent. This cuts per-token price ~2.5x on
    both input and output versus opus, which multiplies directly into
    cost-per-successful-task since cost is dominated by token volume across
    many turns.
    """
    return "anthropic/claude-sonnet-5"
