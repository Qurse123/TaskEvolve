"""Register per-token pricing for models LiteLLM's built-in table doesn't know.

LiteLLM prices every call from its built-in ``model_cost`` table. Two models in
this study aren't in it: **Inkling** (open-weight, Together, Arms C/D) and
**Fable 5** (brand-new Anthropic frontier model used as the closed reference
point). For either, ``litellm.completion_cost()`` silently returns ``0.0`` —
recording those tasks at $0 and quietly invalidating the cost-vs-success study.

Registering the prices here makes the *existing* cost path work unchanged: TAU2's
``generate()`` -> ``litellm.completion_cost`` -> ``AssistantMessage.cost`` ->
``SimulationRun.agent_cost`` -> ``results.csv``. No agent/adapter edit needed.

Prices (verify against each provider's page):
    Inkling (together.ai/models/inkling):  $1.00 / 1M in,  $4.05 / 1M out
    Fable 5 (Anthropic):                   $10.00 / 1M in,  $50.00 / 1M out
"""

from __future__ import annotations

import os

import litellm

INKLING_INPUT_COST_PER_TOKEN = 1.0e-6
INKLING_OUTPUT_COST_PER_TOKEN = 4.05e-6

# Arms C/D reach Inkling through Together's OpenAI-compatible endpoint (the
# ``openai/`` provider + api_base) — that path forwards tools intact, unlike the
# ``together_ai`` provider which drops them. Responses report ``model`` as the
# bare ``thinkingmachines/Inkling`` (litellm strips the known ``openai/`` prefix
# on registration too), so that bare id is the canonical key. It is registered
# under the ``openai`` provider because completion_cost fails on a provider
# mismatch — the provider here MUST match the call's custom_llm_provider.
INKLING_MODEL_IDS = ("thinkingmachines/Inkling",)

INKLING_PRICING = {
    model_id: {
        "input_cost_per_token": INKLING_INPUT_COST_PER_TOKEN,
        "output_cost_per_token": INKLING_OUTPUT_COST_PER_TOKEN,
        "litellm_provider": "openai",
        "mode": "chat",
    }
    for model_id in INKLING_MODEL_IDS
}

# Fable 5 — Anthropic's frontier model, used as the closed-weight reference point
# on the cost-success frontier (the "expensive high-accuracy" corner). Reached via
# the native ``anthropic`` provider (ANTHROPIC_API_KEY, no api_base). Responses
# report ``model`` as the bare ``claude-fable-5``; register that under the
# ``anthropic`` provider so completion_cost prices it.
FABLE_INPUT_COST_PER_TOKEN = 10.0e-6
FABLE_OUTPUT_COST_PER_TOKEN = 50.0e-6
FABLE_MODEL_IDS = ("claude-fable-5",)

FABLE_PRICING = {
    model_id: {
        "input_cost_per_token": FABLE_INPUT_COST_PER_TOKEN,
        "output_cost_per_token": FABLE_OUTPUT_COST_PER_TOKEN,
        "litellm_provider": "anthropic",
        "mode": "chat",
    }
    for model_id in FABLE_MODEL_IDS
}


# Tuned Inkling (Arm D) — a LoRA fine-tune of base Inkling, served behind
# AGENT_MODEL/AGENT_API_BASE once the serving path is finalized. Runtime token
# price is unchanged from base Inkling (LoRA adapters don't change per-token
# inference cost); training cost is accounted separately (experiment.md §15.3).
# The served model id isn't finalized yet, so this is a config-driven
# placeholder, not a real endpoint — override via the (future) served id before
# any real Arm D spend.
TUNED_INKLING_MODEL_IDS = ("armd-inkling-lora",)

TUNED_INKLING_PRICING = {
    model_id: {
        "input_cost_per_token": INKLING_INPUT_COST_PER_TOKEN,
        "output_cost_per_token": INKLING_OUTPUT_COST_PER_TOKEN,
        "litellm_provider": "openai",
        "mode": "chat",
    }
    for model_id in TUNED_INKLING_MODEL_IDS
}


def register_tuned_inkling(model_id: str) -> None:
    """Register an arbitrary served id (a Together-assigned tuned-adapter
    name, not known until the console upload happens — see arm_d/serving.py)
    at base Inkling's per-token rate, under the ``openai`` provider (the
    Together OpenAI-compat seam Arm D reuses from Arm C). LoRA adapters don't
    change per-token inference price; training cost is accounted separately
    (experiment.md §15.3)."""
    litellm.register_model(
        {
            model_id: {
                "input_cost_per_token": INKLING_INPUT_COST_PER_TOKEN,
                "output_cost_per_token": INKLING_OUTPUT_COST_PER_TOKEN,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )


def register_pricing() -> None:
    """Register prices LiteLLM doesn't ship with. Idempotent; safe to call at
    import and repeatedly."""
    litellm.register_model(INKLING_PRICING)
    litellm.register_model(FABLE_PRICING)
    litellm.register_model(TUNED_INKLING_PRICING)
    # The actual Arm D served id (once resolved via arm_d.serving +
    # the Together console upload) isn't known at import time, so it's read
    # from the environment. Unset -> no-op; the TUNED_INKLING_PRICING
    # placeholder above still registers harmlessly.
    tuned_model_id = os.environ.get("ARM_D_TUNED_MODEL_ID")
    if tuned_model_id:
        register_tuned_inkling(tuned_model_id)
