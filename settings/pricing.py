"""Register per-token pricing for models LiteLLM's built-in table doesn't know."""
from __future__ import annotations

import os

import litellm

INKLING_INPUT_COST_PER_TOKEN = 1.0e-6
INKLING_OUTPUT_COST_PER_TOKEN = 4.05e-6


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

INKLING_SMALL_INPUT_COST_PER_TOKEN = 0.5e-6
INKLING_SMALL_OUTPUT_COST_PER_TOKEN = 1.2e-6


def register_inkling_small(model_id: str) -> None:
    """Register an Inkling-Small-served id (base or LoRA-tuned — the shim
    routes both through the same ``openai/`` provider seam, see
    arm_d/serving_shim.py) at Inkling-Small's per-token rate."""
    litellm.register_model(
        {
            model_id: {
                "input_cost_per_token": INKLING_SMALL_INPUT_COST_PER_TOKEN,
                "output_cost_per_token": INKLING_SMALL_OUTPUT_COST_PER_TOKEN,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )


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

    tuned_model_id = os.environ.get("ARM_D_TUNED_MODEL_ID")
    if tuned_model_id:
        register_tuned_inkling(tuned_model_id)

    register_inkling_small("armd-inkling-small-tuned")
    register_inkling_small("armd-inkling-small-retail-tuned")  # A3 retail-only adapter
    register_inkling_small("thinkingmachines/Inkling-Small")
