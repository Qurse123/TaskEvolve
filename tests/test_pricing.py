"""Tests for settings.pricing — registering open-weight model prices with LiteLLM.

Inkling (served via Together AI) is not in LiteLLM's built-in price table, so
litellm.completion_cost() returns 0.0 for it. Recording every Arm C/D task at $0
would invalidate the cost-vs-success study. These tests pin the registration so
the existing cost path (TAU2 generate() -> completion_cost -> AssistantMessage.cost)
prices Inkling correctly.

Run from the repo root:
    python -m pytest tests/test_pricing.py
"""

from __future__ import annotations

import importlib

import litellm
import pytest

from settings.pricing import (
    FABLE_INPUT_COST_PER_TOKEN,
    FABLE_OUTPUT_COST_PER_TOKEN,
    INKLING_INPUT_COST_PER_TOKEN,
    INKLING_MODEL_IDS,
    INKLING_OUTPUT_COST_PER_TOKEN,
    TUNED_INKLING_MODEL_IDS,
    register_pricing,
)


def test_register_pricing_populates_model_cost() -> None:
    register_pricing()
    for model_id in INKLING_MODEL_IDS:
        entry = litellm.model_cost[model_id]
        assert entry["input_cost_per_token"] == INKLING_INPUT_COST_PER_TOKEN
        assert entry["output_cost_per_token"] == INKLING_OUTPUT_COST_PER_TOKEN
        # Must match the call's custom_llm_provider (openai-compatible endpoint) —
        # completion_cost raises on a provider mismatch.
        assert entry["litellm_provider"] == "openai"


def test_bare_model_id_is_canonical() -> None:
    # Responses report `model` as the bare id, and litellm stores it there (the
    # known `openai/` prefix is stripped on registration). That bare id is what
    # completion_cost looks up.
    register_pricing()
    assert "thinkingmachines/Inkling" in INKLING_MODEL_IDS
    assert "thinkingmachines/Inkling" in litellm.model_cost


def test_cost_per_token_is_nonzero_and_correct() -> None:
    register_pricing()
    # The real cost path: response.model = "thinkingmachines/Inkling",
    # custom_llm_provider = "openai" (the Together OpenAI-compatible endpoint).
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="thinkingmachines/Inkling",
        custom_llm_provider="openai",
        prompt_tokens=1000,
        completion_tokens=1000,
    )
    assert prompt_cost == pytest.approx(1000 * INKLING_INPUT_COST_PER_TOKEN)
    assert completion_cost == pytest.approx(1000 * INKLING_OUTPUT_COST_PER_TOKEN)
    assert prompt_cost > 0 and completion_cost > 0


def test_fable_priced_under_anthropic_provider() -> None:
    # Fable 5 (closed frontier reference) must price via the native anthropic
    # provider: response.model = "claude-fable-5", custom_llm_provider = "anthropic".
    register_pricing()
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="claude-fable-5",
        custom_llm_provider="anthropic",
        prompt_tokens=1000,
        completion_tokens=1000,
    )
    assert prompt_cost == pytest.approx(1000 * FABLE_INPUT_COST_PER_TOKEN)
    assert completion_cost == pytest.approx(1000 * FABLE_OUTPUT_COST_PER_TOKEN)


def test_config_import_registers_pricing() -> None:
    # Importing settings.config must register pricing (so it is live before any
    # generate() call), not merely define the function. Assert the behavioral
    # contract — Inkling gets priced — which is robust to how LiteLLM normalizes
    # the stored key. Reset every id form first so the reload actually re-registers.
    for model_id in INKLING_MODEL_IDS:
        litellm.model_cost.pop(model_id, None)
    config = importlib.import_module("settings.config")
    importlib.reload(config)
    _, completion_cost = litellm.cost_per_token(
        model="thinkingmachines/Inkling",
        custom_llm_provider="openai",
        prompt_tokens=0,
        completion_tokens=1000,
    )
    assert completion_cost == pytest.approx(1000 * INKLING_OUTPUT_COST_PER_TOKEN)


def test_tuned_inkling_is_priced() -> None:
    # Arm D serves a LoRA-tuned Inkling at an id that isn't finalized (config-
    # driven placeholder, not a real endpoint). It must price nonzero at the
    # same per-token rate as base Inkling once registered.
    register_pricing()
    for mid in TUNED_INKLING_MODEL_IDS:
        entry = litellm.model_cost.get(mid)
        assert entry and entry["input_cost_per_token"] > 0
