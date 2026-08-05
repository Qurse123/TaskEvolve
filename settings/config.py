"""Project-wide runtime configuration for TaskEvolve.

Flat, immutable settings imported wherever needed:

    from settings.config import AGENT_MODEL, PASS_THRESHOLD

To change how runs behave, edit a value here — or set the model in ``.env``.
No classes live in this module: these are plain module-level constants, treated
as read-only by convention (UPPER_SNAKE_CASE). Per-run parameters that vary by
invocation (which task, which seed) are passed as function arguments, not stored
here.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from settings.pricing import register_pricing

# Load .env so the model constants below resolve. Real environment variables
# win over .env values; calling this at import keeps `from settings.config
# import AGENT_MODEL` self-sufficient.
load_dotenv()

# Bridge: the Tinker SDK reads TINKER_API_KEY, but our .env names it TINKER_KEY.
# Only fill in TINKER_API_KEY if it isn't already set, so an explicit
# TINKER_API_KEY in the real environment still wins.
if os.environ.get("TINKER_KEY") and not os.environ.get("TINKER_API_KEY"):
    os.environ.setdefault("TINKER_API_KEY", os.environ["TINKER_KEY"])

# Register per-token prices for open-weight models LiteLLM doesn't know (Inkling,
# Arms C/D). Done at config import — which every run path imports before any
# generate() call — so litellm.completion_cost prices those models instead of
# silently returning 0.0. No-op for closed models LiteLLM already prices.
register_pricing()

# --- Models (sourced from .env; never hardcoded) ---
# Closed LLM under test for the agent. Required — set AGENT_MODEL in .env.
# Validation (fail-fast) happens where a run is built, in benchmark/adapter.py.
AGENT_MODEL = os.environ.get("AGENT_MODEL")
# TAU2 user-simulator model. Optional — None uses TAU2's default.
USER_MODEL = os.environ.get("USER_MODEL")
# Model the iterator's editor (the optimizer, e.g. Claude) uses to propose
# harness edits. Separate from AGENT_MODEL — the iterator runs a strong
# closed-weight model. Required only when running the iterator (Milestone 2).
ITERATOR_MODEL = os.environ.get("ITERATOR_MODEL")
# Some editor models (e.g. Opus 4.8) reject the `temperature` param and litellm's
# drop_params won't strip it. Set ITERATOR_NO_TEMPERATURE=1 to drop temperature
# from the editor's LLM call (mirrors AGENT_NO_TEMPERATURE for the agent).
ITERATOR_NO_TEMPERATURE = os.environ.get("ITERATOR_NO_TEMPERATURE", "").strip().lower() in (
    "1", "true", "yes", "on",
)

# OpenAI-compatible endpoint for the agent model (Arms C/D). When set, the agent
# reaches its model through this base URL with AGENT_API_KEY — e.g. Together's
# https://api.together.xyz/v1 serving Inkling. This path forwards tool schemas
# intact (the together_ai provider drops them). Unset for the closed arms (A/B),
# which use their provider's default routing, so their behavior is unchanged.
AGENT_API_BASE = os.environ.get("AGENT_API_BASE")
# Key for AGENT_API_BASE; falls back to TOGETHER_API_KEY so the open-weight arms
# work with only the Together key set. None when no open-weight endpoint is used.
AGENT_API_KEY = os.environ.get("AGENT_API_KEY") or os.environ.get("TOGETHER_API_KEY")
# Some newer models (e.g. the Opus 4.8 frontier reference) reject the
# ``temperature`` param, and litellm's drop_params doesn't strip it for them. Set
# AGENT_NO_TEMPERATURE=1 to drop temperature from the agent call for those models.
AGENT_NO_TEMPERATURE = os.environ.get("AGENT_NO_TEMPERATURE", "").strip().lower() in (
    "1", "true", "yes", "on",
)

# --- Evaluation ---
# Reward at or above which a task counts as a pass (design §3.3).
PASS_THRESHOLD = 0.5

# --- Harness ---
# Version label for the editable harness surface (prompts, harness.py,
# model_routing.py). Recorded in each task log so results trace to a harness
# state. Bump when an accepted iterator change alters agent behavior. The env
# override exists for the iterator's fresh-process evals: the parent process
# tags each candidate run by exporting HARNESS_VERSION to the eval subprocess.
HARNESS_VERSION = os.environ.get("HARNESS_VERSION", "v0.1")

# --- Run defaults ---
# Default TAU2 domain for proxy/validation splits; the smoke runner uses "mock".
DEFAULT_DOMAIN = "retail"
# Cap on conversation turns; None uses TAU2's default.
MAX_STEPS = None
# Cap on consecutive tool errors; None uses TAU2's default.
MAX_ERRORS = None
