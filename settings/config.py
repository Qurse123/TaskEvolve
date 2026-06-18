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

# Load .env so the model constants below resolve. Real environment variables
# win over .env values; calling this at import keeps `from settings.config
# import AGENT_MODEL` self-sufficient.
load_dotenv()

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

# --- Evaluation ---
# Reward at or above which a task counts as a pass (design §3.3).
PASS_THRESHOLD = 0.5

# --- Observability ---
# Whether to emit Langfuse traces. TaskEvolve reads this flag and wires the
# LiteLLM -> Langfuse callback in target_agent/traces/langfuse_setup.py.
USE_LANGFUSE = os.environ.get("USE_LANGFUSE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# --- Harness ---
# Version label for the editable harness surface (prompts, harness.py,
# model_routing.py). Recorded in each task log so results trace to a harness
# state. Bump when an accepted iterator change alters agent behavior.
HARNESS_VERSION = "v0.1"

# --- Run defaults ---
# Default TAU2 domain for proxy/validation splits; the smoke runner uses "mock".
DEFAULT_DOMAIN = "retail"
# Cap on conversation turns; None uses TAU2's default.
MAX_STEPS = None
# Cap on consecutive tool errors; None uses TAU2's default.
MAX_ERRORS = None
