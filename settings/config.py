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

# --- Evaluation ---
# Reward at or above which a task counts as a pass (design §3.3).
PASS_THRESHOLD = 0.5

# --- Run defaults ---
# Default TAU2 domain for proxy/validation splits; the smoke runner uses "mock".
DEFAULT_DOMAIN = "retail"
# Cap on conversation turns; None uses TAU2's default.
MAX_STEPS = None
# Cap on consecutive tool errors; None uses TAU2's default.
MAX_ERRORS = None
