"""Langfuse tracing setup for TaskEvolve.

The agent's model calls run through LiteLLM, so this module enables LiteLLM's
Langfuse callback when ``USE_LANGFUSE`` is true.

TaskEvolve uses ``LANGFUSE_BASE_URL`` for the Langfuse SDK. LiteLLM's
OpenTelemetry callback reads ``LANGFUSE_HOST``, so we mirror BASE_URL into HOST
when HOST is not already set. ``flush_tracing()`` should be called before
short-lived scripts exit so buffered traces are sent.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import litellm
from langfuse import Langfuse, get_client

# Importing config triggers load_dotenv() before get_client() reads the env.
from settings import config

logger = logging.getLogger(__name__)

# Legacy "langfuse" callback targets the removed v2 API; "langfuse_otel" is the
# OpenTelemetry callback compatible with the installed v3+ SDK.
_LITELLM_CALLBACK = "langfuse_otel"
_ENV_BASE_URL = "LANGFUSE_BASE_URL"
_ENV_HOST = "LANGFUSE_HOST"


def _resolved_host() -> Optional[str]:
    return os.environ.get(_ENV_BASE_URL) or os.environ.get(_ENV_HOST)


def _bridge_host_env() -> None:
    # Without HOST, the LiteLLM callback defaults to US cloud — fill it from
    # BASE_URL so the callback and the SDK target the same region.
    host = os.environ.get(_ENV_BASE_URL)
    if host and not os.environ.get(_ENV_HOST):
        os.environ[_ENV_HOST] = host


def _enable_litellm_callback() -> None:
    callbacks = list(litellm.success_callback or [])
    if _LITELLM_CALLBACK not in callbacks:
        callbacks.append(_LITELLM_CALLBACK)
        litellm.success_callback = callbacks


class LangfuseTracer:
    """Owns the Langfuse client lifecycle and LiteLLM callback registration."""

    def __init__(self) -> None:
        self._client: Optional[Langfuse] = None
        self._initialized = False

    def init(self, *, verify: bool = True) -> Optional[Langfuse]:
        """Enable tracing once per process; no-op when ``USE_LANGFUSE`` is false.

        Args:
            verify: If True, ping Langfuse with ``auth_check()`` and log the
                result. Auth/connection failures are logged, not raised —
                observability must never break an evaluation run.

        Returns:
            The initialized ``Langfuse`` client, or ``None`` when tracing is off.
        """
        if not config.USE_LANGFUSE:
            logger.info("Langfuse tracing disabled (USE_LANGFUSE is not set).")
            return None

        if self._initialized:
            return self._client

        _bridge_host_env()
        self._client = get_client()
        _enable_litellm_callback()
        self._initialized = True

        host = _resolved_host() or "default (US cloud)"
        logger.info(
            "Langfuse tracing enabled via LiteLLM '%s' -> %s", _LITELLM_CALLBACK, host
        )

        if verify:
            self.verify_auth()

        return self._client

    def verify_auth(self) -> bool:
        """Verify credentials via ``auth_check()``. Fail-soft: never raises."""
        if self._client is None:
            logger.warning("Langfuse not initialized; call init_tracing() first.")
            return False
        try:
            ok = self._client.auth_check()
        except Exception as exc:  # noqa: BLE001 — observability must not crash a run.
            logger.warning("Langfuse auth_check failed: %s", exc)
            return False
        if not ok:
            logger.warning("Langfuse auth_check returned False — check API keys/host.")
        return ok

    @property
    def client(self) -> Optional[Langfuse]:
        return self._client

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()


# Single shared tracer. The reference is constant (never rebound), so the
# module-level functions below delegate to it without a `global` statement.
_tracer = LangfuseTracer()


def init_tracing(*, verify: bool = True) -> Optional[Langfuse]:
    return _tracer.init(verify=verify)


def verify_auth() -> bool:
    return _tracer.verify_auth()


def get_langfuse_client() -> Optional[Langfuse]:
    return _tracer.client


def flush_tracing() -> None:
    _tracer.flush()
