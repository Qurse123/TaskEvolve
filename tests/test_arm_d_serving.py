"""Tests for arm_d.serving — export-to-HF argv building + the Together
OpenAI-compat serving seam for the Arm D tuned model.

All external calls are injected: `export_adapter_to_hf` takes a `runner`
callable instead of shelling out for real, so these tests run at $0 (no
`tinker` CLI, no HF push, no Together call). Mirrors arm_d/train.py's
injected-client pattern.
"""
from __future__ import annotations

import pytest

from arm_d.serving import (
    TOGETHER_API_BASE,
    export_adapter_to_hf,
    resolve_tuned_agent_model,
)


class _CapturingRunner:
    """Fake subprocess runner: records the argv it was called with instead of
    shelling out. Mirrors the `check=True` contract of subprocess.run."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        return None


def test_export_builds_correct_argv():
    runner = _CapturingRunner()
    result = export_adapter_to_hf(
        "tinker://run-123/sampler_weights/40",
        "my-org/armd-inkling-lora",
        runner=runner,
    )
    assert result == "my-org/armd-inkling-lora"
    assert len(runner.calls) == 1
    assert runner.calls[0] == [
        "tinker",
        "checkpoint",
        "push-hf",
        "tinker://run-123/sampler_weights/40",
        "--repo",
        "my-org/armd-inkling-lora",
    ]


def test_export_honors_public_flag():
    runner = _CapturingRunner()
    export_adapter_to_hf(
        "tinker://run-123/sampler_weights/40",
        "my-org/armd-inkling-lora",
        public=True,
        runner=runner,
    )
    assert runner.calls[0] == [
        "tinker",
        "checkpoint",
        "push-hf",
        "tinker://run-123/sampler_weights/40",
        "--repo",
        "my-org/armd-inkling-lora",
        "--public",
    ]


def test_export_rejects_non_tinker_checkpoint_path():
    runner = _CapturingRunner()
    with pytest.raises(ValueError):
        export_adapter_to_hf(
            "/local/path/to/checkpoint",
            "my-org/armd-inkling-lora",
            runner=runner,
        )
    assert runner.calls == []


def test_resolve_tuned_agent_model_returns_openai_seam():
    out = resolve_tuned_agent_model("acct/armd-inkling-lora-a1b2")
    assert out == {
        "agent_model": "openai/acct/armd-inkling-lora-a1b2",
        "agent_api_base": TOGETHER_API_BASE,
        "pricing_key": "acct/armd-inkling-lora-a1b2",
    }


def test_together_api_base_constant():
    assert TOGETHER_API_BASE == "https://api.together.xyz/v1"
