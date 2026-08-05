"""Arm D serving wiring — export a trained Tinker LoRA checkpoint to the HF
Hub, then point the existing eval seam at the Together-served adapter exactly
like Arm C (openai/ + api_base, never the together_ai/ provider — it drops
tool schemas; see settings/pricing.py's header and Arm C notes).

Verified chain (docs/superpowers/notes/2026-08-03-arm-d-tinker-serving-decision.md
§4, resolved 2026-08-04):
    1. train() (arm_d/train.py) -> tinker://<run-id>/sampler_weights/<step>
    2. export_adapter_to_hf(...) -> `tinker checkpoint push-hf` -> HF repo
       (standard PEFT LoRA adapter; needs `huggingface_hub` + `hf auth login`
       at real run time — a user/paid step, never invoked by these tests)
    3. Together console upload (USER step, documented only — not this module's
       job; see the decision note and YAGNI note below)
    4. resolve_tuned_agent_model(...) -> the AGENT_MODEL/AGENT_API_BASE pair
       for the eval seam (benchmark/adapter.py::_agent_llm_args), plus the
       pricing key for settings/pricing.py::register_tuned_inkling.

No Together REST client here by design (SDK not installed; the console upload
is a paid, account-specific user step — automating it is out of scope/YAGNI).
"""
from __future__ import annotations

import subprocess
from typing import Callable, Optional

TOGETHER_API_BASE = "https://api.together.xyz/v1"

# argv -> None (or any return value; ignored). Defaults to a real subprocess
# call; injectable so tests never shell out.
Runner = Callable[[list[str]], object]


def _default_runner(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def export_adapter_to_hf(
    checkpoint_path: str,
    repo_id: str,
    *,
    public: bool = False,
    runner: Optional[Runner] = None,
) -> str:
    """Publish a Tinker checkpoint's LoRA adapter to the HF Hub.

    Runs `tinker checkpoint push-hf <checkpoint_path> --repo <repo_id>`
    (`--public` appended when `public=True`), which publishes a standard PEFT
    LoRA adapter (`adapter_config.json` + `adapter_model.safetensors` +
    `checkpoint_complete`; `base_model_name_or_path` auto-patched). `runner`
    defaults to a real `subprocess.run(..., check=True)` call; inject a fake
    to keep tests at $0. Returns `repo_id` (the published HF repo).
    """
    if not checkpoint_path.startswith("tinker://"):
        raise ValueError(
            f"checkpoint_path must be a tinker:// URI, got: {checkpoint_path!r}"
        )
    argv = ["tinker", "checkpoint", "push-hf", checkpoint_path, "--repo", repo_id]
    if public:
        argv.append("--public")
    run = runner if runner is not None else _default_runner
    run(argv)
    return repo_id


def resolve_tuned_agent_model(served_id: str) -> dict:
    """Map a Together-served adapter id to the eval seam's AGENT_MODEL/
    AGENT_API_BASE pair (Arm C's exact seam: `openai/` + api_base, never
    `together_ai/` — it drops tool schemas).

    `served_id` is the bare id Together assigns on console upload (the
    `model_name` returned there) and is also the pricing key: litellm strips
    the known `openai/` prefix on registration and responses report `model`
    as the bare id — consistent with `settings.pricing.INKLING_MODEL_IDS`.
    """
    return {
        "agent_model": f"openai/{served_id}",
        "agent_api_base": TOGETHER_API_BASE,
        "pricing_key": served_id,
    }
