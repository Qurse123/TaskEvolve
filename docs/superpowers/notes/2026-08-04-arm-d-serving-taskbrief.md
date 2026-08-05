# Task brief — Arm D serving wiring (path A: Tinker LoRA → Together)

**Decision resolved (2026-08-04):** serve the tuned Inkling LoRA via **Together's
OpenAI-compat endpoint** (path A). Inkling IS a supported Together base
(`thinkingmachines/inkling`) for both serverless and dedicated inference, OpenAI-compat
with native function calling (user-confirmed against their Together console). Primary
target: **serverless multi-LoRA** (per-token, reuses Arm C's exact seam); dedicated
endpoint is the documented fallback.

## The verified end-to-end chain
1. **Train** (already built): `arm_d/train.py` saves a Tinker checkpoint → `tinker://<run-id>/sampler_weights/<step>`.
2. **Export** (build this): `tinker checkpoint push-hf tinker://<run-id>/sampler_weights/<step> --repo <hf-repo>`
   publishes a **standard PEFT LoRA adapter** (`adapter_config.json` + `adapter_model.safetensors`
   + `checkpoint_complete`; base_model_name_or_path auto-patched) to the HF Hub.
   (Needs `huggingface_hub` + `hf auth login` at real run time — a user/paid step.)
3. **Deploy** (USER step, document only — do NOT guess a REST payload): in the Together
   console, Models → Upload a model → type=Adapter, base=`thinkingmachines/inkling`,
   Source URL=the HF repo (+ HF token if private). Together returns a `model_name`.
4. **Serve** (build the wiring): point the existing eval seam at it exactly like Arm C —
   `AGENT_MODEL=openai/<model_name>`, `AGENT_API_BASE=https://api.together.xyz/v1`,
   `TOGETHER_API_KEY` in env. The `together_ai/` provider drops tool schemas, so use the
   `openai/` + api_base path (see `settings/pricing.py` header + Arm C notes).

## Build (all $0, TDD, injected external calls — mirror `arm_d/train.py`'s fake-client pattern)

### `arm_d/serving.py` (new)
- `export_adapter_to_hf(checkpoint_path: str, repo_id: str, *, public: bool = False, runner=None) -> str`
  - Validate `checkpoint_path.startswith("tinker://")` (raise `ValueError` otherwise, mirroring the CLI).
  - Build argv: `["tinker", "checkpoint", "push-hf", checkpoint_path, "--repo", repo_id]` + `["--public"]` if public.
  - `runner` defaults to a `subprocess.run(..., check=True)` wrapper; injectable so tests run at $0. Return `repo_id`.
- `TOGETHER_API_BASE = "https://api.together.xyz/v1"` constant.
- `resolve_tuned_agent_model(served_id: str) -> dict` returning `{"agent_model": f"openai/{served_id}",
  "agent_api_base": TOGETHER_API_BASE, "pricing_key": served_id}` — the bare `served_id` is the pricing key
  (litellm strips the `openai/` prefix; response `model` field is bare), consistent with `INKLING_MODEL_IDS`.
- Keep it small — no Together REST client (SDK not installed; upload is the console user step above).

### `settings/pricing.py`
- Add `register_tuned_inkling(model_id: str) -> None` — registers an arbitrary served id at
  **base Inkling per-token rate** (`INKLING_INPUT/OUTPUT_COST_PER_TOKEN`, provider `openai`, mode `chat`).
  LoRA doesn't change per-token inference price; training cost is separate (experiment.md §15.3).
- In `register_pricing()`: after the existing calls, if env `ARM_D_TUNED_MODEL_ID` is set, call
  `register_tuned_inkling(os.environ["ARM_D_TUNED_MODEL_ID"])` so the *actual* served id prices nonzero.
  Keep the `TUNED_INKLING_PRICING` placeholder registration as-is (harmless).

### `settings/config.py`
- No new logic needed beyond ensuring `register_pricing()` still runs at import (it already does).
  `ARM_D_TUNED_MODEL_ID` is read inside `register_pricing()`; just confirm import order works.

### `.env.example`
- Document `ARM_D_TUNED_MODEL_ID=<together-served-adapter-id>` and the 4-step export→deploy→serve flow
  (AGENT_MODEL=openai/<served_id>, AGENT_API_BASE, TOGETHER_API_KEY).

### Update `docs/superpowers/notes/2026-08-03-arm-d-tinker-serving-decision.md`
- §4: mark path A **CHOSEN + resolved**; record the verified chain above; note the serverless-vs-dedicated
  determination happens at the (paid) console upload; correct the earlier "reuses Arm C's exact seam"
  caveat (serverless multi-LoRA = per-token seam; dedicated = GPU-hour fallback).

### Tests (all $0)
- `tests/test_arm_d_serving.py`: export builds correct argv + honors `public` + rejects non-`tinker://` paths
  (injected runner captures argv); `resolve_tuned_agent_model` returns the openai/ seam + bare pricing key.
- Extend `tests/test_pricing.py`: `register_tuned_inkling("acct/foo-lora")` makes `litellm.model_cost`
  carry it at Inkling's input/output rate under provider `openai`; and with `ARM_D_TUNED_MODEL_ID` set,
  `register_pricing()` registers it (use monkeypatch on env).

## Constraints
- **$0 only.** No training, no HF push, no Together upload, no eval. All external calls injected/mocked.
- **No commit.** Leave everything in the working tree for user review.
- TDD (RED→GREEN). Run `python -m pytest tests/ -q` before reporting; output must be pristine.
- Follow existing patterns (see `arm_d/train.py`, `settings/pricing.py`, `benchmark/adapter.py` `_agent_llm_args`).
- Do NOT touch vendor/, benchmark/splits/, benchmark/adapter.py, or the frozen harness.
- YAGNI: no Together REST client, no dedicated-endpoint automation — those are user/console steps.
