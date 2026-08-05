# Arm D — Tinker SDK surface + serving decision (Task 1 spike)

**Date:** 2026-08-03 · tinker **0.24.0** + tinker_cookbook installed in `.venv` ·
key in `.env` as `TINKER_KEY` (SDK expects `TINKER_API_KEY`; config bridges it).

This note pins everything resolved **at $0 by introspecting the installed SDK**
(no paid API calls). The two items that require a live (paid) call — confirming
`thinkingmachines/Inkling` trains + the exact renderer name, and the serving path —
are marked **PAID / DEFERRED to the user** below.

## 1. Training API (pinned from the installed SDK)

```python
import tinker
sc = tinker.ServiceClient()                      # reads TINKER_API_KEY from env
tc = sc.create_lora_training_client(
        base_model="thinkingmachines/Inkling",   # PAID spike confirms this trains
        rank=32)                                 # LoraConfig rank; default 32
tok = tc.get_tokenizer()                          # -> HF PreTrainedTokenizer
```

Training step loop (all calls return `APIFuture`; call `.result()` to block):
```python
fut = tc.forward_backward(data: list[tinker.Datum], loss_fn, loss_fn_config=None)
tc.optim_step(tinker.AdamParams(...))             # Adam update
sampling_client = tc.save_weights_and_get_sampling_client(name="armd-inkling-lora")
```
- `TrainingClient.forward_backward(data: List[Datum], loss_fn: LossFnType, loss_fn_config: Dict[str,float]|None)`
- `TrainingClient.optim_step(adam_params: AdamParams) -> APIFuture[OptimStepResponse]`
- `TrainingClient.save_state(name, ...)` (resumable full state) vs
  `save_weights_and_get_sampling_client(name)` (sampler weights).
- `TrainingClient.get_tokenizer()` — needed to build the renderer.

## 2. Supervised data / loss mask (the key mechanism — from `tinker_cookbook`)

The assistant-only loss mask is a **library concern**, not hand-rolled:
```python
from tinker_cookbook import renderers, supervised
renderer = renderers.get_renderer(name, tokenizer, model_name="thinkingmachines/Inkling")
datum = supervised.conversation_to_datum(
    conversation: list[renderers.Message],
    renderer,
    max_length,
    train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,  # DEFAULT — what we want
    reduction="mean",
) -> tinker.Datum
```
- `renderers.TrainOnWhat` values: `ALL_ASSISTANT_MESSAGES` (default, correct for SFT —
  trains assistant + its tool-call tokens, masks system/user/tool-result), `ALL_MESSAGES`,
  `ALL_TOKENS`, `LAST_ASSISTANT_MESSAGE`, `LAST_ASSISTANT_TURN`, `CUSTOMIZED`
  (reads a per-message `trainable` bool).
- `renderers.Message` is a **TypedDict**: required `role: str`, `content: str | list[parts]`;
  optional `tool_calls: list[ToolCall]`, `tool_call_id: str`, `name: str`, `trainable: bool`.
- `renderers.ToolCall` = `{type: "function", id: str|None, function: FunctionBody(name, arguments)}`.
- Low-level alternative: `supervised.datum_from_model_input_weights(model_input, weights, ...)`
  if we ever need an explicit token weight tensor (not needed — default TrainOnWhat suffices).
- **Renderer registry is empty at import** (`get_registered_renderer_names() == []`); renderers
  resolve live from the model/tokenizer. **PAID spike confirms the exact renderer name/path for Inkling.**

### Mapping from `arm_d.build_dataset.TrainingExample`
`TrainingExample.messages` (`{role, content, tool_calls}`) → list of cookbook `Message`
TypedDicts (tool results become `{"role":"tool","content":...,"tool_call_id":...}`), then
`conversation_to_datum(..., train_on_what=ALL_ASSISTANT_MESSAGES)`. The `target_mask` we
compute is exactly what `ALL_ASSISTANT_MESSAGES` reproduces, so it is a redundant safety
check — do NOT build token weights by hand.

## 3. Training-cost accounting (§15.3)

The SDK exposes billing/telemetry types: `tinker.types.BillingUsageResponse`,
`BillingUsageSession`, `TrainingBillingEvent`, `sc.get_telemetry()`. Read training
cost from the billing/usage API after the run (or, fallback, record step count ×
a documented per-step rate). Keep it separate from runtime token cost.

## 4. Serving the tuned model — PAID / USER DECISION (top risk)

**Finding:** the SDK's inference is `SamplingClient.sample(prompt: ModelInput, num_samples,
sampling_params) -> SampleResponse` — **raw-token completion over `ModelInput`. There is NO
built-in OpenAI-compatible chat/tools endpoint.** TAU2's agent calls the model through
`litellm.completion` with OpenAI chat + tool schemas, so the tuned model cannot be dropped
into the existing eval seam as-is. Two viable paths (pick one; both need the user):

- **A — Export/deploy on Together (preferred if supported).** The SDK has weight
  export/download surface (`Checkpoint`, `save_state`, `CheckpointArchiveUrlResponse`,
  `WeightsDownloadError`/`WeightsMergeError`/`WeightsAdapterError`). If Together can host the
  exported Inkling LoRA behind its OpenAI-compat endpoint, Arm D reuses the Arm C seam
  unchanged (`AGENT_MODEL=openai/<tuned-id>` + `AGENT_API_BASE=together`). **User must confirm
  in their Together console whether tuned-Inkling LoRA deploys are available.**
- **B — Local OpenAI-compat shim over `SamplingClient` (fallback, no Together dependency).**
  A tiny FastAPI server exposing `/v1/chat/completions` that: renders incoming chat+tools via
  the SAME `renderer` used in training → `ModelInput`, calls `SamplingClient.sample`, parses
  tool calls back out of the completion (`renderers.parse_content_blocks` /
  `classify_parse_failure`). Point the eval seam at it via `AGENT_API_BASE=http://localhost:PORT/v1`.
  More code + format-bug risk; smoke-gated like Arm C.

Whichever is chosen, register the resulting model id in `settings/pricing.py`
(`TUNED_INKLING_MODEL_IDS`) at Inkling's per-token price so runtime cost records nonzero.

## 5. Config bridge (do in code)

`.env` has `TINKER_KEY`; the SDK reads `TINKER_API_KEY`. `settings/config.py` should, at
import, do `os.environ.setdefault("TINKER_API_KEY", os.environ.get("TINKER_KEY",""))` (only
when `TINKER_KEY` is set and `TINKER_API_KEY` is not) so the SDK authenticates without renaming
the `.env` var. `.env.example` documents `TINKER_KEY`.

## 6. Deferred to the user (paid / account-specific)

1. Live round-trip confirming `thinkingmachines/Inkling` trains + the exact renderer name
   (one `create_lora_training_client` + 1 `forward_backward`/`optim_step` + `sample`).
2. Serving path choice (A vs B) — needs the Together account check.
3. The actual training run, Opus distillation generation, and the eval runs (all real spend).
