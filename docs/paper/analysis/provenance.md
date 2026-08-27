# Experimental Provenance & Reproducibility

*Tier 2 validity item #13 — report exact experimental provenance so every reported
number can be reconstructed.* This section is a $0 audit of the artifacts already in
the repo: `experiments/results.csv`, the per-task verdict JSONs under
`experiments/logs/<run_id>/task_*.json`, `settings/config.py`, `settings/pricing.py`,
`.env.example`, the Arm D training record/manifest, and the pre-registration docs.
No new runs were made to write it.

All numbers below are copied from those files. Where a value is unverifiable from the
artifacts it is flagged in **§4 Known provenance gaps** rather than guessed.

---

## Shared harness invariants (all arms)

Resolved from `benchmark/adapter.py` (TAU2 defaults) and `settings/config.py`:

| Setting | Value | Source |
|---|---|---|
| User-simulator model | `gpt-4.1-2025-04-14` | TAU2 `DEFAULT_LLM_USER` (env `USER_MODEL` was **unset** in every run → `user_model: null` in all task JSONs) |
| User-sim / agent temperature | `0.0` | TAU2 `DEFAULT_LLM_ARGS_AGENT = {'temperature': 0.0}` (dropped for models that reject it — see Opus/Arm B rows) |
| Max conversation steps | `200` | TAU2 `DEFAULT_MAX_STEPS` (`config.MAX_STEPS = None`) |
| Max consecutive tool errors | `10` | TAU2 `DEFAULT_MAX_ERRORS` (`config.MAX_ERRORS = None`) |
| Pass threshold | reward `≥ 0.5` | `config.PASS_THRESHOLD` |
| Domain (unless noted) | `retail` | `config.DEFAULT_DOMAIN` |

**Seeds are one-per-repeat, constant across all tasks within a run** (verified: each
`<run_id>` folder's task JSONs share a single `seed`). The seed is **not** a column in
`results.csv` — it lives only in the per-task JSONs (gap G1). Seed conventions:
`proxy 1001–1005`, `validation 2001–2005`, `Arm D held-out eval 2001–2003`,
`iterator proxy 5001+` (disjoint 4-seed blocks/iteration; observed 5005, 5006),
`TAU2 test 4001–4005`.

---

## 1. Per-arm provenance table

Model string = the exact `agent_model` recorded in `results.csv` / task JSONs.
"N" = repeats (rows) that form the reported mean ± std. Dates from `generated_at`.

| Arm / role | Model string (as recorded) | Provider + serving path | Harness | Sampling | Split (tasks) · seeds · N | Run date(s) |
|---|---|---|---|---|---|---|
| **Arm A** baseline | `gpt-4.1` | OpenAI native (LiteLLM built-in) | v0.1 | temp 0.0 | proxy (12) · 1001–1005 · N=5 | 2026-06-12 |
| **Arm A** baseline | `gpt-4.1` | OpenAI native | v0.1 | temp 0.0 | validation (35) · 2001–2005 · N=5 | 2026-06-15 |
| Arm A (iterator baseline re-run) | `gpt-4.1` | OpenAI native | v0.1 | temp 0.0 | proxy (12) · 1 row · N=1 | 2026-07-18 |
| **Arm C** naive open-weight | `openai/thinkingmachines/Inkling` | **Together AI OpenAI-compat** (`AGENT_API_BASE=https://api.together.xyz/v1`, `TOGETHER_API_KEY`) | v0.1 | temp 0.0 | proxy (12) · 1001–1005 · N=5 | 2026-07-22 |
| **Arm C** naive open-weight | `openai/thinkingmachines/Inkling` | Together OpenAI-compat | v0.1 | temp 0.0 | validation (35) · 2001–2005 · N=5 | 2026-07-22 |
| **Arm C** (held-out test) | `openai/thinkingmachines/Inkling` | Together OpenAI-compat | v0.1 | temp 0.0 | test_retail (40) · 4001–4005 · N=5 | 2026-08-20 |
| **Opus reference** | `anthropic/claude-opus-4-8` | Anthropic native (LiteLLM built-in, $5/$25 per 1M) | v0.1 | **temp dropped** (`AGENT_NO_TEMPERATURE=1`) | proxy (12) · 1001–1005 · N=5 | 2026-07-22 |
| **Opus reference** | `anthropic/claude-opus-4-8` | Anthropic native | v0.1 | temp dropped | validation (35) · 2001–2005 · N=5 | 2026-07-22 → 07-23 |
| **Arm B** iterator-optimized | `anthropic/claude-opus-4-8` **(base id; v0.2 harness routes → Sonnet 5)** | Anthropic native | **v0.2** (+ v0.3 exploratory) | temp dropped | proxy (12) · 5001-block · v0.2 (3) + v0.3 (4) rows | 2026-07-28 |
| **Arm B** iterator-optimized | `anthropic/claude-opus-4-8` (v0.2 → Sonnet 5) | Anthropic native | v0.2 | temp dropped | validation (35) · 2001–2005 · N=5 | 2026-07-29 |
| **Arm D teacher gen** (distillation) | `anthropic/claude-opus-4-8` | Anthropic native | v0.1 | temp dropped | train_distill_retail (27), _airline (20), _telecom (49) · 3 rows each | 2026-08-06 → 08-07 |
| **Arm D base control** | `openai/thinkingmachines/Inkling-Small` | **Local Tinker shim** (`arm_d/serving_shim.py`, `AGENT_API_BASE=http://localhost:8100/v1`, no adapter) | v0.1 | temp 0.0 | validation (35), eval_airline (10), eval_telecom (25), transfer_banking (30) · 2001–2003 · N=3 | 2026-08-09 → 08-11 |
| **Arm D tuned** | `openai/armd-inkling-small-tuned` | Local Tinker shim (same process, LoRA adapter) | v0.1 | temp 0.0 | validation (35), eval_airline (10), eval_telecom (25), transfer_banking (30) · 2001–2003 · N=3 | 2026-08-10 → 08-11 |
| **Arm D base** (held-out test) | `openai/thinkingmachines/Inkling-Small` | Local Tinker shim | v0.1 | temp 0.0 | test_retail (40) · 4001–4005 · N=5 | 2026-08-19 → 08-20 |
| **Arm D tuned** (held-out test) | `openai/armd-inkling-small-tuned` | Local Tinker shim | v0.1 | temp 0.0 | test_retail (40) · 4001–4005 · N=5 | 2026-08-19 → 08-20 |

Notes:
- **Arm B model attribution is intentional and must be disclosed.** Rows record
  `agent_model = anthropic/claude-opus-4-8` (the *configured* base), but the v0.2
  harness's `target_agent/model_routing.py::get_model` routed every call to
  **Sonnet 5**. The frontier arm is "Opus-base-routed-to-Sonnet." (The checked-out
  `model_routing.py` is the v0.1 identity router, restored for Arm C/D; the v0.2
  routing lives in commit `cbb0e8c`.)
- **Arm D serving changed mid-study** (path A → path B). The pre-registered plan was
  to serve the tuned LoRA via Together like Arm C; that was replaced by a **local
  OpenAI-compatible Tinker shim** because no vendor serves a custom LoRA serverless
  (`experiments/arm_d_eval_precommit.md`, "Serving path B"). Both the tuned model and
  its base control run through the **same shim process**, so any serving-stack artifact
  cancels in the tuned−base delta.
- Arm C uses **full** `Inkling`; Arm D's base control is the smaller
  `Inkling-Small`. They are different checkpoints — Arm C is a secondary frontier
  reference, not the Arm D control.

### Per-token pricing registered (`settings/pricing.py`)

| Model id | input $/tok | output $/tok | Provider attribution | How |
|---|---|---|---|---|
| `gpt-4.1` | LiteLLM built-in | LiteLLM built-in | openai | native table |
| `anthropic/claude-opus-4-8` | LiteLLM built-in ($5/1M) | LiteLLM built-in ($25/1M) | anthropic | native table |
| `thinkingmachines/Inkling` | 1.0e-6 | 4.05e-6 | **openai** (registered) | `INKLING_PRICING` |
| `thinkingmachines/Inkling-Small` | 0.5e-6 | 1.2e-6 | **openai** (registered) | `register_inkling_small` |
| `armd-inkling-small-tuned` | 0.5e-6 | 1.2e-6 | **openai** (registered) | `register_inkling_small` (LoRA = base per-token price) |
| `claude-fable-5` | 10.0e-6 | 50.0e-6 | anthropic | `FABLE_PRICING` (registered but **unused** — arm dropped) |
| `armd-inkling-lora` | 1.0e-6 | 4.05e-6 | openai | `TUNED_INKLING_PRICING` (registered but **not the id actually served**) |

Registration is required because LiteLLM's built-in table has no Inkling entry and
attributes it to `openai` (not `together_ai`) — without this the open-weight arms would
record cost $0. Inkling-Small pricing uses Together's serverless reference rate
($0.50/$1.20 per 1M) as the basis; Tinker exposes no per-token billing telemetry (gap G3).

---

## 2. Fine-tune provenance (Arm D)

Source of truth: `experiments/arm_d_training/run_20260807_222817/training_record.json`
and `arm_d_sft_manifest.json` (same folder).

| Field | Value |
|---|---|
| Base model | `thinkingmachines/Inkling-Small` |
| Method | LoRA SFT via **Tinker**; teacher = **Opus 4.8** (distillation, passed-only trajectories) |
| LoRA rank | 32 |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Max epochs / patience | 4 / 2 (early-stop) |
| Steps (actual) | 88 |
| Train examples | 169 |
| Holdout examples | 26 |
| Best holdout loss | 10.8144 |
| Checkpoint URI | `tinker://fdc7bf81-d958-529c-9788-2332faf52c1d:train:0/sampler_weights/arm-d-inkling-small-lora-run_20260807_222817` |
| Checkpoint name | `arm-d-inkling-small-lora-run_20260807_222817` |
| Trained (record timestamp) | 2026-08-07T22:37:54Z |
| **Training cost** | **$0.00 recorded — unverified** (`cost_per_step_usd = 0.0`; `cost_note`: "no billing telemetry exposed; fallback = steps × cost_per_step_usd") |

**SFT data lineage (audit trail, `arm_d_sft_manifest.json`):** every training example
carries `task_id`, `domain`, `source_run` (the `train_distill_*` Opus run it was
distilled from), and a `sha256` trajectory hash. Domains balanced across retail /
airline / telecom `train_distill_*` runs (27 / 20 / 49 Opus tasks). No eval-split or
validation/test task enters training by construction (splits are disjoint from
TAU2 `train`).

**Teacher-confound disclosure (required):** Arm D behavior is bounded by what Opus 4.8
demonstrated on the training tasks — this is Opus-distillation SFT, **not**
self-improvement.

---

## 3. Exact run commands per arm

Copied from `CLAUDE.md` and the pre-registration docs. The bare model prefixes match
`.env` at run time.

**Arm A (gpt-4.1, v0.1):**
```
python -m scripts.run_train_eval --split proxy      --repeats 5 --seed-start 1001   # AGENT_MODEL=gpt-4.1
python -m scripts.run_train_eval --split validation --repeats 5 --seed-start 2001
```

**Arm C (naive full Inkling via Together, v0.1):**
```
AGENT_MODEL=openai/thinkingmachines/Inkling \
AGENT_API_BASE=https://api.together.xyz/v1 \
HARNESS_VERSION=v0.1 \
python -m scripts.run_train_eval --split {proxy|validation} --repeats 5 --seed-start {1001|2001}
```

**Opus 4.8 frontier reference (v0.1):**
```
AGENT_MODEL=anthropic/claude-opus-4-8 HARNESS_VERSION=v0.1 AGENT_NO_TEMPERATURE=1 \
python -m scripts.run_train_eval --split {proxy|validation} --repeats 5 --seed-start {1001|2001}
```

**Arm B (iterator; accepted change = Opus→Sonnet, produces v0.2):**
```
# optimization (proxy only, iterator never sees validation/test):
python -m scripts.run_iterator --max-minutes <M> --seed-start 3001 --max-iterations <cap> [--max-search-cost-usd <cap>]
# blind validation event, run once post-hoc:
AGENT_MODEL=anthropic/claude-opus-4-8 HARNESS_VERSION=v0.2 AGENT_NO_TEMPERATURE=1 \
python -m scripts.run_train_eval --split validation --repeats 5 --seed-start 2001
```
(Editor/iterator model set via `ITERATOR_MODEL` — the un-biased autonomous run used
an Opus editor; `ITERATOR_NO_TEMPERATURE=1`.)

**Arm D held-out eval (N=3, base + tuned, via the shim) — `experiments/arm_d_eval_precommit.md`:**
```
python -m arm_d.serving_shim            # start local OpenAI-compat shim on :8100 first
python -m scripts.run_arm_d_eval        # 4 splits × 3 seeds × 2 systems (retail/airline/telecom/banking)
```

**TAU2 held-out test split (executed subset, N=5, seeds 4001–4005) — `experiments/test_split_precommit.md`:**
```
# Arm C:
AGENT_MODEL=openai/thinkingmachines/Inkling AGENT_API_BASE=https://api.together.xyz/v1 \
HARNESS_VERSION=v0.1 python -m scripts.run_train_eval --split test_retail --repeats 5 --seed-start 4001
# Arm D base + tuned (shim running):
for M in thinkingmachines/Inkling-Small armd-inkling-small-tuned; do
  AGENT_MODEL=openai/$M AGENT_API_BASE=http://localhost:8100/v1 HARNESS_VERSION=v0.1 \
    python -m scripts.run_train_eval --split test_retail --domain retail --repeats 5 --seed-start 4001
done
```
(The full pre-registered test design also names gpt-4.1, Opus, and Arm B v0.2 arms +
airline/telecom test; those were **pre-registered as deferred for budget before launch**,
not dropped post-hoc.)

---

## 4. Known provenance gaps

- **G1 — No seed column in `results.csv`.** The seed exists only inside each run's
  per-task JSONs (`task_*.json → "seed"`). Reproducing a specific row means opening its
  log folder. Every run uses one seed for all its tasks (verified).
- **G2 — gpt-4.1 June runs predate transcript/turn logging.** The Arm A proxy
  (2026-06-12) and validation (2026-06-15) runs were made before the cost-aware
  transcript layer (`task_*_messages.json`, `turn_count`/`tool_call_count`) landed in
  M2, so those Arm A rows have no per-turn cost evidence — only the aggregate cost.
- **G3 — Tinker training cost unrecorded.** `training_record.json` reports
  `training_cost_usd = 0.0` with an explicit note that no billing telemetry is exposed;
  the LoRA training spend is genuinely unknown, not zero. Runtime (eval) cost is
  accounted normally; teacher (Opus distillation-gen) cost is captured as the
  `train_distill_*` rows in `results.csv`.
- **G4 — Exact model snapshot dates partly unverified.** `agent_model` records the bare
  aliases (`gpt-4.1`, `anthropic/claude-opus-4-8`) — the underlying provider snapshot
  (e.g. whether `gpt-4.1` resolved to `gpt-4.1-2025-04-14`) is **not** pinned in the
  artifacts. The **user-simulator** snapshot *is* pinned: TAU2's `DEFAULT_LLM_USER =
  gpt-4.1-2025-04-14`. Note a stale doc: `.env.example` says the user-sim "defaults to
  gpt-4o if unset" — that comment is **incorrect**; the resolved default is
  `gpt-4.1-2025-04-14`.
- **G5 — Arm B agent_model is the base, not the served model.** v0.2 rows say
  `claude-opus-4-8` while the harness routed to Sonnet 5 (see §1 note). Reading the CSV
  alone overstates the model; the harness version (`v0.2`) is the disambiguator.
- **G6 — Arm B iterator proxy seeds not on the CLAUDE.md default.** CLAUDE.md documents
  `--seed-start 3001`, but the accepted-run proxy JSONs carry seeds in the 5001+ block
  (observed 5005/5006), consistent with the 4-seed-per-iteration block spacing.
- **G7 — banking_knowledge transfer runs include failed/retried rows.** Several
  `transfer_banking` rows have `pass_rate = 0` with empty/na cost (harness or config
  issues before the retrieval config was settled); the judged Arm D result uses the
  later clean N=3 runs (2026-08-11). Absolute banking scores are near-floor for both
  systems.
