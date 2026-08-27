# Cost-Measurement Methodology

*Tier 3 validity item #18 — cost is the study's central axis, so how every dollar
figure is produced must be auditable. This section is a $0 consolidation of the
cost pipeline already implemented in the repo (`settings/pricing.py`,
`settings/config.py`, `benchmark/adapter.py`) and the analysis notes
(`cost_fairness.md`, `amortized_cost.md`, `provenance.md`). No new runs were made.*

---

## 1. How runtime cost is captured

Runtime (inference) cost is metered by **LiteLLM**, not by us. TAU2's
`run_simulation` prices each model call with LiteLLM's `completion_cost` and
accumulates two roll-ups onto the `SimulationRun`:

- **`agent_cost`** — the sum of `completion_cost` over the *agent's* LLM calls for
  that task. This is the number the study reports.
- **`user_cost`** — the same sum over the *user-simulator's* calls (see §3;
  excluded from the headline).

`benchmark/adapter.py::_normalize` copies `sim.agent_cost` straight onto
`EvalResult.agent_cost` (`adapter.py:148`) with no reweighting. Per task the
verdict + the full transcript are persisted: the transcript
(`task_<id>_messages.json`, written by `_persist_transcript` via
`sim.model_dump_json`) carries top-level `agent_cost` / `user_cost` / `seed` and a
per-message `cost` + `usage` (`prompt_tokens` / `completion_tokens`) breakdown.
Verified on a real transcript: `agent_cost = 0.1435`, `user_cost = 0.0187`,
per-message `cost`/`usage` present on billed turns (`null` on tool-result and some
assistant rows).

Per-task `agent_cost` values are summed by the run logger into
**`total_cost_usd`** — one row per run in `experiments/results.csv`. The two
headline metrics derive from that column:

- `cost_per_task = total_cost_usd / num_tasks`
- `cost_per_successful_task = total_cost_usd / (num_tasks × pass_rate)` — the
  study's reported cost metric.

A baseline is always a **distribution** (mean ± std over N repeated seeds), never a
single run.

**Temperature / determinism.** All arms sample at `temperature = 0.0` (TAU2's
`DEFAULT_LLM_ARGS_AGENT`) for reproducibility. Some newer models (Opus 4.8) reject
the `temperature` parameter and LiteLLM's `drop_params` does not strip it, so the
config flag **`AGENT_NO_TEMPERATURE=1`** makes `_agent_llm_args`
(`adapter.py:92-93`) pop `temperature` before the call. This affects the request
shape only, not how cost is metered.

---

## 2. Pricing tables (per-token, per model, per provider)

LiteLLM prices closed models from its built-in table but has **no entry for the
open-weight models** — worse, it silently attributes `thinkingmachines/*` to the
`openai` provider and, absent a registered price, would record **cost = $0**. So
`settings/pricing.py::register_pricing()` is called at `settings/config.py` import
(before any `generate()`), registering the missing prices. Exact per-token rates as
found in `settings/pricing.py`:

| Model id | input $/token | output $/token | ($/1M in / out) | Provider attribution | Source |
|---|---|---|---|---|---|
| `gpt-4.1` | LiteLLM built-in | LiteLLM built-in | — | openai (native) | native table |
| `anthropic/claude-opus-4-8` | LiteLLM built-in | LiteLLM built-in | $5 / $25 | anthropic (native) | native table |
| `thinkingmachines/Inkling` | `1.0e-6` | `4.05e-6` | $1.00 / $4.05 | **openai (registered)** | `INKLING_PRICING` |
| `thinkingmachines/Inkling-Small` | `0.5e-6` | `1.2e-6` | $0.50 / $1.20 | **openai (registered)** | `register_inkling_small` |
| `armd-inkling-small-tuned` | `0.5e-6` | `1.2e-6` | $0.50 / $1.20 | **openai (registered)** | `register_inkling_small` |
| `armd-inkling-lora` | `1.0e-6` | `4.05e-6` | $1.00 / $4.05 | openai (registered) | `TUNED_INKLING_PRICING` (registered, **not the id actually served**) |
| `claude-fable-5` | `10.0e-6` | `50.0e-6` | $10 / $50 | anthropic (registered) | `FABLE_PRICING` (**registered but unused** — arm dropped for Opus) |

Key facts:

- **Open-weight models must be registered or they record $0.** Registration is
  mandatory precisely because LiteLLM lacks these entries; all open ids are
  registered under the **`openai` provider key** (the Together / Tinker
  OpenAI-compatible seam Arms C/D route through), while closed models are priced
  natively.
- **A LoRA adapter does not change per-token price.** The tuned Inkling-Small id is
  registered at the *base* Inkling-Small rate ($0.50 / $1.20); training cost is
  accounted separately (§3, §5). `register_pricing` also honors a runtime
  `ARM_D_TUNED_MODEL_ID` env var and pre-registers
  `thinkingmachines/Inkling-Small` + `armd-inkling-small-tuned`.
- Inkling-Small's rate uses Together's serverless reference price as the basis;
  Tinker itself exposes no per-token billing telemetry.

---

## 3. What is included vs excluded

**Included — agent inference cost.** The headline `total_cost_usd` /
`cost_per_successful_task` is the sum of `agent_cost` across a run's tasks, i.e.
only the task agent's own LLM calls.

**Excluded — user-simulator cost.** TAU2's user-simulator runs its own model
(`gpt-4.1-2025-04-14`, TAU2's `DEFAULT_LLM_USER`, identical across arms). Its
`user_cost` is captured in every transcript but **left out of the headline**. It is
small in absolute dollars and roughly constant across arms, so it forms a *larger*
share of the cheap open arms than of Opus (from `cost_fairness.md §d`):

| Arm | user-sim $/task | as % of agent $/task |
|---|---:|---:|
| Opus 4.8 (closed frontier) | $0.0115 | 2.2% |
| Inkling (open, Together) | $0.0164 | 85.1% |
| Inkling-Small base (open) | $0.0138 | 30.0% |
| Inkling-Small tuned (open) | $0.0127 | 40.8% |

So the ~$0.011–0.016/task user-sim cost is ~2.2% of Opus's agent cost but ~85% of
Inkling's — meaning the raw agent-cost ratio, if anything, *overstates* the cheap
arms' all-in advantage. Excluding it is defensible (same model, same magnitude
every arm) but is disclosed as a confound.

**Excluded from runtime, handled separately — fine-tuning cost.** The Arm D tuned
model incurs a one-time training cost `T_train` that is *not* part of any per-task
serving figure. It is analyzed separately and **parameterized** in
`amortized_cost.md` (break-even task volume `N* = T_train / delta` over a `T_train`
sweep), because the actual dollar figure is unrecorded (§5). The teacher
(Opus-distillation generation) cost *is* captured normally as the `train_distill_*`
rows in `results.csv`.

---

## 4. Pricing-vs-efficiency distinction

The "~26× cheaper per successful task" headline (Inkling $0.0221/succ vs Opus
$0.5768/succ) is a **commercial-pricing** fact, not a model-efficiency fact, and
`cost_fairness.md` separates the two:

- **As-listed price** compares two different kinds of number: closed models at
  **retail API list price** (with provider margin), open models at the study's own
  **near-cost serving rate** (Together / local Tinker shim). At those prices the
  26× gap is a real invoice difference *today*.
- **Efficiency (equal $/token, pricing held constant)** re-prices every arm's agent
  tokens at one common schedule ($1.00/1M in, $3.00/1M out). Under that schedule
  Opus spends 97,499 agent tokens/task vs Inkling 102,726 (1.05×) — Inkling is
  **0.97×** Opus's cost, i.e. *no* intrinsic efficiency advantage. The ~26× is
  therefore **almost entirely a commercial-pricing artifact**.
- A **genuine efficiency gain survives normalization**: the Opus-distilled tuned
  Inkling-Small uses 60,731 vs 90,725 base agent tokens/task (1.49× fewer) at
  identical serving — a real fine-tuning work reduction, visible only because
  pricing is held constant.

Both framings are reported; the "26×" must be stated as a pricing fact, never as a
model property.

---

## 5. Known measurement gaps

- **Per-message token usage is often `None`** (tool-result rows, some assistant
  rows). Where only a blended per-message `cost` exists, input vs output tokens
  cannot be cleanly separated; token analyses back tokens out of cost or use the
  `usage` split where present (available intact for Opus, Inkling, both
  Inkling-Small arms).
- **gpt-4.1 June runs predate transcript persistence** (`cost_fairness.md §e`,
  `provenance.md G2`). The Arm A proxy (2026-06-12) and validation (2026-06-15)
  runs have no `task_*_messages.json` and no turn/tool logging, so gpt-4.1 has only
  aggregate cost — its ~23,463 tokens/task is *estimated* by dividing cost by a
  ~$2.12/1M blended rate and **cannot split input from output**; treat as
  indicative only.
- **Tinker training-cost telemetry is absent** (`amortized_cost.md`,
  `provenance.md G3`). `training_record.json` records `training_cost_usd = 0.0`
  **not** because training was free but because Tinker exposes no billing
  telemetry (`cost_note: "no billing telemetry exposed; fallback = steps *
  cost_per_step_usd (0.0)"`). The real figure = (Tinker GPU-hours for the 88 LoRA
  steps) × (external hourly rate), both outside this repo — hence the parameterized
  break-even sweep rather than a measured amortized cost.
- **No seed column in `results.csv`** (`provenance.md G1`): the seed lives only in
  each run's per-task JSON, so reproducing a specific cost row means opening its log
  folder.
