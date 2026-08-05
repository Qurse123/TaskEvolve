# Arm D — Specialized Inkling via Tinker, multi-domain (design)

**Date:** 2026-08-03
**Branch:** `ARMD_`
**Status:** design, pending review

## 1. Goal

Fine-tune the open-weight **Inkling** model (`thinkingmachines/Inkling`) with **Tinker**
(LoRA) so it becomes a **generally better TAU2 agent — across all task domains, not
just retail — without overfitting**. This is **Arm D** of the TaskEvolve study:
*specialized open-weight model + static harness* (`experiment.md §13.4`). The naive
counterpart is **Arm C** (naive Inkling, same static v0.1 harness). The **C→D delta,
measured on held-out tasks in every domain plus a fully unseen transfer domain, is
the marginal value of model specialization** — the single measurement this arm exists
to produce.

Non-goal: harness changes. The harness stays **static v0.1** (system prompt every
turn, identity model routing) — identical to Arm C and the Opus reference — so the
only moving variable is the model weights.

## 2. Decisions locked in brainstorming

| Decision | Choice | Why |
|---|---|---|
| Base model | `thinkingmachines/Inkling` via Tinker (LoRA) | Tinker hosts Inkling as a trainable base; keeps C→D base identical (no confound). |
| Teacher / training signal | **Distill Opus 4.8's successful trajectories** | Strongest teacher available; `experiment.md §13.4` "distillation from high-performing closed-model trajectories". |
| Domain scope | **Train on retail + airline + telecom; hold out banking_knowledge entirely as a transfer domain** | General competence across domains + a pure never-seen-domain generalization test. |
| Training-task pool | **A disjoint subset of each trained domain's train tasks** (never the held-out eval tasks) | Both in-domain held-out tasks and the whole banking domain stay out of training → zero train/eval overlap. |
| Comparison basis | **Matched base control**: naive Inkling served through the same Tinker/adapter path as D (no adapter), re-measured on the same multi-domain eval | Removes the base-version/serving confound so the C→D delta is purely the adapter (see §8). Together-served naive Inkling = secondary reference; A/Opus stay retail references. |
| Serving the tuned model | **Decide at build time from docs**: export PEFT adapter → Together OpenAI-compat LoRA deploy (preferred) else Tinker sampling client + chat/tools shim | Reuse the Arm C eval seam if Together supports Inkling LoRA; shim fallback otherwise. |
| Harness | Static **v0.1** | Isolate the model axis; identical to Arm C / Opus reference. |
| Eval repeats | N=5 seeds per domain (proxy-style `1001..1005` for any dev peeking; blind eval `2001..2005`) | Distribution, not one run (`experiment.md §15.5`). |

## 3. Anti-overfitting protocol (the core requirement)

Two overfitting failure modes: memorizing specific training tasks, and **narrow
specialization** — getting better at trained domains while degrading elsewhere.
Three guards address both:

1. **Task + domain disjointness (primary guard).**
   - Within each trained domain (retail, airline, telecom): train only on a subset of
     its train tasks; the remaining tasks form the **in-domain held-out eval** and are
     never trained.
   - **banking_knowledge is never trained at all** — it is the **transfer eval**,
     measuring whether specialization generalizes to an unseen domain.
   - Retail reuses the existing `validation.json` (35 tasks) as its held-out eval; the
     27 unused retail-train tasks feed the training pool. Airline and telecom get new
     train/held-out carves from their train splits.
2. **Capacity + steps.** LoRA rank ~16–32, 1–3 epochs, small LR, **early-stop on a
   held-out slice of the training tasks** (never any eval split). Small, diverse data
   → few steps; watch held-out loss, not train loss.
3. **Verdict rule (the Arm B lesson).** The specialization gain **counts only if it
   holds on the held-out eval — especially the banking transfer domain**, not just on
   tasks resembling the training set. Report per-domain distributions; a
   trained-domain-only improvement that fails to transfer is reported as overfit,
   exactly as Arm B v0.2 was.

## 4. Pipeline (new `arm_d/` package)

### 4.1 Distillation data generation (multi-domain)
- New split files under `benchmark/splits/`:
  - `train_distill_retail.json` — the 27 unused retail-train task IDs.
  - `train_distill_airline.json`, `train_distill_telecom.json` — a subset of each
    domain's train tasks; the complement becomes that domain's held-out eval.
  - `eval_airline.json`, `eval_telecom.json` — the held-out complements above.
  - `transfer_banking.json` — a sampled held-out subset of banking_knowledge (never
    trained).
  - Retail held-out eval reuses the existing frozen `validation.json`.
  - All disjoint from each other by construction; the frozen
    proxy/validation/smoke files are **not modified**.
- Run **Opus 4.8** (`anthropic/claude-opus-4-8`, `HARNESS_VERSION=v0.1`,
  `AGENT_NO_TEMPERATURE=1`) on each `train_distill_*` split, a few seeds, **detached
  via nohup**. Persist full `SimulationRun` transcripts. Keep only **passed** tasks.
- Reuse the ~45 existing Opus retail-proxy traces only if their tasks fall in the
  retail training pool (they are proxy tasks → measurement, so **excluded** to keep
  retail proxy/validation clean).

### 4.2 `arm_d/build_dataset.py`
- Input: Opus `task_*_messages.json` transcripts from the `train_distill_*` runs,
  filtered to `passed == True`, across all three trained domains.
- Output: Tinker SFT examples — render each trajectory into **Inkling's chat
  template** with tool calls, **loss-masked to assistant + tool-call tokens only**
  (system / user / tool-result tokens carry 0 loss). Dedup identical trajectories.
  Domain-balanced (don't let telecom's larger pool swamp airline).
- Pure data transform, unit-tested at $0 on a synthetic transcript.

### 4.3 `arm_d/train.py`
- Tinker training loop: LoRA client on `thinkingmachines/Inkling`,
  `forward_backward` / `optim_step`, early-stop on the held-out training slice,
  `save_weights_and_get_sampling_client()` → checkpoint + exported adapter.
- **Logs training cost separately** from runtime cost (`experiment.md §15.3`):
  emit a `training_cost_usd` / step count / wall-clock record.

### 4.4 Serving wiring
- Export the PEFT adapter; serve behind the **existing `AGENT_MODEL` /
  `AGENT_API_BASE` OpenAI-compat seam** (the Arm C path) so `benchmark/adapter.py`,
  `run_train_eval`, `results/logger.py` are reused **unchanged**.
- Register the tuned model id in `settings/pricing.py` (same per-token price as
  Inkling; training cost tracked separately) so runtime cost records nonzero.
- **Build-time decision**: verify in Tinker Cookbook / Together docs whether an
  Inkling LoRA adapter can be deployed on Together's OpenAI-compat endpoint. If yes,
  use it; if no, stand up a thin shim over Tinker's sampling client presenting an
  OpenAI-compatible chat+tools interface.

### 4.5 Smoke gate (before any eval spend)
- Run `scripts/run_smoke.py` against the served tuned model. **Gate:** 3/3 mock
  tasks complete, tool calls parse (no `harness_error`), **`cost_usd` nonzero**.
  Fix wiring before spending on measurement splits.

### 4.6 Evaluation + reporting (multi-domain)
- Static v0.1 harness, N=5 per split (seeds `2001..2005`), detached. Evaluate BOTH
  **Arm D (tuned)** and the **matched base control** (base Inkling, same serving path,
  no adapter — §8 guard 2) on every eval split: retail `validation.json`,
  `eval_airline`, `eval_telecom`, and `transfer_banking`.
- Report `pass_rate` and `cost_per_successful_task` as **mean ± std per domain** and
  an aggregate; call out the **banking transfer** number specifically.
- Frontier plot per domain + an aggregate: **C (naive Inkling) vs D (tuned Inkling)**
  (with A/Opus retail references), cost vs success, means as ◆ with std error bars.
- Report **training cost alongside runtime cost** (§15.3).

## 5. Files

**New:** `arm_d/build_dataset.py`, `arm_d/train.py`, `arm_d/serve.py` (or serving
config); `benchmark/splits/{train_distill_retail,train_distill_airline,
train_distill_telecom,eval_airline,eval_telecom,transfer_banking}.json`; a split
generator (extend `benchmark/splits.py`) that carves the airline/telecom
train/held-out and banking transfer sets deterministically (fixed seed); tests
(`tests/test_arm_d_dataset.py`, split-disjointness test, serving/pricing tests).
**Changed (additive):** `settings/pricing.py` + `settings/config.py` (register tuned
model id), `.env.example` (`TINKER_API_KEY`, tuned `AGENT_MODEL`),
`scripts/plot_results.py` (Arm D + per-domain labels), `CLAUDE.md` + memory (docs at
sequence end).
**Reused unchanged:** `benchmark/adapter.py`, `target_agent/*` (v0.1 harness),
`scripts/run_train_eval.py`, `results/logger.py`, existing
proxy/validation/smoke splits, `vendor/tau2-bench/` (frozen).

## 6. Verification

1. `python -m pytest tests/` green (dataset/pricing/serving/split-disjointness tests
   + existing).
2. **Split disjointness test:** training pools ∩ (any eval split) = ∅; banking never
   in any training pool.
3. `build_dataset.py` produces correctly loss-masked Inkling examples from a
   synthetic multi-domain transcript ($0 unit test).
4. Training run completes, saves a checkpoint, emits a separate `training_cost_usd`.
5. Smoke gate: 3/3 mock, nonzero cost, no harness errors on the served tuned model.
6. Arm C + Arm D each evaluated N=5 on all four eval splits; `results.csv` gains the
   rows with the correct model id, domain tag, and `harness_version=v0.1`.
7. Per-domain + transfer frontier renders; C→D delta reported per domain, with the
   **banking transfer** result as the binding no-overfit verdict.

## 7. Risks / watchpoints

- **Serving path unknown** — top risk. Resolved at build time; smoke gate catches a
  broken wiring (cost=0 or harness_error) before measurement spend.
- **Tool-call format drift** — Opus (Anthropic) tool calls must normalize to
  Inkling's format in `build_dataset.py`; a mismatch teaches malformed tool syntax.
  Unit-tested; smoke gate surfaces runtime parse failures.
- **Narrow specialization** — the failure mode this arm must avoid; the multi-domain
  training set + banking transfer verdict are the guard. If banking regresses vs naive
  Inkling, report honestly (Arm B precedent); do not tune to trained domains.
- **Domain imbalance** — balance the SFT mix so telecom's larger pool doesn't
  dominate (§4.2).
- **Cost** — multi-domain Opus generation (retail 27 + airline/telecom subsets × seeds)
  plus Arm C + Arm D eval on four splits × N=5. Sized honestly at plan time; runs
  detached. `USER_MODEL` (TAU2 user-sim) stays fixed for valid comparison.
- **Training cost** logged/reported separately (§15.3), never folded into runtime cost.

## 8. Research validity guards

The results must be defensible as a controlled measurement, not an anecdote. Seven
guards, all enforced by build/protocol:

1. **Single-variable comparison.** Arm C and Arm D differ in **exactly one thing —
   the model weights**. Identical harness (v0.1), `USER_MODEL` (TAU2 user-sim), seeds,
   eval splits, eval code, and serving path. Anything else changing invalidates the
   delta.
2. **Matched base control (removes the serving/base confound).** The **primary** Arm C
   control is base Inkling served through the **same Tinker/adapter serving path as
   Arm D but with no LoRA adapter** (or a zeroed adapter). This guarantees C and D
   share identical base weights + serving stack, so the delta is purely the adapter.
   The existing Together-served naive Inkling (Arm C from M3) is kept only as a
   **secondary reference**, not the controlled baseline.
3. **Pre-registration.** Before any tuned-model eval runs, commit
   `experiments/arm_d_eval_precommit.md` fixing: the eval splits, seeds (`2001..2005`),
   metrics (`pass_rate`, `cost_per_successful_task`), the domains to report (all four +
   aggregate + banking transfer), and the verdict rule. No post-hoc changes to any of
   these after seeing results (mirrors `validation_event_v0.2_precommit.md`).
4. **No eval-set tuning (blind eval).** LoRA hyperparameters, epoch count, and
   early-stop are selected using **only the held-out slice of the training tasks**.
   The eval splits — retail validation, airline/telecom held-out, and banking transfer
   — are touched **once**, after the training config is frozen. The model is never
   iterated against eval scores (that would overfit the eval, the exact Arm B failure).
5. **Report everything, delta vs noise.** Report all four domains + aggregate +
   transfer, whatever they show — no cherry-picking the best domain. Report the C→D
   delta **relative to seed variance** (std / simple CI); a within-noise "gain" is
   reported as not significant, not as a win (Arm B lesson).
6. **Reproducibility / audit trail.** Deterministic split generation (fixed seed);
   a logged **training manifest** listing the exact task IDs and trajectory hashes in
   the SFT set; logged LoRA hyperparameters, checkpoint id, and Tinker base-model
   version; and cost accounting separated into generation / training / runtime
   (§15.3). Anyone can reconstruct exactly what went into the model.
7. **Teacher-confound disclosure.** Arm D is **Opus-distilled** — its behavior is
   bounded by what Opus demonstrated on the training tasks. Report the specialization
   as "Opus-distillation SFT," not generic self-improvement, so the mechanism is not
   overclaimed.

## 9. Out of scope

Arm E (tuned Inkling + iterator-optimized harness), Arm F (joint optimization),
TAU2 official hidden-test run (once, at milestone end). RL / corrected-failure
trajectories (`§13.4` alternatives) — SFT distillation only for this arm. Extending
A/Opus references to multi-domain (retail-only references suffice for the C→D story).
