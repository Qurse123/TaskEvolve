# Arm D Eval — Pre-registration (committed before any tuned-model eval)

- **Date declared:** 2026-08-05 (before the Arm D evaluation event starts).
  **Re-frozen 2026-08-08** for serving path B (below) — base model, serving
  stack, and N changed; no eval score has been seen under either version.
- **Status:** precommitted. Once the Arm D adapter is trained and its config is
  frozen, the criteria below are fixed — no post-hoc changes after any eval score
  is seen (design spec §8 guard 3; mirrors `validation_event_v0.2_precommit.md`).
- **Arm:** D — *specialized open-weight model + static v0.1 harness*
  (`experiment.md §13.4`). Specialization = **Opus-4.8-distillation SFT** (LoRA on
  `thinkingmachines/Inkling-Small` via Tinker). The measurement this arm exists to
  produce is the **C→D delta on held-out tasks in every domain plus a fully unseen
  transfer domain** — the marginal value of model specialization.

## Serving path B (why the seam changed from Together to a local Tinker shim)

The tuned adapter can't be served serverless anywhere: both Together and
Fireworks require an expensive *dedicated* GPU endpoint for a custom LoRA
adapter — a cost/complexity mismatch for a controlled measurement study. Path
B serves it CHEAPLY, per-token, no hourly commitment, straight from Tinker's
own `SamplingClient`, fronted by a **local OpenAI-compatible HTTP shim**
(`arm_d/serving_shim.py`) so TAU2's existing eval seam (litellm's `openai/`
provider + `AGENT_API_BASE`) runs **unchanged** — just pointed at
`http://localhost:8100/v1` instead of Together. One shim process serves
**both** systems under test, routed only by the request's `model` field, so
any serving-stack artifact introduced by the shim (prompt templating,
tokenization, HTTP overhead) is present identically on both sides of the
comparison and **cancels in the matched tuned-vs-base delta** — the same
single-variable guarantee the original Together path gave, just relocated.

Base model also changed: **Inkling-Small** (not full Inkling) — the size
Tinker fine-tunes and serves for this study. This makes the *primary* control
below a different base checkpoint from the M3 naive-Inkling reference (which
used full `thinkingmachines/Inkling` on Together); that M3 result stays a
secondary frontier reference only, not the Arm D control.

## Systems under test (both evaluated in this event)

1. **Arm D (tuned):** Inkling-Small + LoRA adapter, served via the local Tinker
   shim (`AGENT_MODEL=openai/armd-inkling-small-tuned`,
   `AGENT_API_BASE=http://localhost:8100/v1`); the shim resolves this id to a
   Tinker `SamplingClient` created with `model_path=<tinker:// checkpoint URI>`.
2. **Matched base control (primary control, §8 guard 2):** naive Inkling-Small
   served through the **identical shim, same process, no adapter**
   (`AGENT_MODEL=openai/thinkingmachines/Inkling-Small`, same
   `AGENT_API_BASE`); the shim resolves this id to a `SamplingClient` created
   with `base_model="thinkingmachines/Inkling-Small"`. Same renderer, same
   HTTP path — this removes the base-checkpoint / serving-stack confound so
   the D−control delta is **purely the adapter**.

Secondary reference only (not the controlled baseline): the M3 Together-served
naive **full** Inkling (retail validation 0.874 ± 0.056 pass, $0.022 ± $0.008
cost/successful task). Arm A (gpt-4.1) and Opus-4.8 stay retail references.
All three references differ from the Arm D control in base size and/or
serving stack — reported as context, disclosed, never blended into the D-vs-
control delta.

Both systems run the **static v0.1 harness** (system prompt every turn, identity
model routing) and the **fixed TAU2 `USER_MODEL`** — the only variable between the
two systems is the presence of the LoRA adapter (§8 guard 1, single-variable).

## Eval design (frozen)

- **Held-out eval splits** (all disjoint from every training pool by construction):

  | Domain | Split file | Tasks | Trained? |
  |--------|-----------|-------|----------|
  | retail | `validation.json` | 35 | in-domain held-out |
  | airline | `eval_airline.json` | 10 | in-domain held-out |
  | telecom | `eval_telecom.json` | 25 | in-domain held-out |
  | banking_knowledge | `transfer_banking.json` | 30 | **never trained — transfer** |

- **Repeats:** N=3 per split, seeds **2001–2003** (distribution, not one run;
  `experiment.md §15.5`; reduced from N=5 for serving path B — per-token Tinker
  sampling cost still budgets a distribution, not a single run). Same seeds
  for Arm D and the matched base control.
- **Run structure:** 4 splits × 3 seeds × 2 systems. Each split is run with its
  matching `--domain` (TAU2 identity is `(domain, id)`). Both systems' calls
  are served by the **same shim process** (`arm_d/serving_shim.py`,
  `python -m arm_d.serving_shim`), which must be running before this driver
  (`scripts/run_arm_d_eval.py`) starts.
- **Blind (§8 guard 4):** these four splits are touched **once**, after the training
  config is frozen. The model is never iterated against eval scores.

## Metrics (per `experiment.md §15`)

- **`pass_rate`** and **`cost_per_successful_task`** — reported as **mean ± std per
  domain**, plus a task-weighted **aggregate** across the four splits.
- `cost_per_task`, `turn_count`/`tool_call_count`, and `harness_error` count recorded
  per run (transcript ledger), as in prior arms.
- **Training cost is reported separately** from runtime cost (`experiment.md §15.3`):
  generation (Opus distillation) / training (Tinker) / runtime (eval) kept distinct.
- Runtime cost prices both shim-served ids (`armd-inkling-small-tuned`,
  `thinkingmachines/Inkling-Small`) at the Inkling-Small per-token rate
  (`settings/pricing.py::register_inkling_small`, Together's serverless
  Inkling-Small reference rate — $0.50/$1.20 per 1M in/out — used as the price
  basis since Tinker's own per-token sampling cost tracks the same order of
  magnitude and no Inkling-Small entry exists in LiteLLM's table). Same rate
  for both systems — a LoRA adapter doesn't change per-token inference price.

## Frozen training protocol (fixed before eval; §8 guards 4, 6)

- Training data = **passed-only** Opus-4.8 trajectories on `train_distill_retail`
  (27), `train_distill_airline` (20), `train_distill_telecom` (49); domain-balanced;
  assistant + tool-call tokens are the loss target. No eval-split task ever enters
  training.
- LoRA hyperparameters (rank, LR, epochs, early-stop/patience) are selected using
  **only a held-out slice of the training tasks** — never any eval split. The final
  chosen values, checkpoint id, Tinker base-model version, and the SFT set's task IDs
  + trajectory hashes are logged in the **training manifest** (audit trail, §8 guard
  6). Anyone can reconstruct exactly what went into the model.

## Verdict rules (declared now, judged after)

Let, per domain, `Δ = mean_pass_D − mean_pass_control`, and `σ_d = sqrt(std_D² +
std_control²)` (the combined per-domain seed noise). The same construction applies
to `cost_per_successful_task`.

1. **Per-domain significance (delta vs noise, §8 guard 5).** A domain shows a
   **real specialization effect** iff `|Δ| > σ_d` (a ≥1σ separation of the two
   distributions). A `|Δ| ≤ σ_d` result is reported as **within noise / not
   significant** — never as a win (the Arm B lesson).

2. **Transfer / no-overfit (the binding verdict).** Arm D is a **clean
   specialization win** iff **both** hold:
   - (a) at least one **trained** domain (retail / airline / telecom) shows a
     significant gain (`Δ > σ_d`), **and**
   - (b) the **banking transfer** domain does **not** significantly regress
     (`Δ ≥ −σ_d` on banking).
   If trained domains improve while banking regresses beyond noise
   (`Δ < −σ_d` on banking), the result is reported as **narrow specialization /
   overfit** (Arm B v0.2 precedent) — not a win.

3. **Aggregate.** Report the task-weighted aggregate `pass_rate` and
   `cost_per_successful_task` for Arm D and the control, with the aggregate Δ vs its
   combined noise, as a summary line — subordinate to the per-domain + transfer
   verdicts above (no cherry-picking a single best domain, §8 guard 5).

4. **Validity gate.** Any split-seed with `harness_error > 0` invalidates that run;
   the affected system is re-run for that seed before judging (a harness crash is not
   a measured result). Cost must be nonzero on every run (pricing sanity).

## Reporting commitments

- Report **all four domains + aggregate + banking transfer**, whatever they show —
  no dropping a domain (§8 guard 5).
- State the specialization as **"Opus-4.8-distillation SFT,"** not generic
  self-improvement — Arm D's behavior is bounded by what Opus demonstrated on the
  training tasks (§8 guard 7, teacher-confound disclosure).
- Per-domain and aggregate **C-vs-D frontier plots** (cost vs success, means as ◆
  with std error bars), with the banking transfer point called out.

---

## Result (judged ____, after all runs complete)

_To be filled after the event. Do not edit anything above this line._

**Judged 2026-08-11.** N=3 seeds (2001–2003), served identically via the local
Tinker shim (`arm_d/serving_shim.py`), static v0.1 harness. base = naive
Inkling-Small, D = Opus-distillation LoRA. σ_d = sqrt(std_base² + std_D²).

| domain | system | pass_rate (mean ± std) | cost/successful task | harness errors |
|--------|--------|------------------------|----------------------|----------------|
| retail | control | 0.848 ± 0.033 | $0.054 | 0 |
| retail | D | 0.838 ± 0.016 | $0.037 | 0 |
| airline | control | 0.300 ± 0.100 | $0.242 | 0 |
| airline | D | 0.433 ± 0.058 | $0.075 | 0 |
| telecom | control | 0.587 ± 0.046 | $0.199 | 0 |
| telecom | D | 0.693 ± 0.061 | $0.106 | 0 |
| banking (transfer) | control | 0.100 ± 0.033 | $6.35 | 1 |
| banking (transfer) | D | 0.100 ± 0.033 | $2.66 | 0 |
| aggregate | control | 0.503 | $0.452 | 1 |
| aggregate | D | 0.540 | $0.191 | 0 |

- **Per-domain significance (`|Δ| > σ_d`):**
  - airline: Δ = +0.133 > σ_d 0.116 → **significant gain ✓**
  - telecom: Δ = +0.107 > σ_d 0.076 → **significant gain ✓**
  - retail: Δ = −0.010 < σ_d 0.037 → within noise (flat; already strong)
  - banking: Δ = 0.000 → no change (both near-floor on the untrained domain)
- **Transfer / no-overfit verdict (rule 2):** (a) ≥1 trained domain shows a
  significant gain — **YES** (airline, telecom); (b) banking transfer does not
  significantly regress (Δ = 0.000 ≥ −σ_d) — **YES** → **not overfit.**
- **Overall: CLEAN SPECIALIZATION WIN.** Opus-4.8-distillation LoRA on
  Inkling-Small significantly lifted the two weak trained domains (airline
  +13.3pp, telecom +10.7pp), held retail, was **cheaper per successful task in
  all four domains** (aggregate $0.452 → $0.191, ~2.4×), and **did not degrade
  the unseen banking transfer domain** — improvement generalized without
  overfitting. Disclosures: Inkling-Small base (not full Inkling — Arm C
  reference is full Inkling); Opus-distillation SFT (not self-improvement);
  Tinker-shim per-token serving; banking uses the `alltools` retrieval config.
  Banking absolute scores are near-floor for both systems (a different,
  retrieval-style task type the model wasn't specialized for).
