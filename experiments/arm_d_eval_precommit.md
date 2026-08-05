# Arm D Eval — Pre-registration (committed before any tuned-model eval)

- **Date declared:** 2026-08-05 (before the Arm D evaluation event starts).
- **Status:** precommitted. Once the Arm D adapter is trained and its config is
  frozen, the criteria below are fixed — no post-hoc changes after any eval score
  is seen (design spec §8 guard 3; mirrors `validation_event_v0.2_precommit.md`).
- **Arm:** D — *specialized open-weight model + static v0.1 harness*
  (`experiment.md §13.4`). Specialization = **Opus-4.8-distillation SFT** (LoRA on
  `thinkingmachines/Inkling` via Tinker). The measurement this arm exists to
  produce is the **C→D delta on held-out tasks in every domain plus a fully unseen
  transfer domain** — the marginal value of model specialization.

## Systems under test (both evaluated in this event)

1. **Arm D (tuned):** Inkling + LoRA adapter, served via the Together OpenAI-compat
   seam (`AGENT_MODEL=openai/<served_id>`, `AGENT_API_BASE=https://api.together.xyz/v1`).
2. **Matched base control (primary Arm C control, §8 guard 2):** base Inkling served
   through the **identical serving path, no adapter**. This removes the base-version
   / serving-stack confound so the D−control delta is **purely the adapter**.

Secondary reference only (not the controlled baseline): the M3 Together-served
naive Inkling (retail validation 0.874 ± 0.056 pass, $0.022 ± $0.008
cost/successful task). Arm A (gpt-4.1) and Opus-4.8 stay retail references.

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

- **Repeats:** N=5 per split, seeds **2001–2005** (distribution, not one run;
  `experiment.md §15.5`). Same seeds for Arm D and the matched base control.
- **Run structure:** 4 splits × 5 seeds × 2 systems. Each split is run with its
  matching `--domain` (TAU2 identity is `(domain, id)`).
- **Blind (§8 guard 4):** these four splits are touched **once**, after the training
  config is frozen. The model is never iterated against eval scores.

## Metrics (per `experiment.md §15`)

- **`pass_rate`** and **`cost_per_successful_task`** — reported as **mean ± std per
  domain**, plus a task-weighted **aggregate** across the four splits.
- `cost_per_task`, `turn_count`/`tool_call_count`, and `harness_error` count recorded
  per run (transcript ledger), as in prior arms.
- **Training cost is reported separately** from runtime cost (`experiment.md §15.3`):
  generation (Opus distillation) / training (Tinker) / runtime (eval) kept distinct.

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

| domain | system | pass_rate (mean ± std) | cost/successful task (mean ± std) | harness errors |
|--------|--------|------------------------|-----------------------------------|----------------|
| retail | control |  |  |  |
| retail | D |  |  |  |
| airline | control |  |  |  |
| airline | D |  |  |  |
| telecom | control |  |  |  |
| telecom | D |  |  |  |
| banking (transfer) | control |  |  |  |
| banking (transfer) | D |  |  |  |
| aggregate | control |  |  |  |
| aggregate | D |  |  |  |

- Per-domain significance (`|Δ| > σ_d`): _…_
- Transfer / no-overfit verdict (rule 2): _…_
- Overall: **clean specialization win / narrow-specialization overfit / no
  significant effect** — _…_
