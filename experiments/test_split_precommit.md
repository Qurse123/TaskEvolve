# Pre-registration — TAU2 official held-out TEST split (blind, run once)

**Status: PREPARED, NOT RUN.** This document is the pre-committed protocol for
Tier 1 item #2 (validity checklist). The run itself is a **paid, one-time,
irreversible blind event** and is deliberately **not launched** — it awaits
explicit user approval to spend. Nothing below may be revised after the first
test-split result is observed; that is the point of pre-registration.

Written 2026-08-19, before any test-split result exists.

---

## 1. Why this run exists

Every number in the study so far is on splits **carved from TAU2's `train`
tasks** (our `proxy` = 12 and `validation` = 35 retail tasks, plus the Arm D
train_distill/eval carves). A research paper cannot report generalization off
development data. TAU2 ships an **official `test` split** disjoint from `train`;
running it once, blind, is the honest headline number.

Verified available (via `get_tasks(task_split_name="test")`):

| Domain | official `train` | official `test` (held-out) |
|--------|------------------|----------------------------|
| retail | 74 | **40** |
| airline | 30 | **20** |
| telecom | 74 | **40** |
| banking_knowledge | (no train/test split; our `transfer_banking` = 30 is already zero-shot) | — |

The `test` tasks are disjoint from `train`, therefore disjoint from our
proxy/validation and from the Arm D distillation set (a subset of `train`). This
is the first truly held-out evaluation in the study.

## 2. What will be run (pre-committed scope)

### AMENDMENT — executed scope (2026-08-19, written before any test result)

The full arm set below (§2b) is the *intended* design; the **actually executed**
run is a **budget-constrained open-weight subset**, chosen because the available
credits are Together ~$16 + Tinker ~$20 + a small OpenAI allowance (no Anthropic
budget confirmed, so Opus / Arm B are deferred; the gpt-4.1 *agent* arm is
dropped). **Executed arms**, all static **v0.1** harness, **retail test (40
tasks)**, **N=5**, seeds `4001–4005`:

1. `Inkling` (Arm C) — served via Together (`AGENT_API_BASE=https://api.together.xyz/v1`).
2. `Inkling-Small` **base** — served via the local Tinker shim (port 8100).
3. `armd-inkling-small-tuned` (Arm D) — served via the same shim.

User-simulator stays OpenAI (unchanged from every prior arm — required for
comparability). Airline/telecom test and all Anthropic/OpenAI-agent arms are
**deferred** (not run) for budget; they remain pre-registered in §2b for later.

**Honest limitation of this subset:** on the *dev* retail carve, fine-tuning was
essentially **flat** (base 0.848 → tuned 0.838, tuned cheaper); the large Arm D
gains were in airline/telecom, which this subset does NOT cover. So this run
tests **"no regression + cheaper on held-out retail,"** not the cross-domain
headline. Reported accordingly.

### §2b — full intended arm set (deferred remainder)

- Model axis (retail test): `gpt-4.1`, `Inkling` (Arm C), `claude-opus-4-8`.
- Harness axis (retail test): Arm B `v0.2` (`claude-opus-4-8` routed → Sonnet 5).
- Specialization (retail + airline + telecom test): Inkling-Small **base** and
  `armd-inkling-small-tuned`, served via the Tinker shim.

**Seeds (pre-committed):** `4001, 4002, 4003, 4004, 4005` — **N = 5** for every
arm, including the Arm D pair (this also fixes the earlier Arm D N=3 weakness;
the statistics analysis showed n=3 cannot clear a corrected significance bar).
Seeds are distinct from proxy (1001–1005) and validation (2001–2005).

**Metric:** `cost_per_successful_task` (headline) and `pass_rate`, reported as
**mean ± std over the 5 seeds**. Cost accounting unchanged (agent cost only;
user-sim cost reported separately per the cost-fairness analysis).

## 3. Pre-committed generation + run commands (NOT executed)

**(a) Generate the frozen test-split JSON files** ($0, deterministic — task ids
only). Add a `test`-split generator mirroring `benchmark/splits.py`, producing
`benchmark/splits/test_retail.json` (40), `test_airline.json` (20),
`test_telecom.json` (40). These become **frozen** the moment they are generated
(same rule as the other splits).

**(b) Run (detached via `nohup`; each is long — launch one at a time):**

```
# Model axis — retail test
AGENT_MODEL=openai/gpt-4.1                 HARNESS_VERSION=v0.1                     python -m scripts.run_train_eval --split test_retail --repeats 5 --seed-start 4001
AGENT_MODEL=openai/thinkingmachines/Inkling AGENT_API_BASE=https://api.together.xyz/v1 HARNESS_VERSION=v0.1 python -m scripts.run_train_eval --split test_retail --repeats 5 --seed-start 4001
AGENT_MODEL=anthropic/claude-opus-4-8 HARNESS_VERSION=v0.1 AGENT_NO_TEMPERATURE=1  python -m scripts.run_train_eval --split test_retail --repeats 5 --seed-start 4001
# Harness axis — retail test (Arm B v0.2)
AGENT_MODEL=anthropic/claude-opus-4-8 HARNESS_VERSION=v0.2 AGENT_NO_TEMPERATURE=1  python -m scripts.run_train_eval --split test_retail --repeats 5 --seed-start 4001
# Specialization — retail/airline/telecom test (shim must be running: python -m arm_d.serving_shim)
for M in thinkingmachines/Inkling-Small armd-inkling-small-tuned; do
  for S in "test_retail retail" "test_airline airline" "test_telecom telecom"; do
    set -- $S; AGENT_MODEL=openai/$M AGENT_API_BASE=http://localhost:8100/v1 HARNESS_VERSION=v0.1 \
      python -m scripts.run_train_eval --split $1 --domain $2 --repeats 5 --seed-start 4001
  done
done
```

(Exact model-string prefixes to be confirmed against `.env` at run time; the
above mirror the validation commands.)

## 4. Pre-committed success criteria (what confirms vs refutes each claim)

### Executed-subset criteria (the open-weight retail-only run)

Pre-committed now, from the dev-retail predictions:

- **D-RETAIL (primary): fine-tuning does not regress held-out retail and is
  cheaper.** CONFIRMED iff tuned `pass_rate` ≥ base − σ_d (no significant
  regression; σ_d = √(σ_base²+σ_tuned²)) AND tuned `cost_per_successful_task` <
  base. (Dev predicted 0.848→0.838, tuned cheaper.) A tuned *gain* > σ_d would be
  a bonus, not required.
- **C-RETAIL (secondary): naive open-weight Inkling holds up on held-out retail.**
  Report Inkling test `pass_rate` vs its dev mean (0.874 ± 0.056); CONFIRMED iff
  within 1σ. Descriptive — Inkling vs Inkling-Small size gap also reported.

The four claims below (§4b) require the deferred arms and are NOT tested by this
run; retained for the later full event.

### §4b — full-design criteria (deferred)

Set **now**, from the validation-split predictions, so the test result can only
confirm or refute — not be reinterpreted after the fact.

1. **Model axis — "open-weight ≈ frontier at far lower cost."**
   CONFIRMED iff, on retail test: Inkling `pass_rate` within **5 pp** of Opus
   (≥ Opus − 0.05) AND Inkling `cost_per_successful_task` **≥ 10× lower** than
   Opus. REFUTED if either fails. (Validation predicted 0.874 vs 0.903, ~26×.)
2. **gpt-4.1 is dominated** by Inkling. CONFIRMED iff Inkling ≥ gpt-4.1 on
   success AND cheaper per successful task. (Validation: 0.874 vs 0.806.)
3. **Arm B (Opus→Sonnet) transfers.** CONFIRMED iff retail-test success within
   **1σ** of its validation mean (0.851 ± 0.051) AND cost/successful task ≤ Opus
   v0.1. (Guards against a second proxy-overfit.)
4. **Arm D specialization holds out-of-development.** CONFIRMED per domain iff
   tuned − base `pass_rate` > σ_d = √(σ_base² + σ_tuned²) with the SAME sign as
   on the dev carve (airline +, telecom +, retail ≈ 0, banking no regression),
   AND tuned cheaper per successful task. Any sign flip on airline/telecom is
   reported as a **failure to generalize** (as with Arm B proxy-overfit) — not
   quietly dropped.

## 5. Blind-once discipline (hard rules)

- Run **exactly once** per arm. No re-runs, no seed additions, no harness edits
  after any test result is seen. A crashed run (harness_error) may be re-launched
  only if it produced **no** scored tasks.
- Do **not** iterate harness/model/prompts against test outcomes — that converts
  test into a second development set and voids its purpose.
- Report all pre-registered arms whether they confirm or refute. Any forced
  deviation from this document is logged in a "deviations" section appended
  *after* the fact, with reasons.
- Results append to `experiments/results.csv` as `test_*` rows and plot into the
  frontier figures as a distinct, clearly-labeled event.

## 6. Rough scope (for the approval decision — not a cost claim)

7 arm×domain run-sets × 5 seeds over 20–40 multi-turn tasks each: real spend on
Anthropic + Together + Tinker and several detached hours. **Approve the scope
(full vs minimum-viable) before launch.** Minimum-viable if constrained:
retail-test model axis (gpt-4.1, Inkling, Opus) + Arm D base/tuned on
retail+airline+telecom — validates the two headline claims; Arm B test follows.

---

## 7. OUTCOME (recorded 2026-08-20, after the blind run)

Executed subset (Inkling + Inkling-Small base + tuned, retail test, N=5, seeds
4001–4005) ran **clean**: every run rc=0, no harness errors, Tinker budget
survived the full 10-run shim chain.

**Results (`test_retail`, N=5, mean ± std):**

| Arm | success | cost / successful task | cost / task |
|-----|---------|------------------------|-------------|
| Inkling (Arm C, Together) | 0.895 ± 0.069 | $0.0110 ± 0.0007 | $0.0098 |
| Inkling-Small base | 0.915 ± 0.038 | $0.0416 ± 0.0023 | $0.0380 |
| Inkling-Small tuned (Arm D) | 0.875 ± 0.040 | $0.0329 ± 0.0021 | $0.0287 |

**Verdict vs the pre-committed criteria (§4 executed-subset):**

- **D-RETAIL (primary) — CONFIRMED.** tuned − base = **−0.040** success,
  σ_d = √(0.038²+0.040²) = **0.055** → within noise (no significant regression;
  tuned 0.875 ≥ base − σ_d = 0.860). Tuned **cheaper per successful task**
  ($0.0329 < $0.0416, −21%). The dev-retail result transfers out-of-sample.
- **C-RETAIL (secondary) — CONFIRMED.** Inkling 0.895 ± 0.069 within 1σ of its
  dev mean 0.874 ± 0.056; also the cheapest per successful task ($0.011).

**Interpretation / limits:**
- Confirms the **modest** retail claim (fine-tuning = cheaper, not more accurate;
  the −4pp success is inside σ_d), **not** the headline cross-domain Arm D gains
  (airline/telecom), which this budget-limited subset did not run.
- No overfitting surprise: retail held out-of-sample as dev predicted.
- **Deviation from §2b:** Opus / Arm B / gpt-4.1-agent arms and airline/telecom
  test were deferred for budget — pre-registered as deferred *before* launch
  (§2 amendment), not a post-hoc drop. They remain available for a funded run.

---

## 8. AMENDMENT 2 — funded follow-up batch (pre-registered 2026-08-22, before results)

A second funded batch executes prior priorities **1–3**. The **gpt-4.1 agent arm
is permanently dropped** (user decision); the OpenAI user-simulator is unchanged
(required for comparability). Arm B stays deferred. All runs: static **v0.1**
harness, **N=5, seeds 4001–4005**, held-out `test_*` splits (generated by
`benchmark.splits_arm_d.generate_test_splits`, verified disjoint from train).

**Runs (all pre-committed here, none yet executed):**
- **Item 1 — Arm D cross-domain held-out.** `Inkling-Small` base + `armd-inkling-small-tuned`
  on `test_airline` (20) + `test_telecom` (40), via the Tinker shim.
- **Item 2 — frontier reference.** `claude-opus-4-8` on `test_retail` (40),
  native anthropic, `AGENT_NO_TEMPERATURE=1`.
- **Item 3 — ablation A3 (domain mix).** Train a **retail-only** Inkling-Small
  LoRA (pointer `experiments/arm_d_distill_runs_retail.json`, else identical
  config: rank 32, lr 1e-4), serve under a distinct id
  `armd-inkling-small-retail-tuned`, eval on `test_airline` + `test_telecom`.

**Pre-committed criteria (set now):**
1. **Item 1 — cross-domain transfer CONFIRMED** iff, per domain, tuned − base
   `pass_rate` > σ_d with the SAME sign as dev (airline +13.3pp, telecom +10.7pp)
   AND tuned cheaper per successful task. A within-noise or sign-flipped result =
   **cross-domain gain did NOT hold out-of-sample** (reported honestly, not dropped).
2. **Item 2 — frontier claim CONFIRMED** iff on `test_retail` the already-measured
   Inkling (0.895 ± 0.069 @ $0.011/succ) is within **5 pp** of Opus AND **≥10×**
   cheaper per successful task. Else refuted / reframed.
3. **Item 3 — mechanism.** Compare retail-only-tuned vs all-3-domain-tuned vs base
   on airline/telecom. Pre-committed reading: if **retail-only-tuned ≈ base**
   (Δ ≤ σ_d) while **all-3-tuned > base**, the cross-domain gains **require
   multi-domain distillation** (transfer is not generic). If retail-only-tuned
   **also** beats base, the gains are **generic distillation transfer** (domain
   mix not required). Either outcome is reportable and informative.

**Serving/pricing note (execution-time):** the new id
`armd-inkling-small-retail-tuned` must be (a) routed in `arm_d/serving_shim.py`
to its own `tinker://` checkpoint path (produced by the A3 training run) and
(b) registered in `settings/pricing.py` via `register_inkling_small(...)` — else
LiteLLM records $0 cost. Both are $0 edits done after training yields the path.
