# Data-contamination audit

Tier 1 validity item #5. Two distinct leakage concerns are addressed separately:

- **(A) Train/eval disjointness** — provable at $0, and it is *proven* below.
- **(B) Benchmark-in-pretraining memorization** — cannot be ruled out at $0;
  discussed honestly with concrete probes.

Reproduce (no network, no paid calls):

```bash
python -m scripts.check_split_disjointness            # prints the matrix, exits nonzero on any overlap
python -m pytest tests/test_check_split_disjointness.py -q
```

---

## (A) Train/eval disjointness — STATED GUARANTEE

**Guarantee.** The set of tasks used to fine-tune Arm D (Opus-distilled traces
drawn from the `train_distill_*` splits) is **disjoint from every split Arm D is
evaluated on** (`validation`, `eval_airline`, `eval_telecom`,
`transfer_banking`). No task appears in both a training and an evaluation role.
This holds both for the *split definitions* and for what was *actually trained
on* per the SFT manifest.

### Keying: `(domain, id)`, not `id`

Task ids in `benchmark/splits/*.json` are flat string lists; the **domain comes
from the file**, and ids are only unique *within* a domain. The numeric id `"7"`
is a retail task in `validation.json`, an airline task in `eval_airline.json`,
and an airline task in `train_distill_airline.json` — three different tasks.
Keying by raw id alone would report **5 false "overlaps"** (raw ids `0, 11, 14,
15, 28` recur across domains); every one dissolves once keyed by `(domain, id)`.
The audit therefore keys strictly by `(domain, id)` (domain map mirrors
`scripts.plot_arm_d.SPLIT_TO_DOMAIN`; `train_distill_<d>` → `<d>`).

### Overlap matrix (cells = |intersection| by `(domain, id)`)

| train ↓ / eval → | validation (retail) | eval_airline | eval_telecom | transfer_banking |
|---|---|---|---|---|
| train_distill_retail  | 0 | 0 | 0 | 0 |
| train_distill_airline | 0 | 0 | 0 | 0 |
| train_distill_telecom | 0 | 0 | 0 | 0 |

**train_distill ∩ eval = ∅.**

### Exact counts

| Set | Unique `(domain, id)` tasks |
|---|---|
| train_distill union (retail 27 + airline 20 + telecom 49) | **96** |
| eval union (validation 35 + eval_airline 10 + eval_telecom 25 + transfer_banking 30) | **100** |
| train_distill ∩ eval | **0** |
| SFT manifest (`experiments/arm_d_sft_manifest.json`), unique tasks | **75** (195 rows incl. repeats/multi-run) |
| SFT manifest tasks **not** in any `train_distill` split | **0** (manifest ⊆ train_distill) |
| SFT manifest ∩ eval union | **0** |

The SFT manifest is the ground truth of *what the model actually saw*: 195 rows
(a task can appear in multiple distillation runs) collapse to 75 unique
`(domain, id)` tasks, all of which are a subset of the `train_distill_*` splits
and none of which appear in any eval split. Manifest domains: retail 70, telecom
69, airline 56 rows.

The script `assert`s all three conditions and exits nonzero on any violation, so
a future split edit that reintroduces overlap fails CI rather than passing
silently. `transfer_banking` is a genuine zero-shot transfer domain — it has **no**
`train_distill_banking` counterpart, so nothing banking was ever trained on.

**Note — proxy/smoke are out of scope for this guarantee.** `proxy` (retail
train, used by the Arm B iterator) and `smoke` (mock wiring) are neither Arm D
training nor Arm D eval sets, so they are reported for completeness but excluded
from the train/eval assertion. (Incidentally `proxy ∩ train_distill_retail = ∅`
too, so the iterator's proxy tasks were not distilled either.)

---

## (B) Benchmark-in-pretraining memorization — honest discussion

TAU2-bench is a public benchmark. The models in this study — Inkling (base and
the Opus-distilled Arm D tune), Opus 4.8 (the distillation teacher and frontier
reference), and gpt-4.1 (Arm A) — all have **unknown, closed pretraining
corpora**. We cannot exclude the possibility that any of them ingested TAU2 task
text, its `wiki`/policy documents, or public solution transcripts during
pretraining. **This cannot be ruled out at $0** (it requires either training-set
access we don't have, or new paid runs on held-out novel tasks).

### What it would bias, and in which direction

- **Absolute success rates would be inflated** for any model that memorized
  tasks or policies — the reported numbers become an upper bound, not a clean
  measure of reasoning-from-policy.
- **Cross-arm *comparisons* are more robust than absolute levels.** All arms run
  the *same* tasks through the *same* harness, so a memorization advantage would
  have to differ *by model* to distort the ranking. It plausibly does differ
  (different corpora/cutoffs), so this is a real but bounded threat to the
  headline comparisons, not just the absolute numbers.
- **The efficiency finding is less exposed than the accuracy finding.** The
  paper's headline is *cost per successful task*. Memorization mainly moves the
  success axis; the cost axis (tokens × price) is comparatively insulated.

### The Opus-as-teacher angle (explicitly not eval leakage)

Arm D is distilled from **Opus** traces. This is worth stating plainly because
it *sounds* like leakage and is not:

- Eval is **not** run on Opus. Opus only generated *training* traces, on the
  **disjoint** `train_distill_*` tasks (proven in §A).
- Distilling from a stronger teacher on tasks disjoint from the test set is a
  **legitimate, standard** knowledge-transfer setup — the student learns a
  policy/behavior, not the answers to graded tasks.
- The one thing to keep honest: if **Opus itself** memorized TAU2 during *its*
  pretraining, that memorized behavior could propagate into the student via
  distillation. That is a variant of concern (B) (teacher pretraining
  contamination), **not** a train/eval split violation — and it is bounded by
  the same probes below (esp. perturbed/novel-task tests, which break memorized
  answers regardless of whether the model or its teacher memorized them).

### Recommended probes (to move (B) from "unaddressed" toward "bounded")

1. **Perturbed-task / counterfactual probe.** Semantically rewrite a sample of
   eval tasks (rename entities, change order/amounts, swap policy constants)
   while preserving difficulty. A model reasoning from policy holds its success
   rate; a model reciting memorized transcripts drops. Run per model — the
   *gap* between original and perturbed is the memorization signal.
2. **Canary / verbatim-recall test.** Prompt each model to continue or complete
   TAU2 task text, wiki/policy passages, or known transcript prefixes. Fluent
   verbatim continuation is direct evidence of ingestion. Cheap and $0-adjacent
   for the closed models via a few probe prompts (still a paid call — precommit
   the probe set first).
3. **Held-out genuinely-novel tasks.** Author a small set of new
   retail/airline/telecom tasks in the TAU2 format that post-date all model
   cutoffs and were never published. Success there is contamination-free by
   construction; compare to public-split success. `transfer_banking` already
   serves partly as a novelty/transfer control (no banking training, and a
   domain the study introduced).
4. **Release-date vs publication-date check.** Record each model's
   pretraining-cutoff / release date against the TAU2 (and predecessor
   TAU-bench) publication and dataset-release dates. A cutoff *before* TAU2
   publication is strong evidence a model could **not** have memorized it; a
   cutoff after is a *necessary* (not sufficient) condition for contamination.
   This is a documentation task (provenance table, Tier 2 item #13) — do it for
   all three model families and state each verdict as could-not / could-have.
5. **Memorization-sensitive scoring.** Where feasible, report success on the
   perturbed set as the *conservative* headline and the public-split success as
   the optimistic bound, so readers see the contamination-adjusted range rather
   than a single possibly-inflated number.

### Honest bottom line

Concern (A) is **closed**: train/eval are provably disjoint by `(domain, id)`,
and the SFT manifest confirms the model trained only on the disjoint training
tasks. Concern (B) is **open and inherent to evaluating closed models on a
public benchmark**; it is not resolvable at $0. The study should (i) state (B)
as a threat to *absolute* success levels, (ii) note the *comparison* and the
*cost*-axis finding are less exposed, and (iii) precommit the probes above —
especially the perturbed-task and release-date checks — before the paid,
run-once test-split event (Tier 1 item #2).
