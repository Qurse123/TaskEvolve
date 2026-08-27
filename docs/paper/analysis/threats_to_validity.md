# Threats to Validity

Tier 2 validity item #8. This section catalogues the study's threats in the
standard empirical software-engineering four categories — **internal**,
**external**, **construct**, and **statistical-conclusion** validity — and for
each threat states the mitigation in place or, where none is possible at $0, an
honest acknowledgement. Numbers are pulled from the analysis notes in this
directory (`statistics.md`, `cost_fairness.md`, `contamination_audit.md`,
`amortized_cost.md`, `related_work_baselines.md`) and from
`experiments/test_split_precommit.md`.

All headline comparisons are on the **validation (retail-train) split, N=5**,
unless marked; cross-domain Arm D is **N=3**; the executed held-out test event is
**retail-only, open-weight-only, N=5**.

---

## 1. Internal validity (are the measured differences caused by the variable we changed?)

**1.1 The shared harness (v0.1) was tuned for gpt-4.1.**
The static v0.1 harness — system prompt, policy summary, tool schemas, history
handling — was authored and iterated against gpt-4.1 during Milestone 1. Running
every other model (Opus, Inkling, Inkling-Small ± tune) in that same harness
holds the harness *constant* (good for a clean model axis) but may
systematically **disadvantage models whose ideal prompting differs** from
gpt-4.1's. A model could score lower not because it is weaker but because the
prompt/tool framing suits it worse.
*Mitigation / status:* Pinning v0.1 across the model axis is a deliberate
single-variable control — it trades "best per-model harness" for "same harness,
comparable models," which is the honest way to isolate the model lever. We
explicitly did **not** run Opus in the gpt-4.1-tuned Arm B v0.2 harness for the
mirror-image reason (that would confound model with harness). The residual bias
is real but bounded and one-directional (it can only *understate* non-gpt-4.1
models); quantifying it would require a per-model harness sweep (Tier 3 #17,
not done). Flagged as an open bound, not resolved.

**1.2 The user-simulator is a fixed strong closed model (confound + uncounted cost).**
Every arm is evaluated against TAU2's stock user-simulator, `gpt-4.1-2025-04-14`.
This is a confound in two ways. (a) The *quality* of the simulated user is held
constant across arms, which is correct for comparability, but it means all
results are conditional on one particular (strong, OpenAI) user model; a weaker
or differently-behaved simulator could shift absolute success and even re-order
arms. (b) The headline agent cost **excludes** the user-sim's own token cost.
Per `cost_fairness.md §(d)`, that excluded cost is roughly constant in absolute
dollars but is a *very different share* of each arm: **2.2% of agent cost for
Opus but ~85% for Inkling, ~30% for Inkling-Small base, ~41% for the tuned
model**. Because it is a large share of the cheap open arms, excluding it
**overstates** the open models' all-in cost advantage.
*Mitigation / status:* the user-sim model is identical across arms, so its effect
on the *ranking* is limited; and `cost_fairness.md` discloses the excluded cost
per arm so a reader can add it back. Acknowledged confound, disclosed and
quantified rather than removed.

**1.3 Provider / serving differences across arms.**
The arms are not served through one uniform stack: closed models (gpt-4.1, Opus
4.8) run on their native vendor APIs at retail list price; Inkling (Arm C) is
served via **Together AI's OpenAI-compatible endpoint**; the tuned Inkling-Small
(Arm D) runs through a **local Tinker serving shim** (port 8100). Serving path
differences can affect latency, sampling defaults, tokenization, tool-schema
handling, and — critically — the cost basis (see construct validity 3.2). A known
concrete instance: LiteLLM's `together_ai` provider silently drops tool schemas,
forcing the `openai/` + `api_base` wiring — evidence that serving-layer behavior
is not neutral across providers.
*Mitigation / status:* wiring was verified per provider (tools confirmed to flow;
Inkling registered in the pricing table so cost is recorded, not $0). The cost
basis mismatch is handled head-on in `cost_fairness.md` (normalized to a common
$/token schedule). Serving-path effects on *success* are not separately measured;
acknowledged.

**1.4 Non-determinism (sampling, temperature, provider variability).**
The benchmark is stochastic: an LLM user-simulator, LLM agents, and provider-side
non-determinism even at temperature 0 (noted in Milestone 1). This is why every
point is reported as a distribution over fixed seeds, not a single run. A subtle
per-arm wrinkle: Opus 4.8 **deprecates the `temperature` parameter**, so it is
run with `AGENT_NO_TEMPERATURE=1` (temperature dropped) while other arms pass
temperature — a small sampling-configuration asymmetry across the model axis.
*Mitigation / status:* fixed seeds (proxy 1001–1005, validation 2001–2005, test
4001–4005) and N=5 (N=3 for Arm D cross-domain) with mean ± std reporting
directly target this threat. The residual is the low n itself (see statistical
validity). The Opus temperature asymmetry is forced by the provider and noted,
not correctable.

**1.5 Benchmark-in-pretraining memorization.**
TAU2 is a public benchmark; all study models have unknown, closed pretraining
corpora, so none can be excluded from having ingested TAU2 task text, policy
docs, or public solution transcripts. This would **inflate absolute success**
(the numbers become an upper bound).
*Mitigation / status:* per `contamination_audit.md §B`, cross-arm *comparisons*
are more robust than absolute levels (all arms run the same tasks through the
same harness; a memorization edge would have to differ by model to distort the
ranking), and the *cost*-axis headline is comparatively insulated because
memorization mainly moves the success axis. Train/eval *disjointness* is
separately **proven** at $0 (`contamination_audit.md §A`: 0 overlaps keyed by
`(domain, id)`; the SFT manifest is a subset of the disjoint `train_distill_*`
splits). Memorization itself cannot be ruled out at $0; probes (perturbed-task,
canary-recall, novel held-out tasks, release-date checks) are pre-committed but
not executed. Open and inherent to evaluating closed models on a public
benchmark.

**1.6 Distillation teacher = the frontier reference (Arm D).**
Arm D is distilled from Opus traces, and Opus is also a reported arm. This *sounds*
like leakage.
*Mitigation / status:* eval is never run on Opus; Opus only generated *training*
traces on the disjoint `train_distill_*` tasks (`contamination_audit.md §A`).
Distilling from a stronger teacher on tasks disjoint from the test set is a
standard, legitimate transfer setup. The one residual — if Opus itself memorized
TAU2, that could propagate into the student — is a variant of 1.5, bounded by the
same (unexecuted) perturbed-task probes.

---

## 2. External validity (do the results generalize?)

**2.1 A single benchmark, mostly one domain.**
All results are on TAU2-bench, and the great majority of runs are **retail**. The
model-axis headlines (Arm A, Opus, Arm B, Arm C) are retail-only; the executed
held-out test event is retail-only. Findings may not transfer to other agent
benchmarks or to non-customer-service task types.
*Mitigation / status:* Arm D reaches beyond retail into airline, telecom, and
(zero-shot) banking, giving *some* cross-domain evidence — but only at N=3 and
only for the fine-tuning question. Single-benchmark scope is acknowledged;
widening to a second benchmark is Tier 3 #16 (not done).

**2.2 Small task counts, and splits are train-derived subsets.**
Proxy is **12 tasks**, validation is **35 tasks** — both seeded subsets of TAU2's
retail *train* pool, **not** the official 115-task retail set (nor the 50 airline
/ 114 telecom sets). This is a difficulty-mix confound: our gpt-4.1 retail 0.806
runs ~6pp above Sierra's published GPT-4.1 pass^1 of 0.74, plausibly because our
subset is easier. Our telecom base (0.587) even *exceeds* Sierra's GPT-4.1
telecom pass^1 (0.34), almost certainly a subset/metric artifact rather than a
real ordering.
*Mitigation / status:* `related_work_baselines.md` positions our numbers against
published references and flags every such gap; the gpt-4.1 retail anchor lands us
"in the right neighborhood," arguing the harness is not broken. Absolute levels
are not claimed as directly comparable to published full-set numbers.

**2.3 The held-out test event was a retail-only, open-weight subset.**
The one truly held-out evaluation (TAU2 official `test` split) was executed only
for the open-weight arms (Inkling, Inkling-Small base + tuned) on **retail test
(40 tasks), N=5** (`test_split_precommit.md §7`). Opus, Arm B, and the gpt-4.1
*agent* arm, plus airline/telecom test, were **deferred for budget** (pre-
registered as deferred *before* launch, not dropped post-hoc). So the held-out
evidence confirms only the *modest* retail claim — "fine-tuning does not regress
held-out retail and is cheaper" (tuned − base = −0.040 success, within σ_d=0.055;
tuned −21% cost/succ) — **not** the headline cross-domain Arm D gains
(airline/telecom), which were never run on held-out data.
*Mitigation / status:* the executed subset was pre-committed with explicit
confirm/refute criteria and ran clean (both criteria CONFIRMED). The deferred
arms remain pre-registered for a funded run. Honestly scoped: the cross-domain
headline rests on *development-carve* (dev) data, not held-out data.

**2.4 Cross-domain Arm D is N=3, and the generalization has a hard floor.**
The airline (+13.3pp) and telecom (+10.7pp) tuned gains — the paper's most
striking generalization claim — are each **N=3**. Banking sits at **0.100 → 0.100**
(no movement): specialization did not transfer to a retrieval/knowledge domain it
never trained on, an honest ceiling on the generalization claim.
*Mitigation / status:* the banking floor is reported as a failure-to-generalize,
not hidden. The N=3 gains are reported with the σ_d significance rule and
explicitly flagged as underpowered (see 4.1–4.2); topping up to N=5 is an open
recommendation.

---

## 3. Construct validity (does `cost_per_successful_task` measure what we claim?)

**3.1 Is cost-per-successful-task the right metric, and what does it hide?**
The headline scalar is `cost_per_successful_task = total_cost / (num_tasks ×
pass_rate)`. It is decision-relevant (enterprises pay per token and care about
completed tasks) and it collapses the (cost, success) frontier into one number —
but that collapse hides several things:
- **Latency.** Cost is tokens×price; it says nothing about wall-clock or
  turn-count time-to-complete. A cheap-but-slow agent and an expensive-but-fast
  agent can look identical. Turn/tool counts are logged but wall-clock latency is
  not a reported axis (Tier 2 #10, not done).
- **Partial credit.** TAU2 scores task success as pass/fail; a task that is 90%
  complete counts identically to one that failed immediately. Both the numerator
  (a failed task still costs tokens) and the denominator (binary success) ignore
  partial progress.
- **Catastrophic-failure weighting.** The metric treats a benign non-completion
  and a harmful wrong action (e.g. a wrong refund) the same — it has no notion of
  failure severity or safety cost.
- **Consistency (pass^k).** Our mean-of-seeds success ≈ *average pass^1*; it does
  **not** capture the run-to-run consistency that TAU2's native `pass^k` (all-of-k)
  metric is designed to surface. An inconsistent model looks better under our
  metric than under pass^k.
*Mitigation / status:* the study frames results as a **frontier, not a single
score**, and reports pass_rate and cost/task alongside the combined metric so a
reader can inspect either axis. Metric justification is Tier 2 #9 and the
pass^k mismatch is disclosed in `related_work_baselines.md`. The hidden
dimensions (latency, partial credit, severity) are acknowledged gaps.

**3.2 The open-vs-closed cost comparison conflates pricing with efficiency.**
The headline "~26× cheaper" (Inkling $0.0221/succ vs Opus $0.5768/succ) compares
**retail closed-API list prices** (which bake in commercial margin) against the
study's **own open-model serving cost** (Together's near-cost rate; local Tinker
shim). Stated as "26× cheaper" it reads as an intrinsic *model* property; it is
mostly a *pricing* artifact. Per `cost_fairness.md §(c)`: at a single common
$/token schedule, Opus spends 97,499 agent tokens/task vs Inkling 102,726 (1.05×)
— re-priced identically, Inkling is **0.97× the cost of Opus**, i.e. **no
efficiency advantage** (marginally more tokens). The ~26× is therefore **almost
entirely commercial pricing**, not model efficiency.
*Mitigation / status:* `cost_fairness.md` re-prices every arm at one common
schedule and states the two claims separately — "26× cheaper at listed prices
today" (a real invoice fact) vs "≈equal work at equal $/token" (efficiency). A
genuine efficiency result survives normalization: the Opus-distilled tuned
Inkling-Small uses **1.49× fewer tokens** than the base at identical serving — a
real model-efficiency gain because pricing is held constant. Threat is directly
addressed and reframed rather than left implicit.

---

## 4. Statistical-conclusion validity (are the statistical inferences sound?)

**4.1 Tiny sample sizes (n = 3–5).**
Each arm has only 3–5 logged runs. Bootstrap 95% CIs are correspondingly wide and
permutation tests have low power (`statistics.md`).
*Mitigation / status:* N=5 was the pre-committed minimum for the model/harness
axes; Arm D cross-domain is N=3 and flagged as underpowered throughout. Non-
significant results are reported as *underpowered, not evidence of no effect*.

**4.2 No comparison survives multiple-comparison correction; and n=3 has a p-floor.**
Across the headline family of 14 tests, **0/14 survive Holm correction at
α=0.05** (`statistics.md`), despite very large effect sizes (e.g. Cohen's d up to
−19 for Inkling-vs-Opus cost). This is a power problem, not an effect-size
problem: with **n=3 vs n=3 the smallest attainable two-sided permutation p is
2/20 = 0.10**, so the Arm-D per-domain tests *cannot reach p<0.05 no matter how
large the effect*. (n=5 vs n=5 floors at ≈0.0079.)
*Mitigation / status:* effect sizes, bootstrap CIs, and both raw and Holm-adjusted
p-values are all reported so the reader sees the effect/power split explicitly.
The paper leans on effect sizes and the pre-committed σ_d rule for Arm D rather
than claiming statistical significance it does not have. The corrective is more
seeds, stated as future work.

**4.3 No seed-level pairing (every comparison is unpaired).**
`results.csv` historically has **no seed column**, so runs cannot be matched
across arms even though seeds were shared. Every comparison is therefore unpaired
(permutation/bootstrap), throwing away the variance reduction a paired test on
shared seeds would give — further weakening already-low power.
*Mitigation / status:* unpaired tests are the honest choice given the ledger; the
recommendation to **add a seed column going forward** (to enable paired tests and
tighter CIs) is recorded in both `statistics.md` and the validity checklist's
"known data limitations."

**4.4 The determinism claim is post-hoc.**
The finding that the tuned model is *more deterministic* (lower run-to-run
pass-rate spread) than the base is **exploratory / post-hoc**, not a pre-
registered hypothesis, and rests on n=3 per domain with std-ratio CIs that are
wide/unstable (`statistics.md` determinism table).
*Mitigation / status:* it is labelled directional-only and post-hoc; Tier 2 #7
requires each research question be tagged pre-registered vs exploratory, and this
one is explicitly the latter. Not reported as a confirmed result.

**4.5 Training cost is unrecorded (affects the amortized-cost conclusion).**
Arm D's break-even analysis depends on the one-time fine-tune cost `T_train`, but
Tinker exposed **no billing telemetry** (`training_cost_usd: 0.0` is a telemetry
gap, not a real zero; `amortized_cost.md`).
*Mitigation / status:* the break-even analysis is **parameterized** over a
`T_train` sweep ($0–$100) rather than asserting a single number, so the
conclusion is stated as a function of an assumption, not as a measured value.

---

## Summary

The strongest threats are **statistical power** (n=3–5, 0/14 survive Holm, no
seed pairing), the **open-vs-closed cost framing** (the 26× is mostly a pricing
artifact, corrected in `cost_fairness.md`), and **external scope** (single
benchmark, mostly retail, held-out event covered only open-weight retail). In
each case the study's response is disclosure and reframing rather than
overclaiming: results are presented as an underpowered but large-effect
**frontier**, cost claims are split into "listed price" vs "equal $/token"
efficiency, and generalization claims are scoped to the data that actually backs
them. The threats that remain genuinely open at $0 — pretraining memorization,
per-model harness fairness, latency as a third axis, and more seeds — are named
here and tracked in the validity checklist rather than left implicit.
