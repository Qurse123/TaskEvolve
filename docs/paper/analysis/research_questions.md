# Research questions & hypotheses (paper Tier-2 item #7)

This study is a controlled measurement of the **cost-vs-success tradeoff of LLM
agents on TAU2-bench** (retail as the primary domain; airline / telecom /
banking-knowledge added for the specialization arm). Success alone is not the
decision-relevant quantity — enterprises pay per token — so the objective metric
throughout is **cost per successful task**,
`cost_per_successful_task = total_cost / (num_tasks × pass_rate)`, and the answer
to every question is read off a **Pareto frontier** of (cost, success) points,
not a single leaderboard score. Each point is a distribution over fixed seeds
(N=5 where affordable, N=3 for the cross-domain Arm D dev carve), reported as
mean ± std; a frozen TAU2 Orchestrator and a fixed user-simulator model hold the
non-agent variables constant across arms.

Every RQ below is labeled **pre-registered / by-design** (the comparison was a
planned single-variable arm and/or its pass criterion was written down before
the result was seen) or **exploratory / post-hoc** (noticed after results, not
pre-committed). The distinction matters and is discussed at the end.

---

## RQ1 — Where do models sit on the cost-success frontier, with the harness held fixed?

Holding the harness at the static v0.1 configuration so "model" is the only lever
(v0.1 was NOT re-tuned per model — Opus was deliberately kept out of the
gpt-4.1-tuned v0.2 harness to avoid confounding model with harness), where do a
closed baseline (`gpt-4.1`), a frontier closed model (Opus 4.8), and a naive
open-weight model (Inkling, 975B/41B MoE) fall on the (cost, success) plane?

- **H1a — a naive open-weight model reaches near-frontier success at far lower
  listed cost.** Validation (retail, N=5): Opus **0.903 @ $0.578/succ**, Inkling
  **0.874 @ $0.022/succ**, gpt-4.1 **0.806 @ $0.062/succ** — Inkling is within
  3pp of Opus success at ~26× lower listed cost.
- **H1b — gpt-4.1 is dominated by Inkling** (Inkling ≥ on success AND cheaper per
  successful task). Holds on validation (0.874 vs 0.806; $0.022 vs $0.062).

**Label: pre-registered / by-design.** The three-model axis is a planned
single-variable comparison, and both claims have pre-committed held-out
criteria in `experiments/test_split_precommit.md §4b.1–2` (Inkling within 5pp of
Opus AND ≥10× cheaper; Inkling dominates gpt-4.1). The executed budget-limited
test-split subset (retail, N=5, seeds 4001–4005) confirmed the open-weight arm
holds out-of-sample (Inkling **0.895 ± 0.069 @ $0.011/succ**, within 1σ of its
dev mean — criterion C-RETAIL). The full model-axis test event (Opus / gpt-4.1
on the hidden test split) remains pre-registered but deferred for budget.

**Statistical caveat (applies to all RQs):** across the 14-test Holm-corrected
family, **0 comparisons survive at α=0.05** despite very large effect sizes
(Cohen's d up to ~19 on cost) — n=5/n=3 simply lacks the power (n=3 has a
permutation p-floor of 0.10). Read the frontier gaps as large but underpowered,
not as confirmed significant.

---

## RQ2 — Does an autonomous harness optimizer, seeing only proxy logs, find *durable* cost reductions?

An un-biased iterator (model choice is one of five equally-weighted edit levers,
not a hand-fed answer) optimizes the harness against the 12-task proxy under a
fixed budget, then validation is run once, blind. Does an accepted proxy
improvement transfer to held-out validation without regressing success?

- **H2 — an accepted proxy win transfers to blind validation within guardrails.**
  The reframed iterator autonomously proposed swapping the whole agent
  **Opus→Sonnet 5**; blind validation (N=5) gave **0.851 ± 0.051 @ $0.215/succ**
  vs a proxy prediction of 0.833 @ $0.219 — **transferred with zero overfit.**
  This wins *within the closed-Anthropic axis* (a new frontier point, not a
  dominating one — Inkling still owns the efficiency corner).

**Label: pre-registered / by-design.** The blind-validation overfitting guard is
the core pre-registered protocol: the optimizer never sees validation, and the
pass criterion is written in `experiments/validation_event_v0.2_precommit.md`
before launch. The guard is credible precisely because it also **caught a
failure**: the *first* Arm B run (v0.2, "send system prompt only on turn 1")
looked good on proxy but regressed validation success **0.806→0.720** (below the
pre-committed 0.766 floor) — reported as proxy-overfit, not quietly dropped.

---

## RQ3 — Does specialization (LoRA distillation) extend the frontier, including into untrained domains and held-out tasks?

Arm D fine-tunes a smaller open model (Inkling-Small) via Opus-4.8-distilled
LoRA and evaluates tuned vs base per domain on the same static v0.1 harness (the
local Tinker serving shim serves both sides, so any serving artifact cancels in
the matched delta). Does tuning move the model up-and-left, and does the gain
appear in domains it did not train on and hold on held-out tasks?

- **H3a — tuning yields cross-domain success gains at lower cost.** Dev carve
  (N=3): airline **0.300→0.433 (+13.3pp)**, telecom **0.587→0.693 (+10.7pp)**,
  retail 0.848→0.838 (flat), banking (untrained transfer) 0.100→0.100 (no
  regression); aggregate cost/successful task **$0.452→$0.192**, cheaper in all
  four domains.
- **H3b — the gain holds on truly held-out tasks.** Executed test-split subset
  (retail, N=5): tuned **0.875 ± 0.040** vs base **0.915 ± 0.038** — within
  σ_d = 0.055 (no significant regression) and **−21% cost/succ** ($0.033 vs
  $0.042) → criterion D-RETAIL **CONFIRMED**.

**Label: pre-registered / by-design.** The tuned-vs-base delta, its per-domain
sign predictions, the "no regression on unseen banking" clause, and the
significance rule σ_d = √(σ_base²+σ_tuned²) are all pre-committed in
`experiments/arm_d_eval_precommit.md` and `test_split_precommit.md §4b.4 /
§4 executed-subset`. **Honest scope limit:** the executed held-out run covered
**retail only** (budget), which on the dev carve was the *flat* domain — so it
confirms "no regression + cheaper on held-out retail," **not** the headline
cross-domain (airline/telecom) gains, which remain dev-split (N=3) and
pre-registered-but-deferred for the hidden test split.

---

## RQ4 — Is the "~26× cheaper" advantage a model-efficiency property or a pricing artifact?

The headline compares **retail API list prices** (closed: Opus, gpt-4.1) against
the study's **own serving cost** (open: Together / local shim) — two different
kinds of number. Which factor drives the ratio: intrinsic model efficiency
(tokens of work per task) or commercial $/token markup?

- **Finding (correction, not a confirmed hypothesis): the ~26× is almost
  entirely a commercial-pricing artifact.** Re-priced at one common schedule,
  Opus spends 97,499 agent tokens/task vs Inkling 102,726 — Inkling is **0.97×**
  Opus per task, i.e. **no efficiency advantage**. A genuine efficiency result
  does survive normalization: the tuned Inkling-Small uses **1.49× fewer tokens**
  than its base at identical serving — a real, pricing-independent gain from
  fine-tuning.

**Label: exploratory / post-hoc.** This is a construct-validity correction found
during the Tier-1 cost-fairness analysis (`cost_fairness.md`), after the "26×"
headline existed. It was not a pre-registered hypothesis; it reframes how RQ1's
cost gap must be *stated* (a real invoice fact at today's prices, not an
intrinsic model property). It is confirmatory only in the weak sense that it is a
deterministic re-pricing of already-collected token counts.

---

## RQ5 — Does fine-tuning reduce run-to-run outcome variance ("determinism")?

Does the tuned model produce lower pass-rate spread across seeds than the base?

- **Observation:** pass-rate std-ratio (tuned/base) is <1 on retail (0.499) and
  airline (0.577) but >1 on telecom (1.323) and =1 on banking — mixed, not a
  clean reduction.

**Label: exploratory / post-hoc — HARKing risk, needs confirmation.** This was
noticed *after* results were in, was never pre-registered, and rests on **n=3**
per domain with 95% CIs the analysis itself flags as "wide/unstable, low power."
It is a **hypothesis-generating** observation only. If reported, it must be
stated as such and confirmed on fresh, larger-N seeds before any "fine-tuning
makes agents more deterministic" claim is made. Presenting it as a pre-planned
result would be textbook HARKing (hypothesizing after results are known).

---

## Why the labels matter

Pre-registered / by-design questions (RQ1–RQ3) were set up as single-variable
arms with pass criteria committed to disk **before** the deciding result was
observed (`*_precommit.md`), so their outcomes are **confirmatory**: the data can
only confirm or refute a claim it could not have been reverse-fitted to. The
blind-validation guard earns this status by having *failed loudly once* (Arm B
run 1 proxy-overfit) rather than always passing.

Exploratory / post-hoc findings (RQ4, RQ5, and the mid-study methodology reframe
from a single-objective accept rule to cost-per-successful-task with a low
absolute success floor) are **hypothesis-generating, not confirmatory**. They
are legitimate and worth reporting — the pricing-artifact correction (RQ4)
materially changes how the headline is phrased — but they must be labeled so a
reader does not mistake a pattern spotted in the data for a prediction the study
was designed to test. RQ5 in particular (determinism, n=3, unstable CIs) is the
clearest HARKing risk and is flagged as requiring fresh-seed confirmation.

Finally, a study-wide power caveat colors every confirmatory claim: **no headline
comparison survives Holm correction at α=0.05** (0/14), because n=3–5 is
underpowered even for the large effect sizes observed. The pre-registered claims
are directionally supported with large effects and (for RQ2/RQ3-retail) clean
out-of-sample transfer, but should be reported as *underpowered*, not as
statistically confirmed at conventional thresholds.
