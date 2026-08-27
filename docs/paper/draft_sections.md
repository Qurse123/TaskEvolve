# TaskEvolve — Paper Draft Sections (skeleton)

Reusable prose skeleton for the TaskEvolve paper, in the Anthropic-research-post
style prescribed by `outline.md` (lead with the result, thematic result
subsections, first-class integrity section, close on reproducibility). Every
quantitative claim below is copied from the grounding docs and traceable to a
source: `CLAUDE.md`, `docs/paper/outline.md`, `docs/paper/analysis/*.md`, and
`experiments/test_split_precommit.md`. Sections not drafted here (framing §1,
setup §2, results §3, integrity §4, conclusion §5) live in `outline.md` and are
cross-referenced; this file supplies the pieces `outline.md` does not yet spell
out in prose: **Abstract, Contributions, Formal problem definition, Related Work
pointer, Broader Impact / Ethics.**

Convention (matches `outline.md`): all headline arm comparisons are the
**validation (retail-train) split, N=5** (seeds 2001–2005) unless marked;
cross-domain Arm D is **N=3** (seeds 2001–2003); the executed held-out TAU2 test
event is **retail-only, open-weight-only, N=5** (seeds 4001–4005).

---

## Abstract (~200 words)

Benchmark leaderboards rank LLM agents by task success alone, but enterprises pay
per token — so the decision-relevant quantity is *cost per successful task*, and
the answer is a **Pareto frontier of (cost, success) points, not one winner**. We
run a controlled measurement study on TAU2-bench (retail primary; airline,
telecom, and banking-knowledge for specialization) across **six configurations**:
a closed baseline (gpt-4.1), a frontier closed reference (Opus 4.8), an
autonomously-optimized harness (Arm B), a naive open-weight model (Inkling), and a
smaller open model before and after LoRA distillation (Arm D base/tuned). A frozen
Orchestrator, a fixed user-simulator, shared seeds, and a harness pinned at v0.1
keep the model and harness levers separated. Headline (validation, N=5): naive
open-weight Inkling reaches **0.874** success at **$0.022** per successful task —
within 3pp of Opus (**0.903**) at a listed-price gap of **~26×**. We show that
gap is **largely a commercial-pricing artifact**: re-priced at one common
$/token schedule the two models spend essentially equal work (Inkling 0.97×
Opus). Fine-tuning extends the frontier into *untrained* domains (airline
+13.3pp, telecom +10.7pp) and gets cheaper, but on retail buys **cost, not
accuracy**, and is slower; a blind held-out retail run confirmed only that modest
claim. Our contribution is a **measurement methodology** — the cost-per-
successful-task frontier, pre-registered blind validation, and honestly-reported
negative results — not a modeling breakthrough.

---

## Contributions

Honestly scoped: this is a measurement methodology plus a set of confirmed and
negative empirical results, **not** a new model or a performance breakthrough.

1. **A cost-vs-success *frontier* framing with a single decision-relevant scalar.**
   We argue and adopt `cost_per_successful_task = total_cost / (num_tasks ×
   pass_rate)` as the quantity an enterprise buyer actually optimizes, report it
   next to the raw 2-D (cost, success) Pareto frontier that it compresses, and
   document what the scalar *hides* (latency, partial credit, failure severity,
   pass^k consistency, pricing basis) so the frontier — not the scalar — is the
   ground truth for judging dominance (`metric_justification.md`).

2. **A six-arm controlled measurement on TAU2-bench that separates the model
   lever from the harness lever.** Holding the harness at a static v0.1
   configuration across every model gives a clean model axis; deliberately *not*
   running Opus in the gpt-4.1-tuned v0.2 harness avoids confounding model with
   harness (`threats_to_validity.md §1.1`, `RQ1`).

3. **A pre-registered blind-validation protocol that demonstrably catches
   overfitting.** The optimizer sees proxy only; validation runs once, post-hoc,
   against a criterion written to disk *before* launch. The guard is credible
   because it **failed loudly once**: the first Arm B change looked good on the
   12-task proxy but regressed blind validation success 0.806→0.720 (below the
   pre-committed 0.766 floor) and was reported as proxy-overfit, not dropped
   (`RQ2`, `experiments/validation_event_v0.2_precommit.md`).

4. **A construct-validity correction on the headline cost gap.** We show the
   "~26× cheaper" open-vs-closed claim is *almost entirely a commercial-pricing
   artifact*: at one common $/token schedule Opus spends 97,499 agent tokens/task
   vs Inkling 102,726 (Inkling 0.97× — no efficiency advantage). A genuine
   efficiency result survives normalization — the tuned Inkling-Small uses 1.49×
   fewer tokens than its base at identical serving (`cost_fairness.md`, `RQ4`).

5. **An autonomous harness optimizer that discovered a real, transferable
   tradeoff point.** An un-biased iterator (model choice is one of five equally-
   weighted edit levers) autonomously proposed swapping the whole agent
   Opus→Sonnet 5; the accepted proxy win transferred to blind validation with
   zero overfit (proxy predicted 0.833 @ $0.219; validation 0.851 @ $0.215) — a
   new frontier *point* within the closed-Anthropic axis, not a dominating one
   (`RQ2`).

6. **Honest negative and null results reported as first-class findings.** LoRA
   distillation bought cost-not-accuracy on retail (flat 0.848→0.838), did **not**
   transfer to the untrained banking domain (0.100→0.100), and **no** headline
   comparison survived Holm correction at α=0.05 (0/14) despite large effect
   sizes — an underpowered-not-null verdict we state plainly rather than
   bury (`statistics.md`, `error_analysis.md §5`, `research_questions.md`).

---

## Formal problem definition

**Setting.** A task agent is evaluated on a benchmark of multi-turn, tool-using,
policy-constrained tasks (TAU2-bench). A frozen Orchestrator drives the turn loop
and scores each task pass/fail (reward ≥ 0.5 = pass; `provenance.md`), against a
fixed LLM user-simulator (`gpt-4.1-2025-04-14`). An **arm** is one controlled
configuration — a (model, harness, serving) choice — evaluated over a fixed set
of seeds.

**Per-arm measurement.** For arm `a` run over `N` seeds on a split of `T` tasks,
let `pass_rate(a)` be the mean per-run task-success fraction and `total_cost(a)`
the mean agent token cost per run. Each arm is a **distribution**, reported as
mean ± std over the `N` fixed seeds (N=5 for the model/harness axes, N=3 for the
Arm D cross-domain carve), not a single run — the benchmark is stochastic (LLM
user-sim + provider non-determinism even at temperature 0).

**Headline objective.** The scalar the study minimizes is

```
cost_per_successful_task = total_cost / (num_tasks × pass_rate)
                         = cost_per_task / pass_rate
```

— the marginal dollar price of one delivered success. It penalizes both failure
modes of a bad agent (burning tokens, missing tasks) and collapses the tradeoff
into one comparable number. It is **lossy**: it explodes/undefined as
`pass_rate → 0` (e.g. Arm-D banking at 0.10 gives $6.35/$2.65 with CIs so wide
they are barely a measurement), treats all failures alike, gives no partial
credit, and ignores latency and pricing basis (`metric_justification.md`).

**The frontier and dominance.** The ground truth is the 2-D **Pareto frontier**:
each arm is a point `(cost_per_task, pass_rate)`. Arm `a` **Pareto-dominates**
arm `b` iff `a` is no worse on both axes and strictly better on at least one
(`cost_per_task(a) ≤ cost_per_task(b)` and `pass_rate(a) ≥ pass_rate(b)`, with one
strict). Non-dominated arms form the frontier. This distinction is load-bearing:
Opus→Sonnet (Arm B) and Inkling are *different frontier points*, neither strictly
dominating; but Inkling **does** dominate gpt-4.1 (higher success AND cheaper per
successful task). The scalar ranks; the frontier judges dominance.

**Absolute success floor.** Because minimizing `cost_per_successful_task` alone
would reward a near-zero-success agent that is trivially cheap, the study imposes
a **low absolute success floor** (0.5) as a guardrail — success must clear the
floor before cost is optimized. (This replaced an earlier single-objective accept
rule — "accept iff cheaper AND ≥0.95× best success" — which structurally rejected
every cost-for-precision tradeoff in what is really a multi-objective frontier
study; the reframe to cost-per-successful-task with an absolute floor is itself a
reported methodology correction, `CLAUDE.md`, `research_questions.md`.)

**Significance rule (pre-committed).** For the Arm D tuned-vs-base deltas, a
per-domain change is treated as real iff `|Δ pass_rate| > σ_d = √(σ_base² +
σ_tuned²)` (`experiments/arm_d_eval_precommit.md`, `test_split_precommit.md §4`).
Study-wide power caveat: across a 14-test Holm-corrected family, **0 comparisons
survive at α=0.05** despite Cohen's d up to ~19 on cost — n=3–5 lacks the power
(n=3 vs n=3 has a two-sided permutation p-floor of 2/20 = 0.10; n=5 vs n=5 floors
at ≈0.0079). Frontier gaps are **large but underpowered**, not confirmed
significant (`statistics.md`).

---

## Related Work (pointer — full detail in `related_work_baselines.md`)

This section summarizes and defers to `docs/paper/analysis/related_work_baselines.md`,
which holds the cited reference table, source URLs, and the metric/task-set caveats.
Four threads position the study:

- **Agent benchmarking with simulated users.** τ-bench (Yao et al., 2024,
  arXiv 2406.12045) and its successor τ²-bench (Barres et al., 2025,
  arXiv 2506.07982) established the tool-agent-user, policy-constrained, `pass^k`
  evaluation we build on. Their headline metric is `pass^k` (fraction of `k`
  independent runs that *all* succeed); **ours is mean-of-seeds per-run success
  (≈ average pass^1)** — loosely comparable to their pass^1 column, *not* to
  pass^k (k≥2). Our strongest external anchor, GPT-4.1 retail pass^1 = 74%, lands
  our gpt-4.1 baseline (0.806) "in the right neighborhood" (a sanity check, not a
  match — our splits are seeded subsets of retail-*train*, a difficulty-mix
  confound).

- **Cost-vs-success framing.** FrugalGPT (Chen, Zaharia, Zou, 2023,
  arXiv 2305.05176) is the canonical prior on cost-per-quality via prompt
  adaptation, LLM approximation, and cascades. We share the cost-per-quality
  objective but move it to the *multi-turn, tool-using, policy-constrained agent*
  regime and report **cost per successful task** on a Pareto frontier. Our Arm B
  (an autonomous iterator selecting an Opus→Sonnet whole-agent swap) is
  effectively a degenerate one-model cascade selected by search.

- **Agent-harness / input optimization on the same benchmark.** Input
  reformulation for tool-use accuracy on τ-bench (arXiv 2508.20931) is
  conceptually adjacent to our iterator; cited as related work, not a numeric
  baseline (numbers unverified).

- **Distillation / specialization for agents.** Arm D (per-domain LoRA
  distillation of Inkling-Small via Tinker, teacher = Opus 4.8) is a small
  instance of open-model agentic specialization — the LLM-approximation thesis of
  FrugalGPT, measured on a multi-turn agent benchmark across four TAU2 domains.

*Honest scope (from `related_work_baselines.md §4`): every external number is
**cited, not reproduced** — we never re-ran an external agent inside our harness.
Citation-only positioning cannot separate model differences from harness/
task-set/metric differences; a single same-harness GPT-4.1 retail re-run is the
recommended next step to convert the strongest anchor into a controlled
comparison.*

---

## Broader Impact / Ethics

**Automation of customer-service work.** TAU2-bench is a proxy for tool-heavy
customer-service tasks (refunds, account changes, order handling). A central
finding — a *naive* open-weight model reaching near-frontier success at a fraction
of listed cost, and fine-tuning making a small open model cheaper still — lowers
the cost barrier to deploying agents in exactly these roles. That has a direct
**labor-displacement** dimension for customer-service work; we surface it rather
than treat cost reduction as an unqualified good.

**Silent wrong-outcome failures are the safety concern, not refusals.** Our error
analysis (`error_analysis.md`) shows the dominant genuine-failure locus is
**final database-state mismatch** (db+action loci account for the large majority
of genuine failures), and critically, that these failures often terminate as
`user_stop` — a polite conversational close that **masks a wrong write**. In the
zero-shot banking transfer case, the agent converses to a natural end while
leaving the bank's database in the wrong state (wrong product selection,
wrong-argument writes); termination reason alone would badly under-count these,
and only the reward-breakdown locus exposes them. For real deployments this is the
load-bearing safety point: an agent that *looks* like it helped but silently
committed a wrong, possibly irreversible action (e.g. a wrong refund, a wrong
account opened) is more dangerous than one that visibly gives up. The
cost-per-successful-task metric explicitly **cannot see failure severity** — it
weights a benign give-up and a harmful wrong action identically
(`metric_justification.md §b`, `threats_to_validity.md §3.1`) — so cost-frontier
optimization must not be read as a safety argument.

**Open-weight cost accessibility.** The pricing-artifact correction (`RQ4`,
`cost_fairness.md`) cuts both ways: the ~26× gap is mostly commercial markup, not
model efficiency, so the accessibility benefit of open weights is real but should
be stated as an **invoice fact at today's prices**, not as an intrinsic model
property. The genuine efficiency win we do report (fine-tuning → 1.49× fewer
tokens at equal serving) is the pricing-independent lever.

**Contamination and honesty.** TAU2 is a public benchmark and all models have
unknown closed pretraining corpora; memorization cannot be ruled out at $0 and
would inflate absolute success (`contamination_audit.md §B`). We report absolute
numbers as an upper bound, note that cross-arm comparisons and the cost axis are
less exposed, and pre-commit (but did not execute) perturbed-task / release-date
probes. Train/eval disjointness *is* proven at $0 (0 overlaps by `(domain, id)`;
SFT manifest ⊆ disjoint train_distill splits), and the Opus-teacher setup is
disclosed as distillation on disjoint tasks, not eval leakage.

**Dual-use.** Minimal here. The artifacts are a benchmark harness, an evaluation
methodology, and domain-specialized customer-service agents; there is no
capability uplift toward a distinct high-risk domain. The primary responsible-use
obligations are (a) disclosing the silent wrong-DB failure mode to any deployer,
and (b) not overstating the open-weight cost advantage as intrinsic efficiency.

---

## Cross-references (not re-drafted here)

- Title, executive summary, hero figure, framing (§1), setup (§2), the four
  result subsections (§3.1 model axis, §3.2 Arm B, §3.3 Arm D, §3.4 failures),
  methodology integrity (§4), conclusion (§5), reproducibility: `outline.md`.
- Headline validation numbers and per-arm CIs: `statistics.md`,
  `cost_fairness.md §d`, `metric_justification.md §d`.
- Held-out test event and its pre-committed verdicts (D-RETAIL / C-RETAIL both
  CONFIRMED): `experiments/test_split_precommit.md §7`.
- Provenance, seeds, serving paths, pricing registration, known gaps:
  `provenance.md`. Amortized fine-tune break-even: `amortized_cost.md`.
  Latency third axis: `latency.md`. Threats catalogue: `threats_to_validity.md`.
</content>
</invoke>
