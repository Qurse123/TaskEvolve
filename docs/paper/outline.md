# TaskEvolve paper — section-by-section outline

Format modeled on Anthropic research posts (e.g. *"How Claude is accelerating
protein design and analytical chemistry"*, Aug 2026). Their signature moves,
which this outline preserves:

1. **Lead with the result, not the method** — the headline number and the hero
   figure come before any methodology.
2. **Results organized into thematic subsections**, each opening with its number:
   a competitive win, a hard-case win, a generalization result, and an explicit
   *failures* subsection.
3. **Limitations / integrity get their own first-class section**, not footnotes.
4. **Close with reproducibility** — links to code, data, splits, pre-registration.

Every number below is pulled from `CLAUDE.md` / `experiments/results.csv` as of
2026-08-19. All headline arm comparisons are the **validation (retail) split,
N=5** unless marked; Arm D cross-domain is **N=3**.

---

## Title + summary (no heading)

- **Working title:** "How much does agent success cost? Measuring the
  cost–performance frontier of LLM agents on TAU2-bench."
- **Subtitle / one-liner:** A controlled measurement study across six
  configurations — closed frontier models, a naive open-weight model, an
  autonomously-optimized harness, and a fine-tuned open-weight model.
- **Executive summary (one paragraph, headline-first):** On TAU2-bench retail, a
  *naive* open-weight model (Inkling) reaches **0.874** task success at
  **$0.022** per successful task — within 3pp of frontier Opus 4.8 (**0.903**)
  at **~26× lower cost**. Targeted LoRA fine-tuning of a smaller open model
  (Arm D) then extends the frontier into *untrained* domains (+13.3pp airline,
  +10.7pp telecom) while getting cheaper. The study's method contribution: cost
  and success are a **frontier, not a single score**, and we report **cost per
  successful task** as the scalar that captures the tradeoff.

## Hero figure (immediately after summary)

- `experiments/plots/all_arms_frontier_20260811.png` — cost-vs-success scatter,
  all six arms, validation/retail, arm mean ± std. This *is* the thesis in one
  image. Caption must state: validation (retail) split, N=5 (Arm D N=3), x-axis
  = average cost per task.

---

## 1. Why the cost axis is the real question (framing)

- What TAU2-bench is: tool-heavy customer-service tasks, LLM user-sim, a frozen
  Orchestrator that scores task success. Domains: retail, airline, telecom,
  banking-knowledge.
- The gap this fills: benchmark leaderboards report success only. Enterprises
  pay per token. The decision-relevant quantity is **cost per successful task**,
  and the answer is a **Pareto frontier** of (cost, success) points — not one
  winner.
- Define the headline metric plainly here so results read cleanly:
  `cost_per_successful_task = total_cost / (num_tasks × pass_rate)`.

## 2. The setup (their "The campaign")

- **The six arms**, one table, each a single controlled variable:
  - Arm A — closed baseline, `gpt-4.1`, static v0.1 harness.
  - Opus 4.8 — frontier closed reference, same static v0.1 harness.
  - Arm B — iterator-optimized harness (v0.2); the optimizer autonomously swapped
    the whole agent Opus→Sonnet 5.
  - Arm C — naive open-weight Inkling (975B/41B MoE), static v0.1 harness.
  - Arm D base — naive Inkling-Small (matched control).
  - Arm D tuned — Inkling-Small + Opus-distilled LoRA, same static harness.
- **Controls that make the axes clean:** frozen Orchestrator (never modified),
  fixed user-sim model, identical seeds across arms, harness pinned at v0.1 for
  the model axis so "model" and "harness" are separated levers.
- **Statistical protocol:** every point is a distribution, N=5 fixed seeds
  (proxy 1001–1005, validation 2001–2005), reported mean ± std. Significance for
  Arm D deltas: |Δ| > σ_d = sqrt(σ_base² + σ_tuned²).
- **What the agent has:** the harness (system prompt, policy, tool schemas),
  routed through TAU2's `generate()`. No Orchestrator access.

## 3. Results

Open with the headline, then thematic subsections.

### 3.1 The model axis: open-weight nearly matches frontier at a fraction of cost

- Validation, all v0.1 harness: **Opus 0.903 @ $0.578/succ · Inkling 0.874 @
  $0.022/succ · gpt-4.1 0.806 @ $0.062/succ.**
- Takeaway: naive Inkling is the efficiency winner — higher success than
  gpt-4.1 AND ~26× cheaper than Opus. gpt-4.1 is dominated.
- Figure: `frontier_validation_3model_20260723.png`.

### 3.2 An autonomous optimizer discovers a real tradeoff point (Arm B)

- The un-biased iterator, exploring five edit levers, **autonomously proposed
  swapping Opus→Sonnet 5.** Blind validation N=5: **0.851 ± 0.051 @ $0.215/succ**
  — proxy predicted 0.833 @ $0.219, so it **transferred with zero overfit.**
- Frame honestly: this wins *within the closed-Anthropic axis*; Inkling still
  owns the efficiency corner. A new frontier *point*, not a dominating one.
- Figure: `frontier_validation_20260729_044810.png`, trajectory plot.

### 3.3 Specialization extends the frontier into untrained domains (Arm D)

- Opus-distilled LoRA on Inkling-Small, evaluated base vs tuned per domain (N=3):
  - retail: 0.848 → 0.838 (flat, cheaper)
  - airline: 0.300 → **0.433 (+13.3pp, > σ)**
  - telecom: 0.587 → **0.693 (+10.7pp, > σ)**
  - banking (transfer, untrained): 0.100 → 0.100 (Δ=0, **no regression**)
  - aggregate cost per successful task: **$0.452 → $0.191** (cheaper in all four)
- Takeaway: fine-tuning bought cross-domain gains AND lower cost — a genuine
  Pareto move, verified as a clean specialization win (not overfit).
- Figure: `arm_d_c_vs_d_frontier_20260811.png`.

### 3.4 Where it failed (dedicated failures subsection)

- **Arm B v0.2 proxy-overfit (the first run):** an accepted change (send system
  prompt only on turn 1) cut cost on the 12-task proxy but regressed success
  0.806→0.720 on blind validation — a cost-for-success trade the small proxy
  couldn't detect. Reported as proxy-overfit per pre-committed criterion.
- **Banking transfer floor:** both base and tuned sit at 0.100 — specialization
  didn't transfer to a retrieval/knowledge domain it never trained on. Honest
  ceiling on the generalization claim.

## 4. Methodology integrity (their "dual-use" analogue)

This is the section that signals rigor — treat it as first-class.

- **Blind validation as the overfitting guard:** the optimizer sees proxy only;
  validation runs once, post-hoc, against a **pre-committed** pass criterion
  (`experiments/*_precommit.md`). Cite this throughout.
- **We found and corrected our own methodology flaw.** The original acceptance
  rule was single-objective (accept iff cheaper AND ≥0.95×best success), which
  *structurally rejects every cost-for-precision tradeoff* in what is really a
  multi-objective frontier study. We reframed the objective to cost per
  successful task with a low absolute success floor, and un-biased the optimizer
  (model choice = one lever among five, not a hand-fed answer). Present this as a
  strength: the frontier framing came from catching the flaw.
- **Single-variable discipline:** why Opus was run only in v0.1 and not v0.2
  (v0.2 was tuned for gpt-4.1 — Opus-in-v0.2 would confound model with harness).

## 5. Conclusion

- Synthesis: the frontier has a clear shape — open-weight models dominate the
  efficiency corner; frontier closed models buy the top few points of success at
  a steep premium; targeted fine-tuning shifts a small open model up-and-left,
  including into untrained domains.
- Honest scope caveats: proxy is 12 tasks / validation 35 (retail); cross-domain
  Arm D is N=3; the official TAU2 hidden-test split was **deliberately deferred**
  (not run), so all numbers are on our frozen train-derived splits.
- Future direction: run the official hidden-test split once for headline
  reporting; widen the frontier with more open-weight bases; test whether the
  distillation gains hold at larger N.

## Further reading / reproducibility

- Repo + frozen splits (`benchmark/splits/*.json`), canonical `results.csv`.
- Pre-registration docs: `experiments/validation_event_v0.2_precommit.md`,
  `experiments/arm_d_eval_precommit.md`.
- All frontier figures under `experiments/plots/`.
- Design + spec: `systems_design.md`, `experiment.md`.

## Footnotes (methodology granularity)

- Exact seeds, N per arm, σ_d significance rule, pricing registration (how
  open-weight cost is computed), the serving shim for the tuned adapter, and the
  train/eval split-disjointness guarantee.

---

### Open decisions to confirm before drafting prose

1. **Audience/venue** — Anthropic-style blog post (this format) vs a formal
   paper (would add Related Work + formal Methods). This outline is the blog
   shape; say the word if it should be arXiv-formal.
2. **Headline framing** — lead with "open-weight nearly matches frontier at 26×
   less" (§3.1) or with "how much does success cost?" (the metric). Currently
   both are in the summary; pick one to lead the title.
3. **Arm D N=3 vs N=5** — the cross-domain result is N=3; decide whether to
   caveat prominently or top up to N=5 before publishing.
