# TaskEvolve paper — research-validity checklist

Tracks what's needed to make the study a *valid* research paper (grounded in
empirical-research best practice), tiered by how much each gap blocks
credibility. **Tier 1** = a serious reviewer rejects without it; **Tier 2** =
expected in any competent empirical paper; **Tier 3** = strengthens/polishes.

Status: ☐ not started · ◐ in progress · ✅ done (as reviewable change) ·
⏸ prepared, blocked on a paid/irreversible run the user must approve.

Work for the Tier 1 pass lives on branch `tier1`; analysis artifacts under
`docs/paper/analysis/`, scripts under `scripts/`, tests under `tests/`.

---

## Tier 1 — Blocking validity gaps

| # | Item | Status | Deliverable |
|---|------|--------|-------------|
| 1 | **Cost-comparison fairness** — headline "26× cheaper" compares closed retail API prices vs our own open-model serving cost (conflates model efficiency with commercial markup). Normalize to a common basis / reframe. | ✅ | `scripts/cost_fairness.py`, `tests/test_cost_fairness.py`, `docs/paper/analysis/cost_fairness.md`. **Finding: ~26× is a pricing artifact; at equal $/token Opus≈Inkling (Inkling +5% tokens). Real efficiency win = tuned uses 1.49× fewer tokens than base.** |
| 2 | **Run official TAU2 held-out test split, once (blind).** All current numbers are on train-derived proxy/validation splits. | ◑ | **Executed subset done (2026-08-20):** open-weight arms on retail test (40 tasks, N=5) — `benchmark/splits/test_retail.json`, outcome in `experiments/test_split_precommit.md §7`. **CONFIRMED:** tuned holds retail success within noise & −21% cost/succ; naive Inkling holds out-of-sample. **Deferred (budget):** Opus/Arm B/gpt-4.1 arms + airline/telecom test — pre-registered, need funding. |
| 3 | **External baselines / positioning** — everything is self-referential. Position against published TAU2 numbers + prior work. | ✅ | `docs/paper/analysis/related_work_baselines.md`. Cited (not re-run); our gpt-4.1 retail 0.806 ≈ Sierra's 0.74 pass^1 → sanity-passes. Same-harness re-run of a reference agent is the stronger, still-deferred form. |
| 4 | **Statistical significance** — report tests + effect sizes + CIs, not just mean ± std; correct for multiple comparisons. | ✅ | `scripts/paper_stats.py`, `tests/test_paper_stats.py`, `docs/paper/analysis/statistics.md`. **Finding: 0/14 comparisons survive Holm — huge effect sizes, but n=3–5 lacks the power (n=3 has a p-floor of 0.10).** |
| 5 | **Data-contamination audit** — prove train/eval disjointness; discuss benchmark-in-pretraining memorization. | ✅ | `scripts/check_split_disjointness.py`, `tests/test_check_split_disjointness.py`, `docs/paper/analysis/contamination_audit.md`. Disjointness **proven** (0 overlaps, keyed by (domain,id)); memorization discussed with probes. |
| 6 | **Amortized training-cost accounting (Arm D)** — per-task cost ignores one-time fine-tune cost; report break-even. | ✅ | `scripts/amortized_cost.py`, `tests/test_amortized_cost.py`, `docs/paper/analysis/amortized_cost.md`, `experiments/plots/arm_d_breakeven.png`. Parameterized by assumed T_train (Tinker cost unrecorded). |

**Note on #2/#3:** the *changes/analysis* for Tier 1 are completable at $0 and are
what's up for review. The actual **paid, blind-once test-split run (#2)** and an
optional **same-harness baseline re-run (#3)** are spend + (for #2) irreversible
events — prepared here but deliberately **not executed**, per the standing
"stop before spending / run-once-blind" rule.

---

## Tier 2 — Expected in any competent empirical paper

| # | Item | Status | Deliverable |
|---|------|--------|-------------|
| 7 | Explicit research questions + hypotheses; label **pre-registered** vs **exploratory/post-hoc**. | ✅ | `docs/paper/analysis/research_questions.md` (RQ1–3 pre-registered; RQ4 pricing-artifact + RQ5 determinism labeled post-hoc/HARKing-risk). |
| 8 | Dedicated **Threats to Validity** (internal / external / construct / statistical-conclusion). | ✅ | `docs/paper/analysis/threats_to_validity.md` (17 threats, each with mitigation/status). |
| 9 | Justify the metric (`cost_per_successful_task`); what it hides; show alternative view. | ✅ | `docs/paper/analysis/metric_justification.md` + `scripts/plot_metric_views.py` (+test) + `experiments/plots/metric_views.png`. |
| 10 | Measure **latency** (third axis). | ✅ | `scripts/latency_analysis.py` (+test) + `docs/paper/analysis/latency.md` + `experiments/plots/latency_by_arm.png`. **Finding: cheap ≠ fast — tuned model is slowest retail arm (76.7s vs Inkling 18.7s).** |
| 11 | **Error analysis / failure taxonomy** (esp. banking 0.10 floor). | ✅ | `scripts/error_taxonomy.py` (+test) + `docs/paper/analysis/error_analysis.md`. **Findings: dominant failure = wrong DB write (792/995); `user_stop` masks 76% of failures; banking floor is genuine (0.10, unmoved by tuning) — agent over-retrieves then commits wrong writes, not retrieval starvation.** |
| 12 | **Ablations** — Arm D (rank, distill size, domain mix); Arm B (edit levers). | ⏸ | **Not run, and no plan document was written.** Needs paid Tinker re-training plus re-eval; Arm B lever attribution may be $0 if the iteration logs survive. |
| 13 | Exact experimental provenance (versions, providers, sampling, dates). | ✅ | `docs/paper/analysis/provenance.md` + `scripts/provenance.py` (+test). Flags 7 gaps incl. stale `.env.example` user-sim note + Arm B model-label discrepancy. |

## Tier 3 — Strengthens / polishes

| # | Item | Status | Deliverable |
|---|------|--------|-------------|
| 14 | Paper scaffolding: abstract, contributions, problem/frontier definition, Related Work, Broader Impact/Ethics. | ✅ | `docs/paper/draft_sections.md` (abstract, 6 contributions, formal frontier def, related-work pointer, ethics). |
| 15 | Reproducibility artifact + checklist. | ✅ | `docs/paper/reproducibility.md` (env pins, per-arm repro commands, frozen inputs, NeurIPS checklist, limitations; flagged no lockfile + vendor plain-clone + `srt`/`rg` maybe not in code path). |
| 16 | Widen external validity — second benchmark. | ⏸ | **Not run, and no plan document was written.** Needs a new benchmark integration plus paid runs. (TAU2's 4 domains = partial breadth already.) |
| 17 | Fairness/neutrality of the shared harness. | ✅ | `docs/paper/analysis/harness_neutrality.md`. **v0.1 is model-generic (empty few-shots, deterministic policy extractor); bias is one-directional (can only favor gpt-4.1, which is dominated) → open/frontier advantage is a lower bound.** Empirical per-model adapted-harness sweep deferred (paid). |
| 18 | Cost-measurement methodology subsection. | ✅ | `docs/paper/analysis/cost_methodology.md` (capture path, exact pricing table, included/excluded, pricing-vs-efficiency, gaps). |

---

### Known data limitations surfaced during the Tier 1 pass
- `results.csv` has **no seed column** → cross-arm stats are unpaired
  (permutation/bootstrap), not paired. Recommend adding a seed column going forward.
- `training_record.json` has `training_cost_usd: 0.0` (Tinker exposed no billing
  telemetry) → amortized cost (#6) is **parameterized** by assumed training cost.
- Per-message `usage` (token counts) is not populated in transcripts → cost
  fairness (#1) backs tokens out of cost + known prices where possible, else uses
  a turn/tool-call work proxy.
