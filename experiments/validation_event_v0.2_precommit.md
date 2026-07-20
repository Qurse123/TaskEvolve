# Blind Validation Event — Arm B v0.2 (precommitted before launch)

- **Date declared:** 2026-07-19 (before the validation run starts)
- **System under test:** harness v0.2 at commit `458fe57` (send system prompt
  only on first turn; accepted in `iter_0004`, proxy double-run seeds 3313/3314)
- **Design (frozen, experiment.md §17/§21.15):** validation split (35 retail
  tasks) × 5 repeats, seeds 2001–2005, run once, blind — no reruns, no
  iteration on these results. Runs tagged `HARNESS_VERSION=v0.2`.
- **Arm A reference distribution (M1, frozen):** pass_rate 0.806 ± 0.047,
  cost_per_successful_task $0.062 ± $0.004 (n=5, seeds 2001–2005, v0.1).

## Pass criterion (declared now, judged after)

Arm B v0.2 is a **validated improvement** iff, over the 5 validation repeats:

1. mean pass_rate ≥ 0.95 × 0.806 = **0.766** (same success-floor fraction as
   the proxy acceptance rule), AND
2. mean cost_per_successful_task < **$0.062** (Arm A validation mean), AND
3. zero harness_error tasks across all repeats.

Any other outcome is reported as **proxy-overfit** (experiment.md §20 tail) —
including partial outcomes (e.g. cheaper but below the success floor).

---

## Result (judged 2026-07-19, after all 5 repeats)

Event note: the run was interrupted externally after seeds 2001–2002 (not by
the operator); seeds 2003–2005 were completed in a detached process with no
intermediate changes to the system under test. All 5 seeds ran exactly once.

| seed | pass_rate | cost/successful task | cost/task | harness errors |
|------|-----------|----------------------|-----------|----------------|
| 2001 | 0.743 (26/35) | $0.0506 | $0.0376 | 0 |
| 2002 | 0.657 (23/35) | $0.0603 | $0.0396 | 0 |
| 2003 | 0.743 (26/35) | $0.0511 | $0.0380 | 0 |
| 2004 | 0.743 (26/35) | $0.0527 | $0.0391 | 0 |
| 2005 | 0.714 (25/35) | $0.0507 | $0.0362 | 0 |

**Aggregate: pass 0.720 ± 0.037, cost/successful task $0.0531 ± $0.0041,
cost/task $0.0381 ± $0.0013.**

- Criterion 1 (pass ≥ 0.766): **FAIL** (0.720)
- Criterion 2 (cost/successful task < $0.062): pass ($0.0531)
- Criterion 3 (zero harness errors): pass

## Verdict: PROXY-OVERFIT

Per the precommitted rule, Arm B v0.2 is reported as proxy-overfit: the −18%
proxy cost gain transferred to validation (−24% cost/task vs Arm A's ~$0.050),
but task success regressed 0.806 → 0.720 (−8.6 pp, ~1.8× Arm A's validation
std) — a cost-for-success trade the 12-task proxy could not detect. v0.2 is a
*different point on the cost–success frontier* (0.720, $0.0531) than Arm A
(0.806, $0.062), not a dominating improvement.
