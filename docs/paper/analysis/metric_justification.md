# Headline-metric justification: `cost_per_successful_task` (paper Tier-2 #9)

_$0 analysis — reads only the local `experiments/results.csv`. Companion plot:
`scripts/plot_metric_views.py` -> `experiments/plots/metric_views.png`._

The study's headline number is

```
cost_per_successful_task = total_cost / (num_tasks * pass_rate)
                         = cost_per_task / success_rate
```

This note establishes its **construct validity** — why it is the right scalar to
lead with — and, just as importantly, **what it hides**, so the frontier (not the
scalar) is understood to be the ground truth.

## (a) Why this metric

The study's decision-maker is someone paying per token who derives value **only
from tasks that actually complete**. For that buyer a failed task is not a
discount — it is wasted spend that still has to be re-attempted (by a human, a
retry, or another agent). The quantity they care about is therefore *dollars per
delivered success*, which is exactly `total_cost / (num_tasks * pass_rate)`.

Its virtues:

- **Decision-relevant.** It is the marginal price of one useful outcome, the
  number that goes into a "can we afford to run this agent in production" call.
- **Collapses the tradeoff into one comparable scalar.** Cost and success move in
  opposite directions across arms (Opus buys the top success at a steep price;
  Inkling is cheap but a few points lower). A single ratio lets six arms be
  ranked on one axis instead of argued pairwise.
- **Penalizes both failure modes of a bad agent** — burning tokens *and* missing
  tasks — in the direction a buyer feels them: more spend or fewer successes both
  push it up.

## (b) What it HIDES — failure modes

The scalar is a *lossy compression* of a 2-D (cost, success) point. What it
discards:

1. **It explodes / goes undefined as success -> 0.** The denominator is
   `num_tasks * pass_rate`; at `pass_rate = 0` it is division by zero, and near
   zero it is dominated by denominator noise. This is not hypothetical here: the
   Arm-D **banking** rows sit at `pass_rate = 0.10` and yield
   `cost_per_successful_task` of **$6.35 (base)** and **$2.65 (tuned)** — an order
   of magnitude above every retail number — with bootstrap CIs so wide
   (`[4.26, 8.80]`) they are barely a measurement. A single lucky/unlucky pass
   swings the ratio enormously. In the low-success regime the scalar says more
   about the pass count than about cost, and should be read as "large/undefined,"
   not as a precise price.
2. **It treats all failures as identical.** A catastrophic policy violation (the
   agent takes a wrong, irreversible action) and a benign give-up (the agent
   asks for help and stops) both count as one non-success. A buyer cares about
   the *kind* of failure; the ratio cannot see it.
3. **No partial credit.** TAU2 scores a task as pass/fail, so a task that is 90%
   complete contributes exactly as much as one that never started — zero. Agents
   that make consistent partial progress are indistinguishable from agents that
   flail.
4. **It ignores latency.** Two arms at the same `cost_per_successful_task` can
   differ several-fold in wall-clock time per task; the dollar ratio is blind to
   it. See the latency analysis (Tier-2 #10) for the time axis.
5. **It is sensitive to the pricing basis.** The ratio inherits every caveat of
   the cost numerator. Closed models are priced at retail API list (with
   commercial margin); open models at the study's own serving cost. Re-priced at
   a single common $/token schedule, the headline "~26x cheaper" collapses to
   ~0.97x — i.e. almost entirely a pricing artifact, not model efficiency. See
   `cost_fairness.md`. The scalar reports *invoice cost today*, not intrinsic
   efficiency.

## (c) Alternative views to report alongside

Because the scalar is a summary, the paper reports it **next to** the views it
compresses:

- **The raw 2-D cost-vs-success frontier (Pareto).** This is the *truth*: each
  arm as a `(cost_per_task, success_rate)` point. The frontier shows which arms
  are actually dominated vs. which sit on a genuine tradeoff curve — information
  the scalar erases by projecting onto one line. (E.g. Opus->Sonnet and Inkling
  are *different frontier points*, not one strictly better than the other.)
- **`cost_per_task` and `success_rate` reported separately**, each with its own
  mean ± std / CI (see `statistics.md`). This keeps the numerator and denominator
  legible and makes the low-success explosion in (b.1) obvious.

The rule of thumb: **the scalar is a summary for ranking; the frontier is the
truth for judging dominance.** Lead with the ratio, but never let it stand
alone — always show the 2-D point behind it.

## (d) The retail-arm view (validation split, n=3–5)

Same six arms as `statistics.md`, ordered by the scalar (cheapest first). The
2-D columns are what the scalar compresses; the plot renders both side by side.

| Arm | pass_rate | cost/task | **cost/successful task** |
|-----|----------:|----------:|-------------------------:|
| Inkling (ArmC, v0.1) | 0.874 | $0.0193 | **$0.0219** |
| ArmD tuned (Inkling-Small) | 0.838 | $0.0310 | **$0.0371** |
| ArmD base (Inkling-Small) | 0.848 | $0.0460 | **$0.0543** |
| gpt-4.1 (ArmA, v0.1) | 0.806 | $0.0497 | **$0.0619** |
| Opus->Sonnet (ArmB, v0.2) | 0.851 | $0.1820 | **$0.2146** |
| Opus 4.8 (v0.1) | 0.903 | $0.5208 | **$0.5785** |

Note how the scalar's ranking (Inkling cheapest, Opus most expensive) hides that
Opus has the **highest success** and Inkling only the third — the exact
information the 2-D frontier restores.
