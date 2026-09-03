# Working Structure for the Paper

## 1. Summary / Opening

### 1.1 Thesis

State the main result immediately.

- Small specialized open-weight models can match or outperform a frontier closed model on these agentic customer-service tasks.
- They do so at substantially lower cost per successful task.
- However, optimizing an agent harness against a narrow proxy distribution can produce cost improvements that do not generalize.

### 1.2 Why this question matters

Explain why existing benchmarks are insufficient.

- Most benchmarks optimize/report task success.
- Real deployments care about both capability and inference cost.
- Therefore, the relevant quantity is the **cost-performance frontier**, not task success alone.

### 1.3 Headline result

Introduce **Figure 1 immediately**.

Give only the most important numbers:

- Retail
- Airline
- Telecom
- Fine-tuned Inkling vs Opus
- Iterator failure outside the optimization domain

Do not deeply explain *why* yet.

---

## 2. Measuring the Cost-Performance Frontier

**Question:** How should agentic systems be compared fairly?

This is your common experimental setup.

### 2.1 Benchmark

- TAU2-bench
- Retail, airline, telecom
- Official held-out test splits
- Number of tasks per domain

### 2.2 Metrics

Define:

- Task success rate
- Cost
- Cost per successful task
- Why you use this metric

### 2.3 Controlled comparison

Explain what stays fixed:

- User simulator
- Seeds 4001–4005
- Task splits
- Evaluation procedure
- Agent harness, except where the harness itself is the experimental variable
- Model/provider settings where applicable --> impetus of using inkiling 

This is the **apples-to-apples** section from your current outline.

### 2.4 Avoiding test contamination / overfitting

Briefly establish the rules of the experiment:

- What data each system was allowed to see
- Proxy/train vs held-out test separation
- No optimization against held-out results
- Any other controls you actually used

Keep this general here. Put experiment-specific mitigations in their respective sections.

---

## 3. Can Harness Optimization Move the Frontier?

This is **Experiment 1**.

### 3.1 Motivation

Ask the question explicitly:

> Can we improve the cost-performance frontier without changing the underlying frontier model?

### 3.2 Iterator design

Explain:

- What the autonomous iterator is
- What information it receives
- What it can change
- What objective it optimizes
- Why you used the 12-task retail proxy split

### 3.3 Guardrails / potential failure modes

Explain the relevant issues here:

- Proxy overfitting
- Search over a small task distribution
- Why held-out airline and telecom are useful tests of generalization

Do **not** put all overfitting discussion into a generic methods section. It matters specifically to this experiment.

### 3.4 Results

Show what happened.

- Retail improvement / cost reduction
- Airline: 0.330
- Telecom: 0.210
- Compare with baseline

### 3.5 Interpretation

This is the conclusion of the experiment:

> The iterator successfully optimized its stated objective, but the resulting solution specialized to the proxy distribution rather than improving the general cost-performance frontier.

That sets up Experiment 2.

---

## 4. Can Model Specialization Move the Frontier?

This is **Experiment 2** and probably the central section of the paper.

### 4.1 Motivation

Bridge directly from the iterator result:

> If optimizing the harness against a narrow proxy does not generalize, can specialization of the underlying model produce a more robust shift in the frontier?

### 4.2 Base model

Explain:

- Inkling-Small
- Why you chose it
- Base-model performance/cost before specialization

### 4.3 Fine-tuning methodology

Explain:

- Training data
- How it was generated/constructed
- Training objective
- Fine-tuning method
- Any relevant hyperparameters

### 4.4 Preventing overfitting / leakage

Put the fine-tuning-specific safeguards here:

- Separation between training data and held-out test tasks
- Validation strategy
- Any domain balancing
- Early stopping / checkpoint selection / whatever you actually did
- Anything used to prevent learning benchmark-specific artifacts

This corresponds to the fine-tuning and overfitting material already in your proposed outline.

### 4.5 Results

Now show:

**Retail**

- Fine-tuned Inkling: 0.875
- Opus: 0.875
- 15.5× lower cost per successful task

**Airline**

- Inkling: 0.750
- Opus: 0.690
- >13× lower cost

**Telecom**

- Inkling: 0.660
- Opus: 0.545
- >13× lower cost

### 4.6 Interpretation

Answer the question:

> Unlike harness optimization against the retail proxy, model specialization produces improvements that persist across the held-out domains.

Be careful about saying *why* unless your experiments establish causality.

---

## 5. What Moves the Frontier?

Now synthesize the four arms.

This should be the conceptual payoff rather than another giant results dump.

Compare:

**Opus 4.8 + static harness**

→ expensive, strong general baseline

**Inkling-Small base**

→ cheaper, but weaker without specialization

**Opus 4.8 + autonomous iterator**

→ cost optimized on retail proxy, poor cross-domain generalization

**Inkling-Small fine-tuned**

→ low cost + strong held-out performance

Then answer:

### 5.1 Cost vs capability

What happens when you simply use a cheaper model?

### 5.2 Harness optimization vs model specialization

What changes when you optimize the surrounding system versus changing the model itself?

### 5.3 Generalization

Which gains survive distribution/domain changes?

### 5.4 The resulting frontier

Return to your central concept:

> Which systems are actually Pareto-efficient in cost and task success?

This is where Figure 1 becomes more than a benchmark graph.

---

## 6. Implications

Only **now** broaden from your experiments to the industry.

### 6.1 Implications for enterprise agents

For sufficiently narrow, high-volume workflows:

- specialization may matter more than frontier-scale general capability
- inference economics become increasingly important

### 6.2 Implications for open-weight models

Your defensible claim is not:

> Open models will replace frontier models.

It is closer to:

> As open-weight models become sufficiently capable, domain specialization can make them economically preferable for some bounded production workloads.

### 6.3 Implications for frontier labs

This is where your larger thesis about frontier labs belongs:

- frontier labs push general capability
- application/FDE companies have domain-specific failure data
- specialization can happen closer to the application layer
- frontier models remain valuable where frontier capabilities are actually necessary

Your existing draft already starts exploring this argument.

---

## 7. Limitations

I would give this its own small section before the conclusion.

Be explicit:

- Only three TAU2 domains
- One frontier model comparison
- One small open-weight model / family
- Particular fine-tuning recipe
- Particular cost assumptions/providers
- Small proxy split for iterator
- Benchmark tasks are not equivalent to all production agent workloads
- Results do not establish that specialized small models dominate frontier models generally

This section actually makes the broader claims **more credible**, not weaker.

---

## 8. Conclusion

Do not introduce new arguments.

Bring it back to the question:

> What should organizations optimize when choosing models for agentic workloads?

Then summarize the two findings:

**Finding 1:**

Optimizing an agent system against a narrow proxy can improve its measured economics without producing a general improvement.

**Finding 2:**

Domain specialization of a small open-weight model can shift the cost-performance frontier substantially, matching or exceeding the tested frontier model at far lower cost on these held-out tasks.

End with the larger implication:

> The model with the highest general capability is not necessarily the model that produces the best production system.

---

## Paper Flow at a Glance

**1. Summary + Figure 1**

↓

**2. How do we measure the frontier?**

↓

**3. Can harness optimization move it?**

→ Method → safeguards → result → interpretation

↓

**4. Can model specialization move it?**

→ Method → safeguards → result → interpretation

↓

**5. What moves the frontier?**

→ Compare all four systems

↓

**6. Implications**

↓

**7. Limitations**

↓

**8. Conclusion**

---

# Appendix A. Verified numbers

Validity-gated from `experiments/results.csv`. 9 of 129 ledger rows are crashed or superseded runs
excluded per the pre-registered validity gate. Held-out test splits, N=5, seeds 4001 to 4005.

| System | Retail, 40 tasks | Airline, 20 tasks | Telecom, 40 tasks |
|---|---|---|---|
| Opus 4.8, static harness | 0.875 ± 0.031 at $0.5109 | 0.690 ± 0.074 at $0.6741 | 0.545 ± 0.037 at $1.4051 |
| Opus 4.8 plus agent iterator, v0.4 | 0.740 ± 0.060 at $0.0901 | 0.330 ± 0.057 at $0.1606 | 0.210 ± 0.052 at $0.2423 |
| Inkling-Small base | 0.915 ± 0.038 at $0.0416 | 0.700 ± 0.100 at $0.0763 | 0.675 ± 0.043 at $0.1774 |
| Inkling-Small fine-tuned | 0.875 ± 0.040 at $0.0329 | 0.750 ± 0.061 at $0.0414 | 0.660 ± 0.045 at $0.1028 |

Cost figures are cost per successful task. Dev-split validation, retail, N=5: Opus 0.903 at $0.5785,
iterator v0.4 0.732 at $0.0950.

Iterator run: 6 iterations, 3 accepted, v0.1 climbing to v0.4, editor search cost $0.17. It routed
the whole agent to Haiku 4.5 and dropped the system prompt after the first assistant turn.

# Appendix B. Figure map

| Figure | Content | Section | File |
|---|---|---|---|
| Figure 1 | Four systems, three held-out domains, every seed run shown | 1.3, revisited in 5.4 | `fig1_test_all_runs_four_systems` |
| Figure 2 | Iterator trajectory across its 6 iterations | 3.4 | to build |
| Figure 3 | Base versus fine-tuned by domain, cost and success panels | 4.5 | to build |

Figure style follows the Anthropic research post convention: cream canvas, letterspaced figure
eyebrow, bold headline title carrying the claim, muted palette with the comparator in warm gray,
values printed on the marks, no in-image chart titles, captions below stating the finding then the
conditions. No em dashes and no brackets anywhere in prose, labels or captions.
