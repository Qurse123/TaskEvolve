# External Baselines & Literature Positioning (Tier 1 #3)

**Purpose.** Every number in this study is currently self-referential — our six arms measured only
against each other. This document positions our results against *published* reference points so a
reader can judge whether our absolute numbers are good, average, or broken. All numbers below were
fetched from the cited URLs on 2026-08-19; where a number could not be verified from a primary
source it is explicitly flagged.

> **Scope caveat (read first).** These are *cited, not reproduced* baselines. We did **not** re-run
> any external model or agent inside our own harness. Citation-only positioning is materially weaker
> than a same-harness re-run: the published numbers below use different task sets, different agent
> harnesses, and (critically) a different headline metric than ours. Treat this as a sanity-check
> and a positioning device, not as a controlled comparison. See "Metric mismatch" and "What we
> still lack" below.

---

## 1. Published TAU2 / TAU-bench reference points

### Metric definition (essential for like-for-like reading)

- **TAU2-bench headline metric is `pass^k`** = "the fraction of `k` independent runs that **all**
  succeed" (all-of-k), reported for k = 1..4. `pass^1` is therefore the average single-run task
  success rate; `pass^k` for k>1 is *strictly stricter* than a mean and drops sharply for
  inconsistent models (Sierra report GPT-4o retail falling from ~60% at pass^1 toward ~25% by
  pass^8). Source: τ²-Bench paper and the benchmarkingagents write-up (both cited below).
- **Our metric is different.** We report the **mean over N=5 seeds of the per-run task success rate**
  on our seeded retail-*train* subset (35 tasks for validation, 12 for proxy). In expectation this
  approximates **average pass^1**, NOT pass^k for k>1. So our numbers are only loosely comparable to
  Sierra's pass^1 column and are **not** comparable to their pass^k (k≥2) columns.
- **User simulator:** TAU2 uses `gpt-4.1-2025-04-14` as the simulated user. (We also use TAU2's
  stock user simulator via the vendored orchestrator, so this axis is aligned.)
- **Task counts (official domains):** retail **115**, airline **50**, telecom **114**. Our splits are
  seeded subsets of the retail *train* pool (proxy 12 / validation 35), **not** the official 115-task
  retail set — a difficulty-mix confound on top of the metric mismatch.

### Reference table

| Model | Domain | Metric | Score | Source |
|-------|--------|--------|-------|--------|
| GPT-4.1 | retail | pass^1 | **74%** | τ²-Bench paper (arXiv 2506.07982) |
| GPT-4.1 | airline | pass^1 | **56%** | τ²-Bench paper (arXiv 2506.07982) |
| GPT-4.1 | telecom | pass^1 | **34%** | τ²-Bench paper (arXiv 2506.07982) |
| Claude 3.7 Sonnet | retail | pass^1 | ~60% | τ²-Bench paper (arXiv 2506.07982) |
| Claude 3.7 Sonnet | airline | pass^1 | ~49% | τ²-Bench paper (arXiv 2506.07982) |
| Claude 3.7 Sonnet | telecom | pass^1 | 49% | τ²-Bench paper (arXiv 2506.07982) |
| o4-mini | retail | pass^1 | ~55% | τ²-Bench paper (arXiv 2506.07982) |
| o4-mini | airline | pass^1 | ~45% | τ²-Bench paper (arXiv 2506.07982) |
| o4-mini | telecom | pass^1 | ~42% | τ²-Bench paper (arXiv 2506.07982) |
| GPT-4.1-mini | retail | pass^1 | ~50% | τ²-Bench paper (arXiv 2506.07982) |
| GPT-4.1-mini | airline | pass^1 | ~40% | τ²-Bench paper (arXiv 2506.07982) |
| GPT-4.1-mini | telecom | pass^1 | ~50% | τ²-Bench paper (arXiv 2506.07982) |
| Claude 3.5 Sonnet (2024-10-22) | retail | pass^1 | **69.2%** | Original τ-bench frozen board (arXiv 2406.12045) |
| Claude 3.5 Sonnet (2024-10-22) | airline | pass^1 | **46.0%** | Original τ-bench frozen board (arXiv 2406.12045) |
| GPT-4o | retail | pass^1 | **60.4%** | Original τ-bench frozen board (arXiv 2406.12045) |
| GPT-4o | airline | pass^1 | **42.0%** | Original τ-bench frozen board (arXiv 2406.12045) |
| Simia-Tau (Qwen2.5-32B) | retail | pass^1 | 61.7% | community leaderboard snippet (unverified — see note) |

**Precision notes.**
- Values marked `~` were read off **Figure 3** of the τ²-Bench paper (a bar chart), not a numeric
  table — treat them as approximate (±2–3 pp). The three **GPT-4.1** values (74 / 56 / 34) are the
  most robust: they appear as explicit numbers in both the paper and secondary write-ups.
- The 74% GPT-4.1 retail figure is the single most important external anchor for this study and is
  corroborated across two independent sources (the paper HTML and the AISBench documentation).
- The **original τ-bench board is frozen at late-2024 models** — Claude 3.5 Sonnet still tops it
  (69.2% retail / 46.0% airline). Newer frontier models (Claude 4.x, GPT-5, Gemini 2.5/3.x) are not
  on that frozen board; their TAU-bench numbers come from vendor self-reports or the successor
  boards and are **not directly comparable** to it.
- The Simia-Tau (Qwen2.5-32B) retail 61.7% figure surfaced only in a search snippet and was **not
  confirmed** against a primary table; include only with that caveat.
- Live community aggregators exist (taubench.com, benchlm.ai, ctrlaltdebrief.com) but the
  benchlm.ai board explicitly warns it "mixes a large third-party telecom snapshot with smaller
  provider-published slices" under different harnesses, so its top-25 (many 95–99% scores) are
  **telecom-weighted and not comparable** to a retail-only reading. Do not cite those headline
  percentages as retail reference points.

**Sources**
- τ²-Bench: *"τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment,"* Barres,
  Dong, Ray, Si, Narasimhan, 2025. https://arxiv.org/abs/2506.07982 (PDF: https://arxiv.org/pdf/2506.07982)
- Original τ-bench (frozen board numbers): *"τ-bench: A Benchmark for Tool-Agent-User Interaction
  in Real-World Domains,"* Yao et al., 2024. https://arxiv.org/abs/2406.12045
- Sierra Research repo & leaderboard pointer: https://github.com/sierra-research/tau2-bench
  (README, RELEASE_NOTES; live board at https://taubench.com)
- Independent write-up of pass^k + frozen-board numbers: https://benchmarkingagents.com/tau-bench-retail-airline/
  and https://benchmarkingagents.com/tau-bench/
- AISBench τ²-bench documentation (corroborates GPT-4.1 74/56/34): https://ais-bench-benchmark.readthedocs.io/en/latest/extended_benchmark/agent/tau2_bench.html
- Community aggregators (non-comparable; listed for completeness): https://benchlm.ai/benchmarks/tau2-bench ,
  https://ctrlaltdebrief.com/tools/benchmarks/tau2

---

## 2. Placing OUR numbers against these references

**Our validation numbers** (retail-train subset, N=5 unless noted; mean-of-seeds per-run success):
gpt-4.1 **0.806**; Inkling (open, Arm C) **0.874**; claude-opus-4-8 **0.903**; Opus→Sonnet 5
(Arm B) **0.851**; Inkling-Small base **0.848** (N=3), tuned **0.838** (N=3).

- **gpt-4.1 retail: ours 0.806 vs Sierra pass^1 0.74.** Same ballpark, ours ~6 pp higher. This is a
  *reassuring sanity check, not a match*, and the gap is fully explained by confounds we cannot
  remove without a re-run: (a) our 35 tasks are a seeded subset of retail-*train*, plausibly an
  easier mix than the official 115-task retail set; (b) our v0.1 harness differs from Sierra's
  reference agent; (c) metric — our mean-of-seeds approximates *average pass^1*, so it is comparable
  in kind to Sierra's pass^1 column but would look much higher than their pass^k (k≥2) columns. Net:
  our gpt-4.1 baseline is *consistent with* the published GPT-4.1 retail result — it is not broken
  and not implausibly high.
- **Frontier closed models (Opus 0.903, Sonnet-routed 0.851) and open Inkling (0.874)** all sit
  above Sierra's GPT-4.1 74% retail pass^1. That ordering is expected: these are 2026 frontier /
  strong-open models measured on a (likely easier) retail-train subset with average-pass^1-style
  scoring, versus a 2025 GPT-4.1 pass^1 on the full set. Their absolute levels (85–90%) are
  plausible for current frontier models on retail, but again are **not** a controlled comparison.
- **Cross-domain (Arm D, Inkling-Small, N=3) vs Sierra GPT-4.1 pass^1:**
  - *Airline:* ours base 0.300 → tuned 0.433, vs GPT-4.1 0.56. A small open model trailing a
    frontier closed model on airline is expected; fine-tuning closes part of the gap.
  - *Telecom:* ours base 0.587 → tuned 0.693, vs GPT-4.1 0.34. **Our telecom base is higher than
    Sierra's published GPT-4.1 telecom pass^1 — flag this.** Almost certainly a task-subset /
    metric artifact (our telecom-train subset and mean-of-seeds scoring vs their 114-task pass^1),
    not evidence that Inkling-Small beats GPT-4.1 on telecom. Do not report it as such without a
    same-set re-run.
  - *Banking:* ours 0.100 → 0.100 (no movement). Banking is TAU2's separate `banking_knowledge`
    (knowledge-retrieval) domain; we found **no** clean published GPT-4.1 banking number to anchor
    against, so this point currently has **no external reference**.

**Bottom line:** the one anchor we can lean on with confidence — GPT-4.1 retail — lands our
baseline in the right neighborhood (0.806 vs 0.74), which argues the harness is not broken. Every
other cross-reference is directional only, blocked by the metric + task-subset mismatches below.

---

## 3. Related work / positioning

**Agent benchmarking with simulated users (the benchmark itself).** τ-bench (Yao et al., 2024,
arXiv 2406.12045) and its successor τ²-bench (Barres et al., 2025, arXiv 2506.07982) established the
tool-agent-user, policy-constrained, `pass^k` evaluation we build on. τ²-bench's core contribution —
`pass^k` consistency scoring — is directly relevant to a *caveat* in our work: our mean-of-seeds
metric hides the consistency failures that pass^k is designed to surface.

**Cost-vs-success / cost-efficiency framing for LLM systems.** FrugalGPT (Chen, Zaharia, Zou, 2023,
arXiv 2305.05176) is the canonical prior work: it reduces inference cost via prompt adaptation, LLM
approximation, and LLM cascades, reporting up to ~98% cost savings while matching best-single-model
performance. Our study shares its *cost-per-quality* objective but differs in setting: FrugalGPT
optimizes single-turn query routing on QA-style datasets, whereas we optimize a *multi-turn,
tool-using, policy-constrained agent* and measure **cost per successful task** on a frontier agent
benchmark. Our Arm B (autonomous iterator discovering an Opus→Sonnet whole-agent swap) is
effectively a *degenerate one-model cascade* selected by search rather than a learned router —
positioning our contribution as "cascade/model-selection ideas transferred to the agent-harness
regime, measured on a Pareto cost-vs-success frontier."

**Open-vs-closed cost comparison.** A central finding of our study (Arm C: naive open-weight Inkling
Pareto-beating gpt-4.1 — higher success at ~10–26× lower cost) sits in the broader open-vs-closed
efficiency literature. Community trade-off write-ups on τ²-bench (e.g. the "stop trusting headline
scores, measure trade-offs" argument — Alan/Medium; HTTP-403 at fetch time, cited by title only)
make the same methodological point we adopt: a single success percentage is misleading without the
paired cost axis. We could not verify that article's specific numbers.

**Agent-harness / input optimization on the same benchmark.** There is at least one arXiv paper
studying *input reformulation to improve tool-usage accuracy on τ-bench* (arXiv 2508.20931, 2025) —
a same-benchmark harness-optimization reference point conceptually adjacent to our iterator (Arm B).
We were unable to extract its exact baseline/improved numbers (fetch exceeded the content-size
limit), so it is cited as *existing related work* only, not as a numeric baseline. Related
agent-training-via-simulation and RL-for-tool-use work also cites τ²-bench (e.g. arXiv 2511.01824,
arXiv 2508.18669) and could supply additional open-model retail reference points if fetched and
verified in a follow-up.

**Distillation / specialization for agents.** Our Arm D (fine-tuning Inkling-Small per domain via
Tinker) is a small instance of task/domain specialization of an open model for agentic use. The
cost-vs-success benefit of distilling/specializing smaller models to approach larger ones is the
same thesis as the LLM-approximation branch of FrugalGPT; our contribution is measuring it on a
multi-turn agent benchmark across four TAU2 domains rather than on static classification/QA.

**Sources (this section)**
- FrugalGPT: https://arxiv.org/abs/2305.05176
- Input reformulation on τ-bench: https://arxiv.org/pdf/2508.20931 (numbers unverified)
- Agent-training-via-simulation / RL-for-tool-use citing τ²-bench: https://arxiv.org/pdf/2511.01824 ,
  https://arxiv.org/pdf/2508.18669 (not yet extracted)
- τ-bench / τ²-bench: https://arxiv.org/abs/2406.12045 , https://arxiv.org/abs/2506.07982

---

## 4. What external baselines we still LACK (honest limitations)

1. **No same-harness re-run of any external agent.** Every number in §1 is *cited, not reproduced*.
   We never ran GPT-4.1, Claude, o4-mini, or any published agent inside *our* TaskEvolve harness on
   *our* splits. This is the single biggest validity gap: citation-only positioning cannot separate
   "our result differs because the model differs" from "…because the harness/task-set/metric
   differs."
2. **Metric mismatch (pass^k vs mean-of-seeds) is unresolved.** Sierra reports `pass^k` (all-of-k);
   we report mean-of-seeds per-run success (≈ average pass^1). Our numbers are therefore only
   loosely comparable to Sierra's pass^1 column and **not** comparable to pass^k (k≥2). To compare
   like-for-like we would need to compute our own pass^k on repeated seeded runs, or re-run
   Sierra's models under our per-seed-mean protocol.
3. **Task-set mismatch.** We evaluate on seeded subsets of the retail-*train* pool (12 / 35 tasks),
   not the official 115-task retail set (nor the 50 airline / 114 telecom sets). Difficulty mix is
   an uncontrolled confound behind, e.g., the surprising telecom-base > GPT-4.1-telecom reading.
4. **No verified banking (`banking_knowledge`) external anchor.** Arm D banking (0.100→0.100) has no
   published GPT-4.1 reference we could confirm.
5. **Several nearby numbers left unverified.** The Figure-3 `~` values (Claude 3.7, o4-mini,
   gpt-4.1-mini), Simia-Tau retail 61.7%, and the input-reformulation paper's numbers were not
   confirmed against primary numeric tables and are marked as such above.
6. **Frozen-board vs live-board hazard.** The original τ-bench board is frozen at late-2024 models;
   live aggregators mix harnesses/domains. Any frontier-model comparison drawn from those boards is
   at best indicative.

**Recommended next step to strengthen this to a Tier-1 baseline:** run *one* external reference
agent (e.g. the stock TAU2 GPT-4.1 agent) on our exact splits under our exact metric — converting
the strongest anchor (GPT-4.1 retail) from a cited number into a same-harness reproduced number.
That single re-run would neutralize gaps #1–#3 for the most load-bearing comparison in the paper.
