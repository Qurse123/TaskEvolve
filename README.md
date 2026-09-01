# TaskEvolve

A controlled measurement study of the **cost–performance tradeoffs** of LLM agents on
[TAU2-bench](https://github.com/sierra-research/tau2-bench). We hold the benchmark fixed
and vary one thing at a time — the model, the harness, or a fine-tuned adapter — to answer
a single question:

> **How cheaply can an LLM agent complete real tasks without giving up task success —
> and which lever (a stronger closed model, an auto-optimized harness, or a specialized
> open-weight model) buys the most success per dollar?**

Every result is a **distribution, not a single run** (mean ± std over fixed seeds), because
the benchmark is stochastic (an LLM simulates the user). The headline metric is
**cost per successful task** = runtime cost ÷ tasks passed.

---

## The setup

The *task agent* plays the agent side of a TAU2 conversation (a customer-service
episode); TAU2's frozen `Orchestrator` drives the turn loop and an LLM plays the user. We
never modify the orchestrator — only the agent and its harness — so every arm is
comparable.

Three eval splits (all frozen at generation time):

| Split | Tasks | Role |
|-------|-------|------|
| **Proxy** | 12 (retail) | What the iterator optimizes against |
| **Validation** | 35 (retail) | Blind post-hoc overfitting check |
| Smoke | 3 (mock) | Wiring check at ~$0 |

Plus held-out airline / telecom / banking splits used only for the Arm D specialization study.

**Guardrails that make the numbers trustworthy:** the iterator sees *proxy logs only*
during optimization; validation and test splits are touched **once**, blind, after all
optimization completes, against a criterion committed to a file beforehand. This is what
lets us catch proxy-overfitting instead of reporting it as a win.

---

## Experimental arms

| Arm | What it tests | Model / harness |
|-----|---------------|-----------------|
| **A** | Baseline | `gpt-4.1`, static v0.1 harness |
| **Opus ref** | Strong closed-model reference | `claude-opus-4-8`, v0.1 harness |
| **B** | Can an autonomous iterator cut cost at held success? | Iterator edits prompts/harness/model routing; validated blind |
| **C** | Naive open-weight baseline | Inkling (975B/41B MoE) via Together, v0.1 harness |
| **D** | Does fine-tuning an open-weight model help? | Inkling-Small + Opus-distillation LoRA (Tinker), vs a matched base control |

---

## Findings

### 1. Retail frontier (validation split, v0.1 harness, N=5)

| Arm | Setup | Pass rate | Cost / successful task |
|-----|-------|-----------|------------------------|
| Opus ref | `claude-opus-4-8` | **0.903 ± 0.048** | $0.578 ± $0.041 |
| **C — naive Inkling** | open-weight, Together | 0.874 ± 0.056 | **$0.022 ± $0.008** |
| B — iterator (auto) | Opus→Sonnet whole-agent swap | 0.851 ± 0.051 | $0.215 ± $0.018 |
| A — baseline | `gpt-4.1` | 0.806 ± 0.047 | $0.062 ± $0.004 |

**The efficiency winner is a naive open-weight model.** Inkling nearly matches frontier
Opus on success (−3pp) at **~26× lower cost per successful task**, and it *held on the
blind validation set* — the advantage was real, not proxy-overfit. `gpt-4.1` is
dominated (both less accurate and pricier per success than Inkling). Opus buys the top
success rate, but at a steep price.

### 2. The autonomous iterator (Arm B)

We reframed Arm B from a strict single-objective rule into a general
**autonomous hypothesis-tester**: it reads proxy transcripts, proposes one edit per
iteration to any of five surfaces (prompts, harness, model routing), double-runs proxy,
and keeps the change only if it lowers cost per successful task while holding an absolute
success floor. Model choice is just one lever among five — not hand-fed.

- Left to explore, the iterator **autonomously discovered a whole-agent Opus→Sonnet swap**
  (proxy cost/successful task **−71%**, success held) — and it **transferred cleanly to
  blind validation with zero overfit** (0.851 @ $0.215, vs a proxy prediction of
  0.833 @ $0.219).
- An earlier strict-rule variant produced a change that looked good on the 12-task proxy
  (−18% cost) but **regressed success on validation** (0.806 → 0.720) — reported honestly
  as **proxy-overfit**, not a win. Small proxies hide cost-for-success trades; the blind
  validation event is what surfaced it.

**Takeaway:** an auto-optimizer wins *within its search axis* (here, the closed-Anthropic
model family), but the biggest efficiency gain still came from switching model *class*
(→ open-weight Inkling), which was outside the iterator's edit surface.

### 3. Fine-tuning an open-weight model — Arm D (clean specialization win)

Arm D distills passed-only `claude-opus-4-8` trajectories into a LoRA adapter on
**Inkling-Small** (via Tinker), then compares the tuned model against an
**identically-served base control** — the only variable is the adapter. Both are served
through one local OpenAI-compatible shim so any serving artifact cancels in the delta.
Held-out, blind, N=3 per domain:

| Domain | Base control | Tuned (D) | Δ pass | Verdict |
|--------|--------------|-----------|--------|---------|
| airline | 0.300 ± 0.100 | **0.433 ± 0.058** | +13.3pp | significant gain ✓ |
| telecom | 0.587 ± 0.046 | **0.693 ± 0.061** | +10.7pp | significant gain ✓ |
| retail | 0.848 ± 0.033 | 0.838 ± 0.016 | −1.0pp | within noise (already strong) |
| banking *(never trained)* | 0.100 ± 0.033 | 0.100 ± 0.033 | 0.0pp | no transfer regression ✓ |
| **aggregate** | 0.503 @ $0.452 | **0.540 @ $0.191** | — | ~2.4× cheaper/success |

**Opus-distillation SFT lifted the two weak trained domains significantly, held the strong
one, was cheaper per successful task in all four domains, and did not degrade the unseen
banking transfer domain** — specialization generalized without overfitting. (Disclosures:
the base is Inkling-*Small*, not the full Inkling of Arm C; behavior is bounded by what the
Opus teacher demonstrated; banking is near-floor for both systems.)

---

## Repository layout

```
target_agent/     The TAU2 task agent + its editable harness (prompts, model routing)
iterator_agent/   The autonomous optimizer: propose → double-run proxy → accept/revert
arm_d/            Distillation dataset build, Tinker training, local serving shim
benchmark/        Split generation + the TAU2 orchestrator adapter (frozen seam)
settings/         Config + model pricing table
results/, DB/     Per-run JSON/CSV ledger + a regenerable SQLite mirror
scripts/          Runners (train/eval, iterator, Arm D) + analysis & plotting
experiments/      Committed results, precommit docs, and per-run provenance
docs/paper/       The research writeup (methods, analyses, threats to validity)
tests/            Unit tests for every module (34 files)
```

Design and spec live in **`systems_design.md`** (architecture) and **`experiment.md`**
(research protocol). `CLAUDE.md` is the running lab notebook.

---

## Reproduce

```bash
# 1. Install TAU2-bench once (vendored, never committed)
git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
cd vendor/tau2-bench && uv sync --extra knowledge --extra gym --extra dev && cd -

# 2. Configure keys/models (see .env.example for every knob)
cp .env.example .env   # then fill in OPENAI_API_KEY / ANTHROPIC_API_KEY / TOGETHER_API_KEY

# 3. Wiring check at ~$0
python -m scripts.run_smoke

# 4. A baseline arm (each --repeat = one seeded run; prints mean ± std)
python -m scripts.run_train_eval --split validation --repeats 5 --seed-start 2001

# 5. Run the autonomous iterator on proxy (proxy-only, budgeted)
python -m scripts.run_iterator --max-iterations <N> --seed-start 3001

# 6. Plot the frontier (auto cost-vs-success scatter once ≥2 arms exist)
python -m scripts.plot_results
```

> Baselines require an OpenAI Tier-2 account (≈450k TPM); Tier-1's 30k throttles mid-task.

---

## Notes

- **Observability is local files only.** `results.csv` + per-task verdict JSON are
  canonical; per-turn detail comes from persisting TAU2's transcript locally. No hosted
  tracing service.
- **Run artifacts are regenerable and gitignored** (`experiments/logs/`, `plots/`,
  `results.csv`, `run_logs/`). Official numbers live in `docs/paper/` and the precommit
  files under `experiments/`.
