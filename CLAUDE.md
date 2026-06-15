# TaskEvolve — Working Reference

## What This Repo Is

A controlled measurement study of LLM agent cost-performance tradeoffs on TAU2-bench (retail domain). We optimize a task agent across 6 experimental arms and measure cost per successful task vs task success rate.

Full design: `systems_design.md` | Research spec: `experiment.md`

---

## Current Status: Milestone 1 — Arm A Baseline

Build the harness, run the agent, get a baseline score. Nothing else. when reading the build sequence and starting a new task ensure that you look at /Users/mihirsawhney/Projects/TaskEvolve/systems_design.md and /Users/mihirsawhney/Projects/TaskEvolve/experiment.md to ensure that what you are building is compliant with these documents, use subagent driven development, after complete each sequence stop so I can review/give feedback for making edits

### Build Sequence

- [x] `systems_design.md`
- [x] `pyproject.toml` + `.env.example`
- [x] `benchmark/splits.py` — generate smoke/proxy/validation split JSON files
- [x] `target_agent/agent.py` — `TaskEvolveAgent(HalfDuplexAgent)`
- [x] `target_agent/prompts/system_prompt.j2` — initial system prompt
- [x] `benchmark/adapter.py` — inject `TaskEvolveAgent` into TAU2's `Orchestrator`, run via frozen `run_simulation()`
- [x] `target_agent/traces/langfuse_setup.py` — Langfuse client + LiteLLM `langfuse_otel` callback
- [x] **Generate split JSONs** — `benchmark/splits/{smoke,proxy,validation}.json` written (3 mock / 12 + 35 retail-train, seed 42; proxy⟂validation disjoint). Now frozen.
- [x] `results/logger.py` — per-run folder JSON logs (start_run / log_task / finalize_run)
- [x] `DB/storage.py` — SQLite mirror of `results.csv` (`results` table keyed by `run_id`); sync via `python -m DB.storage`, open `experiments/results.db` in DBeaver. CSV stays canonical; DB is a regenerable query layer (no server — YAGNI for ~100 run-rows).      
- [x] `scripts/run_smoke.py` — 3 mock tasks, verify wiring at zero cost (built; imports + split-load verified at $0). Run via `python -m scripts.run_smoke`.
- [x] **Run smoke test** ← PASSED 3/3 (mock domain, ~$0.009 total). Full chain verified: splits → agent → orchestrator → evaluator → results logger (JSON+CSV) → SQLite mirror. cost_per_successful_task flowing.
- [x] `scripts/run_train_eval.py` — proxy/validation runner (`--split --seed-start S --repeats N`; each repeat = one `results.csv` row; prints mean ± std). Built + verified at $0 (helpers, split-load, arg-validation). Run: `python -m scripts.run_train_eval --split proxy --repeats 5 --seed-start 1001`.
- [x] **Run proxy baseline ×5** — seeds `1001..1005`, gpt-4.1, harness v0.1. **Arm A proxy baseline: pass_rate 0.583 ± 0.083, cost_per_successful_task $0.083 ± $0.016** (n=5; per-seed passes 6/8/7/6/8 of 12). Rows in `results.csv` (`proxy_2026-06-12_19*`), plot `experiments/plots/per_metric_proxy.png`. Required OpenAI Tier 2 (450k TPM) — Tier 1's 30k throttled mid-task.
- [x] **Better results visualization** — a single-arm baseline is a poor fit for a cost-vs-success scatter (one cluster crushed into a corner, no frontier to read). `plot_results.py` now defaults to a **per-metric horizontal dot plot** (raw seeds + mean ± std for task success rate and cost per successful task) when only one arm is present, and auto-switches to the **cost-vs-success frontier scatter** (auto-zoomed axes; arm mean drawn as a ◆ with x/y std error bars; `--from-zero` for origin-anchoring) once ≥2 arms exist. M1 baseline figure: `experiments/plots/per_metric_proxy.png`.
- [x] **Run validation baseline ×5** — seeds `2001..2005`, gpt-4.1, harness v0.1 (35 retail tasks each). **Arm A validation baseline: pass_rate 0.806 ± 0.047, cost_per_successful_task $0.062 ± $0.004** (n=5; per-seed passes 30/30/27/27/27 of 35). Clean run on the hardened runner (no transient retries or recorded failures fired). Rows in `results.csv` (`validation_20260615_*`), figure `experiments/plots/per_metric_validation.png`. Note: validation pass_rate (0.806) runs higher than proxy (0.583) — larger/easier task mix; reported per-split as a distribution, not as a proxy-vs-validation frontier (same Arm A system, different task sets — a true cross-arm frontier is M2).
- [x] `scripts/plot_results.py` — render baseline results from `experiments/results.csv` in two modes. **Each run = one point** (`total_cost_usd/num_tasks`, `pass_rate`), grouped into arms by `(split, harness_version)`. (1) **Per-metric** (default for a single arm): horizontal dot plot per headline metric (task success rate, cost per successful task) — raw seeds + mean ± std. (2) **Frontier** (default once ≥2 arms; force with `--frontier`): cost-vs-success scatter, arm mean as ◆ with x/y std error bars, axes auto-zoomed to data (`--from-zero` anchors at origin). Mode auto-picks by arm count; override with `--per-metric`/`--frontier`. Run: `python -m scripts.plot_results [--split proxy]` → PNG in `experiments/plots/`. Needs `matplotlib` (in pyproject). Mean ± std stays the *reported* number; the graph shows the raw distribution.


> **Baseline = a distribution, not one run.** Milestone 1 uses `N=5` as the precommitted minimum repeat count. The benchmark is stochastic (LLM user-sim; OpenAI non-determinism even at temp 0), so every baseline is measured as mean ± std over fixed seeds. The per-run-row `results.csv` already supports this — no schema change, just group rows.

Install TAU2-bench (do this once):
```bash
git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
cd vendor/tau2-bench && uv sync --extra knowledge --extra gym --extra dev     
```

---

## Iterator Agent Edit Surface (Milestone 2+)

The iterator can modify exactly these files — nothing else:

| File | What it controls |
|------|-----------------|
| `target_agent/prompts/system_prompt.j2` | Agent persona, instructions, format rules |
| `target_agent/prompts/policy_summary.j2` | Compressed/rewritten domain policy |
| `target_agent/prompts/few_shot_examples.j2` | Example conversations (empty in M1) |
| `target_agent/harness.py` | History compression, tool filtering |
| `target_agent/model_routing.py` | Model selection (gpt-4.1 vs gpt-4.1-mini) |

### Deferred to M2 (iterator build)

- [ ] **Proper GENERATION traces + task/run grouping in Langfuse.** M1 uses LiteLLM's `langfuse_otel` callback (the only one compatible with Langfuse v4 — the native `"langfuse"` callback needs the removed v2 `langfuse.model`/`langfuse.client` APIs). OTEL spans get typed as `TOOL`/`SPAN` named `litellm_request`, so the dashboard's Generations view + per-model token/cost rollups look empty. The data is still captured (trace-level cost, token usage) and queryable via the REST API, and the headline `cost_per_successful_task` comes from TAU2's accounting in `results.csv`, not Langfuse — so M1 is unaffected. **Fix when building the iterator:** wrap the agent's LiteLLM call in a Langfuse v4 `start_as_current_generation()` (in `target_agent/`) and attach `task_id` + `run_id` metadata. This yields proper GENERATION observations *and* groups traces per task/run so the iterator can slice cost by task/model. (Do **not** downgrade to langfuse v2 to get the native callback.)

---

## Hard Constraints

1. **Never modify the TAU Orchestrator.** Doing so invalidates benchmark comparisons.
2. **Iterator never sees validation or test results during optimization.** Only proxy logs.
3. **Frozen surfaces (iterator and humans alike):**
   - `vendor/tau2-bench/` — everything in here
   - `benchmark/splits/*.json` — task IDs are fixed at generation time
4. **Validation runs once as a blind post-hoc event**, after all iterations complete. That event contains the precommitted repeated run set (`N=5`, seeds `2001..2005`); the iterator never sees these logs during optimization.
5. **TAU2 test split runs once**, at milestone end, via `scripts/run_tau_test.py`.
6. **Double-run rule:** before committing a change, the iterator must run proxy eval twice with different, logged seeds. Both must show improvement against the current best proxy distribution without crossing cost, policy, invalid-action, or latency guardrails.

---

## Eval Splits Quick Reference

| Split | Tasks | Cost | Who uses it |
|-------|-------|------|-------------|
| Smoke | 3 (mock domain) | ~$0 | Wiring check only |
| Proxy | 12 (retail train) | ~$3/run | Iterator trains against this |
| Validation | 35 (retail train) | ~$10 | Post-hoc overfitting check only |
| TAU2 test | ~40 (official) | ~$12 | Final paper reporting, once |

---

## Key Architecture Facts

- TAU2's `Orchestrator` drives the full turn loop. We only implement `generate_next_message()`.
- One call to `generate_next_message()` = one turn. Return a tool call OR text, never both.
- All prompt files are Jinja2 templates (`.j2`), rendered at agent init.
- Langfuse: TAU2 hardcodes `USE_LANGFUSE=False` (frozen). Our `target_agent/traces/langfuse_setup.py` reads `USE_LANGFUSE` from `.env` and wires LiteLLM's `langfuse_otel` callback itself. Call `init_tracing()` at run start, `flush_tracing()` before exit.
- Log structure: `experiments/logs/<split>_<YYYYMMDD>_<HHMMSS>/task_*.json`
