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
- [ ] **Generate split JSONs** — run `benchmark/splits.py` to populate `benchmark/splits/*.json` (currently empty)
- [x] `results/logger.py` — per-run folder JSON logs (start_run / log_task / finalize_run)
- [x] `DB/storage.py` — SQLite mirror of `results.csv` (`results` table keyed by `run_id`); sync via `python -m DB.storage`, open `experiments/results.db` in DBeaver. CSV stays canonical; DB is a regenerable query layer (no server — YAGNI for ~100 run-rows).
- [ ] `scripts/run_smoke.py` — 3 mock tasks, verify wiring at zero cost
- [ ] **Run smoke test** ← gate before spending real money
- [ ] `scripts/run_train_eval.py` — proxy and validation runner (supports `--repeats N`; each repeat = one `results.csv` row, unique `run_id`)
- [ ] **Run proxy baseline ×N** (~$3/run, 12 tasks) — repeat N times (e.g. 5) for an Arm A variance band, not a single point
- [ ] **Run validation baseline ×N** (~$10/run, 35 tasks) — repeat for variance; official Arm A number is the **mean ± std** across repeats
- [ ] `scripts/plot_results.py` — visualizer reading `experiments/results.csv`: group by (`split`, `harness_version`), show baseline spread across the N repeats + cost-vs-success Pareto (experiment.md §25). Build once the baseline repeats exist; Langfuse covers live per-call cost until then.

> **Baseline = a distribution, not one run.** The benchmark is stochastic (LLM user-sim; OpenAI non-determinism even at temp 0), so every baseline and every accepted change is measured as mean ± std over N runs. The per-run-row `results.csv` already supports this — no schema change, just group rows.

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

---

## Hard Constraints

1. **Never modify the TAU Orchestrator.** Doing so invalidates benchmark comparisons.
2. **Iterator never sees validation or test results during optimization.** Only proxy logs.
3. **Frozen surfaces (iterator and humans alike):**
   - `vendor/tau2-bench/` — everything in here
   - `benchmark/splits/*.json` — task IDs are fixed at generation time
4. **Validation runs once**, post-hoc, after all iterations complete.
5. **TAU2 test split runs once**, at milestone end, via `scripts/run_tau_test.py`.
6. **Double-run rule:** before committing a change, the iterator must run proxy eval twice (different seeds). Both must show improvement.

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
