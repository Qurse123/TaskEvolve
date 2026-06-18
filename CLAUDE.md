# TaskEvolve — Working Reference

## What This Repo Is

A controlled measurement study of LLM agent cost-performance tradeoffs on TAU2-bench (retail domain). We optimize a task agent across 6 experimental arms and measure cost per successful task vs task success rate.

Full design: `systems_design.md` | Research spec: `experiment.md`

---

## Current Status: Milestone 2 — Arm B (iterator agent). Milestone 1 ✅ COMPLETE.

**Standing workflow (applies to every milestone):** when starting a new task in a build sequence, read `/Users/mihirsawhney/Projects/TaskEvolve/systems_design.md` and `/Users/mihirsawhney/Projects/TaskEvolve/experiment.md` first to confirm what you build is compliant with those documents. Use subagent-driven development. After completing each sequence, stop so I can review / give feedback before the next.

## Milestone 1 — Arm A Baseline ✅ COMPLETE

Build the harness, run the agent, get a baseline score. Nothing else.

### Build Sequence

- [x] `systems_design.md`
- [x] `pyproject.toml` + `.env.example`
- [x] `benchmark/splits.py` — generate smoke/proxy/validation split JSON files
- [x] `target_agent/agent.py` — `TaskEvolveAgent(HalfDuplexAgent)`
- [x] `target_agent/prompts/system_prompt.j2` — initial system prompt
- [x] `benchmark/adapter.py` — inject `TaskEvolveAgent` into TAU2's `Orchestrator`, run via frozen `run_simulation()`
- [x] ~~`target_agent/traces/langfuse_setup.py`~~ — Langfuse tracing (built in M1; **removed in M2** — observability is local files only, see "Key Architecture Facts")
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

## Milestone 2 — Arm B: Iterator-Optimized Harness

Build the **iterator agent**: a separate process that reads proxy logs, proposes exactly **one** harness change per iteration, **double-runs proxy** (two logged seeds) to confirm the change beats the current-best distribution within guardrails, keeps or reverts, and repeats under a **fixed optimization budget**. The output is the Arm B optimized harness; validation runs **once, blind, post-hoc**. The iterator is the optimizer — it never performs the benchmark task itself.

Spec: `experiment.md` §8.5 (iterator job), §19–20 (edit surface + change-acceptance rule), §21 (iteration loop), §22 (log fields), §23 (`iterator_agent/` layout); `systems_design.md` §3.5 (9-step architecture).

> **Milestone-numbering note:** `experiment.md §14.2` labels "M2" as Arm C/D (open-weight models). We follow the `systems_design.md`/CLAUDE.md framing where **M2 = the iterator agent (Arm B)** — the natural next step after the Arm A baseline (`experiment.md §14.1`: "Arm B starts only after the Arm A baseline is reproducible and reviewed"). Open-weight arms (C/D) move to M3+.

### Iterator Edit Surface — the iterator may modify exactly these files, nothing else

| File | What it controls |
|------|-----------------|
| `target_agent/prompts/system_prompt.j2` | Agent persona, instructions, format rules |
| `target_agent/prompts/policy_summary.j2` | Compressed/rewritten domain policy |
| `target_agent/prompts/few_shot_examples.j2` | Example conversations (empty in M1) |
| `target_agent/harness.py` | History compression, tool filtering |
| `target_agent/model_routing.py` | Model selection (gpt-4.1 vs gpt-4.1-mini) |

### Iterator Architecture — four roles, each a separate module

A run starts when a **ticket** (the kickoff work-item: optimization objective + the fixed budget) is handed to the orchestrator. The orchestrator loads the **Arm A proxy distribution as the current-best baseline**, then drives the loop. **Only the editor calls an LLM** — the accept/reject decision is a deterministic rule, not an LLM judgment (so the loop is reproducible and auditable). Each LLM step gets its own Jinja prompt template under `iterator_agent/prompts/`; the deterministic steps are plain code.

| Role | Module | LLM? | Responsibility |
|------|--------|------|----------------|
| **Orchestrator** | `iterator_agent/run_iteration.py` | no (control flow) | Receives the ticket, loads the current-best proxy baseline, drives one full propose→test→accept/revert cycle, repeats under the budget. |
| **Editor** (diagnose → propose) | `iterator_agent/researcher.py` + `prompts/diagnose.j2`, `prompts/propose_edit.j2` | **yes** | Reads proxy failure feedback, proposes exactly **one** change to one allowed file with `change_summary` + `reason`. Two templates: one summarizes failures, one writes the concrete edit. |
| **Eval-runner** | reuses `scripts/run_train_eval.run_repeats` / `benchmark/adapter.run_eval` | no | Runs the proxy split **twice** (two logged seeds) for the change under test; runs the blind validation event once at the end. Already built in M1 — wrap, don't rewrite. |
| **Comparator / acceptance** | `iterator_agent/acceptance.py` + `baseline.py` | no (deterministic rule) | Accept iff **both** proxy runs beat the current-best proxy distribution within guardrails (`experiment.md §20`); else revert. |

**Accept flow (per iteration):** ticket → load Arm A proxy as current-best → editor proposes one edit → `edit_guard` checks it touches only allowed files → apply → eval-runner runs **proxy ×2** → comparator applies the deterministic rule → **accept** (bump `config.HARNESS_VERSION`, commit, log to `accepted_changes.md`) or **revert** (log to `rejected_changes.md`). The full **validation suite is NOT run here** — it stays blind and runs once after the budget is exhausted (`Hard Constraints` #2/#4). "Improvement" per iteration = the proxy double-run, not validation.

### Build Sequence

**Foundations (observability + logging the iterator depends on):**

- [ ] **Local per-turn debug artifacts (replaces Langfuse).** Langfuse was **removed in M2** — local files are easier to debug, live in git, and need no external service/keys. For per-turn debugging of failed tasks, persist TAU2's `SimulationRun` transcript into the run folder (e.g. `task_<id>_messages.json`) so `feedback.py` and a human can see *why* a task failed, not just the verdict. (Optional/when needed — the verdict JSON + `results.csv` already drive the loop.)
- [ ] **Iterator search-cost accounting.** Capture the editor's per-call LLM cost (LiteLLM `response._hidden_params["response_cost"]`), return it alongside the `ProposedEdit`, and record it per iteration as **iterator search cost** — kept *separate* from runtime task cost (`experiment.md §15.3`). Feeds the budget cap in `scripts/run_iterator.py` and the per-iteration log. (`feedback.py` reads task cost from the `results.csv`/JSON ledger — TAU2's accounting.)
- [x] `iterator_agent/iteration_log.py` — writes one experiment-note entry per iteration with the `experiment.md §22` Arm B field subset (`iteration_id`, `timestamp`, `harness_version`, `changed_surface`/`changed_file`, `change_summary`, `reason_for_change`, `proxy_seeds`, proxy success before/after mean±std, cost-per-successful-task before/after, `accepted_or_rejected`, `reason_accepted_or_rejected`, plus `iterator_search_cost_usd`/`notes`). API: `IterationRecord` (frozen), `write_record(record, *, logs_root=...)` → `experiments/iterations/<iteration_id>/iteration.json`; `write_artifact(iteration_id, filename, content, *, logs_root=...)` (ad-hoc debug files — the local-debug surface that replaced Langfuse); `append_changelog(record, *, experiments_dir=...)` → bullet into `experiments/accepted_changes.md` / `rejected_changes.md` (raises on unknown verdict). Pure persistence, no LLM. The `iteration.json` files (keyed by `iteration_id`) are the source for the per-iteration plots below; per-run `results.csv` rows reused unchanged — no schema break. 8 unit tests in `tests/test_iteration_log.py`, all green.

**Edit guard + frozen config:**

- [x] `iterator_agent/allowed_edits.yaml` — declares the allowed edit surface (the 5 files above), forbidden/frozen paths (`vendor/tau2-bench/`, `benchmark/splits/`, `benchmark/adapter.py`, `results/`, and the guard config itself), and the acceptance **guardrails** block (`task_success_floor_frac_of_best: 0.95` + cost/invalid-action ceilings declared as `null` until `acceptance.py` consumes them). Set before the run and **frozen for the run — the iterator may not edit this file** (`experiment.md §19.3`, §20).
- [x] `iterator_agent/edit_guard.py` — validates a proposed change touches only allowed-surface files; rejects immediately on any frozen/forbidden path, absolute path, or `..` escape (the "Allowed Change Check" in the `experiment.md §12` data flow). API: `load_policy()` → `EditPolicy`, `evaluate(target, policy)` → `EditDecision`, `assert_allowed(target, policy)` (raises `ForbiddenEditError`). 10 unit tests in `tests/test_edit_guard.py`, all green; pinned `pyyaml` + added `iterator_agent` to the wheel packages.

**Iterator core (the four roles):**

- [x] `iterator_agent/baseline.py` — **(comparator input)** loads the current-best proxy distribution (mean ± std of `pass_rate` and `cost_per_successful_task`) for a `(split, harness_version)` from `experiments/results.csv`. API: `load_distribution(split, harness_version, *, csv_path=...)` → `Distribution`; reuses `results.logger`'s columns, ignores cost-less rows for cost stats, raises `ValueError` on no match. 6 unit tests in `tests/test_baseline.py`, all green.
- [x] `iterator_agent/feedback.py` — **(editor input)** reads **only** the most recent proxy run folder and summarizes failures (per-task reward, termination reason, cost) into a `FeedbackSummary`. API: `find_latest_run_dir(split, *, logs_root=...)` (newest `<split>_*` folder by name; only ever the requested split, so proxy-asks can't return validation), `summarize_run(run_dir)`, `build_feedback(split="proxy", *, logs_root=...)`. Ignores `run_summary.json`; ranks termination reasons by frequency. 5 unit tests in `tests/test_feedback.py`, all green. (Per-turn detail, if needed, comes from persisting TAU2's transcript locally — no Langfuse.)
- [x] `iterator_agent/researcher.py` + `iterator_agent/prompts/diagnose.j2`, `iterator_agent/prompts/propose_edit.j2` — **(editor — the only LLM role)** strong closed-weight model via new `config.ITERATOR_MODEL` (.env; separate from `AGENT_MODEL`). `diagnose.j2` turns `feedback` into a failure analysis; `propose_edit.j2` turns that into exactly **one** **full-file-replacement** edit to one allowed file, returned as JSON (`target_file`, `new_content`, `change_summary`, `reason_for_change`). API: `run_editor(feedback, *, policy, complete, repo_root)` → `ProposedEdit`; `render_diagnose`/`render_propose`/`parse_proposal` (strips ``` fences, validates keys). LLM call injected (`complete: Callable[[str],str]`) so tests run at $0; default wraps LiteLLM. Editor only proposes — `edit_guard` is the authoritative gate in the orchestrator. 7 unit tests in `tests/test_researcher.py`, all green.
- [x] `iterator_agent/acceptance.py` — **(comparator — deterministic rule, no LLM)** encodes the double-run rule (`experiment.md §20–21`): accept iff **both** proxy runs improve the objective (`cost_per_successful_task` strictly below current-best mean) **and** clear every guardrail (success floor = `0.95 × best pass_rate mean`; cost / invalid-action ceilings when set). API: `load_guardrails()` → `Guardrails` (reads the `allowed_edits.yaml` block), `check_run(run, best, guardrails)` → `RunCheck`, `evaluate_candidate(runs, best, guardrails)` → `AcceptanceDecision` (needs ≥2 runs). Objective is cost-per-successful-task, which also rewards success gains (total_cost/num_passed drops as passes rise). 8 unit tests in `tests/test_acceptance.py`, all green.
- [ ] `iterator_agent/run_iteration.py` — **(orchestrator)** one full iteration loop: accept the ticket → load best (`baseline`) → build `feedback` → editor proposes → `edit_guard` → apply → eval-runner proxy run (seed A) → if improved, proxy run (seed B) → `acceptance` rule → on accept **bump `config.HARNESS_VERSION`** (via env override the orchestrator sets, so each run's `results.csv` row attributes to its harness state) + commit + write note; on reject revert + log (`experiment.md §21` steps 1–14).

**Driver + budget:**

- [ ] `scripts/run_iterator.py` — run iterations under a **fixed, pre-declared optimization budget** (iteration count and/or cost cap, set before the run; `experiment.md §17.9`). Proxy split only.

**Runs & verification:**

- [ ] **Smoke the iterator loop (~$0)** — dry-run the full ticket→propose→guard→apply→proxy×2→accept/revert→note→commit chain on the mock/smoke split (or with a no-op proposed change) before spending proxy budget.
- [ ] **Run Arm B optimization on proxy** under the fixed budget — accepted changes accumulate into the Arm B harness (`HARNESS_VERSION` bumped per accept). Iterator sees proxy only.
- [ ] **Blind validation event — N=5, seeds `2001..2005`** — run the final Arm B harness once, post-hoc, against the 35-task validation split; compare to the Arm A validation distribution. If the proxy gain does not hold within guardrails, report Arm B as **proxy-overfit** (`experiment.md §20` tail).

**Plots:**

- [ ] **Per-iteration trajectory plot** (new mode in `scripts/plot_results.py`, reads `iteration_log`) — task success rate **and** cost per successful task vs iteration number (`experiment.md §25.1/§25.2`). This is what "cost + precision/recall over each iteration" maps to: per the confirmed metric choice, precision/recall = the two headline metrics (task success rate, cost per successful task). Accepted vs rejected iterations marked distinctly.
- [ ] **Arm A vs Arm B frontier** — with ≥2 arms, the existing `plot_results.py` frontier mode auto-activates (cost-vs-success scatter, arm means as ◆ with std error bars) — the first real Pareto comparison.
- [ ] _(milestone end, optional)_ **TAU2 official test split once** via `scripts/run_tau_test.py` — final Arm B reporting, run once, reported conservatively.

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
- Observability: **local files only — Langfuse was removed in M2** (local debug files are easier to debug, in-git, and need no external service). `results.csv` + per-task verdict JSON are canonical; per-turn detail (when needed) comes from persisting TAU2's `SimulationRun` transcript locally. The iterator is debugged from its per-iteration log folders, not a hosted UI.
- Log structure: `experiments/logs/<split>_<YYYYMMDD>_<HHMMSS>/task_*.json`
