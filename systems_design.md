# TaskEvolve: System Design

## 1. What We Are Building

TaskEvolve is a controlled measurement study of where performance improvements come from in task-oriented LLM agents. We evaluate three levers — harness design, open-weight model specialization, and joint co-optimization of both — against a frozen external benchmark (TAU2-bench). The goal is to map the cost-performance frontier: which system delivers the highest task success per dollar spent?

This document describes the architecture for **Milestone 1**: the baseline evaluation system. It covers the target agent and harness, benchmark integration, task splits, observability, and the build sequence. The iterator agent (Milestone 2) and model specialization pipeline (Milestone 3+) are previewed but not built here.

---

## 2. System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FROZEN (cannot touch)                    │
│  TAU2-bench: domain tools, user simulator, evaluator, DB    │
└─────────────────────┬───────────────────────────────────────┘
                      │ tools: list[Tool], domain_policy: str
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              TARGET AGENT  (editable)                       │
│  TaskEvolveAgent(HalfDuplexAgent)                           │
│  agent.py · harness.py · model_routing.py · prompts/*.j2    │
└─────────────────────┬───────────────────────────────────────┘
                      │ AssistantMessage (tool_call or text)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│             TAU2 ORCHESTRATOR  (frozen)                     │
│  Drives the full turn loop between agent and user sim       │
│  Executes tool calls against domain DB                      │
│  Returns ToolMessage with result JSON                       │
│  Enforces max_steps and max_errors                          │
└─────────────────────┬───────────────────────────────────────┘
                      │ reward: float 0.0–1.0
                      ▼
┌─────────────────────────────────────────────────────────────┐
│             OBSERVABILITY LAYER  (ours)                     │
│  Langfuse: per-LLM-call traces (cost, tokens, latency)     │
│  Structured JSON logs: per-task outcome + tool call trace  │
└─────────────────────┬───────────────────────────────────────┘
                      │ (Milestone 2+)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              ITERATOR AGENT  (future)                       │
│  Reads logs → proposes one harness change → reruns eval    │
│  Double-runs proxy to confirm improvement before committing │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Component Breakdown

### 3.1 TAU2-bench Integration Layer

TAU2-bench is the evaluation environment. We treat it as a frozen external dependency.

**Installation:**
```bash
git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
cd vendor/tau2-bench && uv sync --extra knowledge --extra gym --extra dev
```

We install the `knowledge`, `gym`, and `dev` extras. The `voice` extra is skipped — it needs the `portaudio` system library (via `pyaudio`) we do not require for half-duplex text runs.

**Entry points we use:** we do **not** call TAU2's `run_single_task()` / `run_domain()`, because those resolve the agent by name from TAU2's global registry and cannot accept an agent *instance*. Instead we construct the orchestrator ourselves and inject our agent (constructor injection), then hand it to TAU2's frozen evaluator:
- `tau2.runner.build.build_environment(domain)` → `Environment`
- `tau2.runner.build.build_user("user_simulator", env, task, …)` → user simulator instance
- `tau2.orchestrator.orchestrator.Orchestrator(domain, agent, user, environment, task, …)` — we pass our `TaskEvolveAgent` **instance** here
- `tau2.runner.simulation.run_simulation(orchestrator, evaluation_type=ALL)` → `SimulationRun` with `reward_info` attached (the frozen evaluator)
- `tau2.data_model.simulation.TextRunConfig` — reference only; we read its field defaults from `tau2.config`

**What TAU2-bench scores:**

TAU2's reward is a **float 0.0–1.0** (not binary). The evaluator runs multiple sub-checks:

| Check | What it tests |
|-------|--------------|
| `db_check` | Final DB state matches expected state (e.g., order actually cancelled) |
| `action_checks` | Required tools called in required sequence |
| `env_assertions` | Domain-level assertions (e.g., refund amount correct) |
| `nl_assertions` | Natural language policy compliance (LLM judge) |

The `reward_basis` field in each task's schema determines how sub-checks combine. Many tasks gate on `db_check`: if it fails, reward = 0.0 regardless. Tasks with partial credit can score 0.5 if some actions were correct. For reporting, we bucket as pass (reward ≥ 0.5) / fail (reward < 0.5) and also record the raw float.

**Frozen surfaces (iterator may never touch):**
```
vendor/tau2-bench/src/tau2/domains/retail/tools.py
vendor/tau2-bench/src/tau2/domains/retail/environment.py
vendor/tau2-bench/src/tau2/user/user_simulator.py
vendor/tau2-bench/src/tau2/evaluator/
vendor/tau2-bench/src/tau2/metrics/
vendor/tau2-bench/data/tau2/domains/retail/
```

### 3.2 Target Agent + Harness

The target agent is a `HalfDuplexAgent` subclass. TAU2's `Orchestrator` drives the full turn loop — we do not build our own. Our entire "harness" is what happens inside a single call to `generate_next_message()`.

**The optimization surface is narrow by design:**
1. **System prompt** — biggest lever, rendered from `target_agent/prompts/system_prompt.j2`
2. **Message history management** — which turns to include, how aggressively to compress
3. **Tool list passed to the LLM** — full list or filtered subset with rewritten descriptions
4. **Model selection** — `gpt-4.1` vs cheaper `gpt-4.1-mini` for cost optimization

**Class:** `target_agent/agent.py` → `TaskEvolveAgent`

**Constructor contract (required by TAU2-bench):**
```python
def __init__(self, tools: list[Tool], domain_policy: str, llm: str, llm_args: dict): ...
```

**State:** `TaskEvolveAgentState(BaseModel)` — holds rendered system message and message history.

**Single-turn contract:**

TAU2's `Orchestrator` calls `generate_next_message(message, state)` once per turn and handles everything else:

```
Orchestrator calls generate_next_message(message, state)
  ├── harness.py: build message list (system + history + current message)
  ├── harness.py: optionally compress history if it exceeds token threshold
  ├── harness.py: optionally filter/rewrite tool descriptions
  └── call generate(model, tools, messages) — TAU2's LiteLLM wrapper
        └── returns AssistantMessage (tool_call OR text — never both)
```

If the agent returns a tool call, `Orchestrator` executes it against the frozen domain DB and calls `generate_next_message()` again with the `ToolMessage` result. If the agent returns text, the turn ends and the user simulator takes over. **We do not build or modify the Orchestrator loop.**

**Prompt templates (Jinja2):**

All prompt files use `.j2` extension and are rendered at agent init time using `jinja2.Environment`. This lets the iterator modify prompt structure programmatically.

| Template | Purpose |
|----------|---------|
| `target_agent/prompts/system_prompt.j2` | Agent persona, instructions, format rules |
| `target_agent/prompts/policy_summary.j2` | Compressed/rewritten version of domain policy |
| `target_agent/prompts/few_shot_examples.j2` | Example task completions (empty in M1) |

**Few-shot examples** are 1–3 abbreviated example conversations injected into context showing the LLM what good behavior looks like:

```
## Examples

### Example 1
User: I need to return my order
Agent: [calls get_user_details(user_id="...")]
Tool: {"orders": [...]}
Agent: [calls return_delivered_item_order(order_id="...")]
Tool: {"status": "return initiated"}
Agent: Your return has been processed...
```

Few-shot is NOT required for Milestone 1. `few_shot_examples.j2` starts empty. The iterator may populate it as an optimization.

**Editable surfaces (what the iterator can modify):**

| File | What it controls |
|------|-----------------|
| `target_agent/prompts/system_prompt.j2` | Agent persona, instructions, format rules |
| `target_agent/prompts/policy_summary.j2` | Compressed/rewritten version of domain policy |
| `target_agent/prompts/few_shot_examples.j2` | Example task completions injected into context |
| `target_agent/harness.py` | Context compression, tool filtering logic |
| `target_agent/model_routing.py` | Which model to use (cheap vs strong, per situation) |

**Model:** OpenAI `gpt-4.1` via LiteLLM for Milestone 1.

**Constraint:** Each agent response must be either a tool call OR text — never both. TAU2-bench's validator enforces this. The `generate()` utility handles it correctly.

**Do NOT modify the TAU Orchestrator.** Doing so invalidates the benchmark comparison — other published results use the standard Orchestrator.

### 3.3 Task Splits

TAU2-bench retail has ~115 tasks in its train split. We carve these into three operational subsets. The "hidden test" is TAU2's own official test split, not a holdout we carve ourselves.

| Split | Size | Purpose | Approx. cost per run |
|-------|------|---------|---------------------|
| **Smoke** | 3 tasks (mock domain) | Verify harness wires up at all | ~$0 |
| **Proxy** | 12 tasks (train subset) | Fast iterator feedback after every change | ~$3 |
| **Validation** | 35 tasks (train subset) | Post-hoc overfitting check after optimization completes | ~$10 |
| **TAU2 test split** | ~40 tasks (official) | Final reporting only — run once at milestone end | ~$12 |

The smoke test uses the **mock domain** where the user simulator is a simple rule-based system, not an LLM call. Three tasks complete at near-zero API cost and verify only that our harness wiring is correct.

Proxy and validation task IDs are generated once by `benchmark/splits.py` with a fixed random seed and written to `benchmark/splits/*.json`.

**Access control:**
- **Iterator sees only proxy results** during optimization. It may never read validation or TAU2 test split results.
- **Validation** runs exactly once after all optimization iterations are complete, as a post-hoc overfitting check.
- **TAU2 test split** runs exactly once at milestone end for final reporting via `scripts/run_tau_test.py`.

**Transfer domain:** airline (held for Milestone 4+). After optimizing on retail, we measure cost-performance on airline and telecom with the same harness, un-modified — this directly tests whether improvements are domain-general or domain-specific.

### 3.4 Observability Layer

Two complementary layers serve different audiences.

**Langfuse — for human debugging:**

TAU2-bench natively supports Langfuse via `USE_LANGFUSE=True` in the `.env` file. Once set, every `generate()` call inside our agent emits a trace to the Langfuse dashboard showing the full prompt, model response, token counts, dollar cost, and latency. This is the right tool for answering "why did the agent call the wrong tool on turn 6?"

See Section 7.1 (Known Tradeoffs) for why OpenTelemetry is not used instead.

**Structured JSON task log — for the iterator agent:**

`observability/logger.py` writes one JSON file per task run, organized into per-run folders:

```
experiments/
  logs/
    proxy_20260529_143022/       ← run_id = <split>_<YYYYMMDD>_<HHMMSS>
      task_retail_001.json
      task_retail_002.json
      ...
      run_summary.json
    proxy_20260529_162511/       ← second run (different seed, for double-run validation)
      ...
  results.csv                    ← one row per run (aggregate metrics)
```

Each task JSON:

```json
{
  "task_id": "retail_task_42",
  "domain": "retail",
  "split": "proxy",
  "harness_version": "v0.1",
  "agent_model": "gpt-4.1",
  "reward": 0.0,
  "passed": false,
  "cost_usd": 0.14,
  "termination_reason": "max_steps",
  "seed": 300,
  "timestamp": "2026-05-29T10:32:00+00:00",
  "run_id": "proxy_20260529_143022"
}
```

The per-task JSON is **verdict-only** — reward, pass/fail, cost, and identity. Per-LLM-call telemetry (prompts, responses, token counts, latency, tool/retrieval steps) is **not** duplicated here; it lives in Langfuse and is correlated by `task_id` + `run_id`. This keeps our logs minimal and avoids re-deriving data the tracer already owns.

`run_summary.json` contains aggregate metrics for the run. `results.csv` appends one summary row per run. The iterator reads the most recent run folder for a given split (plus Langfuse for call-level detail) to generate its feedback.

### 3.5 Iterator Agent (Milestone 2 Preview)

Not built in Milestone 1. Architecture described here for planning continuity.

The iterator is a separate Python process that:
1. Reads the most recent proxy run folder in `experiments/logs/`
2. Identifies failure patterns across tasks
3. Proposes exactly one change to one file in the allowed edit surface
4. Validates against `allowed_edits.yaml` — rejects if the change touches frozen files
5. Runs proxy eval (seed A) and checks if aggregate reward improved over current best
6. If improved: runs proxy eval again (seed B) to confirm the gain is not noise
7. If both runs show improvement: commits the change
8. If either run does not show improvement (or drops success below the cost-floor threshold): reverts
9. Repeats until the optimization budget is exhausted

**After all iterations:** run validation benchmark once to check for overfitting. If validation confirms gains held, run TAU2 official test split once for final reporting.

**The iterator never sees validation or test split results during optimization.** This is enforced structurally: the proxy runner and the validation/test runners are separate scripts with separate result directories, and `allowed_edits.yaml` cannot point to either.

---

## 4. Data Flow

```
TAU2 Orchestrator calls generate_next_message(message, state)
            |
            v
TaskEvolveAgent.generate_next_message(message, state)
  ├── harness.py: build messages = [system_prompt] + compressed_history + [message]
  ├── harness.py: optionally filter or rewrite tool descriptions
  └── generate(model="gpt-4.1", tools=tools, messages=messages)
        [LiteLLM -> OpenAI API]
        [Langfuse trace emitted automatically]
            |
            v AssistantMessage
       tool_call? ──yes──> return to TAU2 Orchestrator
            |                    |
            no            Orchestrator executes tool against domain DB
            |                    |
            v             Orchestrator calls generate_next_message again with ToolMessage
      text response
            |
            v
TAU2 Evaluator: db_check + action_checks + assertions -> reward 0.0–1.0
            |
            v
observability/logger.py
  ├── experiments/logs/<run_id>/task_<id>.json   (full task record)
  ├── experiments/logs/<run_id>/run_summary.json  (updated after each task)
  └── experiments/results.csv                     (append summary row after run)  
```

---

## 5. Repository Structure

```
TaskEvolve/
├── systems_design.md          # this document
├── experiment.md              # research specification
├── pyproject.toml             # project dependencies
├── .env.example               # required environment variables (incl. AGENT_MODEL)
│
├── settings/                  # project-wide config (flat immutable constants)
│   ├── __init__.py
│   └── config.py              # AGENT_MODEL (from .env), PASS_THRESHOLD, DEFAULT_DOMAIN, MAX_STEPS…
│
├── target_agent/              # EDITABLE surface (iterator can modify these)
│   ├── __init__.py
│   ├── agent.py               # TaskEvolveAgent(HalfDuplexAgent)
│   ├── harness.py             # context compression, tool filtering helpers
│   ├── model_routing.py       # model selection logic (stubbed in M1)
│   └── prompts/
│       ├── system_prompt.j2   # Jinja2 template — agent persona and instructions
│       ├── policy_summary.j2  # Jinja2 template — compressed domain policy
│       └── few_shot_examples.j2  # Jinja2 template — empty in M1
│
├── benchmark/
│   ├── __init__.py
│   ├── adapter.py             # run_eval() injects agent into TAU2 Orchestrator, runs run_simulation()
│   ├── splits.py              # generates and loads split JSON files
│   └── splits/
│       ├── smoke.json         # 3 mock task IDs
│       ├── proxy.json         # 12 retail task IDs (iterator training set)
│       └── validation.json    # 35 retail task IDs (post-hoc overfitting check)
│
├── observability/
│   ├── __init__.py
│   ├── logger.py              # writes per-run folder with JSON task logs + CSV row
│   └── langfuse_setup.py      # Langfuse client initialization
│
├── iterator_agent/            # MILESTONE 2 — not built yet
│   ├── researcher.py
│   ├── allowed_edits.yaml
│   └── run_iteration.py
│
├── experiments/
│   ├── logs/                  # one folder per run: <split>_<YYYYMMDD>_<HHMMSS>/
│   └── results.csv            # one row per run (aggregate metrics)
│
└── scripts/
    ├── run_smoke.py           # 3 mock tasks — verify wiring, zero LLM cost
    ├── run_train_eval.py      # run proxy or validation split, write results
    └── run_tau_test.py        # run TAU2 official test split — once at milestone end
```

---

## 6. Milestone 1 Build Sequence

1. Write `systems_design.md` ✓
2. `pyproject.toml` + `.env.example` — establish deps and required env vars
3. `benchmark/splits.py` — generate and write split JSON files from TAU2-bench task lists
4. `target_agent/agent.py` — minimal `TaskEvolveAgent` using TAU2's `generate()` utility
5. `target_agent/prompts/system_prompt.j2` — initial agent system prompt (Jinja2)
6. `benchmark/adapter.py` — `run_eval()` injects `TaskEvolveAgent` into TAU2's `Orchestrator`, runs `run_simulation()` (frozen evaluator), returns a verdict-only `EvalResult`
7. `observability/langfuse_setup.py` — wire Langfuse before first run
8. `observability/logger.py` — per-run folder JSON task log writer
9. `scripts/run_smoke.py` — 3 mock tasks, end-to-end wiring check
10. **Run smoke test** — confirm logs and Langfuse traces appear before spending real budget
11. `scripts/run_train_eval.py` — proxy and validation runner
12. **Run proxy baseline** (~$3, 12 tasks) — first real score (Arm A proxy baseline)
13. **Run validation baseline** (~$10, 35 tasks) — official Arm A baseline

Install TAU2-bench:
```bash
git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
cd vendor/tau2-bench && uv sync --extra knowledge --extra gym --extra dev
```

---

## 7. Known Tradeoffs

### 7.1 Why Not OpenTelemetry

OpenTelemetry is a distributed tracing standard for microservices. It requires a collector process, an OTLP exporter, and span context propagation across service boundaries.

Our system is a single Python process. The iterator agent's feedback loop needs two things at different grains: the per-task **verdict** (reward, pass/fail, cost) and per-call **telemetry** (which tool was called, which turn failed, the prompt/response). We split these by owner: the verdict goes to a flat per-task JSON file we write; the telemetry is owned by Langfuse, which TAU2's `run_simulation()` populates automatically. We do not copy Langfuse-owned data into our JSON.

For LLM-call-level visibility (prompts, tokens, cost), **Langfuse** is purpose-built for this and TAU2-bench has native support via `USE_LANGFUSE=True`. It requires no infrastructure beyond `pip install langfuse` and a Langfuse API key.

OpenTelemetry would add operational complexity — a collector, an exporter sink, cardinality config — for no signal the iterator or a human debugger cannot get from Langfuse + JSON task logs. We revisit OTel if we later build an HTTP API service wrapping the agent.

### 7.2 Why Not Run Full Benchmark Per Iteration

Running all 115 retail tasks costs ~$28 per run (agent + user simulator both call GPT-4o, average 12 turns per task). At 50 iterator iterations that is $1,400 in eval costs before counting the iterator agent's own model calls.

The proxy set (12 tasks, ~$3/run) gives sufficient signal to detect whether a system prompt change helps or hurts. Validation runs only post-hoc after all optimization is complete, not per-iteration. This reduces expected cost per iteration from $28 to $3–6 (proxy once or twice for the double-run check).

### 7.3 Why Three Evaluation Splits, Not Two

The "two test suites" framing conflates three distinct needs:

- **Smoke test** — Does the harness produce any output? Tests wiring, not task performance. Uses the mock domain where the user simulator costs nothing. Three tasks is enough.
- **Proxy** (seen by iterator) — Generates optimization signal. The iterator trains against this.
- **Validation** (never seen during optimization) — Runs once post-hoc to check for overfitting. If the iterator ever observes validation results during optimization, the generalization claims become invalid.

Conflating proxy and smoke wastes API budget on wiring checks that cost nothing on mock. Conflating validation and proxy invalidates the experiment by letting the iterator overfit to validation signal.

### 7.4 The Optimization Surface Is Narrower Than It Looks

TAU2-bench's `Orchestrator` handles the full turn loop: calling the agent, executing tools, enforcing step limits, passing results back. We cannot modify the Orchestrator without invalidating benchmark comparisons.

This means our entire harness lives inside one method call: `generate_next_message()`. The levers available are:
- The system prompt (Jinja2 template)
- Which messages from history to include (context management in `harness.py`)
- Which tools to pass to the LLM (tool filtering in `harness.py`)
- Which model to call (model routing in `model_routing.py`)

Everything else — retry logic, tool execution, turn sequencing — is owned by the Orchestrator. This constraint is a feature: it means our results are directly comparable to any other system evaluated on the same Orchestrator.
