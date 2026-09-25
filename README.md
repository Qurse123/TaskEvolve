# TaskEvolve

A cost-performance measurement study of LLM agents on τ²-bench. The benchmark stays frozen
and one thing varies at a time, the model or the harness or a fine-tuned adapter, so the
study can answer a single question:

> What does a completed customer service task cost, and which lever buys the most task
> success per dollar: a stronger closed model, an autonomously optimized harness, or a
> specialized open-weight model?

Every reported number is a distribution over five fixed seeds rather than one run, because
the benchmark is stochastic: an LLM plays the customer. The headline metric is **cost per
successful task**, which is agent inference cost divided by the number of tasks passed.

Paper: [TaskEvolve.pdf](https://github.com/user-attachments/files/32667105/TaskEvolve.pdf)

---

## Headline result

Four systems, each run five times over the full official held-out test split of three
domains, seeds 4001 to 4005. Task success is the mean pass rate. Cost is agent inference
only, priced at published per-token rates.

### Retail, 40 tasks

| System | Task success | Cost per successful task |
|---|---|---|
| Opus 4.8, static harness | 0.875 ± 0.031 | $0.5109 ± $0.0115 |
| Opus 4.8 plus agent iterator | 0.850 ± 0.025 | $0.4537 ± $0.0261 |
| Inkling-Small base | **0.915 ± 0.038** | $0.0416 ± $0.0023 |
| Inkling-Small fine-tuned | 0.875 ± 0.040 | **$0.0329 ± $0.0021** |

### Airline, 20 tasks

| System | Task success | Cost per successful task |
|---|---|---|
| Opus 4.8, static harness | 0.690 ± 0.074 | $0.6741 ± $0.0178 |
| Opus 4.8 plus agent iterator | 0.600 ± 0.050 | $0.5411 ± $0.0419 |
| Inkling-Small base | 0.700 ± 0.100 | $0.0763 ± $0.0082 |
| Inkling-Small fine-tuned | **0.750 ± 0.061** | **$0.0414 ± $0.0024** |

### Telecom, 40 tasks

| System | Task success | Cost per successful task |
|---|---|---|
| Opus 4.8, static harness | 0.545 ± 0.037 | $1.4051 ± $0.0943 |
| Opus 4.8 plus agent iterator | 0.330 ± 0.021 | $1.0711 ± $0.1118 |
| Inkling-Small base | **0.675 ± 0.043** | $0.1774 ± $0.0216 |
| Inkling-Small fine-tuned | 0.660 ± 0.045 | **$0.1028 ± $0.0113** |

What the three tables say:

- Base Inkling-Small matches or exceeds Opus 4.8 on task success in all three domains at
  7.9 to 12.3 times lower cost per successful task.
- Fine-tuning on successful Opus trajectories lowers cost further, reaching 13.7 to 16.3
  times below Opus, while task success moves a few points in both directions.
- Harness optimization lowers Opus's cost per successful task by 11.2%, 19.7% and 23.8%
  across retail, airline and telecom, and gives up 2.5, 9.0 and 21.5 percentage points of
  task success. It stays dominated by both open-weight systems on cost and on task success.

One further system ran on the held-out retail split alone. Full Inkling, the larger sibling
of Inkling-Small, reached 0.895 ± 0.069 task success at $0.0110 ± $0.0007 per successful
task. Retail-only coverage keeps it out of the three-domain comparison.

### Development-split reference

Earlier arms were measured on a 35-task retail split carved out of τ²-bench train, five
seeds, seeds 2001 to 2005. These are development numbers and are superseded as the headline
by the held-out tables above.

| System | Task success | Cost per successful task |
|---|---|---|
| Opus 4.8, static harness | 0.903 ± 0.048 | $0.5785 ± $0.0407 |
| Full Inkling | 0.874 ± 0.056 | $0.0219 ± $0.0077 |
| gpt-4.1, static harness | 0.806 ± 0.047 | $0.0619 ± $0.0037 |

---

## How a system is identified in the ledger

`experiments/results.csv` is the canonical ledger, 352 rows, one row per run. A system is
the triple of split, agent model and harness version. The mapping is not self-evident, so
it is written out here.

| System in the tables above | `split` | `agent_model` | `harness_version` |
|---|---|---|---|
| Opus 4.8, static harness | `test_*` | `anthropic/claude-opus-4-8` | `v0.1` |
| Opus 4.8 plus agent iterator | `test_*` | `anthropic/claude-opus-4-8` | `v0.5` |
| Inkling-Small base | `test_*` | `openai/thinkingmachines/Inkling-Small` | `v0.1` |
| Inkling-Small fine-tuned | `test_*` | `openai/armd-inkling-small-tuned` | `v0.1` |
| Full Inkling, retail only | `test_retail` | `openai/thinkingmachines/Inkling` | `v0.1` |

Three groups sit in the ledger and belong to no table above. Opus at `v0.4` is an earlier
iterator generation that routed every call to Haiku 4.5. `openai/armd-inkling-small-retail-tuned`
is a retail-only adapter ablation on `test_airline`, and three of its five runs passed zero
tasks and recorded no cost, so the figure scripts gate them out. Rows on `search_*`,
`eval_*`, `proxy`, `validation`, `train_distill_*` and `transfer_banking` are development
splits.

Harness versions are labels on rows rather than code. `v0.1` is the frozen static harness,
which is the state of `target_agent/` on this branch. `v0.5` is that harness plus the three
accepted edits recorded in `experiments/accepted_changes.md` as `iter_0014`, `iter_0020` and
`iter_0021`, whose net effect is to send the domain policy on the first turn alone. The
`v0.2` through `v0.4` harness source was not preserved, so those rows cannot be re-run
byte-for-byte from this tree. See "Known gaps".

---

## What a clean clone gives you

| Tracked, so free | Not tracked, so it costs money or disk |
|---|---|
| `experiments/results.csv`, the 352-row ledger | `experiments/logs/`, roughly 856 MB of per-task transcripts |
| `experiments/analysis/test_task_stats.csv`, 2000 per-task rows for the held-out splits | `experiments/plots/`, every rendered figure |
| `benchmark/splits/*.json`, all frozen task ids | `experiments/iterations/`, per-iteration search records |
| the three pre-registration records under `experiments/` | the LoRA adapter, which is scoped to the training account |
| every script that builds a figure, table or statistic | |

So **every figure, table and statistic in the study regenerates from a clean clone at zero
cost**. Re-running the measurements themselves is what costs money. The full four-system
held-out sweep, twelve system-and-domain cells at five seeds each, measured **$527.36 in
agent inference and 36.7 hours of sequential wall clock**.

---

## Reproducing the study

### 0. Install

```bash
git clone <this-repo> TaskEvolve && cd TaskEvolve

# τ²-bench is a plain clone, not a submodule, so pin it yourself.
git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
git -C vendor/tau2-bench checkout 1746a25db265724f6fed2260934bb67f1514fad7

# Sync at the repo root. pyproject declares tau2[knowledge,gym] as an editable
# path dependency, so this installs the vendored benchmark and TaskEvolve together.
uv sync --extra dev
```

Python 3.12 is required. τ²-bench itself pins `>=3.12,<3.14`.

Fine-tuning and open-weight serving additionally need the Tinker SDK, `tinker` and
`tinker_cookbook`, which are absent from `pyproject.toml` and the lockfile and must be
installed separately from Thinking Machines. Everything except the fine-tuning arm runs
without them.

### 1. Credentials

```bash
cp .env.example .env
```

`.env.example` documents every variable and which arm needs it. The short version:

| Arm | Variables |
|---|---|
| Any arm | `OPENAI_API_KEY`, because the user simulator is always OpenAI |
| gpt-4.1 baseline | `AGENT_MODEL=gpt-4.1` |
| Opus 4.8, and the iterator | `ANTHROPIC_API_KEY`, `AGENT_MODEL=anthropic/claude-opus-4-8`, `AGENT_NO_TEMPERATURE=1` |
| Full Inkling | `TOGETHER_API_KEY`, `AGENT_API_BASE=https://api.together.xyz/v1` |
| Inkling-Small, base or tuned | `TINKER_KEY`, `AGENT_API_BASE=http://localhost:8100/v1` |
| The iterator's editor | `ITERATOR_MODEL`, `ITERATOR_NO_TEMPERATURE=1` |

Two wiring facts that cost real runs to learn. Opus 4.8 rejects the `temperature` parameter
and litellm will not strip it, so `AGENT_NO_TEMPERATURE=1` is mandatory for every Opus run.
Open-weight models must go through the `openai/` prefix plus `AGENT_API_BASE`, because
litellm's `together_ai` provider silently drops tool schemas, which leaves the agent with no
tools and fails every task at full cost.

The baselines need an OpenAI Tier 2 account, roughly 450k tokens per minute. Tier 1 at 30k
throttles mid-task.

### 2. Splits

All task ids are frozen and committed under `benchmark/splits/`. Nothing needs regenerating
to reproduce the tables. The generators are kept for provenance and are deterministic.

| Command | Produces | Seed |
|---|---|---|
| `python -m benchmark.splits` | `smoke` 3, `proxy` 12, `validation` 35 | 42 |
| `python -m benchmark.splits_arm_d` | `train_distill_*`, `eval_*`, `transfer_banking` | 42 |
| `python -m benchmark.splits_search` | `search_retail` 20, `search_airline` 10, `search_telecom` 20 | 20260905 |
| `python -c "from benchmark.splits_arm_d import generate_test_splits; generate_test_splits()"` | `test_retail` 40, `test_airline` 20, `test_telecom` 40 | read from τ²-bench |

The held-out test splits come from inside τ²-bench through
`get_tasks(task_split_name="test")`, so there is nothing to download. The generator asserts
the counts 40, 20 and 40, which is the guard against vendor drift.

Verify the splits never overlap the fine-tuning set:

```bash
python -m scripts.check_split_disjointness
```

### 3. Check the wiring before spending anything

```bash
python -m scripts.smoke_iterator   # truly $0, every external call faked
python -m scripts.run_smoke        # 3 mock tasks, about $0.01, expect "3/3 passed"
```

`run_smoke` returns 0 regardless of pass count, so read the `3/3 passed` line rather than
the exit code.

### 4. Run the arms

Held-out test splits, four systems, five seeds each. This is the block that produces the
headline tables.

```bash
# System 1, Opus 4.8 in the static harness
for S in "test_retail retail" "test_airline airline" "test_telecom telecom"; do set -- $S
  AGENT_MODEL=anthropic/claude-opus-4-8 AGENT_NO_TEMPERATURE=1 HARNESS_VERSION=v0.1 \
    python -m scripts.run_train_eval --split $1 --domain $2 --repeats 5 --seed-start 4001
done

# System 2, Opus 4.8 in the iterator-optimized harness.
# Apply the three accepted edits from experiments/accepted_changes.md first.
# HARNESS_VERSION only labels the rows, it does not change the code.
for S in "test_retail retail" "test_airline airline" "test_telecom telecom"; do set -- $S
  AGENT_MODEL=anthropic/claude-opus-4-8 AGENT_NO_TEMPERATURE=1 HARNESS_VERSION=v0.5 \
    python -m scripts.run_train_eval --split $1 --domain $2 --repeats 5 --seed-start 4001
done

# Systems 3 and 4, base and fine-tuned Inkling-Small, served from one shim process
# so any serving artifact appears on both sides of the comparison.
python -m arm_d.serving_shim --port 8100 --base-model thinkingmachines/Inkling-Small &
for M in thinkingmachines/Inkling-Small armd-inkling-small-tuned; do
  for S in "test_retail retail" "test_airline airline" "test_telecom telecom"; do set -- $S
    AGENT_MODEL=openai/$M AGENT_API_BASE=http://localhost:8100/v1 HARNESS_VERSION=v0.1 \
      python -m scripts.run_train_eval --split $1 --domain $2 --repeats 5 --seed-start 4001
  done
done

# Retail-only reference, full Inkling through Together
AGENT_MODEL=openai/thinkingmachines/Inkling AGENT_API_BASE=https://api.together.xyz/v1 \
HARNESS_VERSION=v0.1 \
  python -m scripts.run_train_eval --split test_retail --domain retail --repeats 5 --seed-start 4001
```

The development-split arms, for the reference table:

```bash
# gpt-4.1 baseline, proxy then validation
AGENT_MODEL=gpt-4.1 HARNESS_VERSION=v0.1 \
  python -m scripts.run_train_eval --split proxy --repeats 5 --seed-start 1001
AGENT_MODEL=gpt-4.1 HARNESS_VERSION=v0.1 \
  python -m scripts.run_train_eval --split validation --repeats 5 --seed-start 2001

# Opus 4.8 reference
AGENT_MODEL=anthropic/claude-opus-4-8 AGENT_NO_TEMPERATURE=1 HARNESS_VERSION=v0.1 \
  python -m scripts.run_train_eval --split validation --repeats 5 --seed-start 2001
```

The autonomous harness search. It reads search-split evidence only, never the held-out
splits, and it reverts every candidate, so the winner is chosen afterwards from the ranking.
Run the `v0.1` baseline on the three search splits first, since the campaign builds its
starting distribution from those rows.

```bash
AGENT_MODEL=anthropic/claude-opus-4-8 AGENT_NO_TEMPERATURE=1 \
ITERATOR_MODEL=anthropic/claude-opus-4-8 ITERATOR_NO_TEMPERATURE=1 \
  python -m scripts.run_armb_campaign --max-minutes 1440 --seed-start 7101 \
  --max-iterations 500 --agent-model anthropic/claude-opus-4-8
python -m scripts.rank_candidates
```

The reported search ran for 24 hours and produced 23 iterations with 3 accepted.

Fine-tuning, end to end:

```bash
# 1. Opus teacher trajectories on the distillation splits. Paid.
AGENT_MODEL=anthropic/claude-opus-4-8 AGENT_NO_TEMPERATURE=1 HARNESS_VERSION=v0.1 \
  nohup python -m scripts.run_distillation > distill.log 2>&1 &

# 2. Select passing trajectories into an SFT set. $0, a pure transform.
python -m scripts.build_distill_dataset

# 3. LoRA SFT on Inkling-Small through Tinker. Paid.
#    Pass --base-model explicitly: TrainConfig defaults to full Inkling, and the
#    study trained Inkling-Small.
nohup python -m scripts.train_arm_d --base-model thinkingmachines/Inkling-Small \
  --lora-rank 32 --lr 1e-4 --max-epochs 4 --patience 2 > train_arm_d.log 2>&1 &

# 4. Serve the adapter your own run produced. The tinker:// path is account-scoped,
#    so yours differs from the one hardcoded as the shim default.
python -m arm_d.serving_shim --port 8100 \
  --tuned-path 'tinker://<your-run>/sampler_weights/<name>' \
  --base-model thinkingmachines/Inkling-Small

# 5. Development-split eval, tuned against the matched base control. Paid.
ARM_D_TUNED_MODEL_ID=armd-inkling-small-tuned \
  nohup python -m scripts.run_arm_d_eval --model-id armd-inkling-small-tuned \
  --base-model-id thinkingmachines/Inkling-Small --seed-start 2001 --repeats 3 \
  > arm_d_eval.log 2>&1 &
```

Reported training run: LoRA rank 32, learning rate 1e-4, batch 8, 4 epochs, patience 2,
finishing in 88 optimizer steps over 169 training trajectories with 26 further trajectories
held out to track validation loss, for 195 selected in total. The record is at
`experiments/arm_d_training/run_20260807_222817/training_record.json`.

Seed blocks, since `results.csv` carries no seed column: proxy 1001 to 1005, validation 2001
to 2005, fine-tuning development eval 2001 to 2003, held-out test 4001 to 4005, distillation
5001 upward, iterator campaign 7101 upward in disjoint blocks of four.

### 5. Regenerate the figures, tables and statistics

All of these read the tracked ledgers and cost nothing.

```bash
# Per-task cache for figures 2 through 4. Needs experiments/logs/, so run it only
# if you re-ran the arms; the resulting CSV is already tracked.
python -m scripts.build_test_task_stats

# The four paper figures, into experiments/plots/paper/
python -m scripts.plot_paper_fig1_frontier          # cost against task success, every seed
python -m scripts.plot_paper_fig2_tokens_vs_cost    # input read per task against dollars paid
python -m scripts.plot_paper_fig3_telecom_handoff   # telecom handoffs and how those runs end
python -m scripts.plot_paper_fig4_conversation_length

# Statistics and validity analyses, into docs/paper/analysis/
python -m scripts.paper_stats        # bootstrap CIs, permutation tests, Holm correction
python -m scripts.cost_fairness      # the same comparison at a common per-token schedule
python -m scripts.amortized_cost     # when fine-tuning pays back its training cost
python -m scripts.latency_analysis
python -m scripts.error_taxonomy
python -m scripts.provenance

# Exploratory views
python -m scripts.plot_results [--frontier | --per-metric | --trajectory] [--split SPLIT]
python -m scripts.plot_arm_d
python -m scripts.plot_metric_views
```

Recompute the headline tables straight from the ledger:

```bash
python - <<'PY'
import csv, statistics as st
rows = list(csv.DictReader(open('experiments/results.csv')))
SYS = [('anthropic/claude-opus-4-8', 'v0.1'), ('anthropic/claude-opus-4-8', 'v0.5'),
       ('openai/thinkingmachines/Inkling-Small', 'v0.1'),
       ('openai/armd-inkling-small-tuned', 'v0.1')]
for split in ('test_retail', 'test_airline', 'test_telecom'):
    for model, harness in SYS:
        rs = [r for r in rows if r['split'] == split
              and r['agent_model'] == model and r['harness_version'] == harness]
        p = [float(r['pass_rate']) for r in rs]
        c = [float(r['cost_per_successful_task']) for r in rs if r['cost_per_successful_task']]
        print(f"{split:13} {model:40} {harness}  n={len(rs)}  "
              f"success={st.mean(p):.3f}±{st.stdev(p):.3f}  "
              f"cost/success=${st.mean(c):.4f}±${st.stdev(c):.4f}")
PY
```

### 6. Tests

```bash
uv run pytest -q     # 291 tests, all green
```

---

## Guardrails that make the numbers mean something

1. **τ²-bench's orchestrator is never modified.** Changing it invalidates every comparison.
2. **Frozen surfaces for humans and the iterator alike:** all of `vendor/tau2-bench/` and
   every file in `benchmark/splits/`.
3. **The iterator sees search-split evidence only** during optimization. Validation and test
   logs stay hidden from it.
4. **Validation and the held-out test splits run once each, blind, after optimization
   finishes**, against criteria written to a file beforehand. The three pre-registration
   records are `experiments/validation_event_v0.2_precommit.md`,
   `experiments/arm_d_eval_precommit.md` and `experiments/test_split_precommit.md`.
5. **No re-runs, no added seeds, no harness edits after a test result is seen.** A crashed
   run may be relaunched only when it scored zero tasks.
6. **The fine-tuned adapter and its base control share one shim process**, one renderer and
   the static `v0.1` harness, so the only difference between them is the adapter.
7. **Every pre-registered arm is reported** whether it confirms or refutes the hypothesis.
   This is how the study caught its own proxy-overfit result rather than publishing it as a
   win, and how it reports that the telecom fine-tuning gain seen on the development split
   did not survive the held-out split.

A permanently failing task is recorded as a failure rather than dropped, so do not rescue a
run by deleting rows.

---

## Repository layout

| Path | Contents |
|---|---|
| `target_agent/` | The agent under test: `agent.py`, plus the five files the iterator may edit |
| `benchmark/` | Split generators, the frozen split ids, and the adapter that injects the agent into τ²-bench |
| `iterator_agent/` | The autonomous harness optimizer: editor, deterministic acceptance rule, edit guard, preflight |
| `arm_d/` | Fine-tuning: dataset build, LoRA training, and the local serving shim |
| `settings/` | Configuration and per-token pricing registration |
| `results/` | Run logging into per-run JSON plus the CSV ledger |
| `scripts/` | Runners, the iterator driver, figure scripts and validity analyses |
| `experiments/` | The tracked ledgers and pre-registration records. Logs, plots and iteration records stay local |
| `docs/paper/` | Analysis notes and the deeper per-arm reproducibility record |
| `tests/` | 291 unit tests |

Observability is local files only. Per-run verdict JSON and the CSV ledger are canonical,
and per-turn detail comes from τ²-bench transcripts persisted next to them.

---

## Known gaps

These are stated so a replicator is not surprised.

- **The `v0.2` through `v0.4` harness source is gone.** Those labels appear in the ledger,
  but the code that produced them was reverted and never committed, so those rows cannot be
  re-run from this tree. `v0.1` and `v0.5` both reconstruct: `v0.1` is the current
  `target_agent/`, and `v0.5` is that plus the three accepted diffs under
  `experiments/iterations/`, which are local rather than tracked.
- **The LoRA adapter is account-scoped.** The `tinker://` checkpoint belongs to the training
  account, so a third party must re-run the SFT and will get a different adapter.
- **Closed model snapshots are not pinned.** `claude-opus-4-8` and `gpt-4.1` can drift under
  the same name, so exact re-measurement is not guaranteed.
- **Cost excludes the user simulator.** `cost_per_successful_task` counts agent inference
  only. Real out-of-pocket spend is higher, and the simulator's share is tracked separately.
- **Runs are not deterministic** even at temperature 0, which is why every result is
  reported as a distribution over five seeds.
- **`results.csv` carries no seed column**, so cross-arm comparisons are unpaired. Seeds are
  recoverable from the per-task JSON under `experiments/logs/`.
- **Three of the five `armd-inkling-small-retail-tuned` runs on `test_airline` recorded zero
  passes and no cost.** The figure scripts gate them out, and that ablation appears in no
  table above.
