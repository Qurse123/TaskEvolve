# Reproducibility Artifact & Checklist

*Tier 3 validity item #15 — a single place from which every reported number can be
reconstructed: environment pins, one-command-ish repro path per arm, the frozen
inputs, an ML-reproducibility Q&A, and the honest limits on exact reproduction.*

This is a **$0 assembly** from the real repo — `pyproject.toml`, `settings/config.py`,
`settings/pricing.py`, `.env.example`, `benchmark/splits/*.json`, `CLAUDE.md`, the
pre-registration docs under `experiments/*_precommit.md`, and the vendored TAU2 tree.
No runs were made to write it. Where a value cannot be reconstructed from the artifacts
it is flagged as a gap (§4/§5) rather than guessed. Provenance detail (per-arm model
strings, dates, seeds, pricing registration) lives in
`docs/paper/analysis/provenance.md`; the no-seed-column statistics consequence lives in
`docs/paper/analysis/statistics.md`. Both are cited rather than duplicated here.

---

## 1. Environment pins

### 1.1 Interpreter + Python dependencies

| Pin | Value | Source |
|---|---|---|
| Python | **>=3.12** (declared); **3.12.13** (this machine) | `pyproject.toml` `requires-python`; `python3 --version` |
| Build backend | `hatchling` | `pyproject.toml` `[build-system]` |
| `tau2[knowledge,gym]` | vendored, editable (`tool.uv.sources`) — see §1.2 | `pyproject.toml` dep + `[tool.uv.sources]` |
| `litellm` | `>=1.40` | `pyproject.toml` |
| `jinja2` | `>=3.1` | `pyproject.toml` |
| `python-dotenv` | `>=1.0` | `pyproject.toml` |
| `matplotlib` | `>=3.8` | `pyproject.toml` |
| `pyyaml` | `>=6.0` | `pyproject.toml` (pinned for the iterator's `allowed_edits.yaml` guard) |
| `numpy` | `>=1.26` | `pyproject.toml` |
| `pytest` | `>=8.0` (dev extra) | `pyproject.toml` `[project.optional-dependencies]` |

`numpy` is used by the statistics and figure scripts (`scripts/paper_stats.py`,
`scripts/plot_paper_fig2_tokens_vs_cost.py`, `scripts/plot_paper_fig3_telecom_handoff.py`)
and is now declared directly rather than relied on transitively. `pydantic` and
`pytest-asyncio` were declared with no importer anywhere in the repo and were dropped in
the 2026-09 cleanup. **`scipy` is deliberately NOT a dependency** — `paper_stats.py`
implements percentile-bootstrap CIs and permutation tests in pure numpy/stdlib
specifically to avoid it (see `docs/paper/analysis/statistics.md` header).

The versions above are **floor pins (`>=`), not exact lockfile pins.** A byte-exact
environment would require committing `uv.lock`; that is a known reproducibility gap
(§5). The dependency that most affects behavior is `litellm` (it owns provider routing,
`drop_params`, and the built-in price table).

### 1.2 One-time TAU2-bench install

TAU2-bench is vendored under `vendor/tau2-bench/` as a **plain clone (not a git
submodule** — there is no `.gitmodules`). Install once:

```bash
git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
cd vendor/tau2-bench && uv sync --extra knowledge --extra gym --extra dev
```

**Pinned commit actually used in this study:**
`1746a25db265724f6fed2260934bb67f1514fad7` (`2026-06-01`, "Update cascaded voice
display (#339)"). Because it is a plain clone, the commit is **not recorded in this
repo's git metadata** — after cloning, check it out explicitly to reproduce:

```bash
git -C vendor/tau2-bench checkout 1746a25db265724f6fed2260934bb67f1514fad7
```

`vendor/tau2-bench/` is a **frozen surface** — never modified by humans or the iterator
(CLAUDE.md Hard Constraint #1/#3). Modifying the TAU2 Orchestrator invalidates all
cross-arm comparisons.

### 1.3 Banking-domain (agentic-shell) OS prerequisites

The `banking_knowledge` domain (used by the `transfer_banking` zero-shot split in
Arm D) has an **agentic-search / shell** retrieval variant that reaches outside Python
to OS-level binaries the pip/uv install does **not** provide:

- **`ripgrep`** (`rg`) — the fast regex search backing the domain's `grep` retrieval
  tool. Install via `brew install ripgrep` (macOS) / `apt install ripgrep` (Debian).
- **`srt`** — a search/retrieval CLI installed via **npm** (`npm i -g srt`), needed for
  the shell-based knowledge-base search path.

**Honest caveat:** in the checked-out TAU2 commit the `banking_knowledge` retrieval
tools inspected (`src/tau2/domains/banking_knowledge/retrieval*.py`,
`db_query.py`) resolve `grep` in-process (Python regex) and do **not** visibly shell
out to `rg`/`srt`; these binaries are prerequisites of the domain's *agentic-shell*
retrieval variant rather than of the code path this study exercised. This repo does not
pin their versions (§5). The `transfer_banking` results were near-floor for both Arm D
systems regardless (pass ~0.10; `docs/paper/analysis/statistics.md`), and several early
banking rows are known-bad (provenance gap G7) — so banking is reported as a
transfer-stress probe, not a headline, and its environment is the least load-bearing.

### 1.4 API credentials (per arm)

Copy `.env.example` → `.env` and fill only what the target arm needs (`settings/config.py`
loads `.env` at import via `python-dotenv`):

| Env var | Needed by | Notes |
|---|---|---|
| `OPENAI_API_KEY` | **every arm** | agent (Arm A/B) *and* the TAU2 user-simulator (all arms). Requires **OpenAI Tier 2 / 450k TPM** — Tier 1 (30k) throttles mid-task (memory: TPM requirement). |
| `AGENT_MODEL` | every arm | the LLM under test; per-arm values in §2. |
| `TOGETHER_API_KEY` | Arm C, Arm D-via-Together (path A, unused) | serves Inkling over the OpenAI-compat endpoint. |
| `AGENT_API_BASE` | open-weight arms | `https://api.together.xyz/v1` (Arm C) or `http://localhost:8100/v1` (Arm D shim). |
| `AGENT_NO_TEMPERATURE=1` | Opus 4.8, Arm B | Opus rejects `temperature`; litellm's `drop_params` won't strip it. |
| `ITERATOR_MODEL` / `ITERATOR_NO_TEMPERATURE` | Arm B iterator only | editor model (autonomous run used an Opus editor). |
| `TINKER_KEY` | Arm D training + shim | bridged to `TINKER_API_KEY` at config import. |
| `ARM_D_TUNED_MODEL_ID` | Arm D (Together path A) | only if serving the LoRA via Together; unused under the shim path actually run. |

---

## 2. One-command-ish repro path per arm

All commands assume `.env` is set (§1.4) and are copied verbatim from `CLAUDE.md`,
`docs/paper/analysis/provenance.md`, and the pre-registration docs. The bare model
prefixes match `.env` at run time. **The iterator/optimization runs and every paid eval
require API budget** (see the ~cost column in the "Eval Splits" table, §3.3).

**Arm A — gpt-4.1, static v0.1 harness** (baseline; `OPENAI_API_KEY`):
```bash
AGENT_MODEL=gpt-4.1 python -m scripts.run_train_eval --split proxy      --repeats 5 --seed-start 1001
AGENT_MODEL=gpt-4.1 python -m scripts.run_train_eval --split validation --repeats 5 --seed-start 2001
```

**Arm C — naive open-weight Inkling via Together, v0.1** (`TOGETHER_API_KEY`):
```bash
AGENT_MODEL=openai/thinkingmachines/Inkling \
AGENT_API_BASE=https://api.together.xyz/v1 \
HARNESS_VERSION=v0.1 \
python -m scripts.run_train_eval --split {proxy|validation} --repeats 5 --seed-start {1001|2001}
```
The `openai/` + `AGENT_API_BASE` path is mandatory: litellm's `together_ai` provider
silently drops tool schemas, so tools must flow through the OpenAI-compat seam.

**Opus 4.8 — frontier closed-model reference, v0.1** (`OPENAI_API_KEY` for the user-sim +
Anthropic access; native litellm routing):
```bash
AGENT_MODEL=anthropic/claude-opus-4-8 HARNESS_VERSION=v0.1 AGENT_NO_TEMPERATURE=1 \
python -m scripts.run_train_eval --split {proxy|validation} --repeats 5 --seed-start {1001|2001}
```

**Arm B — iterator-optimized, produces v0.2** (accepted change = whole-agent
Opus→Sonnet routing; `OPENAI_API_KEY` + Anthropic + `ITERATOR_MODEL`):
```bash
# optimization — proxy only; the iterator NEVER sees validation/test (Hard Constraint #2):
python -m scripts.run_iterator --max-minutes <M> --seed-start 3001 --max-iterations <cap> [--max-search-cost-usd <cap>]
# blind validation event, run once post-hoc (v0.2 harness routes the Opus base id to Sonnet 5):
AGENT_MODEL=anthropic/claude-opus-4-8 HARNESS_VERSION=v0.2 AGENT_NO_TEMPERATURE=1 \
python -m scripts.run_train_eval --split validation --repeats 5 --seed-start 2001
```
Note: v0.2 rows record `agent_model=claude-opus-4-8` (the configured base) while
`target_agent/model_routing.py` in commit `cbb0e8c` routes calls to Sonnet 5 — the arm
is "Opus-base-routed-to-Sonnet" (provenance §1 note / gap G5). The checked-out
`model_routing.py` is the v0.1 identity router.

**Arm D — tuned + base control, via the local Tinker shim, v0.1** (`TINKER_KEY`;
`experiments/arm_d_eval_precommit.md`):
```bash
python -m arm_d.serving_shim            # start local OpenAI-compat shim on :8100 first
python -m scripts.run_arm_d_eval        # 4 splits × 3 seeds × 2 systems (retail/airline/telecom/banking)
```
Both the tuned adapter (`openai/armd-inkling-small-tuned`) and its base control
(`openai/thinkingmachines/Inkling-Small`) run through the **same shim process**, routed
by the request `model` field, so any serving-stack artifact cancels in the tuned−base
delta. Serving is **path B** (local shim) because no vendor serves a custom LoRA
serverless; the pre-registered Together path A (`.env.example` Arm D block,
`ARM_D_TUNED_MODEL_ID`) was replaced before any eval score was seen.

Arm D training (reproduces the adapter itself; `TINKER_KEY`):
```bash
python -m scripts.build_distill_dataset   # distill Opus-4.8 passed-only trajectories -> SFT set
python -m scripts.train_arm_d             # LoRA SFT on Inkling-Small via Tinker
```

**TAU2 official held-out test split — run once, blind** (`experiments/test_split_precommit.md`;
**executed subset** = open-weight retail only, N=5, seeds 4001–4005):
```bash
# Arm C:
AGENT_MODEL=openai/thinkingmachines/Inkling AGENT_API_BASE=https://api.together.xyz/v1 \
HARNESS_VERSION=v0.1 python -m scripts.run_train_eval --split test_retail --repeats 5 --seed-start 4001
# Arm D base + tuned (shim running on :8100):
for M in thinkingmachines/Inkling-Small armd-inkling-small-tuned; do
  AGENT_MODEL=openai/$M AGENT_API_BASE=http://localhost:8100/v1 HARNESS_VERSION=v0.1 \
    python -m scripts.run_train_eval --split test_retail --domain retail --repeats 5 --seed-start 4001
done
```
gpt-4.1, Opus, Arm B v0.2, and airline/telecom test were **pre-registered as deferred
for budget before launch** — not dropped post-hoc.

**Wiring smoke tests (≈$0, run these first to verify the chain):**
```bash
python -m scripts.run_smoke        # 3 mock-domain tasks, full chain at ~$0
python -m scripts.smoke_iterator   # iterator loop with every external call faked, $0
```

---

## 3. Frozen inputs

### 3.1 Split files (frozen at generation time; CLAUDE.md Hard Constraint #3)

All splits are JSON **arrays of task-id strings** under `benchmark/splits/`. Generated
with seed 42; proxy ⟂ validation disjoint; every carve is disjoint from the TAU2
`test` split by construction. Never edited after generation.

| File | Tasks | Domain | Role |
|---|---|---|---|
| `smoke.json` | 3 | mock | wiring check only (~$0) |
| `proxy.json` | 12 | retail | iterator trains against this; the only split it sees |
| `validation.json` | 35 | retail | blind post-hoc overfitting check |
| `test_retail.json` | 40 | retail | official held-out test (run once) |
| `train_distill_retail.json` | 27 | retail | Arm D distillation source (Opus teacher) |
| `train_distill_airline.json` | 20 | airline | Arm D distillation source |
| `train_distill_telecom.json` | 49 | telecom | Arm D distillation source |
| `eval_airline.json` | 10 | airline | Arm D held-out eval |
| `eval_telecom.json` | 25 | telecom | Arm D held-out eval |
| `transfer_banking.json` | 30 | banking_knowledge | Arm D zero-shot transfer (never trained) |

Disjointness is checkable via `scripts/check_split_disjointness.py`.

### 3.2 Seed conventions (provenance §"Shared harness invariants")

One seed **per repeat**, constant across all tasks within a run (verified — every task
JSON in a run folder shares one `seed`). Blocks:
`proxy 1001–1005`, `validation 2001–2005`, `Arm D held-out eval 2001–2003`,
`TAU2 test 4001–4005`, `iterator proxy 5001+` (disjoint 4-seed blocks per iteration).
**The seed is not a column in `results.csv`** — it lives only in the per-task JSONs
(gap G1; the paired-test consequence is in `docs/paper/analysis/statistics.md`).

### 3.3 N=5 protocol + eval-split cost/audience

The benchmark is stochastic (LLM user-sim; provider non-determinism even at temp 0), so
every baseline is a **distribution, not a single run**: mean ± std over fixed seeds,
with a precommitted minimum of **N=5** repeats (Arm D held-out uses N=3 for budget —
see the statistics floor-on-p caveat). Shared invariants (all arms, from provenance):
user-sim `gpt-4.1-2025-04-14`, temp `0.0` (dropped where rejected), max 200 steps, max
10 consecutive tool errors, pass reward ≥ 0.5.

| Split | Tasks | ~Cost/run | Who uses it | Budget? |
|---|---|---|---|---|
| Smoke | 3 (mock) | ~$0 | wiring | no |
| Proxy | 12 (retail) | ~$3 | iterator optimization | yes |
| Validation | 35 (retail) | ~$10 | blind overfitting check | yes |
| TAU2 test | ~40 (official) | ~$12 | final reporting, once | yes |

---

## 4. ML-reproducibility checklist (NeurIPS-style, answered honestly)

| Question | Answer | Where / gap |
|---|---|---|
| **Code released?** | **Yes** — full harness, iterator, Arm D training/serving, analysis scripts in this repo. | `target_agent/`, `iterator_agent/`, `arm_d/`, `scripts/`, `settings/`. |
| **Environment pinned?** | **Partial.** Floor pins (`>=`) in `pyproject.toml`; TAU2 pinned to a known commit. `uv.lock` is committed. | §1. |
| **Data / splits available?** | **Yes** — all split files are in-repo as task-id lists; disjointness scriptable; TAU2 tasks come from the pinned vendor commit. | §3.1; `benchmark/splits/`, `scripts/check_split_disjointness.py`. |
| **Seeds reported?** | **Yes, but off-ledger.** Conventions documented and stored in per-task JSONs; **no `seed` column in `results.csv`** → runs can't be paired across arms. | §3.2; provenance G1; statistics.md (unpaired-only + recommendation to add the column). |
| **Compute / cost reported?** | **Mostly.** Runtime (eval) cost recorded per run in `results.csv`; per-token prices registered in `settings/pricing.py`; iterator search-cost tracked separately. **Tinker LoRA training cost is unrecorded** (`training_cost_usd=0.0`, no billing telemetry). | provenance §"Fine-tune provenance", G3. |
| **Error bars?** | **Yes.** Mean ± std over N=5 (N=3 Arm D) is the reported number; bootstrap 95% CIs in statistics.md. | `docs/paper/analysis/statistics.md`; plots via `scripts/plot_results.py`. |
| **Stats method disclosed?** | **Yes.** Percentile-bootstrap CIs + Holm-corrected permutation tests, pure numpy (no scipy); small-n low power and the p-floor (2/20=0.10 at n=3) disclosed up front. | `docs/paper/analysis/statistics.md`; `scripts/paper_stats.py`. |
| **Model versions pinned?** | **Partial.** User-sim snapshot is pinned (`gpt-4.1-2025-04-14`). Agent closed models record bare aliases (`gpt-4.1`, `claude-opus-4-8`) — underlying provider snapshot **not** pinned. | provenance G4; §5. |
| **Ablations / controls?** | **Yes.** Arm D uses a matched base control through the identical shim; harness held at v0.1 across the model axis for a clean model comparison. | `experiments/arm_d_eval_precommit.md`. Further ablations were scoped but not run. |
| **Pre-registration?** | **Yes.** Blind events pre-committed before any score seen. | `experiments/{validation_event_v0.2,arm_d_eval,test_split}_precommit.md`. |

---

## 5. Repro limitations (what cannot be reproduced exactly)

1. **Closed-model drift / no pinned snapshots.** `gpt-4.1` and `claude-opus-4-8`
   record bare aliases; the provider can move the underlying snapshot under the alias,
   so exact regeneration of Arm A / Opus / Arm B numbers is not guaranteed even with
   identical code and seeds (provenance G4). Only the user-sim snapshot is pinned.
2. **Tinker `tinker://` checkpoint is account-scoped.** The Arm D adapter URI
   (`tinker://fdc7bf81-…/sampler_weights/arm-d-inkling-small-lora-run_20260807_222817`)
   resolves only within the training account; a third party must **re-run the LoRA SFT**
   (`scripts/train_arm_d`) from the distillation set to obtain their own adapter. The
   SFT data lineage (task_id + domain + source_run + sha256 per example) is auditable in
   `arm_d_sft_manifest.json`, but the exact adapter weights are not portable, and the
   training spend is unrecorded (G3). Arm D is **Opus-distillation SFT, not
   self-improvement** — behavior is bounded by what the Opus teacher demonstrated.
3. **Non-determinism even at temperature 0.** LLM user-simulation plus provider
   non-determinism means run-to-run variance is real; this is exactly why every result
   is an N-run distribution, not a point (§3.3).
4. ~~**No committed lockfile.**~~ Resolved: `uv.lock` is committed. Floor pins remain in `pyproject.toml` (§1.1), and the lockfile is what makes a resolution reproducible. Original note:
   byte-exact dependency reproduction.
5. **Banking agentic-shell binaries unpinned.** `ripgrep`/`srt` versions are not fixed
   by this repo (§1.3); banking is a near-floor transfer probe, so this is low-impact.
6. **TAU2 vendor is a plain clone, not a submodule.** The exact commit is documented
   here (§1.2) but not enforced by repo metadata — a fresh clone lands on TAU2 `HEAD`
   unless checked out to `1746a25`.
```
