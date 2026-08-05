# Arm D — Specialized Inkling via Tinker (multi-domain) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LoRA-fine-tune Inkling via Tinker on Opus-distilled multi-domain trajectories so it becomes a generally better TAU2 agent (retail + airline + telecom, with banking held out as a transfer test), measured as a controlled, research-valid C→D delta.

**Architecture:** A new `arm_d/` package with three pure/injectable seams — `build_dataset.py` (TAU2 transcripts → normalized SFT examples, $0-testable), `train.py` (normalized examples → Tinker LoRA loop, tested with an injected fake Tinker client), and serving wiring that reuses the existing `AGENT_MODEL`/`AGENT_API_BASE` eval seam unchanged. New deterministic split files carve train/held-out/transfer sets. Evaluation reuses `scripts/run_train_eval` and `results.csv` untouched.

**Tech Stack:** Python 3, Tinker SDK (`tinker` + `tinker_cookbook`), LiteLLM (existing cost path), TAU2-bench (frozen vendor), pytest, matplotlib (existing plots).

## Global Constraints

- **Harness stays static v0.1** for every Arm C-control and Arm D run: `HARNESS_VERSION=v0.1`, `target_agent/harness.py` and `target_agent/model_routing.py` unmodified (system prompt every turn, identity routing).
- **Never modify** `vendor/tau2-bench/`, `benchmark/splits/{smoke,proxy,validation}.json` (frozen), `benchmark/adapter.py` behavior, or the TAU2 Orchestrator.
- **`USER_MODEL` (TAU2 user-sim) stays fixed** across all runs (identical to Arm A/C) — never set/override it in Arm D runs.
- **Base model string:** `thinkingmachines/Inkling` (Tinker). **Auth:** the key lives in `.env` as `TINKER_KEY`; `settings/config.py` reads it and exposes it to the SDK as `TINKER_API_KEY` (the SDK's expected name). Do not rename the `.env` var.
- **Training pool excludes every eval task:** banking_knowledge is never trained; retail `validation.json`, `eval_airline.json`, `eval_telecom.json` are never trained.
- **No eval-set tuning:** LoRA hyperparameters / early-stop use only the held-out slice of the *training* tasks. Eval splits are run once, after the config is frozen and pre-registered.
- **Cost accounting separated** into generation / training / runtime (`experiment.md §15.3`); training cost never folded into runtime cost.
- **Eval seeds:** `2001..2005` (N=5), same as every other arm. Report `pass_rate` and `cost_per_successful_task` as mean ± std per domain.
- **Commits:** no `Co-Authored-By: Claude` trailer (project convention). Work stays on branch `ARMD_`.
- **Long runs (>30 min) launch detached** via `nohup ... &` (macOS, no setsid); check for un-reverted edits after any kill.
- Spec of record: `docs/superpowers/specs/2026-08-03-arm-d-inkling-tinker-design.md`.

---

### Task 0: Restore v0.1 identity model routing (corrective — do first)

This `ARMD_` branch carries the Arm B **v0.2** change: `target_agent/model_routing.py::get_model` returns `"anthropic/claude-sonnet-5"` unconditionally, ignoring `configured_model`. In that state every Arm C-control/Arm D run would silently execute on Sonnet instead of the tuned Inkling — an invalid, silent confound. Restore v0.1 identity routing (what Arm C used) so `AGENT_MODEL` is honored. This also greens the two currently-failing `tests/test_harness.py` cases.

**Files:**
- Modify: `target_agent/model_routing.py` (`get_model`)
- Test: `tests/test_harness.py` (already asserts identity — must pass after the change)

**Interfaces:**
- Produces: `get_model(configured_model, history=None) -> str` returns `configured_model` unchanged, regardless of history.

- [ ] **Step 1: Run the failing tests to confirm the starting state**

Run: `python -m pytest tests/test_harness.py::test_get_model_is_identity tests/test_harness.py::test_get_model_accepts_and_ignores_history -v`
Expected: FAIL (get_model returns Sonnet, not the configured model).

- [ ] **Step 2: Restore identity routing**

Replace the body of `get_model` in `target_agent/model_routing.py` so it returns the configured model unchanged (v0.1 behavior), keeping the module docstring's note that the iterator *may* modify this file in later milestones:
```python
def get_model(configured_model: str, history=None) -> str:
    """Return the model to use for the current generation call.

    v0.1 (static harness for Arms A/C/D): identity routing — always use the
    configured AGENT_MODEL. Do not override the model here; the model axis is
    controlled entirely by AGENT_MODEL so open-weight arms actually run their
    own model.
    """
    return configured_model
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/test_harness.py -v`
Expected: PASS (identity + ignores-history both green).

- [ ] **Step 4: Confirm the full suite is green again**

Run: `python -m pytest tests/ -q`
Expected: 0 failures (the 2 pre-existing failures resolved; nothing else regressed).

- [ ] **Step 5: Commit**

```bash
git add target_agent/model_routing.py
git commit -m "arm-d: restore v0.1 identity model routing (honor AGENT_MODEL for open-weight arms)"
```

---

### Task 1: De-risk spike — pin the Tinker SDK surface and the serving path

No code lands from downstream tasks until the two external unknowns are resolved and written down: (a) the exact Tinker SFT API (renderer, datum type, loss-mask/weights, train + save calls) for `thinkingmachines/Inkling`, and (b) whether the tuned adapter can be served behind the Together OpenAI-compat endpoint (preferred) or needs a Tinker-sampling-client chat/tools shim.

**Files:**
- Create: `docs/superpowers/notes/2026-08-03-arm-d-tinker-serving-decision.md`
- Modify: `.env.example` (document `TINKER_API_KEY` and the Arm D `AGENT_MODEL`/serving vars)

**Interfaces:**
- Produces (documented in the decision note, consumed by Tasks 6–7): the exact symbols `train.py` will call — service/training client constructor (e.g. `tinker.ServiceClient().create_lora_training_client(base_model="thinkingmachines/Inkling", rank=R)`), the chat renderer + supervised-datum builder (e.g. `tinker_cookbook.renderers` / `tinker_cookbook.supervised`), `forward_backward`, `optim_step`, `save_weights_and_get_sampling_client`, and the adapter-export call. Plus the chosen serving recipe: model id string + env vars the eval seam needs.

- [ ] **Step 1: Confirm Tinker is installed and reachable**

Run:
```bash
cd /Users/mihirsawhney/Projects/TaskEvolve
python -c "import tinker, tinker_cookbook; print('tinker', getattr(tinker,'__version__','?'))"
```
If import fails, install per the Tinker Cookbook (`pip install tinker` or the cookbook's instructions) and record the exact install command in the note. Requires `TINKER_API_KEY` in the environment.

- [ ] **Step 2: Pin the SFT API surface from the installed SDK (not the web)**

Run and capture real signatures:
```bash
python -c "import tinker; print([s for s in dir(tinker) if not s.startswith('_')])"
python -c "import tinker_cookbook as c; print([s for s in dir(c) if not s.startswith('_')])"
python -c "import inspect, tinker; print(inspect.signature(tinker.ServiceClient.__init__))"
```
Also open the Tinker Cookbook's supervised-learning example. Record in the note: the training-client constructor + LoRA rank arg; the renderer used for a chat conversation with tool calls; the datum/example type and **how the loss weight mask is set so only assistant + tool-call tokens carry loss**; and the `forward_backward`/`optim_step`/`save_weights_and_get_sampling_client` signatures.

- [ ] **Step 3: Minimal end-to-end round-trip at trivial cost**

Write a throwaway scratch script (in the session scratchpad, not the repo) that: builds a LoRA training client on `thinkingmachines/Inkling`, runs ONE `forward_backward` + `optim_step` on a single hand-made 2-message example, then `save_weights_and_get_sampling_client()` and `sample()` one short completion. Confirm it returns without error. Record the working call sequence verbatim in the note.

- [ ] **Step 4: Resolve the serving path**

Determine, from Together's docs/console and the Tinker adapter-export docs, whether an exported Inkling PEFT adapter can be deployed behind Together's OpenAI-compat endpoint (so the eval seam `AGENT_MODEL=openai/<id>` + `AGENT_API_BASE=together` is reused unchanged). If yes, record the deploy steps + resulting model id. If no, record the fallback: a thin local OpenAI-compat shim over Tinker's `SamplingClient` (endpoint that accepts chat+tools, renders to Inkling format, parses tool calls back). Pick one; write the decision + the exact `AGENT_MODEL`/`AGENT_API_BASE`/`AGENT_API_KEY` values Arm D will use.

- [ ] **Step 5: Document `.env.example`**

Add commented lines to `.env.example`:
```
# --- Arm D (Tinker fine-tune of Inkling) ---
# TINKER_API_KEY=...            # from the Tinker Console; required to train
# Arm D serving (see docs/superpowers/notes/2026-08-03-arm-d-tinker-serving-decision.md):
# AGENT_MODEL=openai/<tuned-inkling-id>
# AGENT_API_BASE=<serving endpoint>
```

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/notes/2026-08-03-arm-d-tinker-serving-decision.md .env.example
git commit -m "arm-d: pin Tinker SFT API + serving decision (spike)"
```

**Deliverable gate:** the decision note names every Tinker symbol Task 6 will call and the concrete serving recipe. Downstream tasks reference this note; do not proceed with a guessed API.

---

### Task 2: Generalize split loading to arbitrary named splits

`benchmark.splits.load_split` hard-allows only `{smoke, proxy, validation}` and `run_train_eval`'s `--split` choices are `("proxy","validation")`. Multi-domain eval needs new split files loaded by name, with the domain passed explicitly. This task widens both without touching the frozen files.

**Files:**
- Modify: `benchmark/splits.py:68-88` (`load_split`)
- Modify: `scripts/run_train_eval.py:38` and `:252` (`TRAIN_SPLITS`, `--split` choices)
- Test: `tests/test_splits.py` (create if absent)

**Interfaces:**
- Produces: `load_split(name)` returns the ID list for any `<name>.json` present in `benchmark/splits/`, still raising on unknown names; `run_train_eval` accepts any such split via `--split <name> --domain <domain>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splits.py
import json
from pathlib import Path
import pytest
from benchmark import splits

def test_load_split_reads_any_existing_file(tmp_path, monkeypatch):
    d = tmp_path / "splits"
    d.mkdir()
    (d / "eval_airline.json").write_text(json.dumps(["airline_0", "airline_1"]))
    monkeypatch.setattr(splits, "SPLITS_DIR", d)
    assert splits.load_split("eval_airline") == ["airline_0", "airline_1"]

def test_load_split_unknown_name_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        splits.load_split("does_not_exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_splits.py -v`
Expected: FAIL — `load_split("eval_airline")` raises `ValueError` (not in the hardcoded allow-list).

- [ ] **Step 3: Implement — drop the allow-list, keep the file check**

In `benchmark/splits.py`, replace the `allowed = {...}` guard in `load_split` with a name-safety check and rely on file existence:
```python
def load_split(name: str) -> list[str]:
    """Return the task IDs for a named split file in SPLITS_DIR.

    Any ``<name>.json`` present is loadable (frozen proxy/validation/smoke plus
    the Arm D multi-domain splits). Unknown names raise FileNotFoundError.
    """
    if not name or "/" in name or ".." in name:
        raise ValueError(f"Invalid split name: {name!r}")
    path = SPLITS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run `python -m benchmark.splits` (frozen splits) or "
            "`python -m benchmark.splits_arm_d` (Arm D splits) to generate it."
        )
    return json.loads(path.read_text())
```

- [ ] **Step 4: Widen `run_train_eval` split choices**

In `scripts/run_train_eval.py`: change the `--split` argument to accept any string and update the `TRAIN_SPLITS` doc constant to note it is advisory. Keep `--domain` supplied by the caller for non-retail splits (`--domain airline` etc.):
```python
parser.add_argument("--split", required=True,
    help="Split file name in benchmark/splits/ (proxy, validation, eval_airline, ...).")
```
Remove the `choices=TRAIN_SPLITS` restriction.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_splits.py tests/test_run_train_eval.py -v`
Expected: PASS (new split tests pass; existing run_train_eval tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add benchmark/splits.py scripts/run_train_eval.py tests/test_splits.py
git commit -m "arm-d: load_split accepts arbitrary split files; run_train_eval --split unrestricted"
```

---

### Task 3: Deterministically generate the Arm D multi-domain splits

Produce the train/held-out/transfer split files with a fixed seed and provable disjointness. Retail reuses the frozen `validation.json` as its held-out eval; the 27 unused retail-train tasks become the retail training pool.

**Files:**
- Create: `benchmark/splits_arm_d.py`
- Create (generated): `benchmark/splits/{train_distill_retail,train_distill_airline,train_distill_telecom,eval_airline,eval_telecom,transfer_banking}.json`
- Test: `tests/test_splits_arm_d.py`

**Interfaces:**
- Consumes: `tau2.domains.{retail,airline,telecom,banking_knowledge}.environment.get_tasks(task_split_name="train")`; the frozen `benchmark/splits/{proxy,validation}.json`.
- Produces: `generate_arm_d_splits(*, seed=42)` writes the six files; `arm_d_split_manifest()` returns a dict of split-name → task-id list for the disjointness test.

- [ ] **Step 1: Write the failing test (disjointness is the research-validity guard)**

```python
# tests/test_splits_arm_d.py
import json
from pathlib import Path
import pytest
from benchmark import splits_arm_d

def test_generates_disjoint_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(splits_arm_d, "SPLITS_DIR", tmp_path)
    # copy the frozen retail validation so retail held-out is available
    (tmp_path / "validation.json").write_text(json.dumps(["r_val_0", "r_val_1"]))
    splits_arm_d.generate_arm_d_splits(seed=42)
    m = splits_arm_d.arm_d_split_manifest()
    train = set(m["train_distill_retail"]) | set(m["train_distill_airline"]) | set(m["train_distill_telecom"])
    evalsets = set(m["eval_airline"]) | set(m["eval_telecom"]) | set(m["transfer_banking"]) | {"r_val_0", "r_val_1"}
    assert train.isdisjoint(evalsets), "training pool overlaps an eval split"
    assert set(m["transfer_banking"]) and not (set(m["transfer_banking"]) & train), "banking must be untrained"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_splits_arm_d.py -v`
Expected: FAIL — `benchmark.splits_arm_d` does not exist.

- [ ] **Step 3: Implement the generator**

```python
# benchmark/splits_arm_d.py
"""Generate the Arm D multi-domain train/held-out/transfer splits (deterministic).

Trained domains: retail, airline, telecom (each carved into a training pool and a
held-out eval). Transfer domain: banking_knowledge (never trained). Retail's held-out
eval is the frozen validation.json; retail's training pool is the retail-train tasks
in neither proxy nor validation (the 27 unused).
"""
from __future__ import annotations
import json, random
from pathlib import Path

SPLITS_DIR = Path(__file__).parent / "splits"
# fraction of a trained non-retail domain's train tasks kept for held-out eval
HELDOUT_FRAC = 0.35
BANKING_TRANSFER_SIZE = 30

def _train_ids(domain: str) -> list[str]:
    mod = __import__(f"tau2.domains.{domain}.environment", fromlist=["get_tasks"])
    return [t.id for t in mod.get_tasks(task_split_name="train")]

def _write(ids: list[str], name: str) -> None:
    (SPLITS_DIR / f"{name}.json").write_text(json.dumps(ids, indent=2))

def generate_arm_d_splits(*, seed: int = 42) -> None:
    rng = random.Random(seed)
    # retail: training pool = train tasks in neither proxy nor validation
    retail = _train_ids("retail")
    proxy = set(json.loads((SPLITS_DIR / "proxy.json").read_text())) if (SPLITS_DIR / "proxy.json").exists() else set()
    val = set(json.loads((SPLITS_DIR / "validation.json").read_text()))
    _write([t for t in retail if t not in proxy and t not in val], "train_distill_retail")
    # airline / telecom: shuffle, carve held-out then train pool
    for domain in ("airline", "telecom"):
        ids = _train_ids(domain)[:]
        rng.shuffle(ids)
        n_eval = max(1, int(len(ids) * HELDOUT_FRAC))
        _write(ids[:n_eval], f"eval_{domain}")
        _write(ids[n_eval:], f"train_distill_{domain}")
    # banking: pure transfer, never trained
    banking = _train_ids("banking_knowledge")[:]
    rng.shuffle(banking)
    _write(banking[:BANKING_TRANSFER_SIZE], "transfer_banking")

def arm_d_split_manifest() -> dict[str, list[str]]:
    names = ["train_distill_retail", "train_distill_airline", "train_distill_telecom",
             "eval_airline", "eval_telecom", "transfer_banking"]
    return {n: json.loads((SPLITS_DIR / f"{n}.json").read_text()) for n in names}

if __name__ == "__main__":
    generate_arm_d_splits()
    print("Arm D splits written to", SPLITS_DIR)
```

- [ ] **Step 4: Run tests + generate the real files**

Run:
```bash
python -m pytest tests/test_splits_arm_d.py -v
python -m benchmark.splits_arm_d
python -c "from benchmark.splits_arm_d import arm_d_split_manifest as m; print({k: len(v) for k,v in m().items()})"
```
Expected: test PASS; six files written; printed counts are sane (retail_train ≈ 27, airline/telecom split ~35/65, banking = 30).

- [ ] **Step 5: Commit**

```bash
git add benchmark/splits_arm_d.py tests/test_splits_arm_d.py benchmark/splits/train_distill_retail.json benchmark/splits/train_distill_airline.json benchmark/splits/train_distill_telecom.json benchmark/splits/eval_airline.json benchmark/splits/eval_telecom.json benchmark/splits/transfer_banking.json
git commit -m "arm-d: deterministic multi-domain train/held-out/transfer splits"
```

---

### Task 4: Generate Opus distillation transcripts (operational run)

Run Opus 4.8 on the three training-pool splits to produce the successful trajectories the dataset is distilled from. Operational, not code — but gated.

**Files:** none (writes to `experiments/logs/`).

**Interfaces:**
- Produces: per-task `SimulationRun` transcripts under `experiments/logs/<split>_<ts>/task_*_messages.json` for the three `train_distill_*` splits, from which Task 5 keeps only passed tasks.

- [ ] **Step 1: Smoke one task per domain first (cheap wiring check)**

For each trained domain, run a single seed over its train_distill split's first task to confirm Opus + the domain load and produce a transcript. Example (airline):
```bash
cd /Users/mihirsawhney/Projects/TaskEvolve
AGENT_MODEL=anthropic/claude-opus-4-8 HARNESS_VERSION=v0.1 AGENT_NO_TEMPERATURE=1 \
  python -m scripts.run_train_eval --split train_distill_airline --domain airline --repeats 1 --seed-start 9001
```
Expected: run completes, `task_*_messages.json` written, nonzero `agent_cost`, no `harness_error`.

- [ ] **Step 2: Launch full generation detached (2 seeds/task, per spec)**

For each domain, 2 seeds, detached:
```bash
for d in retail airline telecom; do
  nohup env AGENT_MODEL=anthropic/claude-opus-4-8 HARNESS_VERSION=v0.1 AGENT_NO_TEMPERATURE=1 \
    python -m scripts.run_train_eval --split train_distill_$d --domain $d --repeats 2 --seed-start 9001 \
    > experiments/opus_distill_$d.log 2>&1 &
done
```
(Run domains sequentially if rate limits require; retail uses `--domain retail`.)

- [ ] **Step 3: Verify generation quality gate**

After completion, per domain confirm: transcripts exist, a healthy passed fraction (Opus ~0.7–0.85), zero `harness_error`. Count passed transcripts:
```bash
python -c "
import glob, json
for d in ['retail','airline','telecom']:
    n=p=0
    for f in glob.glob(f'experiments/logs/train_distill_{d}_*/task_*_messages.json'):
        j=json.load(open(f)); n+=1
        ri=j.get('reward_info') or {}; r=ri.get('reward', j.get('reward'))
        if (r or 0) >= 0.5: p+=1
    print(d, 'transcripts', n, 'passed', p)
"
```
Expected: each domain yields a usable count of passed trajectories (dozens total). If a domain under-produces, add seeds for that domain only.

- [ ] **Step 4: No commit** — transcripts live under gitignored `experiments/logs/`. Record the run timestamps in the decision note from Task 1 for the audit trail.

---

### Task 5: `arm_d/build_dataset.py` — transcripts → normalized SFT examples ($0-testable)

Pure transform: read the passed Opus transcripts, normalize each into a chat conversation with tool calls and per-message target flags (assistant/tool-call messages are training targets; system/user/tool-result are context), dedup, domain-balance, and emit a training manifest for the audit trail. No Tinker dependency here — that keeps this fully unit-testable at $0.

**Files:**
- Create: `arm_d/__init__.py`, `arm_d/build_dataset.py`
- Test: `tests/test_arm_d_dataset.py`

**Interfaces:**
- Consumes: `SimulationRun` JSON (`messages` list of `{role, content, tool_calls, ...}`, plus `reward_info.reward`/`reward` and `termination_reason`).
- Produces:
  - `@dataclass TrainingExample(messages: list[dict], target_mask: list[bool], task_id: str, domain: str, source_run: str)` — `messages[i]` is `{"role","content","tool_calls"}`; `target_mask[i]` True iff message i is a training target.
  - `build_examples(transcript_dirs: list[Path], *, domains: dict[str,str]) -> list[TrainingExample]` — passed-only, deduped.
  - `balance(examples, *, seed=0, k=2) -> list[TrainingExample]` — downsample dominant domains to the smallest-domain count × k.
  - `write_manifest(examples, path) -> None` — JSON list of `{task_id, domain, source_run, sha256}` (the audit trail; `sha256` over normalized messages).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arm_d_dataset.py
import json
from pathlib import Path
from arm_d.build_dataset import build_examples, balance, write_manifest, TrainingExample

def _transcript(tmp, name, reward, msgs):
    d = tmp / name; d.mkdir(parents=True, exist_ok=True)
    (d / "task_1_messages.json").write_text(json.dumps(
        {"task_id": "t1", "reward_info": {"reward": reward}, "termination_reason": "done", "messages": msgs}))
    return d

def _msgs():
    return [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "cancel my order"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "cancel", "arguments": "{}"}}]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "Done."},
    ]

def test_keeps_only_passed_and_masks_assistant_targets(tmp_path):
    p = _transcript(tmp_path, "train_distill_retail_x", 1.0, _msgs())
    f = _transcript(tmp_path, "train_distill_retail_y", 0.0, _msgs())  # failed -> dropped
    ex = build_examples([p, f], domains={"train_distill_retail_x": "retail", "train_distill_retail_y": "retail"})
    assert len(ex) == 1
    e = ex[0]
    # only the two assistant messages (idx 2 and 4) are targets
    assert e.target_mask == [False, False, True, False, True]
    assert e.domain == "retail"

def test_manifest_has_hash_per_example(tmp_path):
    p = _transcript(tmp_path, "train_distill_retail_x", 1.0, _msgs())
    ex = build_examples([p], domains={"train_distill_retail_x": "retail"})
    out = tmp_path / "manifest.json"
    write_manifest(ex, out)
    rows = json.loads(out.read_text())
    assert rows and set(rows[0]) == {"task_id", "domain", "source_run", "sha256"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arm_d_dataset.py -v`
Expected: FAIL — `arm_d.build_dataset` does not exist.

- [ ] **Step 3: Implement the transform**

```python
# arm_d/build_dataset.py
"""TAU2 Opus transcripts -> normalized SFT examples (passed-only, deduped).

Pure data transform: no Tinker dependency, so it is fully unit-testable at $0.
Tokenization/rendering into Tinker datums happens in arm_d/train.py.
"""
from __future__ import annotations
import hashlib, json, random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PASS_THRESHOLD = 0.5
TARGET_ROLES = {"assistant"}  # assistant turns (incl. tool_calls) are the targets

@dataclass(frozen=True)
class TrainingExample:
    messages: list[dict]
    target_mask: list[bool]
    task_id: str
    domain: str
    source_run: str

def _reward(rec: dict) -> float:
    ri = rec.get("reward_info") or {}
    return ri.get("reward", rec.get("reward")) or 0.0

def _normalize(m: dict) -> dict:
    return {"role": m.get("role"), "content": m.get("content"),
            "tool_calls": m.get("tool_calls")}

def _hash(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()

def build_examples(transcript_dirs: list[Path], *, domains: dict[str, str]) -> list[TrainingExample]:
    seen: set[str] = set()
    out: list[TrainingExample] = []
    for d in transcript_dirs:
        domain = domains[Path(d).name]
        for f in sorted(Path(d).glob("task_*_messages.json")):
            rec = json.loads(f.read_text())
            if _reward(rec) < PASS_THRESHOLD:
                continue
            msgs = [_normalize(m) for m in rec.get("messages", [])]
            if not any(m["role"] in TARGET_ROLES for m in msgs):
                continue
            h = _hash(msgs)
            if h in seen:
                continue
            seen.add(h)
            mask = [m["role"] in TARGET_ROLES for m in msgs]
            out.append(TrainingExample(msgs, mask, rec.get("task_id", f.stem), domain, Path(d).name))
    return out

def balance(examples: list[TrainingExample], *, seed: int = 0, k: int = 2) -> list[TrainingExample]:
    by_dom: dict[str, list[TrainingExample]] = defaultdict(list)
    for e in examples:
        by_dom[e.domain].append(e)
    if not by_dom:
        return []
    cap = min(len(v) for v in by_dom.values()) * k
    rng = random.Random(seed)
    out: list[TrainingExample] = []
    for v in by_dom.values():
        vv = v[:]; rng.shuffle(vv)
        out.extend(vv[:cap])
    rng.shuffle(out)
    return out

def write_manifest(examples: list[TrainingExample], path) -> None:
    rows = [{"task_id": e.task_id, "domain": e.domain, "source_run": e.source_run,
             "sha256": _hash(e.messages)} for e in examples]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_arm_d_dataset.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add arm_d/__init__.py arm_d/build_dataset.py tests/test_arm_d_dataset.py
git commit -m "arm-d: build_dataset — transcripts to normalized SFT examples (passed-only, deduped)"
```

---

### Task 6: `arm_d/train.py` — Tinker LoRA training loop (fake-client-testable)

Turn normalized examples into Tinker datums via the renderer pinned in Task 1, run the LoRA loop with early-stop on a held-out slice of the *training* tasks, save the adapter, and log training cost + hyperparameters + the manifest. The Tinker client and renderer are injected so the loop logic is testable at $0; the real defaults come from the SDK.

**Files:**
- Create: `arm_d/train.py`
- Test: `tests/test_arm_d_train.py`

**Interfaces:**
- Consumes: `arm_d.build_dataset.TrainingExample`; the Tinker symbols recorded in the Task 1 note.
- Produces:
  - `split_holdout(examples, *, frac=0.15, seed=0) -> tuple[list, list]` (train, holdout) — split **by task_id** so no task leaks across.
  - `@dataclass TrainConfig(base_model="thinkingmachines/Inkling", lora_rank=32, lr=1e-4, max_epochs=3, patience=1)`.
  - `train(examples, cfg, *, client_factory, renderer, log_dir) -> TrainResult` where `@dataclass TrainResult(checkpoint, sampling_client, training_cost_usd, steps, best_holdout_loss)`; writes `log_dir/training_record.json` (cost/steps/hyperparams) and copies the manifest in.
  - Early-stop: stop when held-out loss fails to improve for `patience` epochs.

- [ ] **Step 1: Write the failing test (loop logic with a fake Tinker client)**

```python
# tests/test_arm_d_train.py
from arm_d.build_dataset import TrainingExample
from arm_d.train import split_holdout, train, TrainConfig

def _ex(task_id, dom="retail"):
    m = [{"role": "user", "content": "hi", "tool_calls": None},
         {"role": "assistant", "content": "ok", "tool_calls": None}]
    return TrainingExample(m, [False, True], task_id, dom, "run")

def test_holdout_split_is_by_task():
    exs = [_ex(f"t{i}") for i in range(10)]
    tr, ho = split_holdout(exs, frac=0.2, seed=0)
    tr_ids = {e.task_id for e in tr}; ho_ids = {e.task_id for e in ho}
    assert tr_ids.isdisjoint(ho_ids)
    assert len(ho_ids) >= 1

class _FakeClient:
    def __init__(self): self.steps = 0
    def forward_backward(self, batch): self.steps += 1; return {"loss": 1.0 / (self.steps + 1)}
    def optim_step(self): pass
    def save_weights_and_get_sampling_client(self, name=None): return object()

def test_train_logs_cost_and_stops_early(tmp_path):
    exs = [_ex(f"t{i}") for i in range(8)]
    cfg = TrainConfig(max_epochs=5, patience=1)
    res = train(exs, cfg,
                client_factory=lambda cfg: _FakeClient(),
                renderer=lambda ex: {"tokens": [1, 2], "weights": [0, 1]},
                log_dir=tmp_path)
    assert res.steps >= 1
    assert (tmp_path / "training_record.json").exists()
    assert res.training_cost_usd >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arm_d_train.py -v`
Expected: FAIL — `arm_d.train` does not exist.

- [ ] **Step 3: Implement (real Tinker calls behind the injected factory)**

Write `arm_d/train.py` with: `TrainConfig` dataclass; `split_holdout` grouping by `task_id`; `train(...)` that builds datums via `renderer(example)`, iterates epochs calling `client.forward_backward(batch)` + `client.optim_step()`, evaluates held-out loss each epoch, early-stops on `patience`, then `client.save_weights_and_get_sampling_client(name=...)`; and writes `training_record.json` (`training_cost_usd`, `steps`, `best_holdout_loss`, all `TrainConfig` fields, timestamp) plus copies the dataset manifest into `log_dir`. The default `client_factory` builds the real client using the exact constructor from the Task 1 note, e.g.:
```python
def _default_client_factory(cfg):
    import tinker
    return tinker.ServiceClient().create_lora_training_client(
        base_model=cfg.base_model, rank=cfg.lora_rank)  # confirm signature vs Task 1 note
```
The default `renderer` uses the chat renderer + supervised-datum builder pinned in Task 1, setting the loss weight mask from `TrainingExample.target_mask` so only assistant/tool-call tokens train. Training cost is read from the client/service usage if exposed; else recorded as steps × configured rate (documented in the record).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_arm_d_train.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add arm_d/train.py tests/test_arm_d_train.py
git commit -m "arm-d: Tinker LoRA training loop with by-task holdout, early-stop, cost logging"
```

---

### Task 7: Register the tuned model for pricing + serving, and smoke-gate it

Make the tuned Inkling record nonzero runtime cost and be reachable through the existing eval seam, then prove wiring on 3 mock tasks before any measurement spend.

**Files:**
- Modify: `settings/pricing.py:32` (add the tuned model id key)
- Test: `tests/test_pricing.py`

**Interfaces:**
- Consumes: the serving recipe + tuned model id from the Task 1 note.
- Produces: `litellm.completion_cost` prices the tuned model id nonzero; the model is served behind `AGENT_MODEL`/`AGENT_API_BASE`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pricing.py (add)
import litellm
from settings.pricing import register_pricing, TUNED_INKLING_MODEL_IDS

def test_tuned_inkling_is_priced():
    register_pricing()
    for mid in TUNED_INKLING_MODEL_IDS:
        entry = litellm.model_cost.get(mid)
        assert entry and entry["input_cost_per_token"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pricing.py::test_tuned_inkling_is_priced -v`
Expected: FAIL — `TUNED_INKLING_MODEL_IDS` not defined.

- [ ] **Step 3: Implement — register the tuned id at Inkling's per-token price**

In `settings/pricing.py`, add the tuned model id (the exact string from Task 1's serving decision) at the same per-token price as base Inkling (runtime token price is unchanged; training cost is separate), registered under the provider the serving path uses:
```python
# Tuned Inkling (Arm D). Same per-token price as base Inkling; training cost is
# accounted separately (experiment.md §15.3). Model id from the Task 1 serving note.
TUNED_INKLING_MODEL_IDS = ("<tuned-inkling-id-from-task-1>",)
TUNED_INKLING_PRICING = {
    mid: {"input_cost_per_token": INKLING_INPUT_COST_PER_TOKEN,
          "output_cost_per_token": INKLING_OUTPUT_COST_PER_TOKEN,
          "litellm_provider": "openai", "mode": "chat"}
    for mid in TUNED_INKLING_MODEL_IDS
}
```
and add `litellm.register_model(TUNED_INKLING_PRICING)` in `register_pricing()`.

- [ ] **Step 4: Run test + smoke gate the served model**

Run:
```bash
python -m pytest tests/test_pricing.py -v
AGENT_MODEL=<tuned id> AGENT_API_BASE=<endpoint> HARNESS_VERSION=v0.1 python -m scripts.run_smoke
```
Expected: pricing test PASS; 3/3 mock tasks complete, tool calls parse (no `harness_error`), **`cost_usd` nonzero**. If cost is 0, fix the registered key against the response `model` name before proceeding.

- [ ] **Step 5: Commit**

```bash
git add settings/pricing.py tests/test_pricing.py
git commit -m "arm-d: register tuned-Inkling pricing; smoke-gate the served adapter"
```

---

### Task 8: Pre-register the eval event (research-validity gate)

Freeze the eval protocol in a committed file **before** any tuned-model eval runs, mirroring `experiments/validation_event_v0.2_precommit.md`.

**Files:**
- Create: `experiments/arm_d_eval_precommit.md`

- [ ] **Step 1: Write the pre-registration**

Content: the exact eval splits (`validation` [retail, domain=retail], `eval_airline` [domain=airline], `eval_telecom` [domain=telecom], `transfer_banking` [domain=banking_knowledge]); seeds `2001..2005`; metrics (`pass_rate`, `cost_per_successful_task` mean ± std per domain + aggregate); the two systems compared (matched base-control Inkling, no adapter, same serving path; and tuned Arm D); the harness (`v0.1`); and the **verdict rule**: the specialization counts only if the tuned model's held-out success ≥ base-control within noise on trained domains AND does not regress on the **banking transfer** domain; report all domains regardless of outcome. State that hyperparameters were frozen before this event and no eval-set tuning occurred.

- [ ] **Step 2: Commit BEFORE running eval**

```bash
git add experiments/arm_d_eval_precommit.md
git commit -m "arm-d: pre-register the multi-domain eval event (splits, seeds, verdict)"
```

---

### Task 9: Run the controlled eval — base control + tuned, all four splits (operational)

**Files:** none (writes `results.csv` rows + transcripts).

**Interfaces:**
- Produces: 5-seed rows in `experiments/results.csv` for both systems on all four eval splits, tagged by model id + domain + `harness_version=v0.1`.

- [ ] **Step 1: Matched base control — base Inkling via the SAME serving path, no adapter**

Serve base Inkling through Arm D's serving path (adapter absent/zeroed per Task 1), then, detached, for each eval split:
```bash
for s in "validation retail" "eval_airline airline" "eval_telecom telecom" "transfer_banking banking_knowledge"; do
  set -- $s
  nohup env AGENT_MODEL=<base-inkling-via-D-path> AGENT_API_BASE=<endpoint> HARNESS_VERSION=v0.1 \
    python -m scripts.run_train_eval --split $1 --domain $2 --repeats 5 --seed-start 2001 \
    > experiments/armC_ctrl_$1.log 2>&1 &
done
```

- [ ] **Step 2: Tuned Arm D — same four splits, same seeds**

```bash
for s in "validation retail" "eval_airline airline" "eval_telecom telecom" "transfer_banking banking_knowledge"; do
  set -- $s
  nohup env AGENT_MODEL=<tuned-inkling-id> AGENT_API_BASE=<endpoint> HARNESS_VERSION=v0.1 \
    python -m scripts.run_train_eval --split $1 --domain $2 --repeats 5 --seed-start 2001 \
    > experiments/armD_$1.log 2>&1 &
done
```

- [ ] **Step 3: Verify the runs are valid samples**

Per split × system: 5 rows in `results.csv`, 0 `harness_error`, nonzero cost, tool calls present. Any split with harness errors is re-run before analysis (an invalid sample, per the acceptance-guardrail lesson).

- [ ] **Step 4: No commit of logs** (gitignored). Note run timestamps in the precommit file's appendix.

---

### Task 10: Per-domain frontier plot + Arm D label

**Files:**
- Modify: `scripts/plot_results.py` (arm label carries the tuned/base model + domain)
- Test: `tests/test_plot_results.py`

**Interfaces:**
- Produces: a per-domain + aggregate C→D frontier figure under `experiments/plots/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plot_results.py (add)
from scripts.plot_results import _arm_label  # match the real helper name in the module

def test_arm_label_distinguishes_tuned_from_base_inkling():
    base = _arm_label(split="eval_airline", harness_version="v0.1", agent_model="thinkingmachines/Inkling")
    tuned = _arm_label(split="eval_airline", harness_version="v0.1", agent_model="tuned-inkling-armd")
    assert base != tuned
```
(If `_arm_label`'s real signature differs, adapt the call; the invariant — base ≠ tuned — is what matters.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plot_results.py -v`
Expected: FAIL if base and tuned currently collapse to one arm; else adjust the label helper to include the model tag.

- [ ] **Step 3: Implement the label change + a per-domain frontier entrypoint**

Ensure the arm label includes the short model id so base Inkling and tuned Inkling are distinct arms; add/extend a plotting entrypoint that renders one cost-vs-success frontier per eval domain (retail/airline/telecom/banking) plus an aggregate, base-control ◆ vs tuned ◆ with std error bars.

- [ ] **Step 4: Run tests + render**

Run:
```bash
python -m pytest tests/test_plot_results.py -v
python -m scripts.plot_results --frontier
```
Expected: tests PASS; PNGs written showing base vs tuned per domain.

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_results.py tests/test_plot_results.py
git commit -m "arm-d: per-domain C->D frontier; label distinguishes tuned vs base Inkling"
```

---

### Task 11: Docs + memory (sequence end)

**Files:**
- Modify: `CLAUDE.md` (M3 status + Arm D outcome entry)
- Create/update: memory note for Arm D outcome; update `MEMORY.md` index

- [ ] **Step 1: Write the Arm D outcome into `CLAUDE.md`**

Add an "M3 Arm D" entry: the C→D deltas per domain + the banking transfer verdict (the binding no-overfit result), training cost vs runtime cost, the matched-base-control methodology, and figure names. State the verdict honestly per the precommit rule (improvement, no-change, or overfit).

- [ ] **Step 2: Update memory**

Write `taskevolve-armd-outcome.md` (type: project) summarizing the result + method; add its pointer to `MEMORY.md`. Link `[[tinker-inkling-finetuning]]` and `[[taskevolve-armc-outcome]]`.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "arm-d: document multi-domain specialization outcome + transfer verdict"
```

---

## Self-Review

- **Spec coverage:** §2 decisions → Tasks 1,3,6,7,9; §3 anti-overfit guards → Task 3 (disjointness), Task 6 (by-task holdout + early-stop), Task 8 (verdict); §4.1 data gen → Tasks 3–4; §4.2 dataset → Task 5; §4.3 training → Task 6; §4.4 serving → Tasks 1,7; §4.5 smoke → Task 7; §4.6 eval/report → Tasks 9–10; §8 validity guards → single-variable (Task 9), matched base control (Tasks 1,9), pre-registration (Task 8), no eval tuning (Task 6 holdout only), report-everything (Tasks 9–11), audit trail (Tasks 5,6), teacher disclosure (Task 11). All covered.
- **Placeholders:** the only deferred specifics (exact Tinker symbols, tuned model id string, serving endpoint) are the explicit deliverable of the Task 1 spike and are referenced by name downstream — not vague TODOs.
- **Type consistency:** `TrainingExample(messages, target_mask, task_id, domain, source_run)` defined in Task 5, consumed unchanged in Task 6; `arm_d_split_manifest()` defined + used in Task 3; `TUNED_INKLING_MODEL_IDS` defined in Task 7 and used in its own test.
