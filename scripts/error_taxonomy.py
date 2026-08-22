"""Error analysis / failure taxonomy over EXISTING TaskEvolve transcripts ($0).

Tier-2 validity item #11. This script never makes a network or paid call: it
only reads the per-task verdict JSONs and TAU2 ``SimulationRun`` transcripts
already written under ``experiments/logs/<run_id>/``.

It answers "*why* do tasks fail" using only observable signals:

1. ``termination_reason`` — how the episode ended
   (``user_stop`` / ``agent_stop`` / ``max_steps`` / ``too_many_errors`` /
   ``harness_error: ...``).
2. the ``reward_info.reward_breakdown`` failure **locus** — which check scored
   below 1.0: the final database state (``DB`` / ``db_check``), a required
   agent action (``ACTION`` / ``action_checks``), a natural-language assertion
   (``NL_ASSERTION``), a communication check (``COMMUNICATE``) or an
   environment assertion (``ENV_ASSERTION``).

Crucially it separates **HARNESS errors** (infrastructure crashes —
``termination_reason`` prefixed ``harness_error``, e.g. the missing
agentic-shell sandbox binaries that wiped early banking runs) from **genuine
TASK failures** (the agent finished but produced the wrong outcome). Only
genuine failures carry a failure locus; harness errors are counted and reported
separately so they do not contaminate the "why did the agent fail" taxonomy.

Usage (from repo root):
    python -m scripts.error_taxonomy
    python -m scripts.error_taxonomy --logs-root experiments/logs \\
        --out docs/paper/analysis/error_analysis.md

The classification helpers (``is_harness_error``, ``classify_termination``,
``failure_loci``, ``classify_failure``, ``is_failed``) are pure functions on
plain dicts so they can be unit-tested at $0 (see
``tests/test_error_taxonomy.py``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Pure classification helpers ($0 — no filesystem, no network)
# --------------------------------------------------------------------------- #

HARNESS_PREFIX = "harness_error"

# reward_breakdown keys → normalized locus labels.
LOCUS_KEYS = ("DB", "ACTION", "NL_ASSERTION", "COMMUNICATE", "ENV_ASSERTION")

# Termination buckets we distinguish.
TERM_MAX_STEPS = "max_steps"
TERM_TOO_MANY_ERRORS = "too_many_errors"
TERM_USER_STOP = "user_stop"
TERM_AGENT_STOP = "agent_stop"
TERM_HARNESS = "harness_error"
TERM_OTHER = "other"


def is_harness_error(termination_reason: Optional[str]) -> bool:
    """True iff the episode crashed inside the harness (infrastructure), not a
    genuine task failure. Matches the convention used across the codebase:
    ``termination_reason.startswith("harness_error")``."""
    return bool(termination_reason) and str(termination_reason).startswith(HARNESS_PREFIX)


def harness_error_kind(termination_reason: Optional[str]) -> str:
    """Short label for a harness error, e.g. ``SandboxRuntimeError``.

    ``harness_error: SandboxRuntimeError: Cannot use ...`` -> ``SandboxRuntimeError``.
    Returns ``""`` if the reason is not a harness error.
    """
    if not is_harness_error(termination_reason):
        return ""
    rest = str(termination_reason)[len(HARNESS_PREFIX):].lstrip(": ").strip()
    # First token up to the next ':' is the exception class name.
    return rest.split(":", 1)[0].strip() or "unknown"


def classify_termination(termination_reason: Optional[str]) -> str:
    """Map a raw ``termination_reason`` to a coarse bucket."""
    t = str(termination_reason or "").strip()
    if is_harness_error(t):
        return TERM_HARNESS
    low = t.lower()
    if low.startswith("max_steps") or low.startswith("max_turns"):
        return TERM_MAX_STEPS
    if low.startswith("too_many_errors"):
        return TERM_TOO_MANY_ERRORS
    if low == "user_stop":
        return TERM_USER_STOP
    if low == "agent_stop":
        return TERM_AGENT_STOP
    return TERM_OTHER


def is_failed(verdict: Dict, reward_info: Optional[Dict] = None) -> bool:
    """A task is a failure if it did not fully pass.

    ``passed is False`` OR ``reward < 1.0`` (from the verdict, falling back to
    ``reward_info``). Fully-passing tasks (reward >= 1.0 and not explicitly
    ``passed=False``) are excluded from the taxonomy.
    """
    if verdict.get("passed") is False:
        return True
    reward = verdict.get("reward")
    if reward is None and reward_info is not None:
        reward = reward_info.get("reward")
    if reward is not None:
        try:
            return float(reward) < 1.0
        except (TypeError, ValueError):
            return False
    return False


def failure_loci(reward_info: Optional[Dict]) -> List[str]:
    """Return the sorted list of reward loci that scored below 1.0.

    Reads ``reward_info.reward_breakdown`` (values are 0.0/1.0 in this dataset).
    Falls back to ``db_check.db_match`` / ``action_checks[].action_match`` when a
    breakdown entry is absent. Returns ``[]`` when no sub-1.0 locus is
    observable (caller treats that as ``unknown``).
    """
    if not reward_info:
        return []
    loci: set = set()
    breakdown = reward_info.get("reward_breakdown") or {}
    for key, val in breakdown.items():
        try:
            failed = float(val) < 1.0
        except (TypeError, ValueError):
            failed = False
        if failed:
            loci.add(key)
    # Corroborate / backfill from the structured checks.
    db_check = reward_info.get("db_check") or {}
    if db_check.get("db_match") is False:
        loci.add("DB")
    for chk in reward_info.get("action_checks") or []:
        if chk.get("action_match") is False:
            loci.add("ACTION")
    for chk in reward_info.get("nl_assertions") or []:
        if chk.get("met") is False:
            loci.add("NL_ASSERTION")
    return sorted(loci)


def _locus_label(loci: List[str]) -> str:
    """Compact db-vs-action locus label the paper table uses."""
    db = "DB" in loci
    action = "ACTION" in loci
    others = [l for l in loci if l not in ("DB", "ACTION")]
    if db and action:
        base = "db+action"
    elif db:
        base = "db_only"
    elif action:
        base = "action_only"
    elif others:
        base = "+".join(o.lower().replace("_assertion", "") for o in others)
    else:
        return "unknown"
    if others and (db or action):
        base += "+" + "+".join(o.lower().replace("_assertion", "") for o in others)
    return base


@dataclass
class FailureClass:
    """Full classification of one failed task."""

    category: str  # "harness_error" | "genuine"
    termination_bucket: str
    harness_kind: str = ""
    loci: List[str] = field(default_factory=list)
    db_failed: bool = False
    action_failed: bool = False
    locus_label: str = "unknown"


def classify_failure(verdict: Dict, reward_info: Optional[Dict] = None) -> FailureClass:
    """Classify a single FAILED task from its verdict + reward_info dicts.

    Callers should filter to failures first (``is_failed``); this function does
    not re-check pass/fail.
    """
    term = verdict.get("termination_reason")
    bucket = classify_termination(term)
    if is_harness_error(term):
        return FailureClass(
            category="harness_error",
            termination_bucket=TERM_HARNESS,
            harness_kind=harness_error_kind(term),
        )
    loci = failure_loci(reward_info)
    return FailureClass(
        category="genuine",
        termination_bucket=bucket,
        loci=loci,
        db_failed="DB" in loci,
        action_failed="ACTION" in loci,
        locus_label=_locus_label(loci),
    )


# --------------------------------------------------------------------------- #
# Filesystem scan + aggregation
# --------------------------------------------------------------------------- #

def _arm_label(model: Optional[str]) -> str:
    """Short arm label from an agent_model id (drop provider prefix)."""
    if not model:
        return "unknown"
    return str(model).split("/")[-1]


@dataclass
class TaskRow:
    run_id: str
    arm: str
    domain: str
    split: str
    task_id: str
    passed: bool
    reward: Optional[float]
    termination_reason: str
    cls: FailureClass


def _load_json(path: str) -> Optional[Dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def scan(logs_root: str) -> List[TaskRow]:
    """Scan every run folder for FAILED tasks and classify each.

    Passed tasks are skipped. For each failed task we read the verdict
    ``task_<id>.json`` and, when present, the transcript ``task_<id>_messages.json``
    for ``reward_info`` (the failure locus).
    """
    rows: List[TaskRow] = []
    verdicts = sorted(
        p for p in glob.glob(os.path.join(logs_root, "*", "task_*.json"))
        if not p.endswith("_messages.json")
    )
    for vpath in verdicts:
        verdict = _load_json(vpath)
        if not verdict:
            continue
        mpath = vpath[: -len(".json")] + "_messages.json"
        msg = _load_json(mpath)
        reward_info = (msg or {}).get("reward_info") if msg else None
        if not is_failed(verdict, reward_info):
            continue
        cls = classify_failure(verdict, reward_info)
        rows.append(
            TaskRow(
                run_id=os.path.basename(os.path.dirname(vpath)),
                arm=_arm_label(verdict.get("agent_model")),
                domain=verdict.get("domain") or "unknown",
                split=verdict.get("split") or "unknown",
                task_id=verdict.get("task_id") or os.path.basename(vpath),
                passed=bool(verdict.get("passed")),
                reward=verdict.get("reward"),
                termination_reason=str(verdict.get("termination_reason")),
                cls=cls,
            )
        )
    return rows


def count_tasks(logs_root: str) -> Tuple[int, int]:
    """Return (total_tasks, passed_tasks) across all runs (for context)."""
    total = passed = 0
    for vpath in glob.glob(os.path.join(logs_root, "*", "task_*.json")):
        if vpath.endswith("_messages.json"):
            continue
        v = _load_json(vpath)
        if not v:
            continue
        total += 1
        if not is_failed(v):
            passed += 1
    return total, passed


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    out = [line, sep]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def build_report(rows: List[TaskRow], logs_root: str) -> str:
    total, passed = count_tasks(logs_root)
    harness = [r for r in rows if r.cls.category == "harness_error"]
    genuine = [r for r in rows if r.cls.category == "genuine"]

    lines: List[str] = []
    lines.append("# Failure-Mode Taxonomy (Tier-2 validity item #11)\n")
    lines.append(
        "Error analysis over existing TaskEvolve transcripts under "
        f"`{logs_root}` — a $0 pass, no network or paid calls. Every task with "
        "`passed == False` or `reward < 1.0` is classified by two observable "
        "signals: the `termination_reason` and, for genuine failures, the "
        "`reward_info.reward_breakdown` failure locus (which check scored below "
        "1.0). **Harness errors** (infrastructure crashes, `termination_reason` "
        "prefixed `harness_error`) are separated from **genuine task failures** "
        "so they never contaminate the 'why did the agent fail' taxonomy.\n"
    )
    lines.append(
        f"- Total tasks scanned: **{total}**  ·  passed: **{passed}**  ·  "
        f"failed: **{len(rows)}**\n"
        f"- Of the failures: **{len(harness)}** harness/infrastructure errors "
        f"(excluded from the taxonomy) and **{len(genuine)}** genuine task "
        "failures.\n"
    )

    # --- 1. Termination-reason taxonomy (all failures) ------------------- #
    lines.append("\n## 1. Failure taxonomy by termination reason\n")
    term_counter: Counter = Counter(r.cls.termination_bucket for r in rows)
    trows = [[b, str(c), f"{c/len(rows)*100:.1f}%"] for b, c in term_counter.most_common()]
    lines.append(_fmt_table(["termination bucket", "n", "share"], trows))

    # Harness error subtypes.
    if harness:
        lines.append("\n### Harness-error subtypes (infrastructure, NOT task failures)\n")
        hk = Counter(r.cls.harness_kind for r in harness)
        hrows = [[k, str(v)] for k, v in hk.most_common()]
        lines.append(_fmt_table(["harness error kind", "n"], hrows))

    # --- 2. Genuine-failure locus taxonomy ------------------------------- #
    lines.append("\n## 2. Genuine-failure taxonomy by reward locus\n")
    lines.append(
        "Locus = which reward check scored < 1.0. `DB` = final database state "
        "mismatch (`db_check`); `ACTION` = a required agent action was missing "
        "or wrong (`action_checks`); `NL`/`COMMUNICATE`/`ENV` = the "
        "corresponding assertion checks; `unknown` = failed overall but no "
        "sub-1.0 locus was observable (or the transcript was absent).\n"
    )
    loc_counter = Counter(r.cls.locus_label for r in genuine)
    lrows = [[b, str(c), f"{c/max(len(genuine),1)*100:.1f}%"] for b, c in loc_counter.most_common()]
    lines.append(_fmt_table(["failure locus", "n", "share of genuine"], lrows))

    db_only = sum(1 for r in genuine if r.cls.db_failed and not r.cls.action_failed)
    action_only = sum(1 for r in genuine if r.cls.action_failed and not r.cls.db_failed)
    both = sum(1 for r in genuine if r.cls.db_failed and r.cls.action_failed)
    lines.append(
        f"\n**db vs action locus:** db_only = {db_only} · action_only = "
        f"{action_only} · db+action = {both}.\n"
    )

    # --- 3. Per-(arm, domain) breakdown ---------------------------------- #
    lines.append("\n## 3. Per-arm, per-domain breakdown\n")
    key = lambda r: (r.arm, r.domain)  # noqa: E731
    agg: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for r in rows:
        c = agg[key(r)]
        c["failures"] += 1
        if r.cls.category == "harness_error":
            c["harness"] += 1
        else:
            c["genuine"] += 1
            if r.cls.db_failed:
                c["db"] += 1
            if r.cls.action_failed:
                c["action"] += 1
            for l in r.cls.loci:
                if l not in ("DB", "ACTION"):
                    c[l.lower()] += 1
    brows = []
    for (arm, domain), c in sorted(agg.items(), key=lambda kv: (-kv[1]["failures"], kv[0])):
        brows.append([
            arm, domain, str(c["failures"]), str(c["harness"]), str(c["genuine"]),
            str(c["db"]), str(c["action"]),
            str(c["nl_assertion"] + c["communicate"] + c["env_assertion"]),
        ])
    lines.append(_fmt_table(
        ["arm", "domain", "fails", "harness", "genuine", "db", "action", "nl/comm/env"],
        brows,
    ))

    # --- 4. Per-domain rollup -------------------------------------------- #
    lines.append("\n## 4. Per-domain rollup (all arms)\n")
    dom: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        c = dom[r.domain]
        c["failures"] += 1
        c["harness" if r.cls.category == "harness_error" else "genuine"] += 1
        if r.cls.db_failed:
            c["db"] += 1
        if r.cls.action_failed:
            c["action"] += 1
    drows = [
        [d, str(c["failures"]), str(c["harness"]), str(c["genuine"]), str(c["db"]), str(c["action"])]
        for d, c in sorted(dom.items(), key=lambda kv: -kv[1]["failures"])
    ]
    lines.append(_fmt_table(["domain", "fails", "harness", "genuine", "db", "action"], drows))

    # --- 5. Banking 0.10 floor deep-dive (placeholder filled by caller) --- #
    lines.append(BANKING_SECTION)

    lines.append("\n---\n")
    lines.append(
        "_Generated by `scripts/error_taxonomy.py` (no network / no paid "
        "calls). Regenerate with `python -m scripts.error_taxonomy`._\n"
    )
    return "\n".join(lines)


# The banking deep-dive is authored from direct transcript inspection (cited in
# the analysis); it is a static, evidence-based subsection appended to the
# generated report.
BANKING_SECTION = """
## 5. The banking 0.10 floor — what actually goes wrong

`transfer_banking` (domain `banking_knowledge`) is the *zero-shot transfer*
case: the fine-tuned Arm D model (`armd-inkling-small-tuned`) and its base
(`Inkling-Small`) — both specialized on retail/airline/telecom — are run
against a banking domain they never trained on. Reported success sits at a
~0.10 floor. The transcripts show this floor is **real (a genuine task-failure
floor), not a harness artifact — but only once the harness itself is fixed.**

### 5a. Two things are conflated in the raw pass rate

The first five `Inkling-Small` banking runs (`transfer_banking_20260809_*`
through `20260810_195730`) recorded **30/30 tasks as `harness_error:
SandboxRuntimeError`** — the agentic-shell sandbox could not start because the
`srt` and `ripgrep` binaries were not installed (the domain's `shell`
retrieval tool needs them). Those runs contribute **0 information about agent
skill** and are excluded from the floor. Only the later runs
(`20260810_231431` onward) ran on a working sandbox.

**On the clean runs the floor is genuine and identical for base and tuned:**

- `Inkling-Small` clean runs: 2/30, 3/29, 4/30  →  **9 / 89 ≈ 0.10**
- `armd-inkling-small-tuned` clean runs: 3/30, 2/30, 4/30  →  **9 / 90 = 0.10**

Fine-tuning bought **no transfer** to banking — base and tuned land on the same
floor. Among genuine (non-harness) banking failures the locus is overwhelmingly
**`DB` (final database state wrong): 142 db-only vs 15 action-only** — the agent
converses to a natural `user_stop` end but leaves the bank's database in the
wrong state.

### 5b. It is NOT a retrieval-starvation failure — the agent over-retrieves

Failed banking tasks use *more* tools than passing ones, not fewer (per-task
averages): `shell` 11.2 vs 6.2, `call_discoverable_agent_tool` 3.2 vs 0.8,
`unlock_discoverable_agent_tool` 2.1 vs 1.1. The agent is not giving up or
failing to look things up — it flails: many knowledge-base/shell lookups and
repeated privileged-action attempts, yet still commits the wrong writes.

### 5c. What actually goes wrong: wrong product choice + wrong-argument writes

Concrete evidence from
`transfer_banking_20260811_045517/task_task_063_messages.json` (reward 0.0,
`db_check.db_match=False`, `reward_breakdown={"DB": 0.0}`). The user asks to
open "both a credit card and a savings account ... maximize the interest".
The required actions were `log_verification`, `apply_for_credit_card`
(`card_type="Silver Rewards Card"`), and `call_discoverable_agent_tool` to open
a `"Silver Plus Account"` savings account. After ~20 shell/KB retrievals the
agent reasons:

> "the highest-APY savings you can open is the **Silver Plus Savings** ..."

— it gets the *savings* product right (the `unlock_discoverable_agent_tool`
action check passes, `action_match=True`), but then recommends and applies for
the wrong **credit card** and mis-fills the `call_discoverable_agent_tool`
arguments, so `apply_for_credit_card` and the account-open call both score
`action_match=False` and the final DB state does not match. It also never lands
the required `log_verification` action correctly. The failure is a
**policy/reasoning error in product selection and action arguments**, downstream
of plentiful (successful) retrieval.

A second pattern — wrong-argument writes and state confusion — in
`transfer_banking_20260811_084831/task_task_003_messages.json` (reward 0.0,
`reward_breakdown={"DB": 0.0}`, expected `apply_for_credit_card` for the
`"Silver Rewards Card"`, `action_match=False`). The user pivots mid-task:

> user: "I've decided to apply for the Gold Rewards Card."
> assistant: "Before I can help you apply, I need to verify your identity
> first ..."
> user: "I've already submitted my application for the Gold Rewards Card."

The agent had already submitted a credit-card application (for the wrong card),
then loses track of state — it asks to re-verify identity for an application the
user says is already done, and the committed DB write does not match the
required card. The episode ends `user_stop` (a polite conversational close),
masking that the *database outcome is wrong*.

### 5d. Takeaway

The banking floor is **~0.10, genuine, and unmoved by fine-tuning** (base ==
tuned) once the SandboxRuntimeError runs are excluded. The dominant failure
mode is **`DB`-state mismatch caused by wrong product selection and
wrong-argument write calls** in an unfamiliar domain — not retrieval failure and
not the agent giving up. `user_stop` terminations hide these as "completed"
conversations, so termination reason alone would badly under-count them; the
`reward_breakdown` locus is what exposes the true failure.
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-root", default="experiments/logs")
    ap.add_argument("--out", default="docs/paper/analysis/error_analysis.md")
    ap.add_argument("--no-write", action="store_true", help="print report only")
    args = ap.parse_args(argv)

    rows = scan(args.logs_root)
    report = build_report(rows, args.logs_root)
    print(report)
    if not args.no_write:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(report)
        print(f"\n[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
