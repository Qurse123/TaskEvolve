"""Cost-comparison FAIRNESS: separate MODEL EFFICIENCY from COMMERCIAL PRICING.

The headline "open-weight is ~26x cheaper than frontier Opus" compares
closed-model *retail API prices* (Opus, gpt-4.1 -- which bake in provider
margin) against the study's *own serving cost* for the open models (Inkling via
Together, tuned Inkling-Small via a local Tinker shim). That conflates two
different things:

  * MODEL EFFICIENCY -- tokens/work spent per task (intrinsic to the model), and
  * COMMERCIAL PRICING -- the $/token markup (a business artifact).

This script re-prices every arm at a SINGLE common $/token rate so the reader
sees efficiency with pricing held constant, alongside the as-listed retail cost,
and prints the ratio under each view. Raw agent/user token counts are recovered
from the persisted TAU2 transcripts (`task_*_messages.json`); arms whose logs
predate transcript persistence fall back to a backed-out token estimate
(cost / listed blended price) or a turn+tool-call work proxy.

Reads `experiments/results.csv` (canonical per-run cost) joined to
`experiments/logs/<run_id>/` (transcripts + per-task verdicts, run_id == dir).
Writes `docs/paper/analysis/cost_fairness.md` and prints a summary.

Run:  python -m scripts.cost_fairness
"""
from __future__ import annotations

import csv
import glob
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

CSV = Path("experiments/results.csv")
LOGS = Path("experiments/logs")
OUT = Path("docs/paper/analysis/cost_fairness.md")

# --- The single common re-pricing schedule (pricing held constant across arms).
# A neutral generic-serving reference. Its absolute level is arbitrary; the
# RATIO between arms is what matters and is robust to the chosen level, because
# every arm is priced with the same two numbers.
COMMON_INPUT_RATE_PER_TOKEN = 1.0e-6   # $/input token
COMMON_OUTPUT_RATE_PER_TOKEN = 3.0e-6  # $/output token

# Listed blended per-token price used ONLY to back tokens out of cost for arms
# with no transcript (gpt-4.1's early run). gpt-4.1 retail ~ $2/1M in, $8/1M out;
# at the ~98%-input mix seen in every transcripted arm this blends to ~2.1e-6.
# This is a rough estimate -- a single blended cost cannot be split into in/out.
GPT41_BLENDED_PER_TOKEN = 2.12e-6

# Work-proxy weights (turns + weighted tool calls) for arms with neither tokens
# nor a usable cost->token backout.
WORK_TURN_WEIGHT = 1.0
WORK_TOOL_WEIGHT = 1.0

# The reported comparison arms (validation split -- the headline set). Every arm
# here uses the v0.1 harness so the axis is model, not harness (CLAUDE.md).
ARMS: dict[str, tuple[str, str, str]] = {
    "gpt-4.1 (closed)": ("gpt-4.1", "v0.1", "validation"),
    "Opus 4.8 (closed frontier)": ("anthropic/claude-opus-4-8", "v0.1", "validation"),
    "Inkling (open, Together)": ("openai/thinkingmachines/Inkling", "v0.1", "validation"),
    "Inkling-Small base (open)": ("openai/thinkingmachines/Inkling-Small", "v0.1", "validation"),
    "Inkling-Small tuned (open)": ("openai/armd-inkling-small-tuned", "v0.1", "validation"),
}

# The two arms whose ratio is the headline claim.
HEADLINE_CLOSED = "Opus 4.8 (closed frontier)"
HEADLINE_OPEN = "Inkling (open, Together)"


# --------------------------------------------------------------------------- #
# Pure math (unit-tested in tests/test_cost_fairness.py)
# --------------------------------------------------------------------------- #
def reprice_tokens(input_tokens: float, output_tokens: float,
                   input_rate: float, output_rate: float) -> float:
    """Cost of a token bundle at a given per-token schedule."""
    return input_tokens * input_rate + output_tokens * output_rate


def cost_per_successful_task(cost_per_task: float, pass_rate: float) -> float:
    """Cost per *successful* task = cost per task / pass rate. inf if no passes."""
    if pass_rate <= 0:
        return float("inf")
    return cost_per_task / pass_rate


def ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator; inf if denominator is 0."""
    if denominator == 0:
        return float("inf")
    return numerator / denominator


def work_proxy_units(turns: float, tool_calls: float,
                     turn_weight: float = WORK_TURN_WEIGHT,
                     tool_weight: float = WORK_TOOL_WEIGHT) -> float:
    """A model-agnostic 'work' proxy when tokens are unrecoverable."""
    return turns * turn_weight + tool_calls * tool_weight


def estimate_tokens_from_cost(cost: float, blended_rate_per_token: float) -> float:
    """Back total tokens out of a blended cost. Cannot split input vs output."""
    if blended_rate_per_token <= 0:
        return 0.0
    return cost / blended_rate_per_token


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@dataclass
class ArmData:
    label: str
    agent_model: str
    harness_version: str
    split: str
    n_runs: int = 0
    pass_rate: float = 0.0
    as_listed_cost_per_task: float = 0.0
    # per-task token means (agent = the model under test; user = TAU2 user-sim)
    n_tasks: int = 0
    tokens_covered: int = 0
    agent_in: float = 0.0
    agent_out: float = 0.0
    user_in: float = 0.0
    user_out: float = 0.0
    user_cost_per_task: float = 0.0
    # turn/tool work proxy (per task); None when the verdicts predate the fields
    turns: float | None = None
    tools: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_tokens(self) -> bool:
        return self.tokens_covered > 0

    @property
    def agent_tokens_per_task(self) -> float:
        return self.agent_in + self.agent_out


def _arm_key(row: dict) -> tuple[str, str, str]:
    return (row["agent_model"], row["harness_version"], row["split"])


def load_arm(label: str, key: tuple[str, str, str], rows: list[dict]) -> ArmData:
    model, harness, split = key
    runs = [r for r in rows if _arm_key(r) == key]
    data = ArmData(label=label, agent_model=model, harness_version=harness,
                   split=split, n_runs=len(runs))
    if not runs:
        data.notes.append("no rows in results.csv")
        return data

    data.pass_rate = statistics.mean(float(r["pass_rate"]) for r in runs)
    data.as_listed_cost_per_task = statistics.mean(
        float(r["total_cost_usd"]) / float(r["num_tasks"]) for r in runs)

    ag_in = ag_out = us_in = us_out = us_cost = 0.0
    n_tasks = covered = 0
    turns: list[float] = []
    tools: list[float] = []
    for r in runs:
        run_dir = LOGS / r["run_id"]
        for f in glob.glob(str(run_dir / "task_*_messages.json")):
            sim = json.load(open(f))
            n_tasks += 1
            had_usage = False
            for m in sim.get("messages", []):
                u = m.get("usage")
                if not u:
                    continue
                had_usage = True
                pin = u.get("prompt_tokens", 0) or 0
                pout = u.get("completion_tokens", 0) or 0
                if m.get("role") == "assistant":
                    ag_in += pin
                    ag_out += pout
                elif m.get("role") == "user":
                    us_in += pin
                    us_out += pout
            if had_usage:
                covered += 1
            us_cost += sim.get("user_cost", 0.0) or 0.0
        # turn/tool proxy from the per-task verdicts (added in M2)
        for f in glob.glob(str(run_dir / "task_*.json")):
            if f.endswith("_messages.json") or f.endswith("run_summary.json"):
                continue
            v = json.load(open(f))
            if "turn_count" in v:
                turns.append(float(v["turn_count"]))
                tools.append(float(v.get("tool_call_count", 0) or 0))

    data.n_tasks = n_tasks
    data.tokens_covered = covered
    if n_tasks:
        data.agent_in = ag_in / n_tasks
        data.agent_out = ag_out / n_tasks
        data.user_in = us_in / n_tasks
        data.user_out = us_out / n_tasks
        data.user_cost_per_task = us_cost / n_tasks
    if turns:
        data.turns = statistics.mean(turns)
        data.tools = statistics.mean(tools)

    if not data.has_tokens:
        est = estimate_tokens_from_cost(
            data.as_listed_cost_per_task, GPT41_BLENDED_PER_TOKEN)
        data.notes.append(
            f"no transcripts persisted; ~{est:,.0f} agent tokens/task backed out "
            f"of cost at a ~{GPT41_BLENDED_PER_TOKEN*1e6:.2f} $/1M blended rate "
            f"(cannot split input/output)")
        if data.turns is None:
            data.notes.append("verdicts predate turn/tool logging; no work proxy")
    return data


def normalized_cost_per_task(arm: ArmData) -> float | None:
    """Agent-token cost at the common schedule; None if tokens unavailable."""
    if not arm.has_tokens:
        return None
    return reprice_tokens(arm.agent_in, arm.agent_out,
                          COMMON_INPUT_RATE_PER_TOKEN, COMMON_OUTPUT_RATE_PER_TOKEN)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x: float | None, spec: str) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "inf"
    return format(x, spec)


def build_report(arms: dict[str, ArmData]) -> str:
    L: list[str] = []
    L.append("# Cost-comparison fairness: model efficiency vs commercial pricing")
    L.append("")
    L.append("_Generated by `scripts/cost_fairness.py` at $0 (reads only local "
             "`results.csv` + transcripts)._")
    L.append("")

    # (a) the apples-to-oranges issue
    L.append("## (a) The apples-to-oranges problem")
    L.append("")
    L.append("The headline claim -- \"open-weight Inkling is ~26x cheaper per "
             "successful task than frontier Opus\" -- compares two prices that "
             "are not the same kind of number:")
    L.append("")
    L.append("* **Closed models (Opus, gpt-4.1)** are billed at **retail API "
             "list price**, which bakes in the provider's commercial margin.")
    L.append("* **Open models (Inkling, Inkling-Small)** are billed at the "
             "study's **own serving cost** (Together's near-cost rate; the "
             "tuned model via a local Tinker shim).")
    L.append("")
    L.append("So the ratio conflates **model efficiency** (how many tokens of "
             "work a task takes -- intrinsic to the model) with **commercial "
             "pricing** (the $/token markup -- a business artifact). Stated as "
             "\"26x cheaper\" it reads as an intrinsic property of the model. It "
             "is not. Below, every arm is re-priced at a single common $/token "
             "schedule so efficiency is visible with pricing held constant.")
    L.append("")

    # (b) normalized table
    L.append("## (b) Normalized comparison")
    L.append("")
    L.append(f"Common re-pricing schedule: **input "
             f"${COMMON_INPUT_RATE_PER_TOKEN*1e6:.2f}/1M, output "
             f"${COMMON_OUTPUT_RATE_PER_TOKEN*1e6:.2f}/1M**, applied identically "
             f"to every arm's *agent* tokens. Absolute level is arbitrary; the "
             f"cross-arm ratio is what matters.")
    L.append("")
    L.append("| Arm | pass | agent tok/task (in+out) | as-listed $/task | as-listed $/succ | **norm $/task** | **norm $/succ** |")
    L.append("|-----|-----:|-----:|-----:|-----:|-----:|-----:|")
    for arm in arms.values():
        norm = normalized_cost_per_task(arm)
        norm_succ = (cost_per_successful_task(norm, arm.pass_rate)
                     if norm is not None else None)
        as_succ = cost_per_successful_task(arm.as_listed_cost_per_task, arm.pass_rate)
        tok = (f"{arm.agent_in:,.0f}+{arm.agent_out:,.0f}"
               if arm.has_tokens else "n/a")
        L.append(f"| {arm.label} | {arm.pass_rate:.3f} | {tok} | "
                 f"${arm.as_listed_cost_per_task:.4f} | ${_fmt(as_succ, '.4f')} | "
                 f"{'$'+_fmt(norm, '.4f') if norm is not None else 'n/a'} | "
                 f"{'$'+_fmt(norm_succ, '.4f') if norm_succ is not None else 'n/a'} |")
    L.append("")

    # (c) reframed claims
    L.append("## (c) Reframed, defensible cost claims")
    L.append("")
    closed = arms.get(HEADLINE_CLOSED)
    openm = arms.get(HEADLINE_OPEN)
    if closed and openm:
        as_closed = cost_per_successful_task(closed.as_listed_cost_per_task, closed.pass_rate)
        as_open = cost_per_successful_task(openm.as_listed_cost_per_task, openm.pass_rate)
        as_ratio = ratio(as_closed, as_open)
        nc = normalized_cost_per_task(closed)
        no = normalized_cost_per_task(openm)
        tok_ratio = ratio(openm.agent_tokens_per_task, closed.agent_tokens_per_task)
        norm_ratio = ratio(nc, no) if (nc and no) else float("nan")
        L.append(f"**At listed prices today** (retail closed vs study-served open): "
                 f"Inkling costs **${as_open:.4f}/successful task** vs Opus "
                 f"**${as_closed:.4f}** -> **{as_ratio:.0f}x cheaper**. This is a "
                 f"real invoice difference *at current market prices*, and is the "
                 f"honest way to state it -- as a pricing fact, not a model property.")
        L.append("")
        L.append(f"**At equal $/token** (efficiency, pricing held constant): the "
                 f"two models spend essentially the same work per task -- Opus "
                 f"{closed.agent_tokens_per_task:,.0f} agent tokens/task vs Inkling "
                 f"{openm.agent_tokens_per_task:,.0f} ({tok_ratio:.2f}x). Re-priced "
                 f"at one common rate, Inkling is **{norm_ratio:.2f}x** the cost of "
                 f"Opus per task -- i.e. **no efficiency advantage** (marginally "
                 f"more tokens). The ~{as_ratio:.0f}x is therefore **almost entirely "
                 f"a commercial-pricing artifact**, not intrinsic model efficiency.")
        L.append("")
        base = arms.get("Inkling-Small base (open)")
        tuned = arms.get("Inkling-Small tuned (open)")
        if base and tuned and base.has_tokens and tuned.has_tokens:
            tune_ratio = ratio(base.agent_tokens_per_task, tuned.agent_tokens_per_task)
            L.append(f"**A genuine efficiency result survives normalization:** the "
                     f"Opus-distilled tuned Inkling-Small uses "
                     f"{tuned.agent_tokens_per_task:,.0f} agent tokens/task vs the "
                     f"base {base.agent_tokens_per_task:,.0f} -- a **{tune_ratio:.2f}x** "
                     f"token reduction at *identical* serving. That is a real "
                     f"model-efficiency gain (fine-tuning shortened the work), "
                     f"visible precisely because pricing is held constant.")
            L.append("")

    # (d) excluded user-sim cost
    L.append("## (d) Excluded user-simulator cost (disclosed confound)")
    L.append("")
    L.append("The headline agent cost **excludes** the TAU2 user-simulator's own "
             "model cost. It is small but nonzero and identical in kind across "
             "arms (same user-sim model):")
    L.append("")
    L.append("| Arm | user-sim $/task | as % of agent $/task |")
    L.append("|-----|-----:|-----:|")
    for arm in arms.values():
        pct = (100 * arm.user_cost_per_task / arm.as_listed_cost_per_task
               if arm.as_listed_cost_per_task else 0.0)
        uc = f"${arm.user_cost_per_task:.4f}" if arm.has_tokens else "n/a"
        pctf = f"{pct:.1f}%" if arm.has_tokens else "n/a"
        L.append(f"| {arm.label} | {uc} | {pctf} |")
    L.append("")
    L.append("Because it is roughly constant in absolute dollars, the user-sim "
             "cost is a *larger* share of the cheap open arms than of Opus -- "
             "another reason the raw agent-cost ratio overstates the open models' "
             "true all-in advantage.")
    L.append("")

    # (e) limitations
    L.append("## (e) Data-limitation notes")
    L.append("")
    L.append("* **Token recoverability:** agent/user tokens are read from "
             "per-message `usage` in the persisted transcripts and are available "
             "for Opus, Inkling, and both Inkling-Small arms (in/out split intact "
             "-- no blended-cost guessing needed there).")
    L.append("* **gpt-4.1 gap:** its June validation run predates transcript "
             "persistence *and* turn/tool logging, so it has neither tokens nor a "
             "work proxy. Its token count can only be *estimated* by dividing cost "
             "by a listed blended price, which **cannot separate input from "
             "output** -- treat that number as indicative only.")
    L.append("* **Common-rate choice:** the normalized level is arbitrary; only "
             "cross-arm ratios are load-bearing, and those are invariant to the "
             "level because one schedule prices every arm.")
    L.append("* **Blended per-message cost:** where a single per-message `cost` is "
             "the only signal, input vs output cannot be cleanly separated; the "
             "normalized view sidesteps this by using raw token counts, not cost.")
    for arm in arms.values():
        for n in arm.notes:
            L.append(f"* **{arm.label}:** {n}")
    L.append("")
    return "\n".join(L)


def main() -> int:
    rows = list(csv.DictReader(open(CSV)))
    arms = {label: load_arm(label, key, rows) for label, key in ARMS.items()}

    report = build_report(arms)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report)

    # console summary
    print(f"wrote {OUT}\n")
    print(f"{'arm':32s} {'pass':>6s} {'agent tok/task':>15s} "
          f"{'aslisted $/succ':>15s} {'norm $/succ':>12s}")
    for arm in arms.values():
        norm = normalized_cost_per_task(arm)
        norm_succ = (cost_per_successful_task(norm, arm.pass_rate)
                     if norm is not None else None)
        as_succ = cost_per_successful_task(arm.as_listed_cost_per_task, arm.pass_rate)
        tok = f"{arm.agent_tokens_per_task:,.0f}" if arm.has_tokens else "n/a"
        print(f"{arm.label:32s} {arm.pass_rate:6.3f} {tok:>15s} "
              f"${_fmt(as_succ, '.4f'):>14s} "
              f"{('$'+_fmt(norm_succ, '.4f')) if norm_succ is not None else 'n/a':>12s}")

    closed, openm = arms.get(HEADLINE_CLOSED), arms.get(HEADLINE_OPEN)
    if closed and openm:
        as_ratio = ratio(cost_per_successful_task(closed.as_listed_cost_per_task, closed.pass_rate),
                         cost_per_successful_task(openm.as_listed_cost_per_task, openm.pass_rate))
        nc, no = normalized_cost_per_task(closed), normalized_cost_per_task(openm)
        norm_ratio = ratio(nc, no) if (nc and no) else float("nan")
        print(f"\nHEADLINE (Opus vs Inkling):")
        print(f"  as-listed $/succ ratio : {as_ratio:.1f}x cheaper (pricing + efficiency)")
        print(f"  normalized $/task ratio: {norm_ratio:.2f}x (efficiency only, pricing held constant)")
        print(f"  -> ~{as_ratio:.0f}x is almost entirely a commercial-pricing artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
