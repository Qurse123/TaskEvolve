"""Per-task statistics for the held-out test splits, cached as one CSV.

The paper's detail figures all read the same rows: one row per task attempt
across every seed of every system on the official test splits. Reading the
transcripts once and caching keeps each plotting script cheap.

Run:  uv run python -m scripts.build_test_task_stats
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

RESULTS = Path("experiments/results.csv")
LOGS = Path("experiments/logs")
OUT = Path("experiments/analysis/test_task_stats.csv")

SPLITS = ["test_retail", "test_airline", "test_telecom"]

# label -> (agent_model, harness_version); matches scripts/plot_paper_fig1_frontier.SYSTEMS
SYSTEMS = {
    "opus_static": ("anthropic/claude-opus-4-8", "v0.1"),
    "opus_iterator": ("anthropic/claude-opus-4-8", "v0.5"),
    "inkling_base": ("openai/thinkingmachines/Inkling-Small", "v0.1"),
    "inkling_tuned": ("openai/armd-inkling-small-tuned", "v0.1"),
}

HANDOFF_TOOL = "transfer_to_human_agents"

FIELDS = ["split", "system", "run_id", "task_id", "reward", "passed",
          "agent_cost_usd", "user_cost_usd", "input_tokens", "output_tokens",
          "assistant_messages", "tool_calls", "handoff", "handoff_message_index",
          "termination_reason"]


def valid_runs(rows, split, model, harness):
    """Run ids for one system on one split, under the paper's validity rule."""
    out = []
    for r in rows:
        if r["split"] != split or r["agent_model"] != model:
            continue
        if r["harness_version"] != harness:
            continue
        total = float(r["total_cost_usd"] or 0.0)
        if total == 0.0 and float(r["pass_rate"] or 0.0) == 0.0:
            continue
        out.append(r["run_id"])
    return out


def task_row(path: Path) -> dict:
    sim = json.loads(path.read_text())
    assistant = [m for m in sim["messages"] if m["role"] == "assistant"]
    usage = [m.get("usage") or {} for m in assistant]
    handoff = [i for i, m in enumerate(assistant)
               for call in (m.get("tool_calls") or []) if call["name"] == HANDOFF_TOOL]
    reward = (sim.get("reward_info") or {}).get("reward") or 0.0
    return {
        "task_id": sim["task_id"],
        "reward": reward,
        "passed": int(reward >= 1.0),
        "agent_cost_usd": sim.get("agent_cost") or 0.0,
        "user_cost_usd": sim.get("user_cost") or 0.0,
        "input_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("completion_tokens", 0) for u in usage),
        "assistant_messages": len(assistant),
        "tool_calls": sum(len(m.get("tool_calls") or []) for m in assistant),
        "handoff": int(bool(handoff)),
        "handoff_message_index": handoff[0] if handoff else "",
        "termination_reason": sim.get("termination_reason") or "",
    }


def main() -> int:
    rows = list(csv.DictReader(RESULTS.open()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for split in SPLITS:
            for system, (model, harness) in SYSTEMS.items():
                for run_id in valid_runs(rows, split, model, harness):
                    for path in sorted((LOGS / run_id).glob("task_*_messages.json")):
                        record = task_row(path)
                        record.update(split=split, system=system, run_id=run_id)
                        writer.writerow(record)
                        written += 1
    print(f"{written} task rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
