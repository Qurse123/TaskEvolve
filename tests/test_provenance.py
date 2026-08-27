"""$0 unit tests for scripts.provenance (arm grouping from results.csv)."""
from __future__ import annotations

from pathlib import Path

from scripts.provenance import render_markdown, summarize

_CSV = (
    "run_id,split,domain,agent_model,harness_version,num_tasks,num_passed,"
    "pass_rate,total_cost_usd,cost_per_successful_task,mean_reward,generated_at\n"
    "a1,proxy,retail,gpt-4.1,v0.1,12,6,0.5,0.6,0.1,0.5,2026-06-12T19:00:00+00:00\n"
    "a2,proxy,retail,gpt-4.1,v0.1,12,8,0.6667,0.5,0.06,0.6667,2026-06-12T19:30:00+00:00\n"
    "b1,validation,retail,anthropic/claude-opus-4-8,v0.2,35,30,0.8571,6.0,0.2,0.8571,"
    "2026-07-29T03:00:00+00:00\n"
    "c1,transfer_banking,banking_knowledge,openai/thinkingmachines/Inkling-Small,v0.1,"
    "30,0,0.0,,,0.0,2026-08-09T22:02:09+00:00\n"
)


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "results.csv"
    p.write_text(_CSV)
    return p


def test_groups_repeats_into_one_arm(tmp_path):
    arms = {(a.agent_model, a.harness_version, a.split): a for a in summarize(_write(tmp_path))}
    proxy = arms[("gpt-4.1", "v0.1", "proxy")]
    assert proxy.n_repeats == 2
    assert proxy.num_tasks == "12"
    assert abs(proxy.pass_rate_mean - (0.5 + 0.6667) / 2) < 1e-9
    assert proxy.date_first == "2026-06-12" and proxy.date_last == "2026-06-12"


def test_distinct_arms_separated_and_missing_cost_tolerated(tmp_path):
    arms = summarize(_write(tmp_path))
    keys = {(a.agent_model, a.harness_version, a.split) for a in arms}
    assert ("anthropic/claude-opus-4-8", "v0.2", "validation") in keys
    # a row with empty pass_rate stays valid (0.0 parses fine here)
    banking = next(a for a in arms if a.split == "transfer_banking")
    assert banking.n_repeats == 1


def test_render_markdown_has_header_and_rows(tmp_path):
    md = render_markdown(summarize(_write(tmp_path)))
    assert md.startswith("| agent_model | harness | split |")
    assert "gpt-4.1" in md and "claude-opus-4-8" in md
