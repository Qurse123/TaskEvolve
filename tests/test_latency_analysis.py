"""$0 unit tests for scripts/latency_analysis.py — synthetic transcript dicts only.

Run:  python -m pytest tests/test_latency_analysis.py -q
No network, no disk fixtures required for the core aggregation tests.
"""
from __future__ import annotations

import json

from scripts.latency_analysis import (
    Arm,
    LatencyStats,
    aggregate_durations,
    analyze,
    arm_label,
    load_arms_from_csv,
    model_short,
    summarize_arm,
    transcript_duration,
)


# --------------------------- transcript_duration --------------------------- #
def test_transcript_duration_reads_float():
    assert transcript_duration({"duration": 39.8}) == 39.8


def test_transcript_duration_reads_int_as_float():
    v = transcript_duration({"duration": 12})
    assert v == 12.0 and isinstance(v, float)


def test_transcript_duration_missing_is_none():
    assert transcript_duration({"start_time": "x"}) is None


def test_transcript_duration_explicit_none():
    assert transcript_duration({"duration": None}) is None


def test_transcript_duration_rejects_bool_and_nonfinite():
    assert transcript_duration({"duration": True}) is None
    assert transcript_duration({"duration": float("nan")}) is None
    assert transcript_duration({"duration": float("inf")}) is None


# --------------------------- aggregate_durations --------------------------- #
def test_aggregate_basic_stats():
    s = aggregate_durations([10.0, 20.0, 30.0])
    assert s.n == 3
    assert s.mean == 20.0
    assert s.median == 20.0
    assert abs(s.std - 10.0) < 1e-9  # sample std of 10,20,30
    assert s.has_data


def test_aggregate_drops_none_entries():
    s = aggregate_durations([None, 5.0, None, 15.0])
    assert s.n == 2
    assert s.mean == 10.0
    assert s.median == 10.0


def test_aggregate_single_point_zero_std():
    s = aggregate_durations([7.0])
    assert s.n == 1 and s.mean == 7.0 and s.std == 0.0 and s.median == 7.0


def test_aggregate_median_even_count():
    assert aggregate_durations([1.0, 2.0, 3.0, 4.0]).median == 2.5


def test_missing_duration_arm_reports_no_data_not_zero():
    """The key contract: an all-missing arm is 'no latency data', never 0.0."""
    s = aggregate_durations([None, None, None])
    assert s == LatencyStats(0, None, None, None)
    assert not s.has_data
    assert s.mean is None  # explicitly NOT 0.0


def test_aggregate_empty():
    s = aggregate_durations([])
    assert s.n == 0 and not s.has_data and s.mean is None


# --------------------------- naming helpers -------------------------------- #
def test_model_short_strips_provider_prefix():
    assert model_short("openai/thinkingmachines/Inkling-Small") == "Inkling-Small"
    assert model_short("anthropic/claude-opus-4-8") == "claude-opus-4-8"
    assert model_short("gpt-4.1") == "gpt-4.1"


def test_arm_label_format():
    lbl = arm_label("openai/thinkingmachines/Inkling", "v0.1", "test_retail", "retail")
    assert lbl == "Inkling / v0.1 / test_retail (retail)"


# --------------------------- summarize_arm (synthetic disk) ---------------- #
def _write_task(run_dir, tid, *, duration, turns=5, tools=2, passed=True):
    verdict = {
        "task_id": tid, "turn_count": turns, "tool_call_count": tools,
        "passed": passed, "cost_usd": 0.05,
    }
    (run_dir / f"task_{tid}.json").write_text(json.dumps(verdict), encoding="utf-8")
    # messages file always written; duration may be None to simulate old transcript
    (run_dir / f"task_{tid}_messages.json").write_text(
        json.dumps({"task_id": tid, "duration": duration, "messages": []}),
        encoding="utf-8",
    )


def test_summarize_arm_aggregates_disk(tmp_path):
    logs = tmp_path / "logs"
    run = logs / "test_retail_20260101_000000"
    run.mkdir(parents=True)
    _write_task(run, "1", duration=10.0, turns=4, tools=1)
    _write_task(run, "2", duration=20.0, turns=6, tools=3)
    arm = Arm("openai/thinkingmachines/Inkling", "v0.1", "test_retail", "retail",
              run_ids=[run.name])
    out = summarize_arm(arm, logs_root=logs)
    assert out.tasks_total == 2
    assert out.tasks_with_duration == 2
    assert out.latency.mean == 15.0
    assert out.mean_turns == 5.0
    assert out.mean_tool_calls == 2.0
    assert out.coverage_ok


def test_summarize_arm_partial_and_missing_duration(tmp_path):
    logs = tmp_path / "logs"
    run = logs / "validation_20260101_000000"
    run.mkdir(parents=True)
    _write_task(run, "1", duration=None)   # old transcript, no duration
    _write_task(run, "2", duration=30.0)
    arm = Arm("gpt-4.1", "v0.1", "validation", "retail", run_ids=[run.name])
    out = summarize_arm(arm, logs_root=logs)
    assert out.tasks_total == 2
    assert out.tasks_with_duration == 1
    assert out.latency.mean == 30.0  # only the covered task
    assert not out.coverage_ok


def test_summarize_arm_all_missing_duration_reports_no_data(tmp_path):
    logs = tmp_path / "logs"
    run = logs / "validation_20260101_000000"
    run.mkdir(parents=True)
    _write_task(run, "1", duration=None)
    _write_task(run, "2", duration=None)
    arm = Arm("gpt-4.1", "v0.1", "validation", "retail", run_ids=[run.name])
    out = summarize_arm(arm, logs_root=logs)
    assert out.tasks_total == 2
    assert out.tasks_with_duration == 0
    assert not out.latency.has_data
    assert out.latency.mean is None  # never fabricated as 0.0


def test_summarize_arm_skips_messages_json_as_verdict(tmp_path):
    """task_*_messages.json must not be counted as a verdict task."""
    logs = tmp_path / "logs"
    run = logs / "test_retail_20260101_000000"
    run.mkdir(parents=True)
    _write_task(run, "1", duration=10.0)
    out = summarize_arm(
        Arm("m", "v0.1", "test_retail", "retail", run_ids=[run.name]), logs_root=logs
    )
    assert out.tasks_total == 1  # not 2


# --------------------------- CSV grouping ---------------------------------- #
def test_load_arms_from_csv_groups_and_parses(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,split,domain,agent_model,harness_version,num_tasks,num_passed,"
        "pass_rate,total_cost_usd,cost_per_successful_task,mean_reward,generated_at\n"
        "r1,test_retail,retail,openai/thinkingmachines/Inkling,v0.1,40,34,0.85,2.0,0.06,0.85,t\n"
        "r2,test_retail,retail,openai/thinkingmachines/Inkling,v0.1,40,36,0.90,2.1,0.058,0.90,t\n"
        "r3,validation,retail,gpt-4.1,v0.1,35,28,0.80,2.2,,0.80,t\n",
        encoding="utf-8",
    )
    arms = load_arms_from_csv(csv_path)
    assert len(arms) == 2
    ink = arms[("openai/thinkingmachines/Inkling", "v0.1", "test_retail", "retail")]
    assert ink.run_ids == ["r1", "r2"]
    assert ink.pass_rates == [0.85, 0.90]
    assert ink.cost_per_succ == [0.06, 0.058]
    gpt = arms[("gpt-4.1", "v0.1", "validation", "retail")]
    assert gpt.cost_per_succ == []  # blank cost cell ignored, not 0


def test_analyze_end_to_end_synthetic(tmp_path):
    logs = tmp_path / "logs"
    run = logs / "test_retail_20260101_000000"
    run.mkdir(parents=True)
    _write_task(run, "1", duration=10.0)
    _write_task(run, "2", duration=20.0)
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,split,domain,agent_model,harness_version,num_tasks,num_passed,"
        "pass_rate,total_cost_usd,cost_per_successful_task,mean_reward,generated_at\n"
        f"{run.name},test_retail,retail,m,v0.1,2,2,1.0,0.1,0.05,1.0,t\n",
        encoding="utf-8",
    )
    results = analyze(csv_path=csv_path, logs_root=logs)
    assert len(results) == 1
    assert results[0].latency.mean == 15.0
