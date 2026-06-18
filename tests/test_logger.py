"""Tests for results.logger transcript path + per-task cost levers."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.adapter import EvalResult
from results.logger import log_task, start_run, transcript_path


def _result(task_id: str = "t1") -> EvalResult:
    return EvalResult(
        task_id=task_id,
        domain="retail",
        split="proxy",
        agent_model="gpt-4.1",
        reward=1.0,
        passed=True,
        agent_cost=0.05,
        termination_reason="user_stop",
        seed=1,
        turn_count=7,
        tool_call_count=3,
        timestamp="2026-06-18T00:00:00+00:00",
    )


def test_transcript_path_sits_beside_task_file(tmp_path: Path) -> None:
    # Arrange
    run = start_run("proxy", logs_root=tmp_path)

    # Act: an id with filesystem-unsafe characters is sanitized like the verdict file.
    path = transcript_path(run, "retail[0]")

    # Assert
    assert path == run.run_dir / "task_retail_0_messages.json"


def test_log_task_records_cost_levers(tmp_path: Path) -> None:
    # Arrange
    run = start_run("proxy", logs_root=tmp_path)

    # Act
    path = log_task(run, _result())
    record = json.loads(path.read_text(encoding="utf-8"))

    # Assert
    assert record["turn_count"] == 7
    assert record["tool_call_count"] == 3
