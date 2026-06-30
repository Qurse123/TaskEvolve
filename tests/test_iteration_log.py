"""Tests for iterator_agent.iteration_log — the per-iteration record sink."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from iterator_agent.iteration_log import (
    IterationRecord,
    append_changelog,
    write_artifact,
    write_record,
)


def _record(**overrides) -> IterationRecord:
    base = IterationRecord(
        iteration_id="iter_0001",
        timestamp="2026-06-18T12:00:00+00:00",
        harness_version="v0.2",
        changed_surface="system_prompt",
        changed_file="target_agent/prompts/system_prompt.j2",
        change_summary="Tighten tool-selection guidance.",
        reason_for_change="3 of 5 failures called the wrong tool first.",
        proxy_seeds=(1001, 1002),
        proxy_task_success_before_mean=0.583,
        proxy_task_success_before_std=0.083,
        proxy_task_success_after_mean=0.667,
        proxy_task_success_after_std=0.0,
        cost_per_successful_task_before=0.083,
        cost_per_successful_task_after=0.071,
        accepted_or_rejected="accepted",
        reason_accepted_or_rejected="both proxy runs improved within guardrails",
    )
    return replace(base, **overrides) if overrides else base


def test_write_record_creates_folder_and_json(tmp_path: Path) -> None:
    # Arrange
    record = _record()

    # Act
    path = write_record(record, logs_root=tmp_path)

    # Assert
    assert path == tmp_path / "iter_0001" / "iteration.json"
    assert path.exists()


def test_write_record_round_trips_all_fields(tmp_path: Path) -> None:
    # Arrange
    record = _record()

    # Act
    path = write_record(record, logs_root=tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    # Assert
    assert loaded["iteration_id"] == "iter_0001"
    assert loaded["harness_version"] == "v0.2"
    assert loaded["changed_file"] == "target_agent/prompts/system_prompt.j2"
    assert loaded["proxy_seeds"] == [1001, 1002]
    assert loaded["proxy_task_success_after_mean"] == 0.667
    assert loaded["cost_per_successful_task_after"] == 0.071
    assert loaded["accepted_or_rejected"] == "accepted"


def test_write_record_round_trips_model_provenance(tmp_path: Path) -> None:
    # Arrange: provenance of which models were in play this iteration.
    record = _record(editor_model="gpt-4.1", agent_model="gpt-4.1-mini")

    # Act
    loaded = json.loads(
        write_record(record, logs_root=tmp_path).read_text(encoding="utf-8")
    )

    # Assert
    assert loaded["editor_model"] == "gpt-4.1"
    assert loaded["agent_model"] == "gpt-4.1-mini"


def test_model_provenance_defaults_to_empty(tmp_path: Path) -> None:
    loaded = json.loads(
        write_record(_record(), logs_root=tmp_path).read_text(encoding="utf-8")
    )

    assert loaded["editor_model"] == ""
    assert loaded["agent_model"] == ""


def test_write_record_keeps_optional_cost_none(tmp_path: Path) -> None:
    # Arrange
    record = _record(cost_per_successful_task_after=None)

    # Act
    loaded = json.loads(
        write_record(record, logs_root=tmp_path).read_text(encoding="utf-8")
    )

    # Assert
    assert loaded["cost_per_successful_task_after"] is None


def test_append_changelog_accepted_goes_to_accepted_file(tmp_path: Path) -> None:
    # Arrange
    record = _record()

    # Act
    path = append_changelog(record, experiments_dir=tmp_path)

    # Assert
    assert path == tmp_path / "accepted_changes.md"
    text = path.read_text(encoding="utf-8")
    assert "iter_0001" in text
    assert "target_agent/prompts/system_prompt.j2" in text
    assert "Tighten tool-selection guidance." in text


def test_append_changelog_rejected_goes_to_rejected_file(tmp_path: Path) -> None:
    # Arrange
    record = _record(
        accepted_or_rejected="rejected",
        reason_accepted_or_rejected="run 2 crossed the success floor",
    )

    # Act
    path = append_changelog(record, experiments_dir=tmp_path)

    # Assert
    assert path == tmp_path / "rejected_changes.md"
    text = path.read_text(encoding="utf-8")
    assert "run 2 crossed the success floor" in text
    assert not (tmp_path / "accepted_changes.md").exists()


def test_append_changelog_appends_not_overwrites(tmp_path: Path) -> None:
    # Arrange
    first = _record(iteration_id="iter_0001")
    second = _record(iteration_id="iter_0002")

    # Act
    append_changelog(first, experiments_dir=tmp_path)
    path = append_changelog(second, experiments_dir=tmp_path)

    # Assert
    text = path.read_text(encoding="utf-8")
    assert "iter_0001" in text
    assert "iter_0002" in text


def test_append_changelog_rejects_unknown_verdict(tmp_path: Path) -> None:
    # Arrange
    record = _record(accepted_or_rejected="maybe")

    # Act / Assert
    with pytest.raises(ValueError):
        append_changelog(record, experiments_dir=tmp_path)


def test_write_artifact_writes_into_iteration_folder(tmp_path: Path) -> None:
    # Arrange
    write_record(_record(), logs_root=tmp_path)

    # Act
    path = write_artifact(
        "iter_0001", "diagnosis.md", "# Root cause\nWrong tool first.", logs_root=tmp_path
    )

    # Assert
    assert path == tmp_path / "iter_0001" / "diagnosis.md"
    assert path.read_text(encoding="utf-8").startswith("# Root cause")
