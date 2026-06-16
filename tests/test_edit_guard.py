"""Unit tests for the iterator's edit guard — the safety gate that rejects any
proposed change outside the allowed harness surface.

Run from the repo root:
    python -m pytest tests/test_edit_guard.py
"""

from __future__ import annotations

import pytest

from iterator_agent.edit_guard import (
    EditPolicy,
    ForbiddenEditError,
    assert_allowed,
    evaluate,
    load_policy,
)

# A small hand-built policy keeps the logic tests independent of the real YAML.
POLICY = EditPolicy(
    allowed_paths=frozenset(
        {
            "target_agent/harness.py",
            "target_agent/prompts/system_prompt.j2",
        }
    ),
    forbidden_prefixes=(
        "vendor/tau2-bench/",
        "benchmark/splits/",
        "iterator_agent/allowed_edits.yaml",
    ),
)


def test_allowed_file_in_surface_is_allowed():
    # Arrange / Act
    decision = evaluate("target_agent/harness.py", POLICY)

    # Assert
    assert decision.allowed is True


def test_frozen_prefix_is_rejected_as_frozen():
    decision = evaluate("vendor/tau2-bench/src/tau2/evaluator/evaluator.py", POLICY)

    assert decision.allowed is False
    assert "frozen" in decision.reason.lower()


def test_unlisted_file_is_rejected_as_outside_surface():
    decision = evaluate("settings/config.py", POLICY)

    assert decision.allowed is False
    assert "surface" in decision.reason.lower()


def test_normalizes_leading_dot_slash():
    decision = evaluate("./target_agent/harness.py", POLICY)

    assert decision.allowed is True


def test_path_traversal_escaping_root_is_rejected():
    decision = evaluate("../secrets.txt", POLICY)

    assert decision.allowed is False
    assert "escape" in decision.reason.lower()


def test_absolute_path_is_rejected():
    decision = evaluate("/etc/passwd", POLICY)

    assert decision.allowed is False


def test_assert_allowed_raises_for_forbidden():
    with pytest.raises(ForbiddenEditError):
        assert_allowed("vendor/tau2-bench/src/tau2/orchestrator/orchestrator.py", POLICY)


def test_assert_allowed_returns_none_for_allowed():
    assert assert_allowed("target_agent/harness.py", POLICY) is None


def test_load_policy_reads_real_yaml():
    policy = load_policy()

    # The five editable harness files from CLAUDE.md must all be present.
    assert "target_agent/prompts/system_prompt.j2" in policy.allowed_paths
    assert "target_agent/prompts/policy_summary.j2" in policy.allowed_paths
    assert "target_agent/prompts/few_shot_examples.j2" in policy.allowed_paths
    assert "target_agent/harness.py" in policy.allowed_paths
    assert "target_agent/model_routing.py" in policy.allowed_paths
    # The frozen benchmark vendor tree is declared off-limits.
    assert any("vendor/tau2-bench" in p for p in policy.forbidden_prefixes)


def test_real_policy_freezes_the_guard_config_itself():
    # The iterator must not be able to edit its own allow-list.
    policy = load_policy()

    decision = evaluate("iterator_agent/allowed_edits.yaml", policy)

    assert decision.allowed is False
