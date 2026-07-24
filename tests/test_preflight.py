"""Unit tests for the iterator's $0 candidate preflight (AutoPK port: validate
before eval — a structurally broken edit must be rejected before any proxy spend).

The preflight runs in a subprocess against a candidate working tree so the
edited modules are imported fresh (no stale-module leakage into the parent).

Run from the repo root:
    python -m pytest tests/test_preflight.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from iterator_agent.preflight import run_preflight

REPO_ROOT = Path(__file__).resolve().parent.parent


def _candidate_tree(tmp_path: Path) -> Path:
    """A minimal working tree the preflight subprocess can import from."""
    root = tmp_path / "tree"
    for pkg in ("target_agent", "iterator_agent"):
        shutil.copytree(
            REPO_ROOT / pkg,
            root / pkg,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    return root


def test_pristine_tree_passes(tmp_path: Path) -> None:
    root = _candidate_tree(tmp_path)

    result = run_preflight(root, "target_agent/harness.py")

    assert result.passed is True, result.reason


def test_broken_build_messages_fails(tmp_path: Path) -> None:
    # Reproduces the July 2026 failure: an edit that builds raw dicts instead of
    # tau2 Message models crashes every task at runtime. Preflight must catch it.
    root = _candidate_tree(tmp_path)
    (root / "target_agent" / "harness.py").write_text(
        "def build_messages(system_messages, history):\n"
        "    msgs = list(system_messages) + list(history)\n"
        "    if len(msgs) > 10:\n"
        "        summary = {'content': '[Summary: older messages omitted.]'}\n"
        "        return [summary] + msgs[-5:]\n"
        "    return msgs\n"
        "def filter_tools(tools):\n"
        "    return list(tools)\n",
        encoding="utf-8",
    )

    result = run_preflight(root, "target_agent/harness.py")

    assert result.passed is False
    assert "harness" in result.reason.lower() or "build_messages" in result.reason


def test_syntax_error_fails(tmp_path: Path) -> None:
    root = _candidate_tree(tmp_path)
    (root / "target_agent" / "harness.py").write_text(
        "def build_messages(:\n", encoding="utf-8"
    )

    result = run_preflight(root, "target_agent/harness.py")

    assert result.passed is False


def test_disallowed_model_fails(tmp_path: Path) -> None:
    # The editor's contract restricts routing to gpt-4.1 / gpt-4.1-mini.
    root = _candidate_tree(tmp_path)
    (root / "target_agent" / "model_routing.py").write_text(
        "def get_model(configured_model, history=None):\n"
        "    return 'gpt-3.5-turbo'\n",
        encoding="utf-8",
    )

    result = run_preflight(root, "target_agent/model_routing.py")

    assert result.passed is False
    assert "model" in result.reason.lower()


def test_routing_off_pool_on_later_turn_fails(tmp_path: Path) -> None:
    # Exp 2: get_model is exercised with a multi-turn history too, so a policy that
    # returns an off-pool model only on later turns is still caught at $0.
    root = _candidate_tree(tmp_path)
    (root / "target_agent" / "model_routing.py").write_text(
        "def get_model(configured_model, history=None):\n"
        "    if history:\n"
        "        return 'gpt-4o'  # off-pool once the conversation has turns\n"
        "    return configured_model\n",
        encoding="utf-8",
    )

    result = run_preflight(root, "target_agent/model_routing.py")

    assert result.passed is False
    assert "model" in result.reason.lower()


def test_broken_template_fails(tmp_path: Path) -> None:
    # StrictUndefined: a template referencing an undefined variable must fail.
    root = _candidate_tree(tmp_path)
    (root / "target_agent" / "prompts" / "system_prompt.j2").write_text(
        "{{ variable_that_does_not_exist }}", encoding="utf-8"
    )

    result = run_preflight(root, "target_agent/prompts/system_prompt.j2")

    assert result.passed is False
    assert "template" in result.reason.lower() or "render" in result.reason.lower()


def test_missing_function_fails(tmp_path: Path) -> None:
    root = _candidate_tree(tmp_path)
    (root / "target_agent" / "harness.py").write_text(
        "def build_messages(system_messages, history):\n"
        "    return list(system_messages) + list(history)\n",
        encoding="utf-8",
    )  # filter_tools dropped

    result = run_preflight(root, "target_agent/harness.py")

    assert result.passed is False
