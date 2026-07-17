"""Candidate preflight — fail fast on a broken edit BEFORE any eval spend.

AutoPK port ("validate before eval"): the highest-value iteration-speed lever is
rejecting a structurally broken candidate in one cheap check instead of paying
for a full proxy run that crashes every task (experiment.md §20 rejection rule 5
"the change breaks the harness"; §21 step 5 "if the change is invalid, reject
immediately"). The July 2026 Arm B run burned five proxy runs on exactly this.

The check itself lives in :mod:`iterator_agent.preflight_check` and executes in
a **subprocess** with cwd at the candidate tree, so the edited modules are
imported fresh — the parent process's module cache never sees them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

PREFLIGHT_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class PreflightResult:
    """The $0 structural verdict on a candidate edit."""

    passed: bool
    reason: str


def run_preflight(
    repo_root: Union[str, Path],
    target_file: str,
    *,
    timeout_s: float = PREFLIGHT_TIMEOUT_S,
) -> PreflightResult:
    """Exercise the edited tree in a fresh interpreter; return the verdict.

    ``target_file`` is the file the candidate edit touched (recorded in the
    reason for auditability); the check always exercises every edit surface,
    since the surfaces interact (templates render into the agent the harness
    builds messages for).
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "iterator_agent.preflight_check"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return PreflightResult(
            False, f"preflight timed out after {timeout_s:.0f}s (edit: {target_file})"
        )

    verdict = _parse_verdict(proc.stdout)
    if verdict is not None:
        return verdict
    tail = (proc.stderr or proc.stdout or "").strip()[-400:]
    return PreflightResult(
        False,
        f"preflight subprocess produced no verdict (exit {proc.returncode}, "
        f"edit: {target_file}): {tail}",
    )


def _parse_verdict(stdout: str) -> "PreflightResult | None":
    """Parse the JSON verdict from the subprocess's last stdout line."""
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "passed" in data:
            return PreflightResult(bool(data["passed"]), str(data.get("reason", "")))
    return None
