"""Edit guard — the iterator's safety gate ("Allowed Change Check").

A proposed change is allowed ONLY if its target path is in the explicit
allow-list (`allowed_edits.yaml`). Everything else is rejected; a path under a
declared frozen prefix is rejected with a clearer "frozen" reason. This runs
BEFORE any edit is applied, so the iterator can never touch the benchmark,
evaluator, splits, results ledger, or its own allow-list
(experiment.md §12/§19; CLAUDE.md "Hard Constraints").
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

DEFAULT_POLICY_PATH = Path(__file__).parent / "allowed_edits.yaml"


@dataclass(frozen=True)
class EditPolicy:
    """The allowed edit surface: an exact allow-list plus frozen prefixes."""

    allowed_paths: frozenset[str]
    forbidden_prefixes: tuple[str, ...]


@dataclass(frozen=True)
class EditDecision:
    """The verdict for one proposed edit target."""

    target: str
    allowed: bool
    reason: str


def _normalize(target: Union[str, Path]) -> str:
    """Return a repo-relative POSIX path, collapsing ``.``/``..`` segments."""
    return os.path.normpath(str(target)).replace(os.sep, "/")


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> EditPolicy:
    """Load the edit policy from a YAML allow-list file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = data.get("allowed_paths") or []
    forbidden = data.get("forbidden_paths") or []
    return EditPolicy(
        allowed_paths=frozenset(_normalize(p) for p in allowed),
        forbidden_prefixes=tuple(_normalize(p) for p in forbidden),
    )


def evaluate(target: Union[str, Path], policy: EditPolicy) -> EditDecision:
    """Decide whether ``target`` may be edited, with a human-readable reason."""
    raw = str(target)
    norm = _normalize(target)

    # An absolute path or one that climbs out of the repo root is never allowed.
    if os.path.isabs(raw) or norm == ".." or norm.startswith("../"):
        return EditDecision(
            norm, False, f"rejected: path escapes the repo root: {raw!r}"
        )

    if norm in policy.allowed_paths:
        return EditDecision(norm, True, "allowed: in the editable harness surface")

    for prefix in policy.forbidden_prefixes:
        # Normalize at comparison time so a hand-built policy with a trailing
        # slash (e.g. "vendor/tau2-bench/") matches the same as a loaded one.
        norm_prefix = _normalize(prefix)
        if norm == norm_prefix or norm.startswith(norm_prefix + "/"):
            return EditDecision(
                norm, False, f"rejected: frozen path (matches {prefix!r})"
            )

    return EditDecision(norm, False, "rejected: outside the allowed edit surface")
