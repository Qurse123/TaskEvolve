"""TAU2 Opus transcripts -> normalized SFT examples (passed-only, deduped).

Pure data transform: no Tinker dependency, so it is fully unit-testable at $0.
Tokenization/rendering into Tinker datums happens in arm_d/train.py.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PASS_THRESHOLD = 0.5
TARGET_ROLES = {"assistant"}  # assistant turns (incl. tool_calls) are the targets


@dataclass(frozen=True)
class TrainingExample:
    messages: list[dict]
    target_mask: list[bool]
    task_id: str
    domain: str
    source_run: str


def _reward(rec: dict) -> float:
    ri = rec.get("reward_info") or {}
    return ri.get("reward", rec.get("reward")) or 0.0


def _normalize(m: dict) -> dict:
    out = {"role": m.get("role"), "content": m.get("content"),
           "tool_calls": m.get("tool_calls")}
    # TAU2 tool-result messages store the responded tool_call id in `id`;
    # preserve it as tool_call_id so the renderer can link result -> call.
    if m.get("role") == "tool":
        out["tool_call_id"] = m.get("tool_call_id") or m.get("id")
    return out


def _hash(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def build_examples(transcript_dirs: list[Path], *, domains: dict[str, str]) -> list[TrainingExample]:
    seen: set[str] = set()
    out: list[TrainingExample] = []
    for d in transcript_dirs:
        domain = domains[Path(d).name]
        for f in sorted(Path(d).glob("task_*_messages.json")):
            rec = json.loads(f.read_text())
            if _reward(rec) < PASS_THRESHOLD:
                continue
            msgs = [_normalize(m) for m in rec.get("messages", [])]
            if not any(m["role"] in TARGET_ROLES for m in msgs):
                continue
            h = _hash(msgs)
            if h in seen:
                continue
            seen.add(h)
            mask = [m["role"] in TARGET_ROLES for m in msgs]
            out.append(TrainingExample(msgs, mask, rec.get("task_id", f.stem), domain, Path(d).name))
    return out


def balance(examples: list[TrainingExample], *, seed: int = 0, k: int = 2) -> list[TrainingExample]:
    by_dom: dict[str, list[TrainingExample]] = defaultdict(list)
    for e in examples:
        by_dom[e.domain].append(e)
    if not by_dom:
        return []
    cap = min(len(v) for v in by_dom.values()) * k
    rng = random.Random(seed)
    out: list[TrainingExample] = []
    for v in by_dom.values():
        vv = v[:]
        rng.shuffle(vv)
        out.extend(vv[:cap])
    rng.shuffle(out)
    return out


def write_manifest(examples: list[TrainingExample], path: Path) -> None:
    rows = [{"task_id": e.task_id, "domain": e.domain, "source_run": e.source_run,
             "sha256": _hash(e.messages)} for e in examples]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2))


def examples_from_distill_pointer(
    pointer_path: Path, *, seed: int = 0, k: int = 2
) -> list[TrainingExample]:
    """Balanced SFT examples described by a ``run_distillation`` pointer file.

    The pointer (``experiments/arm_d_distill_runs.json``) has the shape
    ``{"runs": {run_id: domain}, "logs_root": ...}``. Each run's transcript dir
    is ``logs_root/run_id`` and ``build_examples`` keys ``domains`` by that dir's
    basename (== ``run_id``), so passing ``dict(runs)`` maps every run to its
    domain. Runs ``build_examples`` (passed-only, deduped) then ``balance``.

    This is the single transcript->examples transform shared by
    ``scripts.build_distill_dataset`` (the $0 build + audit step) and the
    training step, so neither reimplements the pointer plumbing.
    """
    pointer = json.loads(Path(pointer_path).read_text())
    runs: dict[str, str] = pointer["runs"]
    logs_root = Path(pointer.get("logs_root", "experiments/logs"))
    transcript_dirs = [logs_root / run_id for run_id in runs]
    examples = build_examples(transcript_dirs, domains=dict(runs))
    return balance(examples, seed=seed, k=k)
