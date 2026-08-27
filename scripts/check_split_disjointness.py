"""Data-contamination audit: prove the Arm D training task set is disjoint from
every evaluation task set (Tier 1 paper-validity item #5, concern A).

Arm D fine-tuned Inkling-Small on Opus-distilled traces drawn from the
`train_distill_*` splits, then evaluated on the eval splits (`validation` /
`eval_airline` / `eval_telecom` / `transfer_banking`). This script proves, at
$0 (no network, no paid calls), that no task can appear in both roles.

Task ids are only unique WITHIN a domain: the numeric id "7" is a retail task in
`validation.json`, an airline task in `eval_airline.json`, and an airline task
in `train_distill_airline.json` -- three different tasks. So every id is keyed
by its `(domain, id)` pair (domain comes from the split file, cf.
`scripts.plot_arm_d.SPLIT_TO_DOMAIN`); a naive id-only key would report false
cross-domain "overlaps".

Prints a train x eval overlap matrix, checks the SFT manifest (what was actually
trained on) against the eval sets, and exits nonzero if any overlap exists.

Run:  python -m scripts.check_split_disjointness
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

SPLITS_DIR = Path("benchmark/splits")
MANIFEST = Path("experiments/arm_d_sft_manifest.json")

# Domain each split file's ids belong to. train_distill_<d> maps to <d>; the
# eval / proxy / smoke files are mapped explicitly (mirrors plot_arm_d.py).
DOMAIN_BY_SPLIT: Dict[str, str] = {
    "validation": "retail",
    "proxy": "retail",
    "test_retail": "retail",
    "test_airline": "airline",
    "test_telecom": "telecom",
    "smoke": "mock",
    "eval_airline": "airline",
    "eval_telecom": "telecom",
    "transfer_banking": "banking",
    "train_distill_retail": "retail",
    "train_distill_airline": "airline",
    "train_distill_telecom": "telecom",
}

# The two roles whose disjointness the audit guarantees.
TRAIN_SPLITS = ["train_distill_retail", "train_distill_airline",
                "train_distill_telecom"]
EVAL_SPLITS = ["validation", "eval_airline", "eval_telecom",
               "transfer_banking"]

Key = Tuple[str, str]  # (domain, task_id)


def domain_for(split_name: str) -> str:
    """Domain for a split file stem (train_distill_<d> derives its suffix)."""
    if split_name in DOMAIN_BY_SPLIT:
        return DOMAIN_BY_SPLIT[split_name]
    if split_name.startswith("train_distill_"):
        return split_name[len("train_distill_"):]
    raise KeyError(f"unknown split '{split_name}' -- add it to DOMAIN_BY_SPLIT")


def keyed_ids(domain: str, raw_ids: List[str]) -> Set[Key]:
    """Turn a flat list of task ids into a set of (domain, id) keys."""
    return {(domain, str(i)) for i in raw_ids}


def load_split(path: Path) -> Set[Key]:
    """Load one split file into a set of (domain, id) keys."""
    raw = json.loads(path.read_text())
    return keyed_ids(domain_for(path.stem), raw)


def load_all_splits(splits_dir: Path = SPLITS_DIR) -> Dict[str, Set[Key]]:
    """Load every `*.json` split into {split_name: {(domain, id), ...}}."""
    out: Dict[str, Set[Key]] = {}
    for path in sorted(splits_dir.glob("*.json")):
        out[path.stem] = load_split(path)
    return out


def load_manifest(path: Path = MANIFEST) -> Set[Key]:
    """Load the SFT manifest into the set of (domain, id) keys trained on."""
    entries = json.loads(path.read_text())
    return {(e["domain"], str(e["task_id"])) for e in entries}


def build_overlap_matrix(
    rows: Dict[str, Set[Key]], cols: Dict[str, Set[Key]]
) -> Dict[Tuple[str, str], Set[Key]]:
    """|intersection| set for every (row split, col split) pair."""
    return {
        (r, c): rows[r] & cols[c]
        for r in rows
        for c in cols
    }


def find_overlaps(
    train: Dict[str, Set[Key]], evals: Dict[str, Set[Key]]
) -> Dict[Tuple[str, str], Set[Key]]:
    """Non-empty train x eval intersections (the disjointness violations)."""
    return {
        pair: shared
        for pair, shared in build_overlap_matrix(train, evals).items()
        if shared
    }


def _print_matrix(rows: Dict[str, Set[Key]], cols: Dict[str, Set[Key]]) -> None:
    col_names = list(cols)
    width = max([len(r) for r in rows] + [len(c) for c in col_names] + [12])
    header = "".ljust(width) + "".join(c.rjust(width + 2) for c in col_names)
    print(header)
    matrix = build_overlap_matrix(rows, cols)
    for r in rows:
        cells = "".join(str(len(matrix[(r, c)])).rjust(width + 2)
                        for c in col_names)
        print(r.ljust(width) + cells)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    splits = load_all_splits(args.splits_dir)
    train = {n: splits[n] for n in TRAIN_SPLITS if n in splits}
    evals = {n: splits[n] for n in EVAL_SPLITS if n in splits}

    train_union: Set[Key] = set().union(*train.values()) if train else set()
    eval_union: Set[Key] = set().union(*evals.values()) if evals else set()

    print("Split sizes (unique (domain, id) keys):")
    for name in sorted(splits):
        print(f"  {name:<24} {len(splits[name]):>4}")
    print(f"\ntrain_distill union: {len(train_union)} unique tasks")
    print(f"eval union:          {len(eval_union)} unique tasks")

    print("\nOverlap matrix  (rows = train_distill, cols = eval; "
          "cells = |intersection| by (domain, id)):")
    _print_matrix(train, evals)

    violations = find_overlaps(train, evals)

    # SFT manifest: what was ACTUALLY trained on.
    manifest = load_manifest(args.manifest)
    manifest_not_in_train = manifest - train_union
    manifest_in_eval = manifest & eval_union
    print(f"\nSFT manifest: {len(manifest)} unique (domain, id) trained tasks")
    print(f"  manifest tasks NOT in any train_distill split: "
          f"{len(manifest_not_in_train)}")
    print(f"  manifest tasks that appear in an eval split:   "
          f"{len(manifest_in_eval)}")

    ok = True
    if violations:
        ok = False
        print("\nFAIL: train_distill and eval splits share tasks:")
        for (r, c), shared in sorted(violations.items()):
            print(f"  {r} INT {c}: {sorted(shared)}")
    if manifest_not_in_train:
        ok = False
        print("\nFAIL: SFT manifest contains tasks outside train_distill:")
        print(f"  {sorted(manifest_not_in_train)}")
    if manifest_in_eval:
        ok = False
        print("\nFAIL: SFT manifest overlaps eval splits (leakage):")
        print(f"  {sorted(manifest_in_eval)}")

    assert not violations, "train_distill and eval splits are not disjoint"
    assert not manifest_in_eval, "SFT manifest overlaps eval splits"
    assert not manifest_not_in_train, "SFT manifest strays outside train_distill"

    if ok:
        print("\nPASS: train_distill INT eval == empty; "
              "SFT manifest subset of train_distill and disjoint from eval.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
