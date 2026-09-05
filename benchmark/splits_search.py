"""Generate the iterator's search splits.

The iterator selects harness edits by measuring them on tasks. Whatever tasks it
selects against get fitted, so it must never see the held-out ``test_*`` splits
that Figure 1 reports. It also must not select against a distribution harder than
the one it is scored on: the previous search set scored 0.4681 for Opus while the
held-out set scores about 0.70, so edits were being chosen for the wrong regime.

These splits mirror the held-out proportions (40 / 20 / 40) at half the size, and
are drawn from each domain's full train pool, which is the union of every
non-test split we own. Sampling is seeded so the splits are reproducible.

Run:  python -m benchmark.splits_search
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SPLITS = Path("benchmark/splits")
SEED = 20260905

# domain -> (train-pool splits to draw from, held-out split, how many to sample)
DOMAINS = {
    "retail": (["proxy", "train_distill_retail", "validation"], "test_retail", 20),
    "airline": (["eval_airline", "train_distill_airline"], "test_airline", 10),
    "telecom": (["eval_telecom", "train_distill_telecom"], "test_telecom", 20),
}


def _load(name: str) -> list[str]:
    data = json.loads((SPLITS / f"{name}.json").read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("task_ids", data)


def generate() -> None:
    rng = random.Random(SEED)
    for domain, (pool_names, test_name, k) in DOMAINS.items():
        pool = sorted({t for n in pool_names for t in _load(n)})
        held_out = set(_load(test_name))

        overlap = pool & held_out if isinstance(pool, set) else set(pool) & held_out
        if overlap:
            raise AssertionError(
                f"{domain}: train pool overlaps {test_name} on {sorted(overlap)[:5]}"
            )
        if len(pool) < k:
            raise AssertionError(f"{domain}: pool has {len(pool)}, need {k}")

        picked = sorted(rng.sample(pool, k), key=lambda t: (len(t), t))
        out = SPLITS / f"search_{domain}.json"
        out.write_text(json.dumps(picked, indent=2), encoding="utf-8")
        print(f"  search_{domain}: {k} of {len(pool)} pool tasks -> {out}")


if __name__ == "__main__":
    generate()
