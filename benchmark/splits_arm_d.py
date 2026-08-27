"""Generate the Arm D multi-domain train/held-out/transfer splits (deterministic).

Trained domains: retail, airline, telecom (each carved into a training pool and a
held-out eval). Transfer domain: banking_knowledge (never trained). Retail's held-out
eval is the frozen validation.json; retail's training pool is the retail-train tasks
in neither proxy nor validation (the 27 unused).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SPLITS_DIR = Path(__file__).parent / "splits"
# fraction of a trained non-retail domain's train tasks kept for held-out eval
HELDOUT_FRAC = 0.35
BANKING_TRANSFER_SIZE = 30


def _train_ids(domain: str) -> list[str]:
    mod = __import__(f"tau2.domains.{domain}.environment", fromlist=["get_tasks"])
    return [t.id for t in mod.get_tasks(task_split_name="train")]


def _write(ids: list[str], name: str) -> None:
    (SPLITS_DIR / f"{name}.json").write_text(json.dumps(ids, indent=2))


def generate_arm_d_splits(*, seed: int = 42) -> None:
    """Carve deterministic train/held-out/transfer splits across four domains.

    TAU2 task identity is ``(domain, id)``, not the raw ``id`` string alone,
    and every split is always run with its matching ``--domain``
    (``run_train_eval --split eval_airline --domain airline`` resolves via
    ``adapter._load_task(domain, task_id)``). So retail task "5" and airline
    task "5" are different tasks even though they share a literal ID string —
    a raw ID coincidentally reused across domains is not a train/eval leak,
    and carving each domain's own train/eval split independently is correct.
    Disjointness is checked *within* each domain (see
    ``tests/test_splits_arm_d.py``), not by unioning raw IDs across domains.
    """
    rng = random.Random(seed)
    # retail: training pool = train tasks in neither proxy nor validation
    retail = _train_ids("retail")
    proxy = (
        set(json.loads((SPLITS_DIR / "proxy.json").read_text()))
        if (SPLITS_DIR / "proxy.json").exists()
        else set()
    )
    val = set(json.loads((SPLITS_DIR / "validation.json").read_text()))
    retail_train_pool = [t for t in retail if t not in proxy and t not in val]
    _write(retail_train_pool, "train_distill_retail")

    # airline / telecom: shuffle, carve held-out then train pool — a clean
    # random sample per domain, independent of any other domain's IDs.
    for domain in ("airline", "telecom"):
        ids = _train_ids(domain)[:]
        rng.shuffle(ids)
        n_eval = max(1, int(len(ids) * HELDOUT_FRAC))
        _write(ids[:n_eval], f"eval_{domain}")
        _write(ids[n_eval:], f"train_distill_{domain}")

    # banking: pure transfer, never trained.
    banking = _train_ids("banking_knowledge")[:]
    rng.shuffle(banking)
    _write(banking[:BANKING_TRANSFER_SIZE], "transfer_banking")


# Official TAU2 held-out `test` splits (disjoint from `train`), expected counts.
TEST_DOMAINS = {"retail": 40, "airline": 20, "telecom": 40}


def _test_ids(domain: str) -> list[str]:
    mod = __import__(f"tau2.domains.{domain}.environment", fromlist=["get_tasks"])
    return [t.id for t in mod.get_tasks(task_split_name="test")]


def generate_test_splits() -> None:
    """Generate the official TAU2 held-out `test` splits (frozen on write).

    These are disjoint from `train` (hence from proxy/validation/train_distill),
    so they are the true held-out evaluation set. Deterministic, $0 (task ids
    only). Mirrors ``generate_arm_d_splits`` but reads ``task_split_name="test"``.
    """
    for domain, expected in TEST_DOMAINS.items():
        ids = _test_ids(domain)
        assert len(ids) == expected, f"{domain} test: {len(ids)} != {expected}"
        _write(ids, f"test_{domain}")


def arm_d_split_manifest() -> dict[str, list[str]]:
    names = [
        "train_distill_retail", "train_distill_airline", "train_distill_telecom",
        "eval_airline", "eval_telecom", "transfer_banking",
    ]
    return {n: json.loads((SPLITS_DIR / f"{n}.json").read_text()) for n in names}


if __name__ == "__main__":
    generate_arm_d_splits()
    print("Arm D splits written to", SPLITS_DIR)
