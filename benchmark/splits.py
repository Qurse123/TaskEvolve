"""Generate and load benchmark task splits.

Requires TAU2-bench to be installed before importing:
    git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
    cd vendor/tau2-bench && uv sync --all-extras

Run once to populate benchmark/splits/*.json:
    python -m benchmark.splits

Then use load_split() in other modules.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tau2.domains.mock.environment import get_tasks as get_mock_tasks  # type: ignore[import]
from tau2.domains.retail.environment import get_tasks as get_retail_tasks  # type: ignore[import]

SPLITS_DIR = Path(__file__).parent / "splits"

SMOKE_SIZE = 3
PROXY_SIZE = 12
VALIDATION_SIZE = 35
SEED = 42


def generate_splits() -> None:
    """Sample task IDs from TAU2 and write split JSON files.

    Requires TAU2-bench to be installed:
        git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
        cd vendor/tau2-bench && uv sync --all-extras
    """
    retail_train_ids: list[str] = [
        t.id for t in get_retail_tasks(task_split_name="train")
    ]
    mock_ids: list[str] = [t.id for t in get_mock_tasks(task_split_name=None)]

    if len(retail_train_ids) < PROXY_SIZE + VALIDATION_SIZE:
        raise ValueError(
            f"Retail train split has only {len(retail_train_ids)} tasks; "
            f"need at least {PROXY_SIZE + VALIDATION_SIZE}."
        )

    rng = random.Random(SEED)
    shuffled = retail_train_ids[:]
    rng.shuffle(shuffled)

    proxy_ids = shuffled[:PROXY_SIZE]
    validation_ids = shuffled[PROXY_SIZE : PROXY_SIZE + VALIDATION_SIZE]
    smoke_ids = mock_ids[:SMOKE_SIZE]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    _write(smoke_ids, "smoke")
    _write(proxy_ids, "proxy")
    _write(validation_ids, "validation")

    print(f"Splits written to {SPLITS_DIR}/")
    print(f"  smoke:      {len(smoke_ids)} tasks (mock domain)")
    print(f"  proxy:      {len(proxy_ids)} tasks (retail train)")
    print(f"  validation: {len(validation_ids)} tasks (retail train)")


def load_split(name: str) -> list[str]:
    """Return the list of task IDs for a named split.

    Args:
        name: One of "smoke", "proxy", "validation".

    Returns:
        List of task ID strings.
    """
    allowed = {"smoke", "proxy", "validation"}
    if name not in allowed:
        raise ValueError(f"Split must be one of {allowed}, got {name!r}")

    path = SPLITS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run `python -m benchmark.splits` to generate splits."
        )

    return json.loads(path.read_text())


def _write(task_ids: list[str], name: str) -> None:
    path = SPLITS_DIR / f"{name}.json"
    path.write_text(json.dumps(task_ids, indent=2))


if __name__ == "__main__":
    generate_splits()
