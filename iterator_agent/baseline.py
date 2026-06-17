"""Baseline loader — the current-best proxy distribution the comparator scores against.

Reads ``experiments/results.csv`` (the per-run ledger written by
``results/logger.py``) and summarizes the runs for a given
``(split, harness_version)`` into a mean ± std :class:`Distribution`. The
iterator's comparator (``acceptance.py``) uses this to decide whether a
candidate's proxy double-run beats the current best (experiment.md §20–21).
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from results.logger import DEFAULT_RESULTS_CSV


@dataclass(frozen=True)
class Distribution:
    """Mean ± std of one arm's runs on one split (the noise floor to beat)."""

    split: str
    harness_version: str
    n: int
    pass_rate_mean: float
    pass_rate_std: float
    cost_per_successful_task_mean: Optional[float]
    cost_per_successful_task_std: Optional[float]
    run_ids: Tuple[str, ...]


def _parse_float(value: object) -> Optional[float]:
    """Parse a CSV cell into a float; blank/None/non-numeric → None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mean_std(values: List[float]) -> Tuple[float, float]:
    """Sample mean and std; std is 0.0 for a single value (matches the runner)."""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def load_distribution(
    split: str,
    harness_version: str,
    *,
    csv_path: Union[str, Path] = DEFAULT_RESULTS_CSV,
) -> Distribution:
    """Summarize the ``results.csv`` rows for one ``(split, harness_version)``.

    Args:
        split: Split label to match (e.g. ``"proxy"``).
        harness_version: Harness version to match (e.g. ``"v0.1"`` for Arm A).
        csv_path: Path to the per-run results ledger.

    Returns:
        A :class:`Distribution` with mean ± std of ``pass_rate`` and
        ``cost_per_successful_task`` across the matching runs. Cost stats are
        ``None`` when no matching row reports a cost.

    Raises:
        ValueError: If the file is missing or no row matches.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise ValueError(f"results CSV not found: {csv_path}")

    pass_rates: List[float] = []
    costs: List[float] = []
    run_ids: List[str] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split or row.get("harness_version") != harness_version:
                continue
            pass_rate = _parse_float(row.get("pass_rate"))
            if pass_rate is None:
                continue
            pass_rates.append(pass_rate)
            run_ids.append(row.get("run_id") or "")
            cost = _parse_float(row.get("cost_per_successful_task"))
            if cost is not None:
                costs.append(cost)

    if not pass_rates:
        raise ValueError(
            f"no rows in {csv_path} for split={split!r} harness_version={harness_version!r}"
        )

    pass_rate_mean, pass_rate_std = _mean_std(pass_rates)
    if costs:
        cost_mean, cost_std = _mean_std(costs)
    else:
        cost_mean, cost_std = None, None

    return Distribution(
        split=split,
        harness_version=harness_version,
        n=len(pass_rates),
        pass_rate_mean=pass_rate_mean,
        pass_rate_std=pass_rate_std,
        cost_per_successful_task_mean=cost_mean,
        cost_per_successful_task_std=cost_std,
        run_ids=tuple(run_ids),
    )
