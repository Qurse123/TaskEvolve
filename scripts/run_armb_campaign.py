"""Launch the Arm B search: aggregate the multi-domain baseline, then iterate.

``baseline.load_distribution`` summarises one split, but the search set spans
three (retail, airline, telecom) because TAU2 evaluates one domain per run. This
builds the aggregate row 0 the same way ``run_iteration._aggregate`` combines a
candidate's seeds: success is total passed over total tasks, so a domain counts in
proportion to its size, and cost is reconstructed from cost-per-task times tasks.

Run:  python -m scripts.run_armb_campaign --max-minutes 1440 --seed-start 7101
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from iterator_agent.baseline import Distribution
from iterator_agent.run_iteration import _default_eval_seed
from scripts.run_iterator import run_iterator

CSV = Path("experiments/results.csv")
SPLITS = Path("benchmark/splits")
TARGETS: Sequence[Tuple[str, str]] = (
    ("search_retail", "retail"),
    ("search_airline", "airline"),
    ("search_telecom", "telecom"),
)


def _f(value: object) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _size(split: str) -> int:
    data = json.loads((SPLITS / f"{split}.json").read_text(encoding="utf-8"))
    return len(data if isinstance(data, list) else data.get("task_ids", data))


def build_baseline(*, harness_version: str, agent_model: str) -> Distribution:
    """Row 0: the unmodified harness, aggregated across the three search splits."""
    wanted = {s for s, _ in TARGETS}
    by_run: Dict[str, List[dict]] = {}
    for row in csv.DictReader(CSV.open()):
        if (row["split"] in wanted and row["harness_version"] == harness_version
                and row["agent_model"] == agent_model):
            by_run.setdefault(row["split"], []).append(row)

    missing = wanted - set(by_run)
    if missing:
        raise SystemExit(f"no baseline rows for {sorted(missing)}; run the baseline first")

    # Pair the k-th run of each split into one seed-level sample.
    n = min(len(v) for v in by_run.values())
    passes, costs_per_task, costs_per_success = [], [], []
    run_ids: List[str] = []
    for i in range(n):
        tasks = cost = passed = 0.0
        for split, _ in TARGETS:
            row = by_run[split][i]
            size = _size(split)
            tasks += size
            passed += (_f(row["pass_rate"]) or 0.0) * size
            cost += _f(row["total_cost_usd"]) or 0.0
            run_ids.append(row["run_id"])
        passes.append(passed / tasks)
        costs_per_task.append(cost / tasks)
        costs_per_success.append(cost / passed if passed else float("nan"))

    def _sd(xs: List[float]) -> float:
        return st.stdev(xs) if len(xs) > 1 else 0.0

    return Distribution(
        split="search", harness_version=harness_version, n=n,
        pass_rate_mean=st.mean(passes), pass_rate_std=_sd(passes),
        cost_per_successful_task_mean=st.mean(costs_per_success),
        cost_per_successful_task_std=_sd(costs_per_success),
        run_ids=tuple(run_ids),
        cost_per_task_mean=st.mean(costs_per_task), cost_per_task_std=_sd(costs_per_task),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-minutes", type=float, required=True)
    ap.add_argument("--max-iterations", type=int, default=500)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--agent-model", default="anthropic/claude-opus-4-8")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    best = build_baseline(harness_version="v0.1", agent_model=args.agent_model)
    print(f"row 0, unmodified v0.1 across {len(TARGETS)} search splits, n={best.n}:")
    print(f"  success            {best.pass_rate_mean:.4f} ± {best.pass_rate_std:.4f}")
    print(f"  cost/successful    ${best.cost_per_successful_task_mean:.4f}")
    print(f"  cost/task          ${best.cost_per_task_mean:.4f}\n")

    results = run_iterator(
        max_iterations=args.max_iterations,
        seed_start=args.seed_start,
        max_minutes=args.max_minutes,
        agent_model=args.agent_model,
        initial_best=best,
        eval_seed=_default_eval_seed(list(TARGETS)),
    )
    print(f"\n{len(results)} candidate(s) measured. Rank them with:")
    print("  python -m scripts.rank_candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
