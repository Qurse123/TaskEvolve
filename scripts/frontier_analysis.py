"""Cost-performance frontier re-analysis (the corrected, multi-objective view).

The iterator's strict acceptance rule (accept iff cheaper AND >=0.95x best success)
is single-objective and hides tradeoffs. A cost-performance study is a Pareto
frontier problem: the answer is the set of non-dominated (cost, success) points and
the best cost/precision operating point on it — not one "best harness".

This script gathers EVERY (cost_per_task, success) point we already measured:
  * model-axis arms      — from experiments/results.csv (proxy split, per agent_model)
  * iterator candidates  — every experiments/iterations/*/iteration.json
                           (cost_per_task_after, proxy_task_success_after_mean)
computes the Pareto frontier (minimize cost, maximize success), picks the best
cost-per-successful-task point above an absolute success floor, and plots it.

Run from the repo root:
    python -m scripts.frontier_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from iterator_agent.iteration_log import DEFAULT_ITERATIONS_ROOT, RECORD_FILENAME
from results.logger import DEFAULT_RESULTS_CSV

logger = logging.getLogger(__name__)

DEFAULT_PLOTS_DIR = Path("experiments/plots")
# Absolute region-of-interest floor: a config with success below this is not a
# useful operating point regardless of how cheap it is (anti-degenerate). Explicit
# and adjustable — this is the honest replacement for the buried 0.95x-best floor.
DEFAULT_SUCCESS_FLOOR = 0.5


@dataclass(frozen=True)
class Point:
    """One measured configuration as a cost-performance point."""

    label: str
    lever: str  # "model" | "harness" | "routing"
    cost_per_task: float
    success: float

    @property
    def cost_per_successful_task(self) -> Optional[float]:
        return self.cost_per_task / self.success if self.success > 0 else None


def _f(v) -> Optional[float]:
    try:
        s = str(v).strip()
        return float(s) if s and s.lower() != "none" else None
    except (TypeError, ValueError):
        return None


def load_model_points(csv_path: Path, *, split: str = "proxy") -> List[Point]:
    """Aggregate results.csv into one mean (cost_per_task, success) point per model."""
    rows_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split:
                continue
            cost = _f(row.get("total_cost_usd"))
            ntask = _f(row.get("num_tasks"))
            succ = _f(row.get("pass_rate"))
            if cost is None or not ntask or succ is None:
                continue
            # Key by (model, harness) so e.g. Opus routing-v0.2 stays a distinct
            # point instead of averaging into raw Opus v0.1.
            key = (row.get("agent_model") or "?", row.get("harness_version") or "?")
            rows_by_key[key].append((cost / ntask, succ))
    points: List[Point] = []
    for (model, harness), pairs in rows_by_key.items():
        cpt = sum(c for c, _ in pairs) / len(pairs)
        succ = sum(s for _, s in pairs) / len(pairs)
        short = model.rsplit("/", 1)[-1]
        # A routing harness is a routing lever; otherwise it's the raw model.
        lever = "routing" if "rout" in harness.lower() else "model"
        label = f"{lever}:{short}/{harness}" if lever == "routing" else f"model:{short}"
        points.append(Point(label, lever, cpt, succ))
    return points


def load_iteration_points(iterations_root: Path) -> List[Point]:
    """Each iterator candidate's proxy (cost_per_task, success) after the edit."""
    points: List[Point] = []
    for record_path in sorted(iterations_root.glob(f"*/{RECORD_FILENAME}")):
        rec = json.loads(record_path.read_text(encoding="utf-8"))
        cost = _f(rec.get("cost_per_task_after"))
        succ = _f(rec.get("proxy_task_success_after_mean"))
        if cost is None or succ is None:
            continue
        surface = (rec.get("changed_file") or "").rsplit("/", 1)[-1]
        lever = "routing" if "model_routing" in surface else "harness"
        points.append(
            Point(f"{rec.get('iteration_id', '?')}:{surface}", lever, cost, succ)
        )
    return points


def pareto_front(points: Sequence[Point]) -> List[Point]:
    """Non-dominated points: minimize cost, maximize success.

    p dominates q iff p is no worse on both axes (cost <=, success >=) and strictly
    better on at least one. Return the points dominated by nobody, sorted by cost.
    """
    front: List[Point] = []
    for p in points:
        dominated = any(
            (o.cost_per_task <= p.cost_per_task and o.success >= p.success)
            and (o.cost_per_task < p.cost_per_task or o.success > p.success)
            for o in points
            if o is not p
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda pt: pt.cost_per_task)


def best_operating_point(
    points: Sequence[Point], *, success_floor: float
) -> Optional[Point]:
    """Best cost/precision point: min cost-per-successful-task above the floor."""
    viable = [
        p for p in points if p.success >= success_floor and p.cost_per_successful_task
    ]
    return min(viable, key=lambda p: p.cost_per_successful_task) if viable else None


def plot_frontier(
    points: Sequence[Point], front: Sequence[Point], best: Optional[Point],
    *, out_path: Path, success_floor: float,
) -> Path:
    colors = {"model": "tab:blue", "harness": "tab:orange", "routing": "tab:green"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for lever, color in colors.items():
        xs = [p.cost_per_task for p in points if p.lever == lever]
        ys = [p.success for p in points if p.lever == lever]
        if xs:
            ax.scatter(xs, ys, c=color, alpha=0.55, s=45, label=f"{lever} candidate")
    fx = [p.cost_per_task for p in front]
    fy = [p.success for p in front]
    ax.plot(fx, fy, color="black", lw=1.5, marker="o", ms=6, zorder=5,
            label="Pareto frontier")
    ax.axhline(success_floor, ls="--", color="grey", alpha=0.7,
               label=f"success floor {success_floor:g}")
    if best is not None:
        ax.scatter([best.cost_per_task], [best.success], s=260, marker="*",
                   c="red", edgecolors="black", zorder=6,
                   label=f"best cost/precision: {best.label}")
    ax.set_xlabel("Cost per task (USD)")
    ax.set_ylabel("Task success (proxy)")
    ax.set_title("Cost-performance frontier — all configs (model / harness / routing)")
    ax.set_xscale("log")  # costs span 2 orders of magnitude ($0.02 Inkling -> $0.6 Opus)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the cost-performance frontier.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--iterations-root", type=Path, default=DEFAULT_ITERATIONS_ROOT)
    parser.add_argument("--split", default="proxy")
    parser.add_argument("--success-floor", type=float, default=DEFAULT_SUCCESS_FLOOR)
    parser.add_argument("--out", type=Path,
                        default=DEFAULT_PLOTS_DIR / "cost_performance_frontier.png")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    points = load_model_points(args.csv, split=args.split)
    points += load_iteration_points(args.iterations_root)
    front = pareto_front(points)
    best = best_operating_point(points, success_floor=args.success_floor)

    logger.info("Total configs: %d | Pareto-frontier points: %d", len(points), len(front))
    logger.info("--- Pareto frontier (cost per task ascending) ---")
    for p in front:
        cps = p.cost_per_successful_task
        logger.info("  %-34s success=%.3f  cost/task=$%.4f  cost/success=$%s",
                    p.label, p.success, p.cost_per_task,
                    f"{cps:.4f}" if cps else "n/a")
    if best is not None:
        logger.info("BEST cost/precision (success>=%.2f): %s -- success=%.3f, "
                    "cost/task=$%.4f, cost/success=$%.4f", args.success_floor,
                    best.label, best.success, best.cost_per_task,
                    best.cost_per_successful_task)
    out = plot_frontier(points, front, best, out_path=args.out,
                        success_floor=args.success_floor)
    logger.info("Frontier plot -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
