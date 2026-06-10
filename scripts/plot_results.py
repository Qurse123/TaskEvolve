"""Plot the cost vs task-success scatter from experiments/results.csv.

Each run is one point (not collapsed to a mean):
    x = average cost per task   (total_cost_usd / num_tasks)
    y = task success rate       (pass_rate)

Points are grouped into arms by (split, harness_version) — one color per arm —
so the spread across repeated runs is visible. Each arm's centroid (mean x,
mean y) is overlaid as a large marker, giving both the cloud and its center.

Reading the plot: up and to the left is better (higher success, lower cost).
Lines of constant cost-per-successful-task radiate from the origin, since
cost_per_task = cost_per_successful_task * pass_rate.

Run from the repo root:
    python -m scripts.plot_results
    python -m scripts.plot_results --split proxy --out experiments/plots/proxy.png
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # file output only; no interactive display needed
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)

from results.logger import DEFAULT_RESULTS_CSV

logger = logging.getLogger(__name__)

DEFAULT_PLOTS_DIR = Path("experiments/plots")
PLOT_TIME_FORMAT = "%Y%m%d_%H%M%S"
X_LABEL = "Average cost per task (USD)"
Y_LABEL = "Task success rate (pass_rate)"
CENTROID_MARKER = "X"


class RunPoint(NamedTuple):
    """One run's coordinates on the cost/success plane."""

    arm: str
    cost_per_task: float
    pass_rate: float


def load_points(csv_path: Path, *, split: Optional[str] = None) -> List[RunPoint]:
    """Read results.csv into RunPoints, skipping rows without usable cost data.

    Args:
        csv_path: Path to the results CSV (one row per run).
        split: If given, keep only rows for this split.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results CSV not found at {csv_path}. Run an eval first "
            "(e.g. python -m scripts.run_train_eval --split proxy ...)."
        )

    points: List[RunPoint] = []
    skipped = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row.get("split") != split:
                continue
            point = _row_to_point(row)
            if point is None:
                skipped += 1
                continue
            points.append(point)

    if skipped:
        logger.warning("Skipped %d row(s) with missing/zero cost or task count.", skipped)
    return points


def _row_to_point(row: Dict[str, str]) -> Optional[RunPoint]:
    """Convert one CSV row to a RunPoint, or None if cost can't be computed."""
    cost = _as_float(row.get("total_cost_usd"))
    num_tasks = _as_int(row.get("num_tasks"))
    pass_rate = _as_float(row.get("pass_rate"))
    if cost is None or not num_tasks or pass_rate is None:
        return None
    arm = f"{row.get('split', '?')} / {row.get('harness_version', '?')}"
    return RunPoint(arm=arm, cost_per_task=cost / num_tasks, pass_rate=pass_rate)


def plot_points(
    points: Sequence[RunPoint],
    *,
    out_path: Path,
    show_centroid: bool = True,
    title: Optional[str] = None,
) -> Path:
    """Render the per-run scatter (colored by arm) and save it to out_path."""
    arms = _group_by_arm(points)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for index, (arm, arm_points) in enumerate(arms.items()):
        color = colors[index % len(colors)]
        xs = [p.cost_per_task for p in arm_points]
        ys = [p.pass_rate for p in arm_points]
        ax.scatter(xs, ys, color=color, alpha=0.75, s=60, label=f"{arm} (n={len(arm_points)})")
        if show_centroid:
            ax.scatter(
                [statistics.mean(xs)],
                [statistics.mean(ys)],
                color=color,
                marker=CENTROID_MARKER,
                s=240,
                edgecolors="black",
                linewidths=1.3,
                zorder=5,
            )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(title or "Cost vs task success (each point = one run)")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="arm (split / harness)", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _group_by_arm(points: Sequence[RunPoint]) -> "OrderedDict[str, List[RunPoint]]":
    """Group points by arm label, preserving first-seen order for stable colors."""
    grouped: "OrderedDict[str, List[RunPoint]]" = OrderedDict()
    for point in points:
        grouped.setdefault(point.arm, []).append(point)
    return grouped


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip() == "" or value.strip().lower() == "none":
        return None
    return float(value)


def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None or value.strip() == "" or value.strip().lower() == "none":
        return None
    return int(value)


def _default_out_path(split: Optional[str]) -> Path:
    stamp = datetime.now(timezone.utc).strftime(PLOT_TIME_FORMAT)
    suffix = f"_{split}" if split else ""
    return DEFAULT_PLOTS_DIR / f"cost_vs_success{suffix}_{stamp}.png"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot cost vs task-success (per-run scatter) from results.csv."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--split", default=None, help="Filter to one split (e.g. proxy).")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--no-centroid", action="store_true", help="Hide per-arm mean markers.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    points = load_points(args.csv, split=args.split)
    if not points:
        logger.error("No plottable runs in %s%s.", args.csv,
                     f" for split={args.split}" if args.split else "")
        return 1

    out_path = args.out or _default_out_path(args.split)
    plot_points(points, out_path=out_path, show_centroid=not args.no_centroid)
    logger.info("Plotted %d run(s) across %d arm(s) -> %s",
                len(points), len(_group_by_arm(points)), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
