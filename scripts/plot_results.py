"""Plot baseline results from experiments/results.csv.

A baseline is a *distribution*, not one run: every arm is measured over N repeated
seeds, so the honest picture is the spread of those runs plus an aggregate with an
uncertainty band (mean ± std). This script renders that two ways:

1. Per-metric dot plot (default for a single arm). One panel per headline metric
   (task success rate, cost per successful task). Each seed is a dot; the mean ±
   std is overlaid. With one arm a Pareto frontier is meaningless — there is
   nothing to trade against — so this view simply shows where the baseline sits
   and how noisy it is. That spread is the bar later arms must clear.

2. Cost-vs-success frontier scatter (default once >=2 arms exist; force with
   --frontier). x = average cost per task, y = task success rate. Each run is a
   point, each arm a cloud, with the arm mean drawn as a marker carrying x/y std
   error bars. Up and to the left is better. Axes auto-zoom to the data range so
   tightly-clustered arms stay legible; pass --from-zero to anchor at the origin
   (iso-cost-per-success lines then radiate from 0).

Run from the repo root:
    python -m scripts.plot_results                      # auto-pick mode
    python -m scripts.plot_results --split proxy
    python -m scripts.plot_results --frontier --from-zero
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # file output only; no interactive display needed
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)

from iterator_agent.iteration_log import DEFAULT_ITERATIONS_ROOT, RECORD_FILENAME
from results.logger import DEFAULT_RESULTS_CSV

logger = logging.getLogger(__name__)

DEFAULT_PLOTS_DIR = Path("experiments/plots")
PLOT_TIME_FORMAT = "%Y%m%d_%H%M%S"

# Frontier-scatter axis padding: 10% of the data range, with a small absolute
# floor so a degenerate (zero-width) range still gets breathing room.
AXIS_MARGIN_FRAC = 0.10
AXIS_MARGIN_FLOOR_X = 0.005  # USD
AXIS_MARGIN_FLOOR_Y = 0.05   # pass-rate units
JITTER_WIDTH = 0.08          # horizontal spread of per-metric dots within an arm


class RunPoint(NamedTuple):
    """One run's metrics. One CSV row -> one RunPoint."""

    arm: str
    cost_per_task: float
    pass_rate: float
    cost_per_successful_task: Optional[float]


class MetricSpec(NamedTuple):
    """A headline metric to draw as its own per-metric panel."""

    attr: str
    label: str
    fmt: Callable[[float], str]


PER_METRIC_PANELS: Tuple[MetricSpec, ...] = (
    MetricSpec("pass_rate", "Task success rate", lambda v: f"{v:.3f}"),
    MetricSpec(
        "cost_per_successful_task",
        "Cost per successful task (USD)",
        lambda v: f"${v:.3f}",
    ),
)


def load_points(csv_path: Path, *, split: Optional[str] = None) -> List[RunPoint]:
    """Read results.csv into RunPoints, skipping rows without usable cost data."""
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
    return RunPoint(
        arm=arm,
        cost_per_task=cost / num_tasks,
        pass_rate=pass_rate,
        cost_per_successful_task=_as_float(row.get("cost_per_successful_task")),
    )


def plot_per_metric(
    points: Sequence[RunPoint], *, out_path: Path, title: Optional[str] = None
) -> Path:
    """Render one dot-plot panel per headline metric (raw seeds + mean ± std)."""
    arms = _group_by_arm(points)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    jitter = random.Random(0)

    fig, raw_axes = plt.subplots(
        1, len(PER_METRIC_PANELS), figsize=(5 * len(PER_METRIC_PANELS), 5)
    )
    axes = list(raw_axes) if hasattr(raw_axes, "__len__") else [raw_axes]

    for ax, metric in zip(axes, PER_METRIC_PANELS):
        _draw_metric_panel(ax, metric, arms, colors, jitter)

    fig.suptitle(title or "Baseline distribution per metric (each dot = one run)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_metric_panel(ax, metric: MetricSpec, arms, colors, jitter) -> None:
    """Draw a single metric's dots + mean ± std per arm, horizontally (value on x)."""
    tick_labels: List[str] = []
    for index, (arm, arm_points) in enumerate(arms.items()):
        values = [
            getattr(p, metric.attr) for p in arm_points if getattr(p, metric.attr) is not None
        ]
        if not values:
            continue
        color = colors[index % len(colors)]
        ys = [index + (jitter.random() - 0.5) * JITTER_WIDTH for _ in values]
        ax.scatter(values, ys, color=color, alpha=0.7, s=55, zorder=3)
        mean, std = _mean_std(values)
        ax.errorbar(
            mean, index, xerr=std, color=color, marker="D", markersize=9,
            capsize=6, elinewidth=1.6, markeredgecolor="black", zorder=4,
        )
        label = f"{arm}\nn={len(values)}\n{metric.fmt(mean)}"
        if std:
            label += f" ± {metric.fmt(std).lstrip('$')}"
        tick_labels.append(label)

    ax.set_yticks(range(len(tick_labels)))
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_ylim(-0.5, len(tick_labels) - 0.5)
    ax.set_xlabel(metric.label)
    ax.set_title(metric.label)
    ax.grid(True, axis="x", alpha=0.3)


def plot_frontier(
    points: Sequence[RunPoint], *, out_path: Path, from_zero: bool = False,
    title: Optional[str] = None,
) -> Path:
    """Render the cost-vs-success scatter: runs as points, arm mean ± std as a marker."""
    arms = _group_by_arm(points)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for index, (arm, arm_points) in enumerate(arms.items()):
        color = colors[index % len(colors)]
        xs = [p.cost_per_task for p in arm_points]
        ys = [p.pass_rate for p in arm_points]
        ax.scatter(xs, ys, color=color, alpha=0.55, s=55, label=f"{arm} (n={len(arm_points)})")
        mean_x, std_x = _mean_std(xs)
        mean_y, std_y = _mean_std(ys)
        ax.errorbar(
            mean_x, mean_y, xerr=std_x, yerr=std_y, color=color, marker="D",
            markersize=11, capsize=5, elinewidth=1.6, markeredgecolor="black", zorder=5,
        )

    ax.set_xlabel("Average cost per task (USD)")
    ax.set_ylabel("Task success rate")
    ax.set_title(title or "Cost vs task success (each point = one run; ◆ = arm mean ± std)")
    _apply_frontier_limits(ax, points, from_zero=from_zero)
    ax.grid(True, alpha=0.3)
    ax.legend(title="arm (split / harness)", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _apply_frontier_limits(ax, points: Sequence[RunPoint], *, from_zero: bool) -> None:
    """Auto-zoom axes to the data (default) or anchor at the origin (--from-zero)."""
    if from_zero:
        ax.set_xlim(left=0)
        ax.set_ylim(-0.02, 1.05)
        return
    xs = [p.cost_per_task for p in points]
    ys = [p.pass_rate for p in points]
    ax.set_xlim(*_padded_range(xs, AXIS_MARGIN_FLOOR_X, clamp_low=0.0))
    ax.set_ylim(*_padded_range(ys, AXIS_MARGIN_FLOOR_Y, clamp_low=0.0, clamp_high=1.0))


def _padded_range(
    values: Sequence[float], floor: float, *, clamp_low: Optional[float] = None,
    clamp_high: Optional[float] = None,
) -> Tuple[float, float]:
    """Min/max of values padded by 10% of their range (or `floor` if degenerate)."""
    lo, hi = min(values), max(values)
    margin = max((hi - lo) * AXIS_MARGIN_FRAC, floor)
    low, high = lo - margin, hi + margin
    if clamp_low is not None:
        low = max(low, clamp_low)
    if clamp_high is not None:
        high = min(high, clamp_high)
    return low, high


def _mean_std(values: Sequence[float]) -> Tuple[float, Optional[float]]:
    """Mean and sample std; std is None when there are fewer than two values."""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std


def _group_by_arm(points: Sequence[RunPoint]) -> "OrderedDict[str, List[RunPoint]]":
    """Group points by arm label, preserving first-seen order for stable colors."""
    grouped: "OrderedDict[str, List[RunPoint]]" = OrderedDict()
    for point in points:
        grouped.setdefault(point.arm, []).append(point)
    return grouped


class IterationPoint(NamedTuple):
    """One iterator iteration, parsed from its iteration.json record."""

    iteration: int
    accepted: bool
    cost_per_task: Optional[float]
    pass_rate: float
    ticket_id: str
    hypothesis: str


def _iteration_index(iteration_id: str) -> int:
    """Extract the trailing integer from an iteration id (``iter_0007`` -> 7)."""
    tail = iteration_id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def load_iterations(iterations_root: Path) -> List[IterationPoint]:
    """Read every ``<iter>/iteration.json`` into IterationPoints, ordered by index."""
    if not iterations_root.exists():
        raise FileNotFoundError(
            f"Iterations dir not found at {iterations_root}. Run the iterator first "
            "(python -m scripts.run_iterator ...)."
        )
    points: List[IterationPoint] = []
    for record_path in iterations_root.glob(f"*/{RECORD_FILENAME}"):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        points.append(
            IterationPoint(
                iteration=_iteration_index(record.get("iteration_id", "")),
                accepted=record.get("accepted_or_rejected") == "accepted",
                cost_per_task=record.get("cost_per_task_after"),
                pass_rate=float(record.get("proxy_task_success_after_mean", 0.0)),
                ticket_id=record.get("ticket_id", ""),
                hypothesis=record.get("hypothesis", ""),
            )
        )
    return sorted(points, key=lambda p: p.iteration)


def plot_trajectory(
    points: Sequence[IterationPoint], *, out_path: Path, title: Optional[str] = None
) -> Path:
    """Render the hill-climb: objective (cost/task) and task success vs iteration.

    Accepted iterations are filled and joined into the running-best line (the climb);
    rejected iterations are hollow markers at the value they attempted.
    """
    ordered = sorted(points, key=lambda p: p.iteration)
    accepted = [p for p in ordered if p.accepted]
    rejected = [p for p in ordered if not p.accepted]

    fig, (cost_ax, succ_ax) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Objective panel: lower is better; the accepted line is the descending climb.
    _scatter_metric(cost_ax, accepted, rejected, attr="cost_per_task")
    acc_cost = [(p.iteration,p.cost_per_task) for p in accepted if p.cost_per_task is not None]
    if acc_cost:
        cost_ax.plot(*zip(*acc_cost), color="tab:green", linewidth=1.6, zorder=2)
    cost_ax.set_ylabel("Mean cost per task (USD) — lower is better")
    cost_ax.set_title(title or "Iterator trajectory (◆ accepted = current-best climb)")
    cost_ax.grid(True, alpha=0.3)
    cost_ax.legend(fontsize=9)

    # Success guardrail panel.
    _scatter_metric(succ_ax, accepted, rejected, attr="pass_rate")
    acc_succ = [(p.iteration,p.pass_rate) for p in accepted]
    if acc_succ:
        succ_ax.plot(*zip(*acc_succ), color="tab:green", linewidth=1.6, zorder=2)
    succ_ax.set_ylabel("Task success rate")
    succ_ax.set_xlabel("Iteration")
    succ_ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _scatter_metric(ax, accepted, rejected, *, attr: str) -> None:
    """Filled accepted vs hollow rejected markers for one metric on ``ax``."""
    acc = [(p.iteration,getattr(p, attr)) for p in accepted if getattr(p, attr) is not None]
    rej = [(p.iteration,getattr(p, attr)) for p in rejected if getattr(p, attr) is not None]
    if acc:
        ax.scatter(*zip(*acc), color="tab:green", s=70, marker="D",
                   edgecolors="black", zorder=4, label="accepted")
    if rej:
        ax.scatter(*zip(*rej), facecolors="none", edgecolors="tab:red", s=55,
                   zorder=3, label="rejected")


def choose_mode(points: Sequence[RunPoint], *, frontier: bool, per_metric: bool) -> str:
    """Pick 'frontier' or 'per_metric'. Explicit flags win; else >=2 arms -> frontier."""
    if frontier and per_metric:
        raise ValueError("Pass at most one of --frontier / --per-metric.")
    if frontier:
        return "frontier"
    if per_metric:
        return "per_metric"
    return "frontier" if len(_group_by_arm(points)) >= 2 else "per_metric"


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip() == "" or value.strip().lower() == "none":
        return None
    return float(value)


def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None or value.strip() == "" or value.strip().lower() == "none":
        return None
    return int(value)


def _default_out_path(split: Optional[str], mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime(PLOT_TIME_FORMAT)
    suffix = f"_{split}" if split else ""
    stem = mode if mode in ("frontier", "trajectory") else "per_metric"
    return DEFAULT_PLOTS_DIR / f"{stem}{suffix}_{stamp}.png"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot baseline results (per-metric distribution or cost-vs-success frontier)."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--split", default=None, help="Filter to one split (e.g. proxy).")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--frontier", action="store_true", help="Force the cost-vs-success scatter.")
    parser.add_argument("--per-metric", action="store_true", help="Force the per-metric dot plot.")
    parser.add_argument("--from-zero", action="store_true",
                        help="Frontier only: anchor axes at the origin instead of auto-zoom.")
    parser.add_argument("--trajectory", action="store_true",
                        help="Plot the iterator hill-climb (cost + success vs iteration) "
                        "from experiments/iterations/*/iteration.json instead of results.csv.")
    parser.add_argument("--iterations-root", type=Path, default=DEFAULT_ITERATIONS_ROOT,
                        help="Trajectory only: folder of per-iteration records.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.trajectory:
        iterations = load_iterations(args.iterations_root)
        if not iterations:
            logger.error("No iteration records under %s.", args.iterations_root)
            return 1
        out_path = args.out or _default_out_path(args.split, "trajectory")
        plot_trajectory(iterations, out_path=out_path)
        accepts = sum(1 for p in iterations if p.accepted)
        logger.info("Plotted %d iteration(s), %d accepted [trajectory] -> %s",
                    len(iterations), accepts, out_path)
        return 0

    points = load_points(args.csv, split=args.split)
    if not points:
        logger.error("No plottable runs in %s%s.", args.csv,
                     f" for split={args.split}" if args.split else "")
        return 1

    mode = choose_mode(points, frontier=args.frontier, per_metric=args.per_metric)
    out_path = args.out or _default_out_path(args.split, mode)
    if mode == "frontier":
        plot_frontier(points, out_path=out_path, from_zero=args.from_zero)
    else:
        plot_per_metric(points, out_path=out_path)
    logger.info("Plotted %d run(s) across %d arm(s) [%s] -> %s",
                len(points), len(_group_by_arm(points)), mode, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
