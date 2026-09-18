"""Figure 1 -- four systems compared on the held-out test splits, every run shown.

Each system contributes 5 seed runs per domain (seeds 4001-4005). Faint marks
are the individual runs; the solid mark is the mean with +/-1 std on both axes.
Showing the raw seeds keeps the spread visible rather than hiding it in a mean.

Run:  .venv/bin/python -m scripts.plot_paper_fig1_frontier
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

from scripts import paper_style as style

CSV = Path("experiments/results.csv")
OCHRE = "#B58A47"

PANELS = [("test_retail", "A", "Retail, 40 tasks"),
          ("test_airline", "B", "Airline, 20 tasks"),
          ("test_telecom", "C", "Telecom, 40 tasks")]

SYSTEMS = [
    ("anthropic/claude-opus-4-8", "v0.1", "Opus 4.8, static harness", style.WARM_GRAY, "o"),
    ("anthropic/claude-opus-4-8", "v0.5", "Opus 4.8 plus agent iterator", OCHRE, "s"),
    ("openai/thinkingmachines/Inkling-Small", "v0.1", "Inkling-Small base", style.STEEL_BLUE, "o"),
    ("openai/armd-inkling-small-tuned", "v0.1", "Inkling-Small fine-tuned", style.TERRACOTTA, "D"),
]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def runs(split, model, harness):
    """Every valid seed run as (cost_per_successful_task, pass_rate)."""
    out = []
    for r in csv.DictReader(CSV.open()):
        if r["split"] != split or r["agent_model"] != model:
            continue
        if r["harness_version"] != harness:
            continue
        c, p, tot = _f(r["cost_per_successful_task"]), _f(r["pass_rate"]), _f(r["total_cost_usd"])
        if tot in (None, 0.0) and p == 0.0:      # validity gate: never executed
            continue
        if c is None:
            continue
        out.append((c, p))
    return out


# Mean labels sit above or below the mark so neighbouring systems stay apart.
LABEL_OFFSET = {"Opus 4.8, static harness": 15, "Opus 4.8 plus agent iterator": -23,
                "Inkling-Small base": 15, "Inkling-Small fine-tuned": -23}


def main() -> int:
    style.apply()
    style.readable()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.1), sharey=True)
    fig.subplots_adjust(left=0.062, right=0.988, top=0.70, bottom=0.235, wspace=0.08)

    for ax, (split, letter, sub) in zip(axes, PANELS):
        for model, harness, label, colour, marker in SYSTEMS:
            pts = runs(split, model, harness)
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.scatter(xs, ys, s=34, color=colour, marker=marker, alpha=0.34,
                       zorder=2, linewidth=0)
            mx, my = st.mean(xs), st.mean(ys)
            ax.errorbar(mx, my,
                        xerr=st.stdev(xs) if len(xs) > 1 else None,
                        yerr=st.stdev(ys) if len(ys) > 1 else None,
                        fmt="none", ecolor=colour, elinewidth=1.3, capsize=3.5, zorder=3)
            ax.scatter(mx, my, s=150, color=colour, marker=marker, zorder=4,
                       edgecolor=style.CREAM, linewidth=1.2)
            ax.annotate(f"{my:.2f}", xy=(mx, my), xytext=(0, LABEL_OFFSET[label]),
                        textcoords="offset points", ha="center",
                        fontsize=11.5, color=colour, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(0.02, 3.0)
        ax.set_ylim(0.05, 1.02)
        style.horizontal_grid_only(ax)
        ax.set_title(f"{letter}   {sub}", fontsize=12.5, color=style.INK,
                     fontweight="bold", loc="left", pad=10)
        ax.set_xlabel("Cost per successful task, USD, log scale", fontsize=11.5)
    axes[0].set_ylabel("Task success rate", labelpad=9)

    style.titled(fig, None, "Every seed run on the held-out test splits, four systems\ncompared")

    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none", markersize=11,
                          markeredgecolor=style.CREAM, label=lab)
               for _, _, lab, c, m in SYSTEMS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.028), columnspacing=2.4)

    pdf, png = style.save(fig, "fig1_test_all_runs_four_systems")
    print(f"wrote {pdf}\nwrote {png}")
    for split, _, _ in PANELS:
        print(f"\n{split}")
        for model, harness, label, *_ in SYSTEMS:
            pts = runs(split, model, harness)
            if pts:
                print(f"   {label:32} n={len(pts)}  success per seed {[round(p[1],3) for p in pts]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
