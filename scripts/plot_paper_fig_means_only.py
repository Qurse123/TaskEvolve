"""Figure 2 -- cost versus precision across the three held-out test domains.

Four systems on TAU2's official held-out test splits, N=5 seeds each
(4001-4005), static harness except where noted, identical user simulator.

Run:  .venv/bin/python -m scripts.plot_paper_fig2
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
    ("anthropic/claude-opus-4-8", "v0.1", "Opus 4.8, closed frontier",  style.WARM_GRAY, "o"),
    ("anthropic/claude-opus-4-8", "v0.4", "Opus 4.8 plus agent iterator", OCHRE,         "s"),
    ("openai/thinkingmachines/Inkling-Small", "v0.1", "Inkling-Small base", style.STEEL_BLUE, "o"),
    ("openai/armd-inkling-small-tuned", "v0.1", "Inkling-Small fine-tuned", style.TERRACOTTA, "D"),
]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _valid(r) -> bool:
    return not (_f(r["total_cost_usd"]) in (None, 0.0) and _f(r["pass_rate"]) == 0.0)


def load(split, model, harness):
    rows = [r for r in csv.DictReader(CSV.open())
            if r["split"] == split and r["agent_model"] == model
            and r["harness_version"] == harness and _valid(r)]
    if not rows:
        return None
    p = [_f(r["pass_rate"]) for r in rows]
    c = [_f(r["cost_per_successful_task"]) for r in rows
         if _f(r["cost_per_successful_task"]) is not None]
    return {"y": st.mean(p), "ye": st.stdev(p), "x": st.mean(c)}


def main() -> int:
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.4), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.70, bottom=0.235, wspace=0.09)

    for ax, (split, letter, sub) in zip(axes, PANELS):
        for model, harness, label, colour, marker in SYSTEMS:
            d = load(split, model, harness)
            if d is None:
                continue
            ax.errorbar(d["x"], d["y"], yerr=d["ye"], fmt="none", ecolor=colour,
                        elinewidth=1.3, capsize=3.5, alpha=0.95, zorder=2)
            ax.scatter(d["x"], d["y"], s=82, color=colour, marker=marker, zorder=3,
                       edgecolor=style.CREAM, linewidth=1.0)
            ax.annotate(f"{d['y']:.2f}", xy=(d["x"], d["y"]), xytext=(0, 11),
                        textcoords="offset points", ha="center", fontsize=8.6,
                        color=style.MUTED)
        ax.set_xscale("log")
        ax.set_xlim(0.02, 3.0)
        ax.set_ylim(0.10, 1.0)
        style.horizontal_grid_only(ax)
        ax.set_title(f"{letter}   {sub}", fontsize=10.5, color=style.INK,
                     fontweight="bold", loc="left", pad=10)
        ax.set_xlabel("Cost per successful task, USD", fontsize=9)
    axes[0].set_ylabel("Task success rate", labelpad=9)

    style.titled(fig, 2, "Outside retail the closed frontier model is beaten\non both accuracy and cost")

    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none", markersize=8,
                          markeredgecolor=style.CREAM, label=lab)
               for _, _, lab, c, m in SYSTEMS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.4, bbox_to_anchor=(0.5, 0.035), columnspacing=2.4)

    pdf, png = style.save(fig, "figX_test_frontier_by_domain")
    print(f"wrote {pdf}\nwrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
