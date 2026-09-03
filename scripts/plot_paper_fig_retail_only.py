"""Figure 1 (hero) — the held-out cost/success frontier.

Every arm on `test_retail`: 40 held-out TAU2 retail tasks, N=5 seeds (4001-4005),
static v0.1 harness, identical user-simulator. Reads the canonical ledger and
applies the validity gate before aggregating.

Run:  .venv/bin/python -m scripts.plot_paper_fig1
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

from scripts import paper_style as style

CSV = Path("experiments/results.csv")
SPLIT = "test_retail"

# label, colour, marker  — Opus is the desaturated comparator; open weights carry colour.
ARMS = [
    ("anthropic/claude-opus-4-8",            "Opus 4.8\nclosed frontier", style.WARM_GRAY,  "o"),
    ("openai/thinkingmachines/Inkling",      "Inkling 975B",                 style.OLIVE,      "o"),
    ("openai/thinkingmachines/Inkling-Small","Inkling-Small base",         style.STEEL_BLUE, "o"),
    ("openai/armd-inkling-small-tuned",      "Inkling-Small fine-tuned",    style.TERRACOTTA, "D"),
]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _valid(r) -> bool:
    """Pre-registered validity gate: a run that spent nothing never executed."""
    return not (_f(r["total_cost_usd"]) in (None, 0.0) and _f(r["pass_rate"]) == 0.0)


def load(model: str) -> dict:
    rows = [r for r in csv.DictReader(CSV.open())
            if r["split"] == SPLIT and r["agent_model"] == model and _valid(r)]
    p = [_f(r["pass_rate"]) for r in rows]
    c = [_f(r["cost_per_successful_task"]) for r in rows]
    return {"n": len(p), "y": st.mean(p), "ye": st.stdev(p),
            "x": st.mean(c), "xe": st.stdev(c)}


def main() -> int:
    style.apply()
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    fig.subplots_adjust(left=0.115, right=0.955, top=0.74, bottom=0.145)

    data = {m: load(m) for m, *_ in ARMS}
    opus, tuned = data["anthropic/claude-opus-4-8"], data["openai/armd-inkling-small-tuned"]

    # The finding, drawn: identical success, one axis-decade apart in cost.
    ax.plot([tuned["x"], opus["x"]], [tuned["y"], opus["y"]],
            color=style.INK, lw=0.9, ls=(0, (4, 3)), zorder=1, alpha=0.55)
    ax.annotate(f"same success, {opus['x'] / tuned['x']:.1f}\u00d7 cheaper",
                xy=((tuned["x"] * opus["x"]) ** 0.5, tuned["y"]),
                xytext=(0, -17), textcoords="offset points",
                ha="center", fontsize=9.5, style="italic", color=style.BODY)

    for model, label, colour, marker in ARMS:
        d = data[model]
        ax.errorbar(d["x"], d["y"], xerr=d["xe"], yerr=d["ye"], fmt="none",
                    ecolor=colour, elinewidth=1.4, capsize=3, alpha=0.85, zorder=2)
        ax.scatter(d["x"], d["y"], s=145, color=colour, marker=marker,
                   zorder=3, edgecolor="white", linewidth=1.2)
        # Their habit: print the value on the mark so the figure reads alone.
        below = label.startswith("Inkling-Small\nfine")
        ax.annotate(f"{label}\n{d['y']:.3f}  ·  ${d['x']:.4f}",
                    xy=(d["x"], d["y"]),
                    xytext=(0, -46 if below else 16), textcoords="offset points",
                    ha="center", va="top" if below else "bottom",
                    fontsize=9.6, color=style.INK, linespacing=1.35)

    ax.set_xscale("log")
    ax.set_xlabel("Cost per successful task in USD, log scale, cheaper to the left", labelpad=11)
    ax.set_ylabel("Task success rate", labelpad=9)
    ax.set_xlim(0.006, 1.35)
    ax.set_ylim(0.80, 0.975)
    style.horizontal_grid_only(ax)
    style.titled(fig, 1, "Matching frontier task success at a fraction\nof the cost")
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0])
    ax.set_xticklabels(["$0.01", "$0.03", "$0.10", "$0.30", "$1.00"])

    pdf, png = style.save(fig, "figX_frontier_test_retail")
    print(f"wrote {pdf}\nwrote {png}")
    for _, label, *_ in ARMS:
        pass
    for model, label, *_ in ARMS:
        d = data[model]
        print(f"  {label.replace(chr(10),' '):32} n={d['n']}  {d['y']:.4f} ± {d['ye']:.4f}   ${d['x']:.4f} ± {d['xe']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
