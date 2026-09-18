"""Figure 2 -- input tokens read against dollars paid, on the held-out splits.

The claim this figure carries: the four systems read comparable amounts of
input on retail and airline, and the open-weight systems read the most on
telecom, while the dollars differ by an order of magnitude. Price per token
drives the cost gap.

Run:  uv run python -m scripts.plot_paper_fig2_tokens_vs_cost
"""
from __future__ import annotations

import statistics as st

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from scripts import paper_style as style
from scripts import test_stats as ts

BAR = 0.19
# Alternating label heights keep four horizontal value labels per group apart.
LIFT = [4, 23, 4, 23]


def main() -> int:
    style.apply()
    style.readable()
    rows = ts.load()
    fig, (ax_tok, ax_cost) = plt.subplots(1, 2, figsize=(13.6, 6.0))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.70, bottom=0.2, wspace=0.22)

    x = np.arange(len(ts.DOMAINS))
    for i, (key, label, colour) in enumerate(ts.SYSTEMS):
        offset = (i - 1.5) * BAR
        toks, costs = [], []
        for split, _ in ts.DOMAINS:
            sel = ts.select(rows, split, key)
            toks.append(st.mean(r["input_tokens"] for r in sel) / 1000.0)
            costs.append(st.mean(r["agent_cost_usd"] for r in sel))
        ax_tok.bar(x + offset, toks, BAR, color=colour, label=label, linewidth=0)
        ax_cost.bar(x + offset, costs, BAR, color=colour, label=label, linewidth=0)
        for xi, v in zip(x + offset, toks):
            ax_tok.annotate(f"{v:.0f}", (xi, v), xytext=(0, LIFT[i]), textcoords="offset points",
                            ha="center", fontsize=11, color=colour, fontweight="bold")
        # Bar tops sit decades apart on the log axis, so the value labels ride on
        # two fixed rows above the panel instead of on the bars themselves.
        row = 0.95 if i % 2 == 0 else 0.87
        for xi, v in zip(x + offset, costs):
            label_text = f"{v:.2f}" if v >= 0.1 else f"{v:.3f}"
            ax_cost.text(xi, row, label_text, transform=ax_cost.get_xaxis_transform(),
                         ha="center", va="center", fontsize=11, color=colour,
                         fontweight="bold")

    for ax, title in ((ax_tok, "A   Input tokens read per task, thousands"),
                      (ax_cost, "B   Agent cost per task, USD, log scale")):
        ax.set_xticks(x, [name for _, name in ts.DOMAINS])
        style.horizontal_grid_only(ax)
        ax.set_title(title, fontsize=12.5, color=style.INK, fontweight="bold", loc="left", pad=12)
    ax_cost.set_yscale("log")
    ax_cost.set_ylim(0.01, 4.5)
    ax_tok.set_ylim(0, 275)

    style.titled(fig, None, "Input read per task, and what each system pays for a task")

    handles = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in ts.SYSTEMS]
    fig.legend(handles=handles, labels=[lab for _, lab, _ in ts.SYSTEMS], loc="lower center",
               ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.022), columnspacing=2.4)

    pdf, png = style.save(fig, "fig2_test_tokens_vs_cost")
    print(f"wrote {pdf}\nwrote {png}")
    for split, name in ts.DOMAINS:
        print(f"\n{name}")
        for key, label, _ in ts.SYSTEMS:
            sel = ts.select(rows, split, key)
            print(f"   {label:32} in {st.mean(r['input_tokens'] for r in sel):9.0f}"
                  f"  out {st.mean(r['output_tokens'] for r in sel):7.0f}"
                  f"  ${st.mean(r['agent_cost_usd'] for r in sel):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
