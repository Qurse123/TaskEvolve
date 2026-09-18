"""Figure 3 -- handing the customer to a human agent, telecom held-out split.

The claim this figure carries: Opus reaches for the handoff tool in most
telecom runs, and a run containing a handoff is far likelier to end with the
phone in the wrong state. Base Inkling-Small hands off least and keeps
troubleshooting, which is where its telecom success advantage shows up.

Run:  uv run python -m scripts.plot_paper_fig3_telecom_handoff
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from scripts import paper_style as style
from scripts import test_stats as ts

SPLIT = "test_telecom"
BAR = 0.36


def rate(rows):
    return sum(r["passed"] for r in rows) / len(rows) if rows else 0.0


def main() -> int:
    style.apply()
    style.readable()
    rows = ts.load()
    fig, (ax_share, ax_pass) = plt.subplots(1, 2, figsize=(13.6, 6.2))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.70, bottom=0.235, wspace=0.22)

    x = np.arange(len(ts.SYSTEMS))
    labels = ["Opus 4.8\nstatic", "Opus 4.8\nplus iterator", "Inkling-Small\nbase",
              "Inkling-Small\nfine-tuned"]

    for i, (key, _, colour) in enumerate(ts.SYSTEMS):
        sel = ts.select(rows, SPLIT, key)
        handed = [r for r in sel if r["handoff"]]
        kept = [r for r in sel if r["handoff"] == 0]

        share = len(handed) / len(sel)
        ax_share.bar(i, share, 0.56, color=colour, linewidth=0)
        ax_share.annotate(f"{share*100:.0f}%", (i, share), xytext=(0, 6),
                          textcoords="offset points", ha="center", fontsize=14,
                          color=colour, fontweight="bold")

        # Left bar: runs that handed off, drawn hatched. Right bar: runs that kept going.
        ax_pass.bar(i - BAR / 2, rate(handed), BAR, color=colour, alpha=0.4,
                    hatch="//", edgecolor=style.CREAM, linewidth=0)
        ax_pass.bar(i + BAR / 2, rate(kept), BAR, color=colour, linewidth=0)
        for off, group, count_colour in ((-BAR / 2, handed, style.INK),
                                        (BAR / 2, kept, style.CREAM)):
            ax_pass.annotate(f"{rate(group):.2f}", (i + off, rate(group)), xytext=(0, 6),
                             textcoords="offset points", ha="center", fontsize=13,
                             color=colour, fontweight="bold")
            ax_pass.annotate(f"{len(group)}", (i + off, 0), xytext=(0, 8),
                             textcoords="offset points", ha="center", fontsize=11.5,
                             color=count_colour, fontweight="bold")

    for ax, title in ((ax_share, "A   Share of runs that handed off, 200 runs per system"),
                      (ax_pass, "B   Task success, by whether the run handed off, run counts inside the bars")):
        ax.set_xticks(x, labels)
        style.horizontal_grid_only(ax)
        ax.set_title(title, fontsize=12.5, color=style.INK, fontweight="bold", loc="left", pad=12)
    ax_share.set_ylim(0, 1.02)
    ax_pass.set_ylim(0, 1.12)
    ax_share.set_ylabel("Share of runs", labelpad=10)
    ax_pass.set_ylabel("Task success rate", labelpad=10)

    key_handed = Rectangle((0, 0), 1, 1, facecolor=style.WARM_GRAY, alpha=0.4, hatch="//",
                           edgecolor=style.CREAM, label="Handoff")
    key_kept = Rectangle((0, 0), 1, 1, facecolor=style.WARM_GRAY,
                         label="No handoff")
    ax_pass.legend(handles=[key_handed, key_kept], loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, -0.155), frameon=False, fontsize=11.5,
                   handlelength=1.8, columnspacing=2.0)

    style.titled(fig, None, "How often each system hands the telecom customer to a human\nagent, and how those runs end")

    pdf, png = style.save(fig, "fig3_test_telecom_handoff")
    print(f"wrote {pdf}\nwrote {png}")
    for key, label, _ in ts.SYSTEMS:
        sel = ts.select(rows, SPLIT, key)
        handed = [r for r in sel if r["handoff"]]
        kept = [r for r in sel if r["handoff"] == 0]
        print(f"   {label:32} handoff {len(handed):3}/{len(sel)}  success with {rate(handed):.3f}"
              f"  without {rate(kept):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
