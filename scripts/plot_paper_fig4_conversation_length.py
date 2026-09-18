"""Figure 4 -- how long each system's conversations run on the held-out splits.

The claim this figure carries: base Inkling-Small holds the longest
conversations, most visibly on telecom, and fine-tuning on Opus trajectories
pulls that distribution back toward the Opus length. Conversation length is
what the fine-tune changed, and each message re-sends the history.

Run:  uv run python -m scripts.plot_paper_fig4_conversation_length
"""
from __future__ import annotations

import statistics as st

import matplotlib.pyplot as plt

from scripts import paper_style as style
from scripts import test_stats as ts


def main() -> int:
    style.apply()
    style.readable()
    rows = ts.load()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.8))
    fig.subplots_adjust(left=0.135, right=0.988, top=0.70, bottom=0.165, wspace=0.09)

    # Short row labels so a reader stops matching colours to the legend.
    row_labels = ["Opus 4.8 static", "Opus 4.8 plus iterator", "Inkling-Small base",
                  "Inkling-Small fine-tuned"]

    order = list(reversed(ts.SYSTEMS))          # first system at the top
    for ax, (split, name) in zip(axes, ts.DOMAINS):
        data = [[r["assistant_messages"] for r in ts.select(rows, split, key)]
                for key, _, _ in order]
        positions = range(1, len(order) + 1)
        box = ax.boxplot(data, positions=list(positions), vert=False, widths=0.62,
                         patch_artist=True, showfliers=False,
                         medianprops={"color": style.CREAM, "linewidth": 1.4},
                         whiskerprops={"color": style.BODY, "linewidth": 1.0},
                         capprops={"color": style.BODY, "linewidth": 1.0})
        for patch, (_, _, colour) in zip(box["boxes"], order):
            patch.set(facecolor=colour, edgecolor=colour, linewidth=0)
        # Headroom on the right so the mean labels clear the longest whisker.
        whisker_max = max(max(cap.get_xdata()) for cap in box["caps"])
        ax.set_xlim(0, whisker_max * 1.45)
        for pos, values in zip(positions, data):
            colour = order[pos - 1][2]
            ax.text(0.985, pos, f"mean {st.mean(values):.1f}", va="center", ha="right",
                    fontsize=11.5, color=colour, fontweight="bold",
                    transform=ax.get_yaxis_transform())
        ticks = list(reversed(row_labels)) if ax is axes[0] else ["" for _ in order]
        ax.set_yticks(list(positions), ticks)
        ax.grid(True, axis="x", color=style.GRID, linewidth=0.9)
        ax.grid(False, axis="y")
        heading = f"{name}, note the wider scale" if split == "test_telecom" else name
        ax.set_title(heading, fontsize=12.5, color=style.INK, fontweight="bold",
                     loc="left", pad=12)
        ax.set_xlabel("Agent messages per task", fontsize=11.5)

    style.titled(fig, None, "Fine-tuning on Opus trajectories shortens Inkling-Small's\nconversations")

    pdf, png = style.save(fig, "fig4_test_conversation_length")
    print(f"wrote {pdf}\nwrote {png}")
    for split, name in ts.DOMAINS:
        print(f"\n{name}")
        for key, label, _ in ts.SYSTEMS:
            vals = [r["assistant_messages"] for r in ts.select(rows, split, key)]
            print(f"   {label:32} mean {st.mean(vals):5.1f}  median {st.median(vals):5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
