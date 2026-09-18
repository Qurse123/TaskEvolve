"""Shared matplotlib style for the TaskEvolve paper figures.

Conventions derived by inspecting the figures in Anthropic's research post
"How Claude is accelerating protein design and analytical chemistry" (Aug 2026):
muted earthy palette with the comparator series in warm gray, no in-image chart
title (the caption carries the claim), top/right spines removed, light warm
horizontal gridlines only, unboxed legends, and values printed directly on the
marks so a figure reads without cross-referencing a table.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Focal series carry colour; the baseline/comparator is deliberately desaturated.
TERRACOTTA = "#C4643F"
STEEL_BLUE = "#5B87B5"
OLIVE = "#7C9450"
WARM_GRAY = "#A6A099"      # reserved for the comparator arm (Opus)
INK = "#1F1E1C"            # titles, axis lines
BODY = "#57534E"           # axis labels, tick labels
MUTED = "#8C8781"          # the "FIGURE n" eyebrow, value sub-labels
CREAM = "#F0EDE6"          # page/canvas background — the Anthropic card
GRID = "#DED9CF"

PALETTE = [TERRACOTTA, STEEL_BLUE, OLIVE]


def apply() -> None:
    """Install the paper style into matplotlib's global rcParams."""
    mpl.rcParams.update({
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "text.color": BODY,
        "axes.labelcolor": BODY,
        "axes.edgecolor": INK,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "xtick.color": BODY,
        "ytick.color": BODY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.facecolor": CREAM,
    })


def readable() -> None:
    """Enlarge in-figure text so a page-size figure reads without zooming.

    Call straight after apply(). The paper prints figures at \\textwidth on a
    1.1in-margin page, so the sizes here are what a reader actually sees.
    """
    mpl.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 12,
    })


def readable() -> None:
    """Enlarge in-figure text so a page-size figure reads without zooming.

    Call straight after apply(). The paper prints figures at \\textwidth on a
    1.1in-margin page, so the sizes here are what a reader actually sees.
    """
    mpl.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 12,
    })


def titled(fig, number, title: str, *, x=0.055, y=0.955) -> None:
    """The Anthropic figure header: a bold title, over an optional eyebrow.

    Pass number=None when the figure sits in a LaTeX float, so the caption owns
    the numbering and a reader sees one number rather than two.
    """
    if number is None:
        fig.text(x, y, title, fontsize=15.5, color=INK,
                 fontweight="bold", va="top", linespacing=1.32)
        return
    fig.text(x, y, f"F I G U R E   {number}", fontsize=7.6, color=MUTED,
             fontweight="medium", va="top")
    fig.text(x, y - 0.052, title, fontsize=15.5, color=INK,
             fontweight="bold", va="top", linespacing=1.32)


def horizontal_grid_only(ax) -> None:
    """Their grid convention: light horizontal rules, no vertical ones."""
    ax.grid(True, axis="y", color=GRID, linewidth=0.9)
    ax.grid(False, axis="x")


def save(fig, stem, *, outdir="experiments/plots/paper"):
    """Write vector PDF (for LaTeX) plus a PNG preview. Returns both paths."""
    from pathlib import Path
    d = Path(outdir); d.mkdir(parents=True, exist_ok=True)
    pdf, png = d / f"{stem}.pdf", d / f"{stem}.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    return pdf, png
