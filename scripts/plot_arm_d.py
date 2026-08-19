"""Arm D C-vs-D frontier: naive Inkling-Small (base control) vs Opus-distilled
tuned Inkling-Small, per TAU2 domain, served identically via the Tinker shim.

Cost-vs-success scatter (x = cost per successful task, log scale; y = task
success rate). One base + one tuned point per domain, with an arrow base->tuned
(up-and-left = better: higher success at lower cost). Reads
`experiments/results.csv`; the 6 stale all-error banking-base runs carry no
`cost_per_successful_task` and are dropped, leaving the 24 clean N=3 runs.

Run:  python -m scripts.plot_arm_d
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

CSV = Path("experiments/results.csv")
OUT = Path("experiments/plots/arm_d_c_vs_d_frontier_20260811.png")
DOMAIN_ORDER = ["retail", "airline", "telecom", "banking"]
DOMAIN_COLOR = {"retail": "#1f77b4", "airline": "#d62728",
                "telecom": "#2ca02c", "banking": "#9467bd"}
SPLIT_TO_DOMAIN = {"validation": "retail", "eval_airline": "airline",
                   "eval_telecom": "telecom", "transfer_banking": "banking"}


def load() -> dict:
    agg: dict = defaultdict(lambda: {"pass": [], "cost": []})
    for r in csv.DictReader(open(CSV)):
        model = r.get("agent_model", "").lower()
        if "inkling-small" not in model or r.get("harness_version") != "v0.1":
            continue
        cs = (r.get("cost_per_successful_task") or "").strip()
        if not cs:  # stale all-error banking-base rows (no successes) -> drop
            continue
        system = "tuned" if "armd" in model else "base"
        domain = SPLIT_TO_DOMAIN.get(r["split"], r["split"])
        agg[(system, domain)]["pass"].append(float(r["pass_rate"]))
        agg[(system, domain)]["cost"].append(float(cs))
    return agg


def _mean(xs):
    return sum(xs) / len(xs)


def main() -> int:
    agg = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    for domain in DOMAIN_ORDER:
        color = DOMAIN_COLOR[domain]
        b = agg.get(("base", domain))
        t = agg.get(("tuned", domain))
        if not b or not t:
            continue
        bx, by = _mean(b["cost"]), _mean(b["pass"])
        tx, ty = _mean(t["cost"]), _mean(t["pass"])
        ax.annotate("", xy=(tx, ty), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.5, lw=1.5))
        ax.scatter([bx], [by], marker="o", s=90, facecolors="none",
                   edgecolors=color, linewidths=2, zorder=3)
        ax.scatter([tx], [ty], marker="*", s=240, color=color, zorder=3)
        ax.annotate(domain, (tx, ty), textcoords="offset points", xytext=(8, 6),
                    color=color, fontsize=10, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("cost per successful task (USD, log scale)  —  cheaper -->")
    ax.set_ylabel("task success rate  —  higher better")
    ax.set_title(
        "Arm D: naive vs Opus-distilled Inkling-Small on TAU2 (N=3, identical serving)\n"
        "o naive  -->  * tuned   (up-and-left = better)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(handles=[
        Line2D([], [], marker="o", color="gray", markerfacecolor="none",
               linestyle="none", markersize=10, label="naive Inkling-Small (base control)"),
        Line2D([], [], marker="*", color="gray", linestyle="none",
               markersize=15, label="tuned (Opus-distillation LoRA)"),
    ], loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    for domain in DOMAIN_ORDER:
        b, t = agg.get(("base", domain)), agg.get(("tuned", domain))
        if b and t:
            print(f"  {domain:8s} base pass={_mean(b['pass']):.3f} $/succ={_mean(b['cost']):.3f}"
                  f"  | tuned pass={_mean(t['pass']):.3f} $/succ={_mean(t['cost']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
