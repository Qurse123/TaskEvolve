"""Two views of the headline metric on the validation/retail arms (paper Tier-2 #9).

Renders side by side, from `experiments/results.csv`:
  (left)  the raw cost-vs-success scatter  -- the 2-D *truth* the scalar hides
  (right) the same arms ranked by cost_per_successful_task -- the *scalar* view

Seeing them together makes plain what the scalar compresses: the ranking on the
right throws away the success axis on the left. Only the retail-domain
validation-split arms are shown (the head-to-head set in statistics.md).

Run:  python -m scripts.plot_metric_views
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CSV = Path("experiments/results.csv")
OUT = Path("experiments/plots/metric_views.png")

# Effective-model tag for a routing harness (base model != what actually ran),
# mirroring scripts.plot_results._ROUTING_HARNESS_MODEL.
_ROUTING_HARNESS_MODEL = {"v0.2": "Opus->Sonnet5"}


def cost_per_successful_task(
    total_cost: float, num_tasks: int, pass_rate: float
) -> Optional[float]:
    """total_cost / (num_tasks * pass_rate).

    Returns None when it is undefined -- pass_rate == 0 (no successful task) or a
    non-positive task count -- rather than raising or returning inf, so callers
    can drop the point. This is the ``success -> 0`` blow-up the metric note flags.
    """
    successes = num_tasks * pass_rate
    if successes <= 0:
        return None
    return total_cost / successes


def _short_model(agent_model: str) -> str:
    return agent_model.rsplit("/", 1)[-1]


def _arm_label(agent_model: str, harness_version: str) -> str:
    tag = _ROUTING_HARNESS_MODEL.get(harness_version, _short_model(agent_model))
    return f"{tag} ({harness_version})"


def _mean(xs):
    return sum(xs) / len(xs)


def load(csv_path: Path = CSV) -> dict:
    """Aggregate retail (validation-split) arms into per-arm mean metrics."""
    agg: dict = defaultdict(lambda: {"pass": [], "cpt": [], "cst": []})
    for r in csv.DictReader(open(csv_path)):
        if r.get("split") != "validation" or r.get("domain") != "retail":
            continue
        num_tasks = int(r["num_tasks"])
        total_cost = float(r["total_cost_usd"])
        pass_rate = float(r["pass_rate"])
        label = _arm_label(r["agent_model"], r["harness_version"])
        cst = cost_per_successful_task(total_cost, num_tasks, pass_rate)
        agg[label]["pass"].append(pass_rate)
        agg[label]["cpt"].append(total_cost / num_tasks)
        if cst is not None:
            agg[label]["cst"].append(cst)
    return {
        k: {
            "pass": _mean(v["pass"]),
            "cpt": _mean(v["cpt"]),
            "cst": _mean(v["cst"]) if v["cst"] else None,
        }
        for k, v in agg.items()
    }


def main() -> int:
    arms = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(arms, key=lambda k: arms[k]["cst"] or float("inf"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_for = {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}

    fig, (scat, bar) = plt.subplots(1, 2, figsize=(14, 6))

    # (left) raw 2-D cost-vs-success scatter -- the truth.
    for lab in labels:
        a = arms[lab]
        scat.scatter([a["cpt"]], [a["pass"]], s=120, color=color_for[lab],
                     edgecolors="black", zorder=3, label=lab)
        scat.annotate(lab, (a["cpt"], a["pass"]), textcoords="offset points",
                      xytext=(8, 4), fontsize=8)
    scat.set_xscale("log")
    scat.set_xlabel("cost per task (USD, log scale)  —  cheaper -->")
    scat.set_ylabel("task success rate  —  higher better")
    scat.set_title("(a) raw cost-vs-success frontier  —  the 2-D truth")
    scat.grid(True, which="both", alpha=0.25)

    # (right) arms ranked by the scalar -- what the scalar compresses to.
    ranked = [lab for lab in labels if arms[lab]["cst"] is not None]
    ys = range(len(ranked))
    bar.barh(list(ys), [arms[lab]["cst"] for lab in ranked],
             color=[color_for[lab] for lab in ranked], edgecolor="black")
    for y, lab in zip(ys, ranked):
        bar.text(arms[lab]["cst"], y, f"  ${arms[lab]['cst']:.4f}",
                 va="center", fontsize=8)
    bar.set_yticks(list(ys))
    bar.set_yticklabels(ranked, fontsize=8)
    bar.invert_yaxis()  # cheapest (best) on top
    bar.set_xlabel("cost per successful task (USD)  —  the scalar")
    bar.set_title("(b) scalar ranking  —  success axis compressed away")
    bar.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Headline metric: the scalar (b) is a lossy summary of the frontier (a)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    for lab in labels:
        a = arms[lab]
        cst = f"${a['cst']:.4f}" if a["cst"] is not None else "undefined"
        print(f"  {lab:22s} pass={a['pass']:.3f} $/task={a['cpt']:.4f} $/succ={cst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
