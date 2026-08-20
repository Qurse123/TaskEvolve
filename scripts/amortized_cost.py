"""Arm D amortized training-cost accounting: break-even task volume for the
Opus-distilled tuned Inkling-Small vs the naive base control.

The Arm D headline reports lower per-task *serving* cost for the tuned model,
but that ignores the ONE-TIME fine-tuning cost. This computes the break-even
task volume N*(T_train) = T_train / delta, where delta is the per-task serving
saving (base - tuned). Because Tinker exposed NO billing telemetry
(`training_record.json: training_cost_usd = 0.0`), the true T_train is
UNRECORDED, so the analysis is PARAMETERIZED over an assumed-cost sweep rather
than committing to a single fabricated number.

Reads `experiments/results.csv` (per-task cost = total_cost_usd / num_tasks),
following `scripts/plot_arm_d.py`: only inkling-small v0.1 rows are used, and
the stale all-error / partial banking rows (no `cost_per_successful_task`) are
dropped, leaving the clean N=3 runs per (system, domain).

Writes:
  - stdout summary table
  - docs/paper/analysis/amortized_cost.md
  - experiments/plots/arm_d_breakeven.png

Run:  python -m scripts.amortized_cost
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CSV = Path("experiments/results.csv")
MD_OUT = Path("docs/paper/analysis/amortized_cost.md")
PLOT_OUT = Path("experiments/plots/arm_d_breakeven.png")
TRAINING_RECORD = Path("experiments/arm_d_training/run_20260807_222817/training_record.json")

DOMAIN_ORDER = ["retail", "airline", "telecom", "banking"]
DOMAIN_COLOR = {"retail": "#1f77b4", "airline": "#d62728",
                "telecom": "#2ca02c", "banking": "#9467bd", "aggregate": "#000000"}
SPLIT_TO_DOMAIN = {"validation": "retail", "eval_airline": "airline",
                   "eval_telecom": "telecom", "transfer_banking": "banking"}

# ASSUMPTIONS, not measured. Tinker exposed no billing telemetry, so the true
# one-time fine-tuning cost is unknown; we sweep a plausible range of dollar
# values to report break-even task volume as a function of it.
T_TRAIN_SWEEP = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]


@dataclass
class Cell:
    """Pooled runs for one (system, domain): task-weighted cost and success."""
    total_cost: float = 0.0
    num_tasks: int = 0
    num_passed: int = 0

    def add(self, total_cost: float, num_tasks: int, num_passed: int) -> None:
        self.total_cost += total_cost
        self.num_tasks += num_tasks
        self.num_passed += num_passed

    @property
    def per_task_cost(self) -> float:
        return self.total_cost / self.num_tasks

    @property
    def pass_rate(self) -> float:
        return self.num_passed / self.num_tasks

    @property
    def per_success_cost(self) -> float:
        if self.num_passed == 0:
            return float("inf")
        return self.total_cost / self.num_passed


@dataclass(frozen=True)
class DomainDelta:
    """Per-task and per-successful-task serving savings for one domain."""
    domain: str
    base_per_task: float
    tuned_per_task: float
    base_pass: float
    tuned_pass: float
    base_per_success: float
    tuned_per_success: float

    @property
    def delta_per_task(self) -> float:
        return self.base_per_task - self.tuned_per_task

    @property
    def delta_per_success(self) -> float:
        return self.base_per_success - self.tuned_per_success


def breakeven_n(t_train: float, delta: float) -> float | None:
    """Break-even task count N*(T_train) = T_train / delta.

    Returns None when delta <= 0 (tuned is not cheaper, so it either never
    breaks even, or — at T_train=0 with delta>0 — is immediately favorable).
    A return of 0.0 means break-even at zero tasks (T_train == 0, delta > 0).
    """
    if delta <= 0:
        return None
    return t_train / delta


def load_cells(rows) -> dict:
    """Aggregate clean inkling-small v0.1 rows into (system, domain) -> Cell.

    Follows scripts/plot_arm_d.py: rows without `cost_per_successful_task`
    (the stale all-error / partial banking runs) are dropped.
    """
    cells: dict = defaultdict(Cell)
    for r in rows:
        model = (r.get("agent_model") or "").lower()
        if "inkling-small" not in model or r.get("harness_version") != "v0.1":
            continue
        if not (r.get("cost_per_successful_task") or "").strip():
            continue  # stale all-error / partial run -> drop
        system = "tuned" if "armd" in model else "base"
        domain = SPLIT_TO_DOMAIN.get(r["split"], r["split"])
        cells[(system, domain)].add(
            float(r["total_cost_usd"]), int(r["num_tasks"]), int(r["num_passed"]))
    return cells


def compute_deltas(cells: dict) -> list:
    """Build per-domain + aggregate DomainDelta rows from the pooled cells."""
    deltas: list = []
    agg_base, agg_tuned = Cell(), Cell()
    for domain in DOMAIN_ORDER:
        base = cells.get(("base", domain))
        tuned = cells.get(("tuned", domain))
        if not base or not tuned:
            continue
        agg_base.add(base.total_cost, base.num_tasks, base.num_passed)
        agg_tuned.add(tuned.total_cost, tuned.num_tasks, tuned.num_passed)
        deltas.append(DomainDelta(
            domain=domain,
            base_per_task=base.per_task_cost, tuned_per_task=tuned.per_task_cost,
            base_pass=base.pass_rate, tuned_pass=tuned.pass_rate,
            base_per_success=base.per_success_cost,
            tuned_per_success=tuned.per_success_cost))
    if agg_base.num_tasks and agg_tuned.num_tasks:
        deltas.append(DomainDelta(
            domain="aggregate",
            base_per_task=agg_base.per_task_cost, tuned_per_task=agg_tuned.per_task_cost,
            base_pass=agg_base.pass_rate, tuned_pass=agg_tuned.pass_rate,
            base_per_success=agg_base.per_success_cost,
            tuned_per_success=agg_tuned.per_success_cost))
    return deltas


def _fmt_n(n: float | None) -> str:
    if n is None:
        return "never"
    if n == 0.0:
        return "0 (immediate)"
    return f"{n:,.0f}"


def _breakeven_table(deltas: list, per_success: bool) -> list:
    """Rows of [domain, delta, N*(T) ...] for the markdown / stdout table."""
    table = []
    for d in deltas:
        delta = d.delta_per_success if per_success else d.delta_per_task
        row = [d.domain, delta]
        for t in T_TRAIN_SWEEP:
            row.append(breakeven_n(t, delta))
        table.append(row)
    return table


def _render_md(deltas: list) -> str:
    lines = []
    lines.append("# Arm D — Amortized Fine-Tuning Cost & Break-Even Task Volume\n")
    lines.append(
        "The Arm D headline reports lower per-task *serving* cost for the "
        "Opus-distilled tuned Inkling-Small than for the naive base control. "
        "That comparison ignores the **one-time fine-tuning cost** "
        "`T_train`. This note reports the break-even task volume "
        "`N*(T_train) = T_train / delta`, where `delta` is the per-task serving "
        "saving (base − tuned).\n")

    lines.append("## Training cost is UNRECORDED\n")
    lines.append(
        f"`{TRAINING_RECORD}` records `training_cost_usd: 0.0` **not** because "
        "training was free, but because **Tinker exposed no billing telemetry** "
        "(`cost_note: \"no billing telemetry exposed; fallback = steps * "
        "cost_per_step_usd (0.0)\"`). Training provenance: base "
        "`thinkingmachines/Inkling-Small`, LoRA rank 32, lr 1e-4, **88 steps**, "
        "169 train + 26 holdout examples, best holdout loss 10.81.\n")
    lines.append(
        "Because the dollar cost is unknown, the analysis below is "
        "**parameterized** over an assumed `T_train` sweep — these are "
        "**ASSUMPTIONS, not measured values**. To obtain the real number, "
        "multiply the Tinker GPU-hours consumed by the 88 training steps "
        "against the Tinker/GPU hourly rate (both external to this repo).\n")

    lines.append("## Per-task serving cost and per-task saving (delta)\n")
    lines.append("Per-task cost = `total_cost_usd / num_tasks`, task-weighted "
                 "over the clean N=3 runs per cell (stale all-error/partial "
                 "banking rows with no `cost_per_successful_task` dropped, per "
                 "`scripts/plot_arm_d.py`).\n")
    lines.append("| domain | base $/task | tuned $/task | delta $/task | "
                 "base pass | tuned pass |")
    lines.append("|---|---|---|---|---|---|")
    for d in deltas:
        lines.append(
            f"| {d.domain} | {d.base_per_task:.4f} | {d.tuned_per_task:.4f} | "
            f"{d.delta_per_task:+.4f} | {d.base_pass:.3f} | {d.tuned_pass:.3f} |")
    lines.append("")

    hdr = "| domain | delta $/task | " + " | ".join(
        f"T=${t:g}" for t in T_TRAIN_SWEEP) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(T_TRAIN_SWEEP)) + "|"
    lines.append("## Break-even task count N*(T_train) — per TASK\n")
    lines.append("Number of tasks the tuned model must serve before its "
                 "serving savings repay the assumed one-time `T_train`.\n")
    lines.append(hdr)
    lines.append(sep)
    for row in _breakeven_table(deltas, per_success=False):
        domain, delta = row[0], row[1]
        cells = " | ".join(_fmt_n(n) for n in row[2:])
        lines.append(f"| {domain} | {delta:+.4f} | {cells} |")
    lines.append("")

    lines.append("## Break-even count N*(T_train) — per SUCCESSFUL task\n")
    lines.append("Because tuned changes **both** cost and success rate, the "
                 "same break-even expressed against `cost_per_successful_task` "
                 "(`$/task ÷ pass_rate`). `delta` here is the per-successful-task "
                 "saving; N* counts **successful** tasks.\n")
    lines.append("| domain | base $/succ | tuned $/succ | delta $/succ | " +
                 " | ".join(f"T=${t:g}" for t in T_TRAIN_SWEEP) + " |")
    lines.append("|---|---|---|---|" + "|".join(["---"] * len(T_TRAIN_SWEEP)) + "|")
    for d in deltas:
        row_ns = [breakeven_n(t, d.delta_per_success) for t in T_TRAIN_SWEEP]
        cells = " | ".join(_fmt_n(n) for n in row_ns)
        lines.append(
            f"| {d.domain} | {d.base_per_success:.4f} | "
            f"{d.tuned_per_success:.4f} | {d.delta_per_success:+.4f} | {cells} |")
    lines.append("")

    lines.append("## Reading the table\n")
    lines.append(
        "- Where `delta > 0`, larger `T_train` pushes break-even to more tasks; "
        "a domain with a large per-task saving (e.g. banking, telecom) amortizes "
        "training far faster than one with a small saving (retail).\n"
        "- Where `delta <= 0`, the tuned model is **not** cheaper to serve, so "
        "it never breaks even on cost alone (reported as `never`).\n"
        "- At `T_train = $0` with `delta > 0`, break-even is immediate (0 tasks): "
        "tuning is favorable from the first task.\n")
    lines.append(f"\n_Plot: `{PLOT_OUT}`. Generated by "
                 "`scripts/amortized_cost.py`._\n")
    return "\n".join(lines)


def _make_plot(deltas: list) -> None:
    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ts = [t for t in T_TRAIN_SWEEP]
    for d in deltas:
        if d.delta_per_task <= 0:
            continue  # never breaks even; nothing to draw
        ys = [breakeven_n(t, d.delta_per_task) for t in ts]
        color = DOMAIN_COLOR.get(d.domain, "#7f7f7f")
        style = "--" if d.domain == "aggregate" else "-"
        ax.plot(ts, ys, style, color=color, marker="o", linewidth=2,
                label=f"{d.domain} (delta=${d.delta_per_task:.4f}/task)")
    ax.set_xlabel("assumed one-time training cost T_train (USD)  —  ASSUMPTION")
    ax.set_ylabel("break-even task count  N*(T_train) = T_train / delta")
    ax.set_title("Arm D: break-even task volume vs assumed fine-tuning cost\n"
                 "(tuned Inkling-Small must serve N* tasks to repay T_train)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    plt.close(fig)


def main() -> int:
    with open(CSV) as fh:
        rows = list(csv.DictReader(fh))
    cells = load_cells(rows)
    deltas = compute_deltas(cells)

    print("Arm D amortized cost — per-task serving saving (delta = base - tuned)")
    print("T_train values are ASSUMPTIONS (Tinker exposed no billing telemetry).\n")
    print(f"{'domain':10s} {'base$/task':>10s} {'tuned$/task':>11s} "
          f"{'delta$/task':>11s}  N*(T=$10)  N*(T=$50)")
    for d in deltas:
        n10 = breakeven_n(10.0, d.delta_per_task)
        n50 = breakeven_n(50.0, d.delta_per_task)
        print(f"{d.domain:10s} {d.base_per_task:10.4f} {d.tuned_per_task:11.4f} "
              f"{d.delta_per_task:+11.4f}  {_fmt_n(n10):>9s}  {_fmt_n(n50):>9s}")

    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text(_render_md(deltas))
    print(f"\nwrote {MD_OUT}")
    _make_plot(deltas)
    print(f"wrote {PLOT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
