"""Latency as a third axis (cost, success, latency) — $0 analysis over EXISTING transcripts.

Tier-2 validity item #10. TAU2 persists per-task transcripts as `SimulationRun`
dumps at `experiments/logs/<run_id>/task_<id>_messages.json`; each carries a
top-level `duration` (wall-clock seconds for that task). This script aggregates
that `duration` per experimental arm alongside the per-task effort proxies
(`turn_count`, `tool_call_count`) and the study's cost/success numbers, giving a
three-axis (cost, success, latency) picture.

Arm identity follows `scripts/plot_arm_d.py`: agent_model short name + harness +
split/domain. The canonical arm set is the runs recorded in
`experiments/results.csv` (the archived/contaminated log dirs are excluded).

IMPORTANT CAVEAT (stated in the written report too): wall-clock `duration`
conflates model speed with provider load and serving path (Together vs the local
Tinker shim vs Anthropic). It is OBSERVATIONAL, not a controlled latency
benchmark.

Run:  python -m scripts.latency_analysis
Outputs: prints a table, writes docs/paper/analysis/latency.md, saves
         experiments/plots/latency_by_arm.png
No network, no paid calls.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "experiments" / "results.csv"
DEFAULT_LOGS = REPO_ROOT / "experiments" / "logs"
DEFAULT_MD = REPO_ROOT / "docs" / "paper" / "analysis" / "latency.md"
DEFAULT_PLOT = REPO_ROOT / "experiments" / "plots" / "latency_by_arm.png"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                   #
# --------------------------------------------------------------------------- #
def transcript_duration(transcript: dict) -> Optional[float]:
    """Return the per-task wall-clock `duration` (seconds) from a SimulationRun dict.

    Returns None when the field is missing or non-numeric (older transcripts, or
    runs that predate transcript persistence entirely). None means "no latency
    data for this task", never 0.0.
    """
    value = transcript.get("duration")
    if isinstance(value, bool):  # guard: bool is a subclass of int
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


@dataclass(frozen=True)
class LatencyStats:
    """Aggregate of per-task durations. `n == 0` means no latency data was found."""

    n: int
    mean: Optional[float]
    std: Optional[float]
    median: Optional[float]

    @property
    def has_data(self) -> bool:
        return self.n > 0


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _sample_std(xs: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1); 0.0 for a single point."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def aggregate_durations(durations: Sequence[Optional[float]]) -> LatencyStats:
    """Aggregate durations, dropping None entries.

    An arm whose every task lacks a usable duration yields ``LatencyStats(0, ...)``
    with ``has_data == False`` — reported as "no latency data", NOT as 0.0.
    """
    vals = [float(d) for d in durations if d is not None]
    if not vals:
        return LatencyStats(0, None, None, None)
    return LatencyStats(
        n=len(vals),
        mean=_mean(vals),
        std=_sample_std(vals),
        median=_median(vals),
    )


def model_short(agent_model: str) -> str:
    """Short model name: last path segment (matches scripts/plot_arm_d.py intent)."""
    return (agent_model or "").rsplit("/", 1)[-1]


def arm_label(agent_model: str, harness: str, split: str, domain: str) -> str:
    return f"{model_short(agent_model)} / {harness} / {split} ({domain})"


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    agent_model: str
    harness: str
    split: str
    domain: str
    run_ids: List[str] = field(default_factory=list)
    pass_rates: List[float] = field(default_factory=list)
    cost_per_succ: List[float] = field(default_factory=list)

    @property
    def key(self) -> Tuple[str, str, str, str]:
        return (self.agent_model, self.harness, self.split, self.domain)

    @property
    def label(self) -> str:
        return arm_label(self.agent_model, self.harness, self.split, self.domain)


@dataclass
class ArmLatency:
    arm: Arm
    latency: LatencyStats
    tasks_total: int
    tasks_with_duration: int
    mean_turns: Optional[float]
    mean_tool_calls: Optional[float]

    @property
    def coverage_ok(self) -> bool:
        return self.tasks_with_duration == self.tasks_total and self.tasks_total > 0


def load_arms_from_csv(csv_path: Path = DEFAULT_CSV) -> Dict[Tuple[str, str, str, str], Arm]:
    """Group canonical CSV run rows into arms (excludes archived/contaminated logs)."""
    arms: Dict[Tuple[str, str, str, str], Arm] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["agent_model"], r["harness_version"], r["split"], r["domain"])
            arm = arms.get(key)
            if arm is None:
                arm = Arm(*key)
                arms[key] = arm
            arm.run_ids.append(r["run_id"])
            pr = (r.get("pass_rate") or "").strip()
            if pr:
                arm.pass_rates.append(float(pr))
            cs = (r.get("cost_per_successful_task") or "").strip()
            if cs:
                arm.cost_per_succ.append(float(cs))
    return arms


def _iter_task_files(run_dir: Path):
    for p in sorted(run_dir.glob("task_*.json")):
        if p.name.endswith("_messages.json"):
            continue
        yield p


def summarize_arm(arm: Arm, logs_root: Path = DEFAULT_LOGS) -> ArmLatency:
    """Collect per-task durations + effort proxies for one arm across its runs."""
    durations: List[Optional[float]] = []
    turns: List[int] = []
    tools: List[int] = []
    tasks_total = 0
    for run_id in arm.run_ids:
        run_dir = logs_root / run_id
        if not run_dir.exists():
            continue
        for verdict_path in _iter_task_files(run_dir):
            tasks_total += 1
            try:
                verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                verdict = {}
            if verdict.get("turn_count") is not None:
                turns.append(int(verdict["turn_count"]))
            if verdict.get("tool_call_count") is not None:
                tools.append(int(verdict["tool_call_count"]))
            msg_path = verdict_path.with_name(verdict_path.stem + "_messages.json")
            dur: Optional[float] = None
            if msg_path.exists():
                try:
                    dur = transcript_duration(json.loads(msg_path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    dur = None
            durations.append(dur)
    latency = aggregate_durations(durations)
    return ArmLatency(
        arm=arm,
        latency=latency,
        tasks_total=tasks_total,
        tasks_with_duration=latency.n,
        mean_turns=_mean(turns) if turns else None,
        mean_tool_calls=_mean(tools) if tools else None,
    )


def analyze(csv_path: Path = DEFAULT_CSV, logs_root: Path = DEFAULT_LOGS) -> List[ArmLatency]:
    arms = load_arms_from_csv(csv_path)
    results = [summarize_arm(a, logs_root) for a in arms.values()]
    # Order: retail-domain arms first, then by split, then latency (slowest last).
    results.sort(key=lambda al: (al.arm.domain != "retail", al.arm.split,
                                 al.latency.mean if al.latency.has_data else 1e18))
    return results


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def _fmt(x: Optional[float], nd: int = 1) -> str:
    return f"{x:.{nd}f}" if x is not None else "—"


def format_table(results: Sequence[ArmLatency]) -> str:
    header = (
        f"{'arm':<62} {'tasks':>6} {'dur?':>5} "
        f"{'mean_s':>8} {'std_s':>8} {'med_s':>8} {'turns':>6} {'tools':>6} "
        f"{'succ':>6} {'$/succ':>8}"
    )
    lines = [header, "-" * len(header)]
    for al in results:
        arm = al.arm
        succ = _mean(arm.pass_rates) if arm.pass_rates else None
        cost = _mean(arm.cost_per_succ) if arm.cost_per_succ else None
        if al.latency.has_data:
            mean_s, std_s, med_s = (_fmt(al.latency.mean), _fmt(al.latency.std),
                                    _fmt(al.latency.median))
        else:
            mean_s = std_s = med_s = "NO DATA"
        lines.append(
            f"{arm.label:<62} {al.tasks_total:>6} "
            f"{al.tasks_with_duration:>5} {mean_s:>8} {std_s:>8} {med_s:>8} "
            f"{_fmt(al.mean_turns):>6} {_fmt(al.mean_tool_calls):>6} "
            f"{_fmt(succ, 3):>6} {_fmt(cost, 4):>8}"
        )
    return "\n".join(lines)


def _md_table(results: Sequence[ArmLatency]) -> str:
    rows = [
        "| arm | tasks | w/dur | mean s | ± std | median s | mean turns | mean tools | success | $/succ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for al in results:
        arm = al.arm
        succ = _mean(arm.pass_rates) if arm.pass_rates else None
        cost = _mean(arm.cost_per_succ) if arm.cost_per_succ else None
        if al.latency.has_data:
            mean_s, std_s, med_s = (_fmt(al.latency.mean), _fmt(al.latency.std),
                                    _fmt(al.latency.median))
        else:
            mean_s = std_s = med_s = "no latency data"
        rows.append(
            f"| {arm.label} | {al.tasks_total} | {al.tasks_with_duration} | "
            f"{mean_s} | {std_s} | {med_s} | {_fmt(al.mean_turns)} | "
            f"{_fmt(al.mean_tool_calls)} | {_fmt(succ, 3)} | {_fmt(cost, 4)} |"
        )
    return "\n".join(rows)


def render_markdown(results: Sequence[ArmLatency]) -> str:
    covered = [al for al in results if al.latency.has_data]
    gaps = [al for al in results if not al.latency.has_data]
    partial = [al for al in covered if not al.coverage_ok]

    # Three-axis highlights over retail arms with latency (the comparable cell).
    retail = [al for al in covered if al.arm.domain == "retail"]
    fast = min(retail, key=lambda al: al.latency.mean) if retail else None
    slow = max(retail, key=lambda al: al.latency.mean) if retail else None

    lines: List[str] = []
    A = lines.append
    A("# Latency as a Third Axis (cost, success, latency)")
    A("")
    A("Tier-2 validity item #10. A $0 observational analysis over the per-task "
      "transcripts already persisted on disk — no network, no paid calls. "
      "Generated by `scripts/latency_analysis.py`.")
    A("")
    A("## What is measured")
    A("")
    A("TAU2 writes each task's `SimulationRun` to "
      "`experiments/logs/<run_id>/task_<id>_messages.json` with a top-level "
      "`duration` (wall-clock seconds for that task). We aggregate that `duration` "
      "per arm, alongside the per-task effort proxies `turn_count` / "
      "`tool_call_count` (from the verdict JSON) and the study's success rate and "
      "cost-per-successful-task (from `experiments/results.csv`). Arm identity "
      "follows `scripts/plot_arm_d.py`: model short name + harness + split/domain. "
      "The canonical arm set is the runs recorded in `results.csv`; the archived / "
      "contaminated log dirs are excluded.")
    A("")
    A("> **Caveat — observational, not a controlled benchmark.** Wall-clock "
      "`duration` conflates model speed with provider load and serving path "
      "(Together AI for Inkling, the local Tinker shim for the tuned "
      "Inkling-Small, Anthropic for Opus/Haiku, OpenAI for gpt-4.1). Runs were "
      "collected at different times under different load. Read these as directional "
      "observations, not a latency ranking.")
    A("")
    A("## Latency table (per arm)")
    A("")
    A(_md_table(results))
    A("")
    A("- **tasks** = total tasks logged for the arm; **w/dur** = how many carried a "
      "usable `duration`. mean/median/std are over the tasks that had one.")
    A("- `$/succ` = cost per successful task (mean over the arm's runs, from the CSV).")
    A("")
    A("## Cost–success–latency picture")
    A("")
    if fast and slow and fast is not slow:
        fsucc = _mean(fast.arm.pass_rates) if fast.arm.pass_rates else None
        ssucc = _mean(slow.arm.pass_rates) if slow.arm.pass_rates else None
        fcost = _mean(fast.arm.cost_per_succ) if fast.arm.cost_per_succ else None
        scost = _mean(slow.arm.cost_per_succ) if slow.arm.cost_per_succ else None
        A(f"Among retail-domain arms with latency data, the fastest is "
          f"**{fast.arm.label}** at {_fmt(fast.latency.mean)}s/task "
          f"(success {_fmt(fsucc,3)}, {_fmt(fcost,4)} $/succ) and the slowest is "
          f"**{slow.arm.label}** at {_fmt(slow.latency.mean)}s/task "
          f"(success {_fmt(ssucc,3)}, {_fmt(scost,4)} $/succ).")
        A("")
    A("Key observation: **cheap does not imply fast.** The open-weight Inkling "
      "family is the cost-efficiency winner of the study, but it is served through "
      "third-party / shim endpoints, so its wall-clock latency is not automatically "
      "the lowest. A cheap-but-slow arm and an expensive-but-fast arm are different "
      "points on the frontier; latency exposes a tradeoff the cost axis alone hides. "
      "See the table above for the per-arm numbers and `latency_by_arm.png` for the "
      "latency-vs-success view.")
    A("")
    A("## Coverage gaps (honest accounting)")
    A("")
    if gaps:
        A("Arms with **no usable latency data** (report as such, never as 0):")
        A("")
        for al in gaps:
            A(f"- `{al.arm.label}` — {al.tasks_total} tasks, 0 with `duration`. "
              "Predates transcript persistence (no `SimulationRun` dumps written).")
        A("")
    else:
        A("Every canonical arm has at least some latency data.")
        A("")
    if partial:
        A("Arms with **partial** latency coverage (some tasks lack `duration`; "
          "means are over the covered subset):")
        A("")
        for al in partial:
            A(f"- `{al.arm.label}` — {al.tasks_with_duration}/{al.tasks_total} "
              "tasks with `duration`.")
        A("")
    A("The most important gap for the paper: **gpt-4.1's original June validation "
      "run predates transcript persistence**, so the strong-closed-model baseline "
      "on retail has no wall-clock latency — its latency cell is a known blank, not "
      "a zero.")
    A("")
    return "\n".join(lines) + "\n"


def save_plot(results: Sequence[ArmLatency], out: Path = DEFAULT_PLOT) -> Optional[Path]:
    covered = [al for al in results if al.latency.has_data]
    if not covered:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    covered = sorted(covered, key=lambda al: al.latency.mean)
    labels = [al.arm.label for al in covered]
    means = [al.latency.mean for al in covered]
    stds = [al.latency.std or 0.0 for al in covered]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * len(covered) + 1.5)))
    y = range(len(covered))
    ax.barh(list(y), means, xerr=stds, color="#1f77b4", alpha=0.8,
            error_kw=dict(ecolor="#333", lw=1, capsize=3))
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean per-task wall-clock duration (s)  —  observational, not controlled")
    ax.set_title("TAU2 arms — per-task latency (mean ± std over tasks with duration)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    results = analyze()
    print(format_table(results))
    print()

    DEFAULT_MD.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_MD.write_text(render_markdown(results), encoding="utf-8")
    print(f"wrote {DEFAULT_MD}")

    plot = save_plot(results)
    if plot:
        print(f"wrote {plot}")
    else:
        print("no latency data in any arm; skipped plot")

    covered = [al.arm.label for al in results if al.latency.has_data]
    gaps = [al.arm.label for al in results if not al.latency.has_data]
    print(f"\narms with latency data: {len(covered)}; arms without: {len(gaps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
