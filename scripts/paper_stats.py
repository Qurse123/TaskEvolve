"""Statistical rigor for the TaskEvolve arm comparisons (paper Tier-1 item #4).

Reads `experiments/results.csv` and produces, for the headline comparisons:
  * per-arm percentile-bootstrap 95% CIs for pass_rate AND cost_per_successful_task;
  * unpaired two-sided permutation-test p-values for the headline deltas
    (exact enumeration when the sample is small, Monte-Carlo otherwise);
  * effect sizes (raw delta + Cohen's d) beside every p-value;
  * a determinism check: bootstrap 95% CI on the pass-rate std-ratio
    (tuned / base) per Arm-D domain, flagging the tiny n;
  * a Holm multiple-comparisons correction across the family of headline tests,
    reporting both raw and adjusted p.

The ledger has NO seed column, so every cross-arm comparison is UNPAIRED.

Run:  python -m scripts.paper_stats     (also writes docs/paper/analysis/statistics.md)
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np

CSV = Path("experiments/results.csv")
OUT = Path("docs/paper/analysis/statistics.md")
HI_CAP = 100.0  # display cap for the std-ratio CI upper bound (n=3 blow-up)

# Domain -> split/domain key in the ledger for the Arm-D per-domain runs.
DOMAIN_SPLIT = {
    "retail": "validation",
    "airline": "eval_airline",
    "telecom": "eval_telecom",
    "banking": "transfer_banking",
}


# --------------------------------------------------------------------------- #
# Pure statistics primitives (numpy/stdlib only — scipy is unavailable).
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    data,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    stat: Callable = np.mean,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `stat` over `data` (resampled with replacement)."""
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return (float("nan"), float("nan"))
    rng = rng if rng is not None else np.random.default_rng(0)
    n = data.size
    boots = np.array([stat(data[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def cohens_d(a, b) -> float:
    """Pooled-std standardized effect size (mean(a) - mean(b)) / s_pooled."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


def permutation_test(
    a,
    b,
    *,
    max_exact: int = 200_000,
    n_mc: int = 20_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Unpaired two-sided permutation test on the difference of means.

    Enumerates every partition exactly when C(N, n) <= max_exact (deterministic,
    the common case here where n <= 5), otherwise falls back to Monte-Carlo.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n, N = a.size, pooled.size
    tol = 1e-9
    total = math.comb(N, n)
    if total <= max_exact:
        s_all = pooled.sum()
        count = 0
        for idx in combinations(range(N), n):
            g = pooled[list(idx)]
            gm = g.mean()
            rm = (s_all - g.sum()) / (N - n)
            if abs(gm - rm) >= abs(obs) - tol:
                count += 1
        return count / total
    rng = rng if rng is not None else np.random.default_rng(0)
    count = 0
    for _ in range(n_mc):
        perm = rng.permutation(pooled)
        d = perm[:n].mean() - perm[n:].mean()
        if abs(d) >= abs(obs) - tol:
            count += 1
    return (count + 1) / (n_mc + 1)


def holm_adjust(pvals) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    pvals = list(pvals)
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def std_ratio_ci(
    tuned,
    base,
    *,
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Point estimate + bootstrap 95% CI for std(tuned)/std(base)."""
    tuned = np.asarray(tuned, dtype=float)
    base = np.asarray(base, dtype=float)
    rng = rng if rng is not None else np.random.default_rng(0)
    sb = base.std(ddof=1)
    point = float(tuned.std(ddof=1) / sb) if sb > 0 else float("nan")
    vals = []
    for _ in range(n_boot):
        ts = tuned[rng.integers(0, tuned.size, tuned.size)]
        bs = base[rng.integers(0, base.size, base.size)]
        d = bs.std(ddof=1)
        if d > 0:
            vals.append(ts.std(ddof=1) / d)
    if not vals:
        return (point, float("nan"), float("nan"))
    vals = np.asarray(vals)
    return (point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# --------------------------------------------------------------------------- #
# Data loading / arm selection.
# --------------------------------------------------------------------------- #
def short_model(agent_model: str) -> str:
    """Canonical short arm name from the ledger's `agent_model` string."""
    m = agent_model.lower()
    if "armd-inkling-small-tuned" in m:
        return "armd-tuned"
    if "inkling-small" in m:
        return "inkling-small"
    if m == "openai/thinkingmachines/inkling":
        return "inkling"
    if "opus-4-8" in m:
        return "opus"
    if m == "gpt-4.1":
        return "gpt-4.1"
    return agent_model


@dataclass
class Arm:
    """One arm's runs for a given (model, harness, split/domain)."""

    label: str
    passes: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.passes)


def load_rows(csv_path: Path = CSV) -> list[dict]:
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def select_arm(rows, *, label, model, harness, split) -> Arm:
    """Rows matching (short model, harness, split); drops broken runs that carry
    no cost_per_successful_task (the stale all-error banking-base rows)."""
    arm = Arm(label=label)
    for r in rows:
        if short_model(r["agent_model"]) != model:
            continue
        if r["harness_version"] != harness or r["split"] != split:
            continue
        cs = (r.get("cost_per_successful_task") or "").strip()
        if not cs:  # broken/all-error run -> exclude from both metrics for a clean n
            continue
        arm.passes.append(float(r["pass_rate"]))
        arm.costs.append(float(cs))
    return arm


# --------------------------------------------------------------------------- #
# Comparison assembly.
# --------------------------------------------------------------------------- #
@dataclass
class CompareResult:
    name: str
    metric: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    delta: float
    d: float
    p_raw: float
    p_holm: float = float("nan")


def compare(name: str, metric: str, a: Arm, b: Arm, *, rng) -> CompareResult:
    av = a.passes if metric == "pass_rate" else a.costs
    bv = b.passes if metric == "pass_rate" else b.costs
    ma, mb = float(np.mean(av)), float(np.mean(bv))
    return CompareResult(
        name=name,
        metric=metric,
        n_a=len(av),
        n_b=len(bv),
        mean_a=ma,
        mean_b=mb,
        delta=ma - mb,
        d=cohens_d(av, bv),
        p_raw=permutation_test(av, bv, rng=rng),
    )


def build_arms(rows) -> dict[str, Arm]:
    """The six retail arms + Arm-D base/tuned across the four domains."""
    arms: dict[str, Arm] = {}
    # Retail (validation split) arms.
    arms["gpt-4.1 (ArmA, v0.1) retail"] = select_arm(
        rows, label="gpt-4.1 (ArmA, v0.1) retail", model="gpt-4.1", harness="v0.1", split="validation")
    arms["Inkling (ArmC, v0.1) retail"] = select_arm(
        rows, label="Inkling (ArmC, v0.1) retail", model="inkling", harness="v0.1", split="validation")
    arms["Opus (v0.1) retail"] = select_arm(
        rows, label="Opus (v0.1) retail", model="opus", harness="v0.1", split="validation")
    arms["Opus->Sonnet (ArmB, v0.2) retail"] = select_arm(
        rows, label="Opus->Sonnet (ArmB, v0.2) retail", model="opus", harness="v0.2", split="validation")
    # Arm-D base + tuned per domain.
    for domain, split in DOMAIN_SPLIT.items():
        arms[f"ArmD base (Inkling-Small) {domain}"] = select_arm(
            rows, label=f"ArmD base (Inkling-Small) {domain}",
            model="inkling-small", harness="v0.1", split=split)
        arms[f"ArmD tuned {domain}"] = select_arm(
            rows, label=f"ArmD tuned {domain}",
            model="armd-tuned", harness="v0.1", split=split)
    return arms


def build_comparisons(arms, *, rng) -> list[CompareResult]:
    """Headline family: 7 comparisons x 2 metrics = 14 tests, Holm-corrected."""
    specs = [
        ("ArmC Inkling vs Opus v0.1 [retail]",
         arms["Inkling (ArmC, v0.1) retail"], arms["Opus (v0.1) retail"]),
        ("ArmC Inkling vs ArmA gpt-4.1 [retail]",
         arms["Inkling (ArmC, v0.1) retail"], arms["gpt-4.1 (ArmA, v0.1) retail"]),
        ("ArmB Opus->Sonnet vs ArmA gpt-4.1 [retail]",
         arms["Opus->Sonnet (ArmB, v0.2) retail"], arms["gpt-4.1 (ArmA, v0.1) retail"]),
    ]
    for domain in DOMAIN_SPLIT:
        specs.append((
            f"ArmD tuned vs base [{domain}]",
            arms[f"ArmD tuned {domain}"], arms[f"ArmD base (Inkling-Small) {domain}"],
        ))
    results: list[CompareResult] = []
    for name, a, b in specs:
        for metric in ("pass_rate", "cost_per_successful_task"):
            results.append(compare(name, metric, a, b, rng=rng))
    adj = holm_adjust([r.p_raw for r in results])
    for r, pa in zip(results, adj):
        r.p_holm = pa
    return results


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def _fmt(x: float, nd: int = 4) -> str:
    return "nan" if x != x else f"{x:.{nd}f}"


def arm_ci_table(arms, *, rng) -> tuple[str, list[str]]:
    header = (
        "| Arm | n | pass_rate mean | pass_rate 95% CI | cost/succ mean | cost/succ 95% CI |\n"
        "|-----|---|----------------|------------------|----------------|------------------|\n"
    )
    lines = []
    for arm in arms.values():
        if arm.n == 0:
            continue
        p_ci = bootstrap_ci(arm.passes, rng=np.random.default_rng(rng.integers(1 << 31)))
        c_ci = bootstrap_ci(arm.costs, rng=np.random.default_rng(rng.integers(1 << 31)))
        lines.append(
            f"| {arm.label} | {arm.n} | {_fmt(np.mean(arm.passes), 3)} | "
            f"[{_fmt(p_ci[0], 3)}, {_fmt(p_ci[1], 3)}] | {_fmt(np.mean(arm.costs), 4)} | "
            f"[{_fmt(c_ci[0], 4)}, {_fmt(c_ci[1], 4)}] |"
        )
    return header, lines


def comparison_table(results) -> tuple[str, list[str]]:
    header = (
        "| Comparison (A vs B) | Metric | nA/nB | mean A | mean B | delta (A-B) | "
        "Cohen's d | raw p | Holm p | verdict |\n"
        "|---------------------|--------|-------|--------|--------|-------------|"
        "-----------|-------|--------|---------|\n"
    )
    lines = []
    for r in results:
        nd = 3 if r.metric == "pass_rate" else 4
        verdict = "significant" if r.p_holm < 0.05 else "n.s."
        lines.append(
            f"| {r.name} | {r.metric} | {r.n_a}/{r.n_b} | {_fmt(r.mean_a, nd)} | "
            f"{_fmt(r.mean_b, nd)} | {_fmt(r.delta, nd)} | {_fmt(r.d, 2)} | "
            f"{_fmt(r.p_raw, 4)} | {_fmt(r.p_holm, 4)} | {verdict} |"
        )
    return header, lines


def variance_table(arms, *, rng) -> tuple[str, list[str]]:
    header = (
        "| Domain | n | base pass std | tuned pass std | std-ratio (tuned/base) | 95% CI | flag |\n"
        "|--------|---|---------------|----------------|------------------------|--------|------|\n"
    )
    lines = []
    for domain in DOMAIN_SPLIT:
        base = arms[f"ArmD base (Inkling-Small) {domain}"]
        tuned = arms[f"ArmD tuned {domain}"]
        if base.n < 2 or tuned.n < 2:
            continue
        bstd = float(np.std(base.passes, ddof=1))
        tstd = float(np.std(tuned.passes, ddof=1))
        point, lo, hi = std_ratio_ci(
            tuned.passes, base.passes, rng=np.random.default_rng(rng.integers(1 << 31)))
        # At n=3 a resampled base std can be ~0, sending the ratio to +inf; report
        # a capped upper bound rather than a meaningless 1e14 (the blow-up itself
        # is the small-n instability the flag warns about).
        hi_txt = f">{HI_CAP:.0f} (unstable)" if hi > HI_CAP else _fmt(hi, 3)
        flag = "n=3 (tiny; CI wide/unstable, low power)" if min(base.n, tuned.n) <= 3 else ""
        lines.append(
            f"| {domain} | {min(base.n, tuned.n)} | {_fmt(bstd, 4)} | {_fmt(tstd, 4)} | "
            f"{_fmt(point, 3)} | [{_fmt(lo, 3)}, {hi_txt}] | {flag} |"
        )
    return header, lines


def render_report(arms, results, *, rng) -> str:
    arm_h, arm_l = arm_ci_table(arms, rng=rng)
    cmp_h, cmp_l = comparison_table(results)
    var_h, var_l = variance_table(arms, rng=rng)
    sig = [r for r in results if r.p_holm < 0.05]

    out = []
    out.append("# TaskEvolve — Statistical Analysis (paper Tier-1 #4)\n")
    out.append(
        "Generated by `scripts/paper_stats.py` over `experiments/results.csv`. "
        "Pure numpy/stdlib (no scipy): percentile bootstrap CIs and exact/Monte-Carlo "
        "unpaired permutation tests, Holm-corrected across the headline family.\n")
    out.append("## Caveats (read first)\n")
    out.append(
        "- **Small n.** Each arm has only **n=3–5** logged runs. Bootstrap CIs are "
        "correspondingly **wide** and the permutation tests have **low power**.\n"
        "- **Floor on p.** With n=3 vs n=3 there are only C(6,3)=20 partitions, so the "
        "smallest attainable two-sided permutation p is **2/20 = 0.10** — the Arm-D "
        "per-domain tests *cannot* reach p<0.05 no matter how large the effect. "
        "With n=5 vs n=5 the floor is 2/252 ≈ 0.0079.\n"
        "- **Unpaired only.** The ledger has **no seed column**, so runs cannot be "
        "paired across arms; every comparison is unpaired (throwing away any "
        "shared-seed variance reduction). **Recommendation: add a `seed` column to "
        "`results.csv` going forward** to enable paired tests and tighter CIs.\n"
        "- Cost is `cost_per_successful_task`; runs with no successful task (empty cell) "
        "are excluded so each arm has a clean n.\n")

    out.append("\n## Per-arm bootstrap 95% CIs\n")
    out.append(arm_h + "\n".join(arm_l) + "\n")

    out.append("\n## Headline comparisons (Holm-corrected family of 14 tests)\n")
    out.append(
        "delta = mean(A) − mean(B). Cohen's d is the pooled-std standardized effect. "
        "`verdict` = significant iff Holm-adjusted p < 0.05.\n\n")
    out.append(cmp_h + "\n".join(cmp_l) + "\n")

    out.append("\n## Determinism check — pass-rate std-ratio (tuned / base) per domain\n")
    out.append(
        "Ratio < 1 means the tuned model's outcomes are *more* deterministic "
        "(lower run-to-run spread) than the base. All n=3 → treat as directional only.\n\n")
    out.append(var_h + "\n".join(var_l) + "\n")

    out.append("\n## Summary\n")
    if sig:
        out.append("Significant after Holm correction (adjusted p < 0.05):\n")
        for r in sig:
            out.append(
                f"- **{r.name}** ({r.metric}): delta={_fmt(r.delta, 4)}, "
                f"d={_fmt(r.d, 2)}, raw p={_fmt(r.p_raw, 4)}, Holm p={_fmt(r.p_holm, 4)}.\n")
    else:
        out.append("No comparison survived Holm correction at alpha=0.05.\n")
    out.append(
        "\nEvery number above is over the stated n (3–5). Treat non-significant "
        "results as *underpowered*, not as evidence of no effect.\n")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def run(csv_path: Path = CSV, out_path: Path = OUT, *, seed: int = 20260819):
    rows = load_rows(csv_path)
    arms = build_arms(rows)
    results = build_comparisons(arms, rng=np.random.default_rng(seed))
    report = render_report(arms, results, rng=np.random.default_rng(seed + 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    # Readable stdout summary.
    print("=== Per-arm bootstrap 95% CIs ===")
    for arm in arms.values():
        if arm.n == 0:
            continue
        print(f"  {arm.label:42s} n={arm.n}  "
              f"pass={np.mean(arm.passes):.3f}  $/succ={np.mean(arm.costs):.4f}")
    print("\n=== Headline comparisons (raw p / Holm p) ===")
    for r in results:
        mark = "*" if r.p_holm < 0.05 else " "
        print(f" {mark} {r.name:44s} {r.metric:24s} "
              f"delta={r.delta:+.4f} d={r.d:+.2f} p={r.p_raw:.4f} holm={r.p_holm:.4f}")
    sig = [r for r in results if r.p_holm < 0.05]
    print(f"\n{len(sig)} of {len(results)} tests significant after Holm.")
    print(f"wrote {out_path}")
    return arms, results


if __name__ == "__main__":
    run()
