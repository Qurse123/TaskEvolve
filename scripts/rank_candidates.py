"""Rank every measured harness candidate against the unmodified baseline.

The iterator no longer accepts or reverts greedily. It applies each candidate,
measures it on both seeds, records it, and reverts, so every row is measured from
the same v0.1 starting point. This script turns those records into the table the
winner is picked from, with the baseline as row 0.

Rows whose tasks crashed, or that never reached evaluation because preflight
caught them, carry no usable measurement and are listed separately rather than
ranked: a run that did no work reports near-zero cost and would otherwise top the
table.

Run:  python -m scripts.rank_candidates
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ITERATIONS = Path("experiments/iterations")


def _rows() -> tuple[list[dict], list[dict], Optional[dict]]:
    ranked: list[dict] = []
    unmeasured: list[dict] = []
    baseline: Optional[dict] = None

    for path in sorted(ITERATIONS.glob("iter_*/iteration.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if baseline is None and rec.get("cost_per_successful_task_before") is not None:
            baseline = {
                "id": "baseline",
                "file": "unmodified v0.1 harness",
                "summary": "change nothing",
                "success": rec.get("proxy_task_success_before_mean"),
                "cost": rec.get("cost_per_successful_task_before"),
            }
        row = {
            "id": rec.get("iteration_id"),
            "file": (rec.get("changed_file") or "?").replace("target_agent/", ""),
            "summary": (rec.get("change_summary") or "").strip(),
            "success": rec.get("proxy_task_success_after_mean"),
            "cost": rec.get("cost_per_successful_task_after"),
            "why": rec.get("reason_accepted_or_rejected") or "",
        }
        (ranked if row["cost"] is not None else unmeasured).append(row)
    return ranked, unmeasured, baseline


def main() -> int:
    ranked, unmeasured, baseline = _rows()
    if baseline is not None:
        ranked.append(baseline)
    ranked.sort(key=lambda r: r["cost"])

    print(f"\n{'':>2}  {'candidate':<12} {'file':<26} {'success':>8} {'$/succ':>9}  change")
    print("-" * 108)
    for i, r in enumerate(ranked):
        mark = "->" if r["id"] == "baseline" else f"{i:>2}"
        print(f"{mark}  {r['id']:<12} {r['file']:<26} {r['success']:>8.4f} "
              f"{r['cost']:>9.4f}  {r['summary'][:44]}")

    if ranked:
        best = ranked[0]
        print()
        if best["id"] == "baseline":
            print("  Winner: the unmodified harness. No candidate beat it, so harness")
            print("  optimization did not move the frontier. That is the result.")
        else:
            delta = (baseline["cost"] - best["cost"]) / baseline["cost"] * 100 if baseline else 0
            print(f"  Winner: {best['id']} ({best['file']}) at ${best['cost']:.4f} per successful task, "
                  f"{delta:.1f}% below baseline.")
            print(f"  Success {best['success']:.4f} against baseline "
                  f"{baseline['success']:.4f}." if baseline else "")

    if unmeasured:
        print(f"\n  {len(unmeasured)} candidate(s) produced no usable measurement:")
        for r in unmeasured:
            print(f"    {r['id']}  {r['file']:<26} {r['why'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
