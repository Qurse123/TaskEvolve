"""Generate Opus-4.8 distillation transcripts for Arm D (PAID: runs Opus on the
`train_distill_*` splits — the teacher signal, `experiment.md §13.4` / design
spec §4.1).

Launch DETACHED (long, multi-domain — background runs get killed otherwise):

    AGENT_MODEL=anthropic/claude-opus-4-8 HARNESS_VERSION=v0.1 AGENT_NO_TEMPERATURE=1 \\
      nohup python -m scripts.run_distillation > distill.log 2>&1 &

Runs each **trained** domain's `train_distill` split with the configured agent
model in the static v0.1 harness, persisting a full TAU2 transcript per task
(`task_*_messages.json`) via `run_train_eval`'s logger. It then writes a pointer
file (`experiments/arm_d_distill_runs.json`) mapping every produced run dir ->
domain, so the next ($0) step — building the SFT set with
`arm_d/build_dataset.py` (passed-only, deduped) — can find the transcripts
without hand-copying directory names.

Seeds default to a distinct base (5001) so these rows stay visually separable
from proxy (1001) / validation (2001) in `results.csv`. Splits + domains are
design-fixed (design spec §4.1); the eval/held-out and banking splits are NEVER
generated here (they must stay untrained).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from results.logger import DEFAULT_LOGS_ROOT
from scripts.run_train_eval import RepeatMetrics, run_repeats
from settings import config

logger = logging.getLogger(__name__)

# The three TRAINED domains and their distillation splits (design spec §4.1).
# banking_knowledge is deliberately absent — it is the untrained transfer eval.
DISTILL_SPLITS: Tuple[Tuple[str, str], ...] = (
    ("train_distill_retail", "retail"),
    ("train_distill_airline", "airline"),
    ("train_distill_telecom", "telecom"),
)
DEFAULT_SEED_START = 5001
DEFAULT_REPEATS = 3
POINTER_PATH = Path("experiments/arm_d_distill_runs.json")


def run_distillation(
    *,
    seed_start: int = DEFAULT_SEED_START,
    repeats: int = DEFAULT_REPEATS,
    splits: Sequence[Tuple[str, str]] = DISTILL_SPLITS,
    runner: Callable[..., List[RepeatMetrics]] = run_repeats,
    pointer_path: Path = POINTER_PATH,
    logs_root: Path = DEFAULT_LOGS_ROOT,
) -> Dict[str, str]:
    """Run each `(split, domain)` distillation eval and record its run dirs.

    Each split gets a disjoint block of `repeats` consecutive seeds
    (`seed_start + i*repeats ..`). Returns a `{run_id: domain}` map and writes it
    to `pointer_path` (the run dir for a run_id is `logs_root/run_id`, so
    `build_dataset` consumes `[logs_root/run_id ...]` with this map as `domains`).
    `runner` is injected so the whole driver is unit-testable at $0.
    """
    run_domains: Dict[str, str] = {}
    for i, (split, domain) in enumerate(splits):
        block_seed = seed_start + i * repeats
        logger.info("distillation: %s (domain=%s) seeds %d..%d",
                    split, domain, block_seed, block_seed + repeats - 1)
        metrics = runner(split, seed_start=block_seed, repeats=repeats, domain=domain)
        for m in metrics:
            run_domains[m.run_id] = domain

    pointer = {
        "runs": run_domains,
        "logs_root": str(logs_root),
        "agent_model": config.AGENT_MODEL,
    }
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    logger.info("wrote %d run->domain entries to %s", len(run_domains), pointer_path)
    return run_domains


def _warn_if_not_opus_v01() -> None:
    """The teacher is meant to be Opus in the static v0.1 harness (design §4.1).
    A silent wrong-model run would waste real spend, so warn loudly — but don't
    hard-fail (the operator owns the env)."""
    model = (config.AGENT_MODEL or "").lower()
    if "opus" not in model:
        logger.warning("AGENT_MODEL=%r is not an Opus model — the design teacher "
                       "is Opus-4.8. Continuing as configured.", config.AGENT_MODEL)
    if config.HARNESS_VERSION != "v0.1":
        logger.warning("HARNESS_VERSION=%r is not v0.1 — Arm D uses the static "
                       "v0.1 harness. Continuing as configured.", config.HARNESS_VERSION)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START,
                        help=f"First seed (default {DEFAULT_SEED_START}).")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"Repeats per domain (default {DEFAULT_REPEATS}).")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _warn_if_not_opus_v01()
    run_domains = run_distillation(seed_start=args.seed_start, repeats=args.repeats)

    logger.info("=" * 60)
    logger.info("distillation generation complete: %d runs across %d domains",
                len(run_domains), len(set(run_domains.values())))
    logger.info("pointer file: %s", POINTER_PATH)
    logger.info("next ($0): build the SFT set (passed-only, deduped) from the "
                "run dirs in that pointer file via arm_d/build_dataset.py.")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
