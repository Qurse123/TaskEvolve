"""Build the Arm D SFT dataset from distillation transcripts ($0, no model calls).

Reads the pointer written by ``scripts/run_distillation.py``
(``experiments/arm_d_distill_runs.json``: ``{"runs": {run_id -> domain}, ...}``),
runs the shared ``arm_d.build_dataset`` transform (passed-only + deduped ->
domain-balanced) via ``examples_from_distill_pointer``, and writes the audit
manifest (``experiments/arm_d_sft_manifest.json``: task_id/domain/source_run/
sha256 per example — design spec §8 guard 6). Prints per-domain + total counts
so the SFT set can be sanity-checked (non-empty, balanced) BEFORE paying to
train. Pure transform over existing transcripts — safe to rerun.

Run:  python -m scripts.build_distill_dataset
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from arm_d.build_dataset import examples_from_distill_pointer, write_manifest

logger = logging.getLogger(__name__)

DEFAULT_POINTER = Path("experiments/arm_d_distill_runs.json")
DEFAULT_MANIFEST = Path("experiments/arm_d_sft_manifest.json")


def build_distill_dataset(
    *,
    pointer_path: Path = DEFAULT_POINTER,
    manifest_path: Path = DEFAULT_MANIFEST,
    seed: int = 0,
    k: int = 2,
):
    """Build + balance the SFT examples from the pointer and write the audit
    manifest. Returns the balanced `list[TrainingExample]` (the training step
    rebuilds these in-process via the same `examples_from_distill_pointer`)."""
    examples = examples_from_distill_pointer(pointer_path, seed=seed, k=k)
    write_manifest(examples, manifest_path)
    counts = Counter(e.domain for e in examples)
    logger.info("SFT dataset: %d examples across %d domains", len(examples), len(counts))
    for domain in sorted(counts):
        logger.info("  %-12s %d", domain, counts[domain])
    if not examples:
        logger.warning("SFT dataset is EMPTY — no passed transcripts found under the "
                       "pointer's run dirs. Run scripts.run_distillation first, or "
                       "check that distillation produced passing tasks.")
    logger.info("audit manifest -> %s", manifest_path)
    return examples


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER,
                        help=f"Distillation pointer file (default {DEFAULT_POINTER}).")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"Output audit manifest (default {DEFAULT_MANIFEST}).")
    parser.add_argument("--seed", type=int, default=0, help="balance() shuffle seed.")
    parser.add_argument("--k", type=int, default=2,
                        help="balance() per-domain cap multiplier (default 2).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_distill_dataset(pointer_path=args.pointer, manifest_path=args.manifest,
                          seed=args.seed, k=args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
