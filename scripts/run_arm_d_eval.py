"""Arm D evaluation driver (PAID: runs the tuned model + matched base control on
the pre-registered eval splits — see experiments/arm_d_eval_precommit.md).

Runs BOTH systems the pre-registration fixes, over the 4 held-out/transfer
splits x seeds 2001-2003, in the static v0.1 harness through the SAME local
Tinker serving shim (arm_d/serving_shim.py) — an `openai/` + AGENT_API_BASE
seam pointed at localhost instead of Together, so the only variable is the
LoRA adapter (§8 guards 1 & 2):
  - base_control : naive Inkling-Small (openai/thinkingmachines/Inkling-Small)
  - tuned        : the Arm D adapter (openai/armd-inkling-small-tuned)

Both ids are routed by arm_d/serving_shim.py to Tinker SamplingClients (tuned
via model_path, base via base_model) — CHEAP per-token serving, since neither
Together nor Fireworks will serve a custom LoRA adapter without a dedicated
GPU endpoint. The shim must be running (`python -m arm_d.serving_shim`)
before this driver is invoked.

Each (system, split) runs in its OWN subprocess so `settings.config` reads that
system's AGENT_MODEL (config resolves env at import; an in-process second run
would reuse the first system's model). Per-repeat metrics are captured via
`run_train_eval --metrics-json`, and an index (experiments/arm_d_eval_index.json)
maps (system, split) -> metrics file + run spec for the plotting step.

Launch DETACHED (8 system x split runs, 3 seeds each):
    nohup python -m scripts.run_arm_d_eval > arm_d_eval.log 2>&1 &

TINKER_KEY must be in .env (config bridges it to TINKER_API_KEY for the shim).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# The 4 pre-registered eval splits with their TAU2 domains (arm_d_eval_precommit.md).
# retail = the frozen validation.json; banking = the untrained transfer domain.
EVAL_SPLITS: Tuple[Tuple[str, str], ...] = (
    ("validation", "retail"),
    ("eval_airline", "airline"),
    ("eval_telecom", "telecom"),
    ("transfer_banking", "banking_knowledge"),
)
SEED_START = 2001
REPEATS = 3
# Local Tinker serving shim (arm_d/serving_shim.py) — path B: per-token, no
# dedicated GPU endpoint. Replaces the Arm C Together seam for Arm D since
# neither Together nor Fireworks will serve a custom LoRA adapter serverless.
SHIM_API_BASE = "http://localhost:8100/v1"
BASE_INKLING_ID = "thinkingmachines/Inkling-Small"  # matched base control (no adapter)
DEFAULT_TUNED_MODEL_ID = "armd-inkling-small-tuned"
DEFAULT_INDEX = Path("experiments/arm_d_eval_index.json")
DEFAULT_METRICS_ROOT = Path("experiments/arm_d_eval_metrics")

Runner = Callable[[List[str], Dict[str, str]], object]


def _default_runner(argv: List[str], env: Dict[str, str]) -> None:
    subprocess.run(argv, env=env, check=True)


def run_arm_d_eval(
    *,
    tuned_model_id: str,
    base_model_id: str = BASE_INKLING_ID,
    splits: Sequence[Tuple[str, str]] = EVAL_SPLITS,
    seed_start: int = SEED_START,
    repeats: int = REPEATS,
    api_base: str = SHIM_API_BASE,
    index_path: Path = DEFAULT_INDEX,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    runner: Runner = _default_runner,
    base_env: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Run every (system, split) eval and write the plotting index.

    Both systems go through the identical local Tinker shim `openai/` seam in
    the v0.1 harness; only the model id differs (base vs adapter). `runner` and
    `base_env` are injected so the whole driver is unit-testable at $0 (no real
    eval spend).
    """
    if not tuned_model_id:
        raise SystemExit(
            "tuned_model_id is required — set ARM_D_TUNED_MODEL_ID (the shim-routed "
            "tuned-adapter id) or pass --model-id. Run scripts.train_arm_d first."
        )

    systems = (("base_control", base_model_id), ("tuned", tuned_model_id))
    metrics_root.mkdir(parents=True, exist_ok=True)
    env0 = dict(base_env if base_env is not None else os.environ)

    index: List[dict] = []
    for system, model_id in systems:
        for split, domain in splits:
            metrics_json = metrics_root / f"{system}__{split}.json"
            env = {
                **env0,
                "AGENT_MODEL": f"openai/{model_id}",
                "AGENT_API_BASE": api_base,
                "HARNESS_VERSION": "v0.1",
            }
            argv = [
                sys.executable, "-m", "scripts.run_train_eval",
                "--split", split, "--domain", domain,
                "--repeats", str(repeats), "--seed-start", str(seed_start),
                "--metrics-json", str(metrics_json),
            ]
            logger.info("eval: system=%s split=%s domain=%s model=%s seeds %d..%d",
                        system, split, domain, model_id,
                        seed_start, seed_start + repeats - 1)
            runner(argv, env)
            index.append({
                "system": system, "split": split, "domain": domain,
                "model_id": model_id, "agent_model": f"openai/{model_id}",
                "metrics_json": str(metrics_json),
                "seed_start": seed_start, "repeats": repeats,
                "harness_version": "v0.1",
            })

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    logger.info("wrote eval index (%d runs) -> %s", len(index), index_path)
    return index


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=os.environ.get("ARM_D_TUNED_MODEL_ID", DEFAULT_TUNED_MODEL_ID),
        help="Shim-routed tuned-adapter id (default: $ARM_D_TUNED_MODEL_ID or "
        f"{DEFAULT_TUNED_MODEL_ID!r}).",
    )
    parser.add_argument("--base-model-id", default=BASE_INKLING_ID,
                        help=f"Matched base control id (default {BASE_INKLING_ID}).")
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_arm_d_eval(tuned_model_id=args.model_id, base_model_id=args.base_model_id,
                   seed_start=args.seed_start, repeats=args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
