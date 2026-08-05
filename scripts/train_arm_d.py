"""Arm D training entrypoint (PAID: runs a real Tinker LoRA fine-tune).

Ties the staged pieces together end-to-end:
  pointer (experiments/arm_d_distill_runs.json)
    -> examples_from_distill_pointer  (passed-only, deduped, balanced)
    -> real Tinker LoRA client on thinkingmachines/Inkling + tml_v0 renderer
    -> arm_d.train.train()  (holdout early-stop, save adapter, training_record.json)

The renderer is bound to the training client's tokenizer, so the client must be
built BEFORE train() runs; train() then receives a factory that returns that
SAME pre-built adapter (it must not create a second client). Every real call is
behind an injected seam (`examples_loader`, `training_factory`), so the wiring
is unit-tested at $0 with fakes — no tinker import on the test path.

Launch DETACHED (real training is long):
    nohup python -m scripts.train_arm_d > train_arm_d.log 2>&1 &
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from arm_d.build_dataset import examples_from_distill_pointer
from arm_d.train import TrainConfig, TrainResult, train

logger = logging.getLogger(__name__)

DEFAULT_POINTER = Path("experiments/arm_d_distill_runs.json")
DEFAULT_MANIFEST = Path("experiments/arm_d_sft_manifest.json")
DEFAULT_TRAINING_ROOT = Path("experiments/arm_d_training")


def build_real_training(cfg: TrainConfig) -> Tuple[Callable, Callable]:
    """Build the real Tinker adapter + `tml_v0` renderer for `train()` (paid;
    tinker imported lazily inside `arm_d.train`). Returns `(client_factory,
    renderer)` where `client_factory` returns the SAME pre-built adapter — the
    renderer is bound to this client's tokenizer, so `train()` must reuse it, not
    create a second client."""
    from arm_d.train import _default_client_factory, build_default_renderer

    adapter = _default_client_factory(cfg)          # creates the LoRA client
    renderer = build_default_renderer(cfg, adapter.tc)  # tml_v0 auto-resolved
    return (lambda _cfg: adapter), renderer


def train_arm_d(
    *,
    pointer_path: Path = DEFAULT_POINTER,
    manifest_path: Path = DEFAULT_MANIFEST,
    log_dir: Optional[Path] = None,
    cfg: Optional[TrainConfig] = None,
    seed: int = 0,
    k: int = 2,
    examples_loader: Callable = examples_from_distill_pointer,
    training_factory: Callable[[TrainConfig], Tuple[Callable, Callable]] = build_real_training,
) -> TrainResult:
    """Load the SFT set from the pointer and run the LoRA fine-tune.

    `examples_loader` / `training_factory` are injected so the whole flow runs at
    $0 with fakes. The audit manifest, if present, is copied into `log_dir`
    beside `training_record.json` by `train()` (audit trail, §8 guard 6).
    """
    cfg = cfg or TrainConfig()
    log_dir = Path(log_dir) if log_dir is not None else _default_log_dir()

    examples = examples_loader(pointer_path, seed=seed, k=k)
    if not examples:
        raise SystemExit(
            f"No SFT examples from {pointer_path}. Run scripts.run_distillation "
            "then scripts.build_distill_dataset first (need passed transcripts)."
        )
    logger.info("training on %d SFT example(s); base=%s rank=%d lr=%g max_epochs=%d",
                len(examples), cfg.base_model, cfg.lora_rank, cfg.lr, cfg.max_epochs)

    client_factory, renderer = training_factory(cfg)
    result = train(
        examples, cfg,
        client_factory=client_factory, renderer=renderer,
        log_dir=log_dir, manifest_path=manifest_path,
    )

    logger.info("=" * 60)
    logger.info("training complete: checkpoint=%s steps=%d best_holdout_loss=%.6f "
                "training_cost_usd=%.6f",
                result.checkpoint, result.steps, result.best_holdout_loss,
                result.training_cost_usd)
    logger.info("record + manifest: %s", log_dir)
    logger.info("next (paid): export the adapter with arm_d.serving.export_adapter_to_hf "
                "using the tinker:// checkpoint path from this run's Tinker output, "
                "then upload to Together (console) and set ARM_D_TUNED_MODEL_ID.")
    logger.info("=" * 60)
    return result


def _default_log_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_TRAINING_ROOT / f"run_{stamp}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="Training output dir (default: experiments/arm_d_training/run_<UTC>).")
    parser.add_argument("--lora-rank", type=int, default=TrainConfig.lora_rank)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--seed", type=int, default=0, help="balance() shuffle seed.")
    parser.add_argument("--k", type=int, default=2, help="balance() per-domain cap multiplier.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = TrainConfig(lora_rank=args.lora_rank, lr=args.lr,
                      max_epochs=args.max_epochs, patience=args.patience)
    train_arm_d(pointer_path=args.pointer, manifest_path=args.manifest,
                log_dir=args.log_dir, cfg=cfg, seed=args.seed, k=args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
