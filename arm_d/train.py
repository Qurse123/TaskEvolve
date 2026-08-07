"""Tinker LoRA training loop for Arm D.

Turns `arm_d.build_dataset.TrainingExample`s into Tinker datums (via an injected
renderer) and runs a LoRA fine-tune with early-stop on a by-task holdout slice
of the *training* set. The Tinker client and the renderer are both injected
callables (`client_factory`, `renderer`) so `train()`'s loop logic — holdout
split, epoch loop, early stopping, cost/record logging — is fully unit-testable
at $0 with a fake client. `_default_client_factory` / `build_default_renderer`
wire the real `tinker` + `tinker_cookbook` symbols pinned in the Task 1 spike
note (docs/superpowers/notes/2026-08-03-arm-d-tinker-serving-decision.md) but
are never invoked by the tests — only reachable by real-usage callers.
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from arm_d.build_dataset import TrainingExample

# Fallback per-step training cost (USD) when the client exposes no billing/
# telemetry surface. 0.0 by default — documented placeholder, not a real price,
# per decision note §3 ("record step count x a documented per-step rate").
# Real-usage callers should set TrainConfig.cost_per_step_usd once Tinker's
# billing API (sc.get_telemetry() / tinker.types.BillingUsageResponse) is
# confirmed against a live run.
DEFAULT_COST_PER_STEP_USD = 0.0

# `forward_backward`'s loss_fn identifier. Confirmed at $0 (2026-08-05) against
# the installed SDK: "cross_entropy" is a valid `tinker.types.LossFnType`
# literal AND the value tinker_cookbook's own supervised loops (sl_loop.py,
# sdft.py, train.py) pass for SFT. No live round-trip needed to pin it.
DEFAULT_LOSS_FN = "cross_entropy"


@dataclass
class TrainConfig:
    base_model: str = "thinkingmachines/Inkling"
    lora_rank: int = 32
    lr: float = 1e-4
    max_epochs: int = 4
    patience: int = 2
    # Minibatch size for SGD: each epoch takes ceil(n_train/batch_size) gradient
    # steps (forward_backward + optim_step per minibatch), not one full-batch
    # step — so a run does many updates, a real fine-tune rather than a few.
    batch_size: int = 8
    # See DEFAULT_COST_PER_STEP_USD above.
    cost_per_step_usd: float = DEFAULT_COST_PER_STEP_USD


@dataclass
class TrainResult:
    checkpoint: str
    sampling_client: Any
    training_cost_usd: float
    steps: int
    best_holdout_loss: float


def split_holdout(
    examples: list[TrainingExample], *, frac: float = 0.15, seed: int = 0
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split examples into (train, holdout) by task_id — no task appears on both
    sides. Holdout size is `round(len(task_ids) * frac)`, clamped to at least 1
    (when there is more than one task) and to leave at least one training task."""
    task_ids = sorted({e.task_id for e in examples})
    if not task_ids:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    max_holdout = max(0, len(task_ids) - 1)  # always keep >=1 training task
    n_holdout = min(max(1, round(len(task_ids) * frac)), max_holdout) if max_holdout else 0
    holdout_ids = set(task_ids[:n_holdout])
    train = [e for e in examples if e.task_id not in holdout_ids]
    holdout = [e for e in examples if e.task_id in holdout_ids]
    return train, holdout


def _block(value: Any) -> Any:
    """Resolve a Tinker `APIFuture` (real API: call `.result()` to block) or pass
    a plain value through unchanged (the fake test client, which returns dicts/
    None directly). Lets the same loop code work against both."""
    result_fn = getattr(value, "result", None)
    return result_fn() if callable(result_fn) else value


def _loss_from_metrics(metrics: Any) -> float | None:
    """Pull a loss value from a `ForwardBackwardOutput.metrics` dict. Prefers an
    exact ``"loss"`` key, else the first key containing ``"loss"`` — the real
    Tinker metrics key is provider-versioned and may be suffixed (e.g.
    ``"loss:sum"``), which is why the smoke saw loss=0.0 with a strict lookup."""
    if not isinstance(metrics, dict) or not metrics:
        return None
    if "loss" in metrics:
        return float(metrics["loss"])
    for key, val in metrics.items():
        if "loss" in str(key).lower():
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _extract_loss(result: Any) -> float:
    """Best-effort loss from a forward_backward result. The fake test client
    returns `{"loss": v}`; the real `ForwardBackwardOutput` carries a `metrics`
    dict (loss under a possibly-suffixed key — see `_loss_from_metrics`), so the
    early-stop signal is read from there, not a strict `metrics["loss"]`."""
    if isinstance(result, dict):
        if "loss" in result:
            return float(result["loss"])
        from_metrics = _loss_from_metrics(result.get("metrics"))
        return from_metrics if from_metrics is not None else 0.0
    loss = getattr(result, "loss", None)
    if loss is not None:
        return float(loss)
    from_metrics = _loss_from_metrics(getattr(result, "metrics", None))
    return from_metrics if from_metrics is not None else 0.0


def _extract_cost(telemetry: Any) -> float | None:
    if telemetry is None:
        return None
    if isinstance(telemetry, dict):
        for key in ("total_cost_usd", "cost_usd", "cost"):
            if key in telemetry:
                return float(telemetry[key])
        return None
    for attr in ("total_cost_usd", "cost_usd", "cost"):
        val = getattr(telemetry, attr, None)
        if val is not None:
            return float(val)
    return None


def _training_cost(client: Any, steps: int, cfg: TrainConfig) -> tuple[float, str]:
    """§15.3: read training cost from client/service billing telemetry when
    exposed; else fall back to steps * cfg.cost_per_step_usd (documented rate,
    kept separate from runtime token cost). The fake test client exposes no
    telemetry, so tests always take the fallback path (cost 0.0 by default)."""
    get_telemetry = getattr(client, "get_telemetry", None)
    if callable(get_telemetry):
        try:
            cost = _extract_cost(get_telemetry())
        except Exception:
            cost = None
        if cost is not None:
            return float(cost), "from client.get_telemetry()"
    return steps * cfg.cost_per_step_usd, (
        f"no billing telemetry exposed; fallback = steps * cost_per_step_usd "
        f"({cfg.cost_per_step_usd})"
    )


def train(
    examples: list[TrainingExample],
    cfg: TrainConfig,
    *,
    client_factory: Callable[[TrainConfig], Any],
    renderer: Callable[[TrainingExample], Any],
    log_dir: str | Path,
    manifest_path: str | Path | None = None,
    holdout_frac: float = 0.15,
    holdout_seed: int = 0,
) -> TrainResult:
    """Run the LoRA training loop.

    Splits `examples` by task_id into train/holdout (`split_holdout`), then for
    up to `cfg.max_epochs` epochs: renders the train split into a batch, calls
    `client.forward_backward(batch)` + `client.optim_step()` (one step/epoch),
    renders + scores the holdout split the same way, and early-stops once
    holdout loss fails to improve for `cfg.patience` consecutive epochs. Saves
    the adapter via `client.save_weights_and_get_sampling_client(name=...)` and
    writes `log_dir/training_record.json` (cost, steps, best holdout loss, every
    `TrainConfig` field, timestamp). If `manifest_path` is given and exists, it
    is copied into `log_dir` alongside the record.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_examples, holdout_examples = split_holdout(
        examples, frac=holdout_frac, seed=holdout_seed
    )

    client = client_factory(cfg)

    steps = 0
    best_holdout_loss = float("inf")
    epochs_without_improvement = 0
    last_train_loss = 0.0

    # Render once (deterministic), then run minibatch SGD: each epoch takes
    # ceil(n_train/batch_size) gradient steps (forward_backward + optim_step per
    # minibatch), so a run does many updates — a real fine-tune, not a few.
    train_datums = [renderer(ex) for ex in train_examples]
    holdout_datums = [renderer(ex) for ex in holdout_examples]
    shuffle_rng = random.Random(holdout_seed)

    for _epoch in range(cfg.max_epochs):
        order = list(range(len(train_datums)))
        shuffle_rng.shuffle(order)
        for start in range(0, len(order), cfg.batch_size):
            batch = [train_datums[i] for i in order[start:start + cfg.batch_size]]
            if not batch:
                continue
            fb_result = _block(client.forward_backward(batch))
            last_train_loss = _extract_loss(fb_result)
            _block(client.optim_step())
            steps += 1

        if holdout_datums:
            # forward-only: the holdout pass must NOT accumulate gradients that
            # would leak into the next epoch's first optim_step.
            ho_result = _block(client.forward(holdout_datums))
            holdout_loss = _extract_loss(ho_result)
        else:
            # Degenerate case (too few distinct tasks for a holdout slice):
            # fall back to last train loss so early-stop still has a signal.
            holdout_loss = last_train_loss

        if holdout_loss < best_holdout_loss - 1e-9:
            best_holdout_loss = holdout_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                break

    checkpoint_name = f"arm-d-{Path(cfg.base_model).name.lower()}-lora-{log_dir.name}"
    # Persist an EXPORTABLE checkpoint (a tinker:// path) then get a sampler from
    # it. save_weights_and_get_sampling_client makes an *ephemeral* checkpoint
    # whose weights can't be exported to HF — arm_d.serving.export_adapter_to_hf
    # needs this persistent path (`tinker checkpoint push-hf <path>`).
    checkpoint_path = client.save_weights_for_sampler(checkpoint_name)
    sampling_client = client.create_sampling_client(checkpoint_path)

    training_cost_usd, cost_note = _training_cost(client, steps, cfg)

    record = {
        **asdict(cfg),
        "steps": steps,
        "best_holdout_loss": best_holdout_loss,
        "training_cost_usd": training_cost_usd,
        "cost_note": cost_note,
        "checkpoint": checkpoint_path,
        "checkpoint_name": checkpoint_name,
        "num_train_examples": len(train_examples),
        "num_holdout_examples": len(holdout_examples),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (log_dir / "training_record.json").write_text(json.dumps(record, indent=2))

    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        if manifest_path.exists():
            shutil.copy2(manifest_path, log_dir / manifest_path.name)

    return TrainResult(
        checkpoint=checkpoint_path,
        sampling_client=sampling_client,
        training_cost_usd=training_cost_usd,
        steps=steps,
        best_holdout_loss=best_holdout_loss,
    )


# --------------------------------------------------------------------------
# Real-usage defaults (Task 1 spike note). Never imported/invoked by the unit
# tests — only reachable by a real caller that wires these in explicitly.
# --------------------------------------------------------------------------


class _RealTrainingClientAdapter:
    """Adapts a live `tinker.TrainingClient` to the same call shape `train()`'s
    loop uses against the fake test client: `forward_backward(batch)` (no
    `loss_fn` arg), `optim_step()` (no `AdamParams` arg), and
    `save_weights_and_get_sampling_client(name=...)`. This hides the extra
    arguments the real SDK requires (decision note §1) behind the injected
    seam instead of branching the loop on real-vs-fake."""

    def __init__(self, tc: Any, *, lr: float, loss_fn: str = DEFAULT_LOSS_FN):
        self.tc = tc
        self._lr = lr
        self._loss_fn = loss_fn

    def forward_backward(self, batch: list[Any]) -> Any:
        return self.tc.forward_backward(data=batch, loss_fn=self._loss_fn)

    def forward(self, batch: list[Any]) -> Any:
        # forward-only (no gradient accumulation) — for holdout eval, so the
        # holdout pass never leaks gradients into the next training step.
        return self.tc.forward(data=batch, loss_fn=self._loss_fn)

    def optim_step(self) -> Any:
        import tinker

        return self.tc.optim_step(tinker.AdamParams(learning_rate=self._lr))

    def save_weights_for_sampler(self, name: str) -> str:
        # Persistent, EXPORTABLE checkpoint. save_weights_and_get_sampling_client
        # makes an ephemeral one (weights can't be exported to HF); this returns
        # the tinker:// path arm_d.serving.export_adapter_to_hf pushes.
        resp = self.tc.save_weights_for_sampler(name=name).result()
        return resp.path

    def create_sampling_client(self, model_path: str) -> Any:
        return self.tc.create_sampling_client(model_path=model_path)

    def get_tokenizer(self) -> Any:
        return self.tc.get_tokenizer()


def _default_client_factory(cfg: TrainConfig) -> _RealTrainingClientAdapter:
    """Builds the real Tinker LoRA training client (decision note §1):
    `tinker.ServiceClient().create_lora_training_client(base_model=..., rank=...)`,
    wrapped in `_RealTrainingClientAdapter` for the shared loop's call shape.
    """
    import tinker

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=cfg.base_model, rank=cfg.lora_rank)
    return _RealTrainingClientAdapter(tc, lr=cfg.lr)


def _to_cookbook_messages(messages: list[dict]) -> list[dict]:
    """Maps `arm_d.build_dataset`'s normalized `{role, content, tool_calls}`
    messages to `tinker_cookbook.renderers.Message` TypedDicts (decision note
    §2's mapping). Passes through `tool_call_id`/`name` when present, though
    `build_dataset._normalize` currently only preserves role/content/tool_calls
    — a gap in the upstream normalizer, not fixed here (build_dataset is a
    frozen, already-shipped module; flagged in the report instead)."""
    out = []
    for m in messages:
        msg: dict = {"role": m.get("role"), "content": m.get("content")}
        if m.get("tool_calls"):
            msg["tool_calls"] = [_to_openai_tool_call(tc) for tc in m["tool_calls"]]
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        if m.get("name"):
            msg["name"] = m["name"]
        out.append(msg)
    return out


def _to_openai_tool_call(tc: dict) -> dict:
    """TAU2 stores tool calls flat: ``{id, name, arguments(dict)}``. The tml_v0
    renderer uses the OpenAI schema: ``{id, type:"function", function:{name,
    arguments(JSON string)}}`` — a call missing ``function`` raises
    ``ValueError: tool_call missing 'function'``. Pass through anything already
    OpenAI-shaped; JSON-encode dict arguments (OpenAI wants a string)."""
    if not isinstance(tc, dict) or "function" in tc:
        return tc
    args = tc.get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args if args is not None else {})
    return {
        "id": tc.get("id"),
        "type": "function",
        "function": {"name": tc.get("name"), "arguments": args},
    }


def build_default_renderer(
    cfg: TrainConfig,
    tc: Any,
    *,
    renderer_name: str | None = None,
    max_length: int = 4096,
) -> Callable[[TrainingExample], Any]:
    """Real-usage renderer builder (decision note §2): resolves the cookbook
    renderer via `renderers.get_renderer(renderer_name, tc.get_tokenizer(),
    model_name=cfg.base_model)` and returns a callable
    `renderer(example) -> tinker.Datum` via `supervised.conversation_to_datum(
    ..., train_on_what=ALL_ASSISTANT_MESSAGES)` — the assistant-only loss mask
    is a library concern, not hand-built (per the note).

    `renderer_name` defaults to the cookbook's recommended renderer for the base
    model (`model_info.get_recommended_renderer_name(cfg.base_model)`). For
    `thinkingmachines/Inkling` this resolves at $0 to `tml_v0` (confirmed
    2026-08-05 from the installed `tinker_cookbook.model_info` table) — the
    earlier PAID/DEFERRED unknown (decision note §6.1) is now resolved. Pass an
    explicit `renderer_name` only to override. `tc` is the raw TrainingClient
    (e.g. `_default_client_factory(cfg).tc`), not the adapter, since only the
    raw client's `get_tokenizer()` is needed here.
    """
    from tinker_cookbook import model_info, renderers, supervised
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(cfg.base_model)
    # Load the tokenizer via the cookbook loader (tml_renderers-backed for
    # Inkling), NOT tc.get_tokenizer(): tinker's path imports `tml_tokenizers`
    # (not published on PyPI) for thinkingmachines/ models and raises
    # ModuleNotFoundError. The cookbook adapter resolves Inkling's tokenizer
    # through tml_renderers instead (confirmed live 2026-08-06). `tc` is no
    # longer needed here but stays in the signature for call-site stability.
    tok = get_tokenizer(cfg.base_model)
    cb_renderer = renderers.get_renderer(renderer_name, tok, model_name=cfg.base_model)

    def _renderer(example: TrainingExample) -> Any:
        conversation = _to_cookbook_messages(example.messages)
        return supervised.conversation_to_datum(
            conversation,
            cb_renderer,
            max_length,
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )

    return _renderer
