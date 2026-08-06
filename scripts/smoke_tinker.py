"""Live Tinker round-trip smoke — Arm D (PAID: makes real Tinker API calls).

The code-level unknowns are already pinned at $0 from the installed SDK, so this
script does NOT exist to discover them:
  - renderer for `thinkingmachines/Inkling` = "tml_v0"
    (`tinker_cookbook.model_info.get_recommended_renderer_name`)
  - SFT loss_fn = "cross_entropy"
    (valid `tinker.types.LossFnType` literal; the value the cookbook's own SFT
    loops — sl_loop.py, sdft.py — pass)

It verifies the one thing only a live call can: that THIS ACCOUNT can LoRA-train
`thinkingmachines/Inkling` end-to-end. It runs the minimal real chain that
`arm_d/train.py` will use — create_lora_training_client -> render one tiny
example (tml_v0, assistant-masked) -> forward_backward(cross_entropy) ->
optim_step -> save_weights_and_get_sampling_client -> best-effort telemetry — and
prints a PASS/FAIL summary. Deliberately tiny (rank 8, one synthetic
conversation, one step) to keep spend minimal.

Requires TINKER_KEY in .env (settings.config bridges it to TINKER_API_KEY).
Run:  python -m scripts.smoke_tinker
"""
from __future__ import annotations

import traceback

import settings.config  # noqa: F401  side effect: bridges TINKER_KEY -> TINKER_API_KEY

from arm_d.build_dataset import TrainingExample
from arm_d.train import TrainConfig, _extract_loss, build_default_renderer

# One tiny, fully synthetic conversation. Nothing task-specific, no eval data —
# just enough for one real forward_backward. Assistant turn is the loss target.
SYNTHETIC = TrainingExample(
    messages=[
        {"role": "system", "content": "You are a helpful retail support agent.", "tool_calls": None},
        {"role": "user", "content": "What is your return window?", "tool_calls": None},
        {"role": "assistant", "content": "Our return window is 30 days from delivery.", "tool_calls": None},
    ],
    target_mask=[False, False, True],
    task_id="smoke-0",
    domain="mock",
    source_run="smoke",
)

EXPECTED_RENDERER = "tml_v0"


def main() -> int:
    import tinker
    from tinker_cookbook import model_info

    cfg = TrainConfig(lora_rank=8, max_epochs=1)

    # [1] Renderer name — $0 local lookup, no paid call yet (sanity/anchor).
    renderer_name = model_info.get_recommended_renderer_name(cfg.base_model)
    print(f"[1] renderer for {cfg.base_model!r}: {renderer_name!r} "
          f"(expected {EXPECTED_RENDERER!r})")
    if renderer_name != EXPECTED_RENDERER:
        print("    WARN: SDK now recommends a different renderer than the pinned "
              "value — update arm_d/train.py + the decision note.")

    # [2] Create the LoRA training client — FIRST paid call (auth + entitlement).
    print(f"[2] creating LoRA training client (paid): base={cfg.base_model!r} "
          f"rank={cfg.lora_rank} ...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=cfg.base_model, rank=cfg.lora_rank)
    print("    ok: training client created.")

    # [3] Render one synthetic example via the real cookbook renderer.
    print(f"[3] rendering one synthetic SFT example ({renderer_name}, "
          "assistant-masked) ...")
    renderer = build_default_renderer(cfg, tc, renderer_name=renderer_name)
    datum = renderer(SYNTHETIC)
    print(f"    ok: datum built ({type(datum).__name__}).")

    # [4] forward_backward -> loss, then optim_step (exactly train()'s calls).
    print("[4] forward_backward + optim_step (loss_fn=cross_entropy) ...")
    fb = tc.forward_backward(data=[datum], loss_fn="cross_entropy").result()
    print(f"    ok: forward_backward returned; loss={_extract_loss(fb)}")
    print(f"    raw metrics keys: {list(getattr(fb, 'metrics', {}) or {})}")
    tc.optim_step(tinker.AdamParams(learning_rate=cfg.lr)).result()
    print("    ok: optim_step applied.")

    # [5] Persist an EXPORTABLE checkpoint (tinker:// path) + get a sampler from
    # it. save_weights_and_get_sampling_client is ephemeral (can't be exported).
    print("[5] save_weights_for_sampler(name='armd-smoke') + create_sampling_client ...")
    ckpt_path = tc.save_weights_for_sampler(name="armd-smoke").result().path
    sampler = tc.create_sampling_client(model_path=ckpt_path)
    print(f"    ok: persistent checkpoint = {ckpt_path}")
    print(f"    ok: sampling client obtained ({type(sampler).__name__}).")

    # [6] Best-effort training-cost telemetry (§15.3) — informational only.
    get_tel = getattr(tc, "get_telemetry", None) or getattr(sc, "get_telemetry", None)
    if callable(get_tel):
        try:
            print(f"[6] telemetry: {get_tel()}")
        except Exception as exc:  # noqa: BLE001
            print(f"[6] telemetry unavailable: {type(exc).__name__}: {exc}")

    print("\nSMOKE PASS: this account can LoRA-train 'thinkingmachines/Inkling' "
          "end-to-end.")
    print("Confirmed for arm_d/train.py: renderer_name=tml_v0, "
          "loss_fn=cross_entropy.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — smoke: surface any failure clearly
        traceback.print_exc()
        print(f"\nSMOKE FAIL: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
