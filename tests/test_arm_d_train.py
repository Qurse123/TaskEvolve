from arm_d.build_dataset import TrainingExample
from arm_d.train import split_holdout, train, TrainConfig


def _ex(task_id, dom="retail"):
    m = [{"role": "user", "content": "hi", "tool_calls": None},
         {"role": "assistant", "content": "ok", "tool_calls": None}]
    return TrainingExample(m, [False, True], task_id, dom, "run")


def test_holdout_split_is_by_task():
    exs = [_ex(f"t{i}") for i in range(10)]
    tr, ho = split_holdout(exs, frac=0.2, seed=0)
    tr_ids = {e.task_id for e in tr}; ho_ids = {e.task_id for e in ho}
    assert tr_ids.isdisjoint(ho_ids)
    assert len(ho_ids) >= 1


class _FakeClient:
    def __init__(self): self.steps = 0
    def forward_backward(self, batch): self.steps += 1; return {"loss": 1.0 / (self.steps + 1)}
    def forward(self, batch): return {"loss": 0.5}  # holdout eval (forward-only), constant
    def optim_step(self): pass
    def save_weights_for_sampler(self, name): return f"tinker://fake/{name}/sampler_weights/final"
    def create_sampling_client(self, model_path): return object()


def test_train_logs_cost_and_stops_early(tmp_path):
    exs = [_ex(f"t{i}") for i in range(8)]
    cfg = TrainConfig(max_epochs=5, patience=1)
    res = train(exs, cfg,
                client_factory=lambda cfg: _FakeClient(),
                renderer=lambda ex: {"tokens": [1, 2], "weights": [0, 1]},
                log_dir=tmp_path)
    assert res.steps >= 1
    assert (tmp_path / "training_record.json").exists()
    assert res.training_cost_usd >= 0.0


def test_minibatch_takes_multiple_steps_per_epoch(tmp_path):
    # 20 tasks, ~15% holdout -> ~17 train; batch_size 4 -> ~5 gradient steps in a
    # single epoch (the point of minibatch SGD: many updates, not one per epoch).
    exs = [_ex(f"t{i}") for i in range(20)]
    cfg = TrainConfig(max_epochs=1, patience=5, batch_size=4)
    res = train(exs, cfg,
                client_factory=lambda cfg: _FakeClient(),
                renderer=lambda ex: {"tokens": [1, 2], "weights": [0, 1]},
                log_dir=tmp_path)
    assert res.steps >= 4  # multiple minibatch steps within one epoch


def test_inkling_renderer_name_pinned_to_tml_v0():
    """The Inkling renderer name is resolved at $0 from the cookbook's model
    table (not a live round-trip). Lock it so an SDK bump that changes it fails
    loudly instead of silently mis-rendering training data."""
    from tinker_cookbook import model_info

    assert model_info.get_recommended_renderer_name("thinkingmachines/Inkling") == "tml_v0"


def test_default_loss_fn_is_a_valid_lossfntype():
    """DEFAULT_LOSS_FN must be a valid tinker LossFnType literal (the SFT value
    the cookbook loops use); guards against a typo or an SDK enum change."""
    import typing

    from tinker.types import LossFnType

    from arm_d.train import DEFAULT_LOSS_FN

    assert DEFAULT_LOSS_FN in typing.get_args(LossFnType)
