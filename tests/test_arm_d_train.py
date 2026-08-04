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
    def optim_step(self): pass
    def save_weights_and_get_sampling_client(self, name=None): return object()


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
