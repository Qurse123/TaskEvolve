"""$0 tests for the Arm D training entrypoint. Injects fake examples + a fake
Tinker client/renderer, so no tinker import, no real training, no spend."""
import pytest

from arm_d.build_dataset import TrainingExample
from arm_d.train import TrainConfig
from scripts.train_arm_d import train_arm_d


class _FakeClient:
    def __init__(self):
        self.steps = 0

    def forward_backward(self, batch):
        self.steps += 1
        return {"loss": 1.0 / (self.steps + 1)}

    def forward(self, batch):  # holdout eval (forward-only)
        return {"loss": 0.5}

    def optim_step(self):
        pass

    def save_weights_for_sampler(self, name):
        return f"tinker://fake/{name}/sampler_weights/final"

    def create_sampling_client(self, model_path):
        return object()


def _ex(task_id, dom="retail"):
    m = [{"role": "user", "content": "hi", "tool_calls": None},
         {"role": "assistant", "content": "ok", "tool_calls": None}]
    return TrainingExample(m, [False, True], task_id, dom, "run")


def _fake_factory(cfg):
    # client_factory returns one shared fake; renderer is a no-op datum builder
    client = _FakeClient()
    return (lambda _cfg: client), (lambda ex: {"tokens": [1, 2], "weights": [0, 1]})


def test_entrypoint_runs_train_and_writes_record(tmp_path):
    examples = [_ex(f"t{i}") for i in range(6)]
    result = train_arm_d(
        pointer_path=tmp_path / "pointer.json",   # ignored: loader is injected
        manifest_path=tmp_path / "missing_manifest.json",  # absent -> not copied
        log_dir=tmp_path / "out",
        cfg=TrainConfig(max_epochs=2, patience=1),
        examples_loader=lambda p, *, seed, k: examples,
        training_factory=_fake_factory,
    )
    assert result.steps >= 1
    assert (tmp_path / "out" / "training_record.json").exists()


def test_entrypoint_errors_when_no_examples(tmp_path):
    with pytest.raises(SystemExit):
        train_arm_d(
            pointer_path=tmp_path / "pointer.json",
            log_dir=tmp_path / "out",
            examples_loader=lambda p, *, seed, k: [],
            training_factory=_fake_factory,
        )


def test_entrypoint_copies_manifest_into_log_dir(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    train_arm_d(
        pointer_path=tmp_path / "pointer.json",
        manifest_path=manifest,
        log_dir=tmp_path / "out",
        cfg=TrainConfig(max_epochs=1),
        examples_loader=lambda p, *, seed, k: [_ex(f"t{i}") for i in range(4)],
        training_factory=_fake_factory,
    )
    assert (tmp_path / "out" / "manifest.json").exists()
