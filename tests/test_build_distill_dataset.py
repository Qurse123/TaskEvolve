"""$0 tests for the Arm D SFT-build wrapper + the shared pointer transform.
Uses synthetic transcript dirs + a synthetic pointer — no model calls, no eval."""
import json

from arm_d.build_dataset import examples_from_distill_pointer
from scripts.build_distill_dataset import build_distill_dataset


def _transcript(run_dir, reward, msgs):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_1_messages.json").write_text(json.dumps(
        {"task_id": "t1", "reward_info": {"reward": reward},
         "termination_reason": "done", "messages": msgs}))


def _msgs(text):
    # `text` varies per run so build_examples' content-hash dedup (global, by
    # trajectory) doesn't collapse two otherwise-distinct runs.
    return [
        {"role": "user", "content": "cancel my order"},
        {"role": "assistant", "content": text},
    ]


def _pointer(tmp_path, runs):
    """Write a run_distillation-shaped pointer + the transcript dirs it names."""
    logs_root = tmp_path / "logs"
    for run_id, reward in runs.items():
        _transcript(logs_root / run_id, reward, _msgs(f"Done for {run_id}."))
    pointer = tmp_path / "pointer.json"
    pointer.write_text(json.dumps({
        "runs": {run_id: "retail" if "retail" in run_id else "airline" for run_id in runs},
        "logs_root": str(logs_root),
        "agent_model": "anthropic/claude-opus-4-8",
    }))
    return pointer


def test_examples_from_pointer_keeps_passed_and_tags_domain(tmp_path):
    pointer = _pointer(tmp_path, {"train_distill_retail_1": 1.0,
                                  "train_distill_airline_1": 1.0})
    ex = examples_from_distill_pointer(pointer)
    assert len(ex) == 2
    assert {e.domain for e in ex} == {"retail", "airline"}


def test_wrapper_writes_manifest_and_drops_failed(tmp_path):
    # retail run passed, airline run failed -> only retail survives
    pointer = _pointer(tmp_path, {"train_distill_retail_1": 1.0,
                                  "train_distill_airline_1": 0.0})
    manifest = tmp_path / "manifest.json"
    ex = build_distill_dataset(pointer_path=pointer, manifest_path=manifest)

    assert [e.domain for e in ex] == ["retail"]
    rows = json.loads(manifest.read_text())
    assert len(rows) == 1
    assert set(rows[0]) == {"task_id", "domain", "source_run", "sha256"}
    assert rows[0]["domain"] == "retail"


def test_wrapper_empty_pointer_writes_empty_manifest(tmp_path):
    pointer = tmp_path / "pointer.json"
    pointer.write_text(json.dumps({"runs": {}, "logs_root": str(tmp_path / "logs")}))
    manifest = tmp_path / "manifest.json"
    ex = build_distill_dataset(pointer_path=pointer, manifest_path=manifest)
    assert ex == []
    assert json.loads(manifest.read_text()) == []
