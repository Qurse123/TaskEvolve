import json
from pathlib import Path

from arm_d.build_dataset import build_examples, balance, write_manifest, TrainingExample


def _transcript(tmp, name, reward, msgs):
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "task_1_messages.json").write_text(json.dumps(
        {"task_id": "t1", "reward_info": {"reward": reward}, "termination_reason": "done", "messages": msgs}))
    return d


def _msgs():
    return [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "cancel my order"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "cancel", "arguments": "{}"}}]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "Done."},
    ]


def test_keeps_only_passed_and_masks_assistant_targets(tmp_path):
    p = _transcript(tmp_path, "train_distill_retail_x", 1.0, _msgs())
    f = _transcript(tmp_path, "train_distill_retail_y", 0.0, _msgs())  # failed -> dropped
    ex = build_examples([p, f], domains={"train_distill_retail_x": "retail", "train_distill_retail_y": "retail"})
    assert len(ex) == 1
    e = ex[0]
    # only the two assistant messages (idx 2 and 4) are targets
    assert e.target_mask == [False, False, True, False, True]
    assert e.domain == "retail"


def test_manifest_has_hash_per_example(tmp_path):
    p = _transcript(tmp_path, "train_distill_retail_x", 1.0, _msgs())
    ex = build_examples([p], domains={"train_distill_retail_x": "retail"})
    out = tmp_path / "manifest.json"
    write_manifest(ex, out)
    rows = json.loads(out.read_text())
    assert rows and set(rows[0]) == {"task_id", "domain", "source_run", "sha256"}
