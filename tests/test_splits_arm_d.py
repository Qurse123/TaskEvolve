import json

from benchmark import splits_arm_d


def test_generates_disjoint_splits(tmp_path, monkeypatch):
    """Disjointness is checked per domain — TAU2 task identity is (domain, id),
    so raw ID strings coincidentally reused across different domains (e.g.
    retail task "5" and airline task "5") are unrelated tasks, not a leak.
    Unioning raw IDs across domains would be a meaningless comparison.
    """
    monkeypatch.setattr(splits_arm_d, "SPLITS_DIR", tmp_path)
    # copy the frozen retail validation so retail held-out is available
    (tmp_path / "validation.json").write_text(json.dumps(["r_val_0", "r_val_1"]))
    splits_arm_d.generate_arm_d_splits(seed=42)
    m = splits_arm_d.arm_d_split_manifest()

    val = set(json.loads((tmp_path / "validation.json").read_text()))
    assert set(m["train_distill_retail"]).isdisjoint(val), (
        "retail training pool overlaps retail's held-out validation split"
    )
    assert set(m["train_distill_airline"]).isdisjoint(set(m["eval_airline"])), (
        "airline training pool overlaps airline's held-out eval split"
    )
    assert set(m["train_distill_telecom"]).isdisjoint(set(m["eval_telecom"])), (
        "telecom training pool overlaps telecom's held-out eval split"
    )

    # banking is a pure transfer domain: never trained, so it can't leak into
    # any train_distill_* split by construction. Just assert it's a real,
    # non-empty sample.
    assert set(m["transfer_banking"]), "transfer_banking must be non-empty"
