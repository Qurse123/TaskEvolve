import json

import pytest

from benchmark import splits


def test_load_split_reads_any_existing_file(tmp_path, monkeypatch):
    d = tmp_path / "splits"
    d.mkdir()
    (d / "eval_airline.json").write_text(json.dumps(["airline_0", "airline_1"]))
    monkeypatch.setattr(splits, "SPLITS_DIR", d)
    assert splits.load_split("eval_airline") == ["airline_0", "airline_1"]


def test_load_split_unknown_name_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        splits.load_split("does_not_exist")
