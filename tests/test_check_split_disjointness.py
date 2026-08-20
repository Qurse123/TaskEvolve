"""$0 tests for the Arm D train/eval disjointness audit.

Run only:  python -m pytest tests/test_check_split_disjointness.py -q
"""
from __future__ import annotations

import pytest

from scripts.check_split_disjointness import (
    EVAL_SPLITS,
    TRAIN_SPLITS,
    build_overlap_matrix,
    domain_for,
    find_overlaps,
    keyed_ids,
    load_all_splits,
    load_manifest,
    main,
)


# --- synthetic: keying + overlap detection --------------------------------

def test_keyed_ids_are_domain_scoped():
    a = keyed_ids("retail", ["1", "2"])
    b = keyed_ids("airline", ["1", "2"])
    # same numeric ids, different domains -> no collision.
    assert a & b == set()
    assert ("retail", "1") in a


def test_find_overlaps_flags_shared_task():
    train = {"t": keyed_ids("retail", ["1", "2", "3"])}
    evals = {"e": keyed_ids("retail", ["3", "4"])}
    overlaps = find_overlaps(train, evals)
    assert overlaps == {("t", "e"): {("retail", "3")}}


def test_find_overlaps_empty_when_disjoint():
    train = {"t": keyed_ids("retail", ["1", "2"])}
    evals = {"e": keyed_ids("retail", ["3", "4"])}
    assert find_overlaps(train, evals) == {}


def test_same_id_across_domains_is_not_an_overlap():
    train = {"t": keyed_ids("airline", ["7"])}
    evals = {"e": keyed_ids("retail", ["7"])}  # different task, same number
    assert find_overlaps(train, evals) == {}


def test_overlap_matrix_covers_all_pairs():
    rows = {"r1": keyed_ids("retail", ["1"]), "r2": keyed_ids("retail", ["2"])}
    cols = {"c1": keyed_ids("retail", ["1"])}
    matrix = build_overlap_matrix(rows, cols)
    assert matrix[("r1", "c1")] == {("retail", "1")}
    assert matrix[("r2", "c1")] == set()


def test_domain_for_derives_train_distill_suffix():
    assert domain_for("train_distill_airline") == "airline"
    assert domain_for("validation") == "retail"
    with pytest.raises(KeyError):
        domain_for("nonexistent_split")


# --- real data: the actual splits are disjoint ----------------------------

def test_real_splits_train_eval_disjoint():
    splits = load_all_splits()
    train = {n: splits[n] for n in TRAIN_SPLITS}
    evals = {n: splits[n] for n in EVAL_SPLITS}
    assert find_overlaps(train, evals) == {}


def test_real_manifest_subset_of_train_and_disjoint_from_eval():
    splits = load_all_splits()
    train_union = set().union(*(splits[n] for n in TRAIN_SPLITS))
    eval_union = set().union(*(splits[n] for n in EVAL_SPLITS))
    manifest = load_manifest()
    assert manifest, "manifest should be non-empty"
    assert manifest <= train_union          # trained only on train_distill
    assert manifest & eval_union == set()   # never trained on an eval task


def test_main_exits_zero_on_real_data():
    assert main([]) == 0
