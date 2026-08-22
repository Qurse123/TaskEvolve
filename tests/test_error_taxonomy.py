"""$0 unit tests for the failure taxonomy classifier (Tier-2 item #11).

Tests operate on synthetic verdict + reward_info dicts — no filesystem, no
network. Run only:

    python -m pytest tests/test_error_taxonomy.py -q
"""

from __future__ import annotations

from scripts.error_taxonomy import (
    TERM_HARNESS,
    TERM_MAX_STEPS,
    TERM_TOO_MANY_ERRORS,
    TERM_USER_STOP,
    classify_failure,
    classify_termination,
    failure_loci,
    harness_error_kind,
    is_failed,
    is_harness_error,
)


# --------------------------------------------------------------------------- #
# termination_reason -> bucket
# --------------------------------------------------------------------------- #

def test_harness_error_detected_by_prefix():
    t = "harness_error: SandboxRuntimeError: required binaries not installed"
    assert is_harness_error(t) is True
    assert classify_termination(t) == TERM_HARNESS
    assert harness_error_kind(t) == "SandboxRuntimeError"


def test_non_harness_reasons_not_flagged_as_harness():
    assert is_harness_error("user_stop") is False
    assert is_harness_error(None) is False
    assert harness_error_kind("user_stop") == ""


def test_termination_bucket_mapping():
    assert classify_termination("user_stop") == TERM_USER_STOP
    assert classify_termination("max_steps") == TERM_MAX_STEPS
    assert classify_termination("max_turns") == TERM_MAX_STEPS
    assert classify_termination("too_many_errors") == TERM_TOO_MANY_ERRORS
    assert classify_termination("something_else") == "other"


# --------------------------------------------------------------------------- #
# is_failed — passed tasks excluded
# --------------------------------------------------------------------------- #

def test_passed_task_is_not_a_failure():
    assert is_failed({"passed": True, "reward": 1.0}) is False


def test_reward_below_one_is_a_failure_even_if_passed_missing():
    assert is_failed({"reward": 0.0}) is True
    assert is_failed({"reward": 0.5}) is True


def test_passed_false_is_failure_regardless_of_reward():
    assert is_failed({"passed": False, "reward": 1.0}) is True


def test_reward_falls_back_to_reward_info():
    assert is_failed({"passed": None}, {"reward": 0.0}) is True
    assert is_failed({"passed": None}, {"reward": 1.0}) is False


# --------------------------------------------------------------------------- #
# failure locus — db vs action vs nl
# --------------------------------------------------------------------------- #

def test_db_locus_from_breakdown_and_dbcheck():
    ri = {"reward_breakdown": {"DB": 0.0}, "db_check": {"db_match": False}}
    assert failure_loci(ri) == ["DB"]


def test_action_locus_from_action_checks():
    ri = {
        "reward_breakdown": {"ACTION": 0.0},
        "action_checks": [{"action_match": False, "tool_type": "write"}],
    }
    assert failure_loci(ri) == ["ACTION"]


def test_both_db_and_action_failed():
    ri = {"reward_breakdown": {"DB": 0.0, "ACTION": 0.0}}
    assert failure_loci(ri) == ["ACTION", "DB"]


def test_passing_locus_not_reported():
    # DB failed, NL passed -> only DB is a locus.
    ri = {"reward_breakdown": {"DB": 0.0, "NL_ASSERTION": 1.0}}
    assert failure_loci(ri) == ["DB"]


def test_no_reward_info_gives_no_locus():
    assert failure_loci(None) == []
    assert failure_loci({}) == []


# --------------------------------------------------------------------------- #
# classify_failure — end to end on synthetic dicts
# --------------------------------------------------------------------------- #

def test_classify_harness_error_has_no_locus():
    verdict = {
        "termination_reason": "harness_error: SandboxRuntimeError: nope",
        "passed": False,
        "reward": 0.0,
    }
    cls = classify_failure(verdict, {"reward_breakdown": {"DB": 0.0}})
    assert cls.category == "harness_error"
    assert cls.harness_kind == "SandboxRuntimeError"
    # harness errors are not assigned a failure locus
    assert cls.db_failed is False
    assert cls.action_failed is False
    assert cls.loci == []


def test_classify_genuine_db_failure_user_stop():
    verdict = {"termination_reason": "user_stop", "passed": False, "reward": 0.0}
    ri = {"reward_breakdown": {"DB": 0.0}, "db_check": {"db_match": False}}
    cls = classify_failure(verdict, ri)
    assert cls.category == "genuine"
    assert cls.termination_bucket == TERM_USER_STOP
    assert cls.db_failed is True
    assert cls.action_failed is False
    assert cls.locus_label == "db_only"


def test_classify_genuine_db_plus_action():
    verdict = {"termination_reason": "user_stop", "passed": False, "reward": 0.0}
    ri = {"reward_breakdown": {"DB": 0.0, "ACTION": 0.0}}
    cls = classify_failure(verdict, ri)
    assert cls.db_failed and cls.action_failed
    assert cls.locus_label == "db+action"


def test_classify_genuine_unknown_locus_when_no_breakdown():
    verdict = {"termination_reason": "user_stop", "passed": False, "reward": 0.5}
    cls = classify_failure(verdict, {"reward_breakdown": {}})
    assert cls.category == "genuine"
    assert cls.locus_label == "unknown"
