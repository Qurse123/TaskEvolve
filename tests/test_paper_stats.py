"""$0 unit tests for scripts.paper_stats — synthetic inputs, no CSV/network."""
from __future__ import annotations

import math

import numpy as np

from scripts.paper_stats import (
    bootstrap_ci,
    cohens_d,
    holm_adjust,
    permutation_test,
    short_model,
    std_ratio_ci,
)


def test_bootstrap_ci_brackets_the_mean():
    data = [10.0, 11.0, 9.0, 12.0, 8.0, 10.5]
    lo, hi = bootstrap_ci(data, n_boot=2000, rng=np.random.default_rng(0))
    mean = float(np.mean(data))
    assert lo <= mean <= hi
    assert lo < hi  # non-degenerate interval


def test_bootstrap_ci_degenerate_constant():
    lo, hi = bootstrap_ci([5.0, 5.0, 5.0], n_boot=500, rng=np.random.default_rng(1))
    assert lo == hi == 5.0


def test_permutation_identical_groups_p_is_one():
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    assert permutation_test(a, b) == 1.0


def test_permutation_well_separated_is_small():
    # n=3 vs n=3: floor is 2/20 = 0.10; a maximally separated split hits the floor.
    a = [10.0, 11.0, 12.0]
    b = [0.0, 1.0, 2.0]
    p = permutation_test(a, b)
    assert math.isclose(p, 0.1, abs_tol=1e-9)


def test_permutation_larger_separated_below_005():
    # n=5 vs n=5: floor is 2/252 ~= 0.0079, so a clean separation is < 0.05.
    a = [10.0, 11.0, 12.0, 13.0, 14.0]
    b = [0.0, 1.0, 2.0, 3.0, 4.0]
    p = permutation_test(a, b)
    assert p < 0.05


def test_permutation_symmetric_two_sided():
    a = [5.0, 6.0, 7.0]
    b = [1.0, 2.0, 3.0]
    assert permutation_test(a, b) == permutation_test(b, a)


def test_holm_ordering_correct():
    # p sorted asc: 0.01, 0.03, 0.04 -> (3*.01, 2*.03, 1*.04)=.03,.06,.04
    # with running max -> .03, .06, .06, mapped back to input order.
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert math.isclose(adj[0], 0.03, abs_tol=1e-12)
    assert math.isclose(adj[1], 0.06, abs_tol=1e-12)
    assert math.isclose(adj[2], 0.06, abs_tol=1e-12)


def test_holm_monotone_and_capped():
    adj = holm_adjust([0.5, 0.6, 0.9])
    assert all(0.0 <= p <= 1.0 for p in adj)
    # Adjusted p never below the raw p.
    assert adj[0] >= 0.5 and adj[2] >= 0.9


def test_holm_adjusted_ge_raw():
    raw = [0.001, 0.02, 0.2, 0.04]
    adj = holm_adjust(raw)
    for r, a in zip(raw, adj):
        assert a >= r - 1e-12


def test_cohens_d_sign_and_zero_variance():
    assert cohens_d([3, 4, 5], [1, 2, 3]) > 0
    assert cohens_d([1, 2, 3], [3, 4, 5]) < 0
    assert cohens_d([5, 5, 5], [5, 5, 5]) == 0.0


def test_std_ratio_equal_spread_near_one():
    point, lo, hi = std_ratio_ci(
        [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], n_boot=1000, rng=np.random.default_rng(0))
    assert math.isclose(point, 1.0, abs_tol=1e-9)
    assert lo <= 1.0 <= hi or lo == hi  # CI contains the point (may be wide at n=3)


def test_short_model_mapping():
    assert short_model("gpt-4.1") == "gpt-4.1"
    assert short_model("openai/thinkingmachines/Inkling") == "inkling"
    assert short_model("openai/thinkingmachines/Inkling-Small") == "inkling-small"
    assert short_model("openai/armd-inkling-small-tuned") == "armd-tuned"
    assert short_model("anthropic/claude-opus-4-8") == "opus"
