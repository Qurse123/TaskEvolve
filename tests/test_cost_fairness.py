"""$0 unit tests for the cost-fairness normalization math (synthetic data only).

Run:  python -m pytest tests/test_cost_fairness.py -q
"""
from __future__ import annotations

import math

from scripts.cost_fairness import (
    cost_per_successful_task,
    estimate_tokens_from_cost,
    ratio,
    reprice_tokens,
    work_proxy_units,
)


# --- reprice_tokens: apply a single common $/token schedule ------------------ #
def test_reprice_tokens_basic():
    # 1000 in @ $1/1M + 100 out @ $3/1M = 0.001 + 0.0003
    assert reprice_tokens(1000, 100, 1e-6, 3e-6) == 0.0013


def test_reprice_at_common_rate_isolates_efficiency():
    # Two arms with nearly identical token bundles priced at ONE common rate
    # should land at nearly identical cost -- efficiency parity, no pricing gap.
    closed = reprice_tokens(95_835, 1_664, 1e-6, 3e-6)
    openm = reprice_tokens(101_971, 755, 1e-6, 3e-6)
    assert math.isclose(ratio(openm, closed), openm / closed, rel_tol=1e-12)
    # open uses marginally MORE effective tokens -> ratio just above 1.0
    assert 0.95 < ratio(openm, closed) < 1.05


def test_reprice_zero_tokens():
    assert reprice_tokens(0, 0, 1e-6, 3e-6) == 0.0


# --- cost_per_successful_task ------------------------------------------------ #
def test_cost_per_successful_task():
    assert cost_per_successful_task(0.10, 0.5) == 0.20


def test_cost_per_successful_task_no_passes_is_inf():
    assert cost_per_successful_task(0.10, 0.0) == float("inf")


# --- ratio ------------------------------------------------------------------- #
def test_ratio_normal():
    assert math.isclose(ratio(0.5768, 0.0221), 0.5768 / 0.0221, rel_tol=1e-12)


def test_ratio_headline_pricing_artifact():
    # as-listed retail vs study-served: big ratio (pricing + efficiency)
    as_listed = ratio(0.5768, 0.0221)
    assert as_listed > 20
    # same models at ONE common rate: ratio ~1 (pricing removed)
    normalized = ratio(
        reprice_tokens(95_835, 1_664, 1e-6, 3e-6),
        reprice_tokens(101_971, 755, 1e-6, 3e-6),
    )
    assert 0.9 < normalized < 1.1
    # the big listed ratio is almost entirely NOT explained by efficiency
    assert as_listed / normalized > 20


def test_ratio_zero_denominator_is_inf():
    assert ratio(1.0, 0.0) == float("inf")


# --- work_proxy_units (fallback when tokens unavailable) --------------------- #
def test_work_proxy_units_default_weights():
    assert work_proxy_units(10, 4) == 14.0


def test_work_proxy_units_custom_weights():
    assert work_proxy_units(10, 4, turn_weight=1.0, tool_weight=2.0) == 18.0


def test_work_proxy_ratio_between_arms():
    a = work_proxy_units(20, 6)   # 26
    b = work_proxy_units(10, 3)   # 13
    assert ratio(a, b) == 2.0


# --- estimate_tokens_from_cost (blended backout, for transcript-less arms) --- #
def test_estimate_tokens_from_cost():
    # $0.0497 at a 2.12e-6 blended rate ~= 23.4k tokens
    est = estimate_tokens_from_cost(0.0497, 2.12e-6)
    assert math.isclose(est, 0.0497 / 2.12e-6, rel_tol=1e-9)


def test_estimate_tokens_from_cost_zero_rate_is_zero():
    assert estimate_tokens_from_cost(0.05, 0.0) == 0.0


# --- tuned-model efficiency gain survives normalization ---------------------- #
def test_tuning_efficiency_gain_visible_at_common_rate():
    base = reprice_tokens(89_836, 889, 1e-6, 3e-6)
    tuned = reprice_tokens(59_785, 946, 1e-6, 3e-6)
    # tuned is genuinely cheaper at the SAME price -> a real efficiency win
    assert tuned < base
    assert ratio(base, tuned) > 1.3
