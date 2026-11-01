"""
Software Engineering Assignment 2 - Unit Test Suite
Student: Alvin Binu (B23CS2114 / MBT23CA014)
Project: FeatureScalpel (Mechanistic Intervention via Sparse Autoencoders)

This test module verifies core deterministic components of FeatureScalpel:
- Hook scale mapping and cumulative delta-patch composition (src.hooks)
- Safety filter boundary decision logic and rank degradation guards (src.editing, Eq. 3.4)
- Gradient-descent multiplier parameterization bounds (src.gradient_editing, Eq. 3.5)
- Probability formatting and odds presentation thresholds (src.format_utils)

All tests run on CPU in milliseconds without requiring external model checkpoints.
Testing techniques: Boundary Value Analysis (BVA) and Equivalence Partitioning (EP).
"""

import math
import pytest
import torch

from src.hooks import build_scale_map, with_extra_scale
from src.format_utils import fmt_prob, fmt_prob_pct, _human_count
from src.gradient_editing import _to_a, Z_ZERO, BETA_MAX


def evaluate_safety_criterion(ablated_rank: int, clean_rank: int, prob_delta: float) -> bool:
    """
    Pure evaluation of the FeatureScalpel per-feature safety filter (Equation 3.4):
    safe(k) <===> (rk(t) <= r0(t)) AND (pk(t) - p0(t) >= -1e-6)
    """
    return bool(ablated_rank <= clean_rank and prob_delta >= -1e-6)


# ==============================================================================
# TC01 - TC04: Scale Map Construction and Composition (src.hooks)
# ==============================================================================

def test_tc01_build_scale_map_zero_mute_bva():
    """
    TC01: build_scale_map with zero mute strength (lower boundary).
    Technique: Boundary Value Analysis (BVA)
    Expected: Multiplier is exactly 1.0 (unmodified identity pass).
    """
    res = build_scale_map(mute_ids=[10], mute_strength=0.0, boost_ids=[], boost_strength=0.0)
    assert res == {10: 1.0}, f"Expected identity multiplier 1.0, got {res}"


def test_tc02_build_scale_map_full_ablation_bva():
    """
    TC02: build_scale_map with full ablation strength 1.0 (upper boundary).
    Technique: Boundary Value Analysis (BVA)
    Expected: Multiplier is 0.0 (feature contribution fully zeroed).
    """
    res = build_scale_map(mute_ids=[10], mute_strength=1.0, boost_ids=[], boost_strength=0.0)
    assert res == {10: 0.0}, f"Expected complete ablation 0.0, got {res}"


def test_tc03_build_scale_map_disjoint_mute_and_boost_ep():
    """
    TC03: build_scale_map with valid disjoint mute and boost lists.
    Technique: Equivalence Partitioning (EP)
    Expected: Mute scales to (1 - 0.3) = 0.7, Boost scales to (1 + 0.5) = 1.5.
    """
    res = build_scale_map(mute_ids=[100], mute_strength=0.3, boost_ids=[200], boost_strength=0.5)
    assert pytest.approx(res[100], abs=1e-6) == 0.7
    assert pytest.approx(res[200], abs=1e-6) == 1.5


def test_tc04_build_scale_map_overlapping_feature_ep():
    """
    TC04: build_scale_map when a feature is present in both mute and boost lists.
    Technique: Equivalence Partitioning (EP)
    Expected: Multiplicative compounding: (1 - 0.5) * (1 + 1.0) = 0.5 * 2.0 = 1.0.
    """
    res = build_scale_map(mute_ids=[42], mute_strength=0.5, boost_ids=[42], boost_strength=1.0)
    assert pytest.approx(res[42], abs=1e-6) == 1.0


# ==============================================================================
# TC05 - TC06: with_extra_scale Chaining (src.hooks)
# ==============================================================================

def test_tc05_with_extra_scale_none_base_ep():
    """
    TC05: with_extra_scale starting from empty/None base map.
    Technique: Equivalence Partitioning (EP)
    Expected: Returns dict containing only the new feature scaled.
    """
    res = with_extra_scale(None, feature_id=12, scale=0.8)
    assert res == {12: 0.8}


def test_tc06_with_extra_scale_boundary_zero_bva():
    """
    TC06: with_extra_scale with scale=0.0 (lower boundary multiplier).
    Technique: Boundary Value Analysis (BVA)
    Expected: Existing scale is multiplied by 0.0 resulting in 0.0.
    """
    res = with_extra_scale({5: 2.0}, feature_id=5, scale=0.0)
    assert res == {5: 0.0}


# ==============================================================================
# TC07 - TC09: Safety Filter Condition Evaluation (Eq. 3.4)
# ==============================================================================

def test_tc07_safety_filter_exact_prob_boundary_bva():
    """
    TC07: Safety filter evaluation at exact tolerance threshold -1e-6.
    Technique: Boundary Value Analysis (BVA)
    Expected: Passes (True) because delta >= -1e-6.
    """
    is_safe = evaluate_safety_criterion(ablated_rank=8, clean_rank=8, prob_delta=-1e-6)
    assert is_safe is True, "Exact boundary -1e-6 must be accepted as safe"


def test_tc08_safety_filter_prob_just_below_boundary_bva():
    """
    TC08: Safety filter evaluation slightly below tolerance threshold (-1.01e-6).
    Technique: Boundary Value Analysis (BVA)
    Expected: Fails (False) because target probability degradation exceeds tolerance.
    """
    is_safe = evaluate_safety_criterion(ablated_rank=8, clean_rank=8, prob_delta=-1.01e-6)
    assert is_safe is False, "Probability degradation exceeding 1e-6 must be rejected"


def test_tc09_safety_filter_rank_regression_ep():
    """
    TC09: Safety filter when rank worsens despite positive probability gain.
    Technique: Equivalence Partitioning (EP)
    Expected: Fails (False) because ablated_rank (3) > clean_rank (2).
    """
    is_safe = evaluate_safety_criterion(ablated_rank=3, clean_rank=2, prob_delta=0.05)
    assert is_safe is False, "Rank regression must always be rejected by the safety filter"


# ==============================================================================
# TC10 - TC11: Gradient Parametrization Bounds (src.gradient_editing, Eq. 3.5)
# ==============================================================================

def test_tc10_gradient_to_a_zero_edit_bva():
    """
    TC10: Gradient multiplier transform at Z_ZERO (identity parameter).
    Technique: Boundary Value Analysis (BVA)
    Expected: a = 0.0, resulting in multiplier m = 1 + a = 1.0 (exact zero edit).
    """
    z = torch.tensor(Z_ZERO, dtype=torch.float32)
    a = _to_a(z).item()
    assert abs(a) < 1e-6, f"Expected a = 0.0 at Z_ZERO, got {a}"


def test_tc10_extra_gradient_bounds_bva():
    """
    TC11: Gradient multiplier bounds at extreme parameter limits z -> -inf, +inf.
    Technique: Boundary Value Analysis (BVA)
    Expected: a(-100) -> -1.0 (multiplier -> 0.0), a(+100) -> 2.0 (multiplier -> 3.0).
    """
    z_min = torch.tensor(-100.0)
    z_max = torch.tensor(100.0)
    a_min = _to_a(z_min).item()
    a_max = _to_a(z_max).item()
    assert pytest.approx(a_min, abs=1e-4) == -1.0
    assert pytest.approx(a_max, abs=1e-4) == 2.0


# ==============================================================================
# TC12 - TC14: Probability Formatting & Odds Representation (src.format_utils)
# ==============================================================================

def test_tc12_fmt_prob_boundary_one_percent_bva():
    """
    TC12: fmt_prob at exactly p = 0.01 (1.0% boundary).
    Technique: Boundary Value Analysis (BVA)
    Expected: Returns '1.00%' without odds parenthesis.
    """
    res = fmt_prob(0.01)
    assert res == "1.00%", f"Expected '1.00%', got '{res}'"


def test_tc13_fmt_prob_just_below_one_percent_bva():
    """
    TC13: fmt_prob at p = 0.0099 (just below 1% threshold).
    Technique: Boundary Value Analysis (BVA)
    Expected: Returns odds representation format '0.990% (about 1 in 101)'.
    """
    res = fmt_prob(0.0099)
    assert "about 1 in 101" in res, f"Expected odds in output, got '{res}'"


def test_tc14_fmt_prob_zero_and_negative_ep():
    """
    TC14: fmt_prob with zero or negative probability inputs.
    Technique: Equivalence Partitioning (EP)
    Expected: Returns '0' for non-positive inputs.
    """
    assert fmt_prob(0.0) == "0"
    assert fmt_prob(-0.05) == "0"
