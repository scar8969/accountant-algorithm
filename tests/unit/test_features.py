"""Unit tests for the structured feature engine."""

from __future__ import annotations

import math

import pytest

from backend.features import schema as S
from backend.features.structured import (answer_completeness, directional_score,
                                         explanation, pref_satisfaction,
                                         profile_confidence, score_pair,
                                         similarity, validate_profile)


def base_profile(**overrides) -> dict:
    profile = {
        "participant_id": "p1",
        "traits": {t: 3.0 for t in S.TRAITS},
        "preferences": {t: {"range": [2, 4], "weight": 3} for t in S.TRAITS},
        "goal": "long_term",
        "flags": {f: False for f in S.FLAGS},
        "dealbreakers": [],
        "interests": ["reading", "hiking"],
        "text": "Quiet weekends, long walks, science fiction.",
    }
    profile.update(overrides)
    return profile


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def test_similarity_bounds_and_symmetry():
    assert similarity(3, 3) == 1.0
    assert similarity(1, 5) == 0.0
    assert similarity(2, 4) == pytest.approx(0.5)
    assert similarity(2, 4) == similarity(4, 2)
    assert 0.0 <= similarity(1, 2) <= 1.0


def test_pref_satisfaction_inside_and_outside_range():
    assert pref_satisfaction([2, 4], 3) == 1.0
    assert pref_satisfaction([2, 4], 4) == 1.0
    assert pref_satisfaction([2, 4], 5) == pytest.approx(0.75)
    assert pref_satisfaction([2, 4], 1) == pytest.approx(0.75)
    assert pref_satisfaction([3, 3], 1) == pytest.approx(0.5)
    assert pref_satisfaction([1, 1], 5) == 0.0


# --------------------------------------------------------------------------- #
# pair scoring
# --------------------------------------------------------------------------- #
def test_identical_profiles_score_high_and_symmetrically():
    a = base_profile()
    b = base_profile(participant_id="p2")
    ps_ab = score_pair(a, b)
    ps_ba = score_pair(b, a)
    assert ps_ab.score is not None
    assert ps_ab.score == ps_ba.score
    assert ps_ab.score > 95
    assert ps_ab.coverage == 1.0
    assert ps_ab.dimensions["values"] == pytest.approx(1.0)


def test_different_profiles_score_lower():
    a = base_profile()
    b = base_profile(participant_id="p2",
                     traits={t: 1.0 for t in S.TRAITS},
                     goal="casual")
    assert score_pair(a, b).score < score_pair(a, base_profile(participant_id="p3")).score


def test_missing_answers_withhold_score_instead_of_imputing():
    a = base_profile(traits={}, preferences={}, goal=None, interests=[])
    b = base_profile(participant_id="p2", traits={}, preferences={}, goal=None,
                     interests=[])
    ps = score_pair(a, b)
    assert ps.score is None
    assert ps.insufficient is True
    assert ps.coverage == 0.0
    assert ps.eligible_a is False
    assert any("coverage" in note for note in ps.limitations)


def test_partial_missingness_lowers_coverage_but_keeps_score():
    traits = {t: 3.0 for t in S.TRAITS}
    for name in list(traits)[:2]:
        traits.pop(name)
    a = base_profile(traits=traits)
    ps = score_pair(a, base_profile(participant_id="p2"))
    assert 0.0 < ps.coverage < 1.0
    assert ps.score is not None


def test_dealbreakers_gate_directionally():
    a = base_profile(dealbreakers=["smoker"])
    b = base_profile(participant_id="p2", flags={**{f: False for f in S.FLAGS},
                                                 "smoker": True})
    ps = score_pair(a, b)
    assert ps.gate_a == ["smoker"]
    assert ps.gate_b == []
    assert ps.eligible_a is False
    assert ps.eligible_b is True
    # hard gate applies to the gated direction only
    assert directional_score(b, a) is not None
    assert directional_score(a, b) == 0.0


def test_directional_score_uses_owner_preferences():
    wants_high = base_profile(preferences={"ambition": {"range": [5, 5], "weight": 5}})
    high = base_profile(participant_id="p2", traits={**{t: 3.0 for t in S.TRAITS},
                                                     "ambition": 5})
    low = base_profile(participant_id="p3", traits={**{t: 3.0 for t in S.TRAITS},
                                                    "ambition": 1})
    assert directional_score(wants_high, high) > directional_score(wants_high, low)

    # asymmetry comes from stated preferences, not from the similarity term
    likes_low = base_profile(participant_id="p4",
                             preferences={"ambition": {"range": [1, 1], "weight": 5}})
    assert directional_score(likes_low, low) > directional_score(wants_high, low)


def test_directional_score_none_when_insufficient():
    empty = base_profile(traits={}, preferences={}, goal=None, interests=[])
    assert directional_score(empty, base_profile()) is None


def test_explanation_points_reconstruct_the_score():
    ps = score_pair(base_profile(), base_profile(participant_id="p2"))
    exp = explanation(ps)
    present = [v for v in ps.leaves.values() if v["s"] is not None]
    denom = sum(v["w"] for v in present)
    rebuilt = sum(v["w"] / denom * 100.0 * v["s"] for v in present)
    assert rebuilt == pytest.approx(ps.score, abs=0.01)
    assert exp["coverage"] == ps.coverage
    assert all(0.0 <= c["points"] <= 100.0 for c in exp["contributions"])


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_validate_accepts_valid_profile():
    assert validate_profile(base_profile()) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda p: p["traits"].__setitem__("ambition", 9), "ambition"),
    (lambda p: p["traits"].__setitem__("ambition", "high"), "ambition"),
    (lambda p: p["preferences"].__setitem__("ambition", {"range": [4, 2]}), "ambition"),
    (lambda p: p["preferences"].__setitem__("ambition", {"range": [0, 3]}), "ambition"),
    (lambda p: p["preferences"].__setitem__("ambition", {"range": [2, 4], "weight": 9}),
     "weight"),
    (lambda p: p.__setitem__("goal", "situationship"), "goal"),
    (lambda p: p.__setitem__("dealbreakers", ["height"]), "deal-breaker"),
    (lambda p: p["flags"].__setitem__("smoker", "yes"), "smoker"),
    (lambda p: p.__setitem__("traits", {"charisma": 4}), "charisma"),
])
def test_validate_rejects_bad_input(mutate, needle):
    profile = base_profile()
    mutate(profile)
    errors = validate_profile(profile)
    assert errors
    assert any(needle in e for e in errors)


# --------------------------------------------------------------------------- #
# confidence
# --------------------------------------------------------------------------- #
def test_completeness_and_confidence_scale_with_missing_answers():
    full = answer_completeness(base_profile())
    assert full == 1.0
    partial = base_profile(goal=None, interests=[])
    assert answer_completeness(partial) < full
    assert 0.0 <= profile_confidence(partial) <= 1.0


def test_recency_decays_confidence_not_score():
    from datetime import datetime, timedelta, timezone
    old = (datetime.now(timezone.utc) - timedelta(days=730)).isoformat()
    a = base_profile(updated_at=old)
    b = base_profile(participant_id="p2", updated_at=old)
    ps = score_pair(a, b, reference_time=datetime.now(timezone.utc))
    assert ps.score is not None
    assert ps.confidence_a < 0.5  # two half-lives of decay
    assert ps.score == score_pair(base_profile(), base_profile(participant_id="p2")).score


def test_weights_are_pre_registered_and_normalised():
    total = (sum(w for _, w in S.SIMILARITY_FEATURES.values())
             + sum(w for _, w in S.COMPLEMENTARY_FEATURES.values())
             + S.GOAL_WEIGHT + S.TEXT_WEIGHT)
    assert math.isclose(total, 1.0, abs_tol=1e-9)
