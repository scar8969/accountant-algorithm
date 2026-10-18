"""Tests for the robustness fixes: hard ranges, shrinkage, uncertainty, guards."""

from __future__ import annotations

import copy

import pytest

from backend.experiments.runner import (DEFAULT_CONFIG, load_config, run_experiment,
                                        validate_config, rows_for_task)
from backend.experiments.simulation import generate_profiles, simulate_pairs
from backend.experiments.splits import (assert_no_leakage, assert_temporal_order,
                                        grouped_pair_split, pair_key,
                                        temporal_pair_split)
from backend.features import schema as S
from backend.features.structured import (directional_score, preference_count,
                                         score_pair, validate_profile)


def profile(**overrides) -> dict:
    base = {
        "participant_id": "p1",
        "traits": {t: 3.0 for t in S.TRAITS},
        "preferences": {t: {"range": [2, 4], "weight": 3} for t in S.TRAITS},
        "goal": "long_term",
        "flags": {f: False for f in S.FLAGS},
        "dealbreakers": [],
        "interests": ["reading"],
        "text": None,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# hard ranges
# --------------------------------------------------------------------------- #
def test_hard_range_acts_as_a_gate():
    owner = profile(preferences={"ambition": {"range": [4, 5], "weight": 5, "hard": True}})
    other = profile(participant_id="p2", traits={**{t: 3.0 for t in S.TRAITS},
                                                 "ambition": 2})
    ps = score_pair(owner, other)
    assert "ambition:hard_range" in ps.gate_a
    assert ps.eligible_a is False
    assert directional_score(owner, other) == 0.0


def test_soft_range_does_not_gate():
    owner = profile(preferences={"ambition": {"range": [4, 5], "weight": 5}})
    other = profile(participant_id="p2", traits={**{t: 3.0 for t in S.TRAITS},
                                                 "ambition": 2})
    ps = score_pair(owner, other)
    assert ps.gate_a == []
    assert ps.eligible_a is True
    assert directional_score(owner, other) > 0.0


def test_unknown_preference_key_is_rejected():
    bad = profile(preferences={"ambition": {"range": [2, 4], "wieght": 5}})
    assert any("unknown keys" in e for e in validate_profile(bad))


def test_non_boolean_hard_flag_is_rejected():
    bad = profile(preferences={"ambition": {"range": [2, 4], "hard": "yes"}})
    assert any("hard must be boolean" in e for e in validate_profile(bad))


# --------------------------------------------------------------------------- #
# directional shrinkage
# --------------------------------------------------------------------------- #
def test_preference_weight_shrinks_when_few_preferences_are_stated():
    one_pref = profile(preferences={"ambition": {"range": [1, 1], "weight": 5}})
    four_prefs = profile(preferences={t: {"range": [1, 1], "weight": 5}
                                      for t in list(S.TRAITS)[:4]})
    target = profile(participant_id="p2", traits={**{t: 3.0 for t in S.TRAITS},
                                                 "ambition": 5})
    assert preference_count(one_pref, target) == 1
    assert preference_count(four_prefs, target) == 4
    # both prefer the wrong thing, but the single-preference score stays closer to
    # the symmetric compatibility score
    assert directional_score(one_pref, target) > directional_score(four_prefs, target)


# --------------------------------------------------------------------------- #
# uncertainty + evidence
# --------------------------------------------------------------------------- #
def test_uncertainty_band_is_deterministic_and_ordered():
    a, b = profile(), profile(participant_id="p2")
    first = score_pair(a, b)
    second = score_pair(a, b)
    assert first.uncertainty == second.uncertainty
    band = first.uncertainty
    assert band["p05"] <= band["p50"] <= band["p95"]
    assert band["method"] == "weight-perturbation"
    assert "not" in band["note"]


def test_uncertainty_absent_when_score_withheld():
    empty = profile(traits={}, preferences={}, goal=None, interests=[])
    ps = score_pair(empty, profile(participant_id="p2", traits={}, preferences={},
                                   goal=None, interests=[]))
    assert ps.score is None
    assert ps.uncertainty == {}
    assert ps.evidence == "low"


def test_evidence_levels_follow_completeness():
    full = score_pair(profile(), profile(participant_id="p2"))
    assert full.evidence == "high"
    partial = profile(traits={t: 3.0 for t in list(S.TRAITS)[:5]},
                      preferences={}, goal=None, interests=[])
    ps = score_pair(partial, profile(participant_id="p2"))
    assert ps.evidence in {"medium", "low"}


# --------------------------------------------------------------------------- #
# runner guards
# --------------------------------------------------------------------------- #
def test_mutual_and_interaction_rows_are_not_duplicated():
    profiles = generate_profiles(30, seed=13)
    pairs = simulate_pairs(profiles, seed=13, exposure_k=8)
    mutual = rows_for_task(pairs, "mutual")
    interaction = rows_for_task(pairs, "interaction")
    assert len({pair_key(r["pair"]) for r in mutual}) == len(mutual)
    assert len({pair_key(r["pair"]) for r in interaction}) == len(interaction)
    directional = rows_for_task(pairs, "directional")
    assert len(directional) >= len(mutual)


def test_validate_config_rejects_unknown_model():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["models"] = {"mutual": ["not_a_model"]}
    with pytest.raises(ValueError, match="unknown model"):
        validate_config(cfg)


def test_validate_config_rejects_bad_ranges_and_keys():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["synthetic"]["n_profiles"] = 5
    with pytest.raises(ValueError, match="n_profiles"):
        validate_config(cfg)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["protocol"]["test_frac"] = 0.9
    with pytest.raises(ValueError, match="test_frac"):
        validate_config(cfg)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["surprise"] = True
    with pytest.raises(ValueError, match="unknown top-level"):
        validate_config(cfg)


def test_grouped_split_refuses_tiny_datasets():
    profiles = generate_profiles(6, seed=1)
    pairs = simulate_pairs(profiles, seed=1, exposure_k=3)
    with pytest.raises(ValueError):
        grouped_pair_split(pairs, 0.3, seed=0)


def test_leakage_assertion_raises_on_overlap():
    profiles = generate_profiles(12, seed=2)
    pairs = simulate_pairs(profiles, seed=2, exposure_k=4)
    train, test, _ = grouped_pair_split(pairs, 0.3, seed=2)
    assert_no_leakage(train, test)  # does not raise

    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(train, test + train[:1], require_disjoint_people=False)
    with pytest.raises(AssertionError, match="shared participants"):
        assert_no_leakage(train, test + train[:1])


def test_temporal_order_assertion_catches_reversed_time():
    profiles = generate_profiles(12, seed=3)
    pairs = simulate_pairs(profiles, seed=3, exposure_k=4)
    train, test = temporal_pair_split(pairs, 0.7)
    assert_temporal_order(train, test)
    with pytest.raises(AssertionError, match="temporal"):
        assert_temporal_order(test, train)


def test_run_records_power_note_and_unduplicated_counts(tmp_path):
    cfg = load_config("configs/experiments/h1_structured_vs_baselines.yaml")
    cfg["synthetic"]["n_profiles"] = 40
    cfg["protocol"]["bootstrap_iters"] = 40
    results = run_experiment(cfg, out_dir=tmp_path)

    assert results["power_note"]["mutual_positives"] > 0
    assert "PR-AUC" in results["power_note"]["statement"]
    assert results["comparison"]["mutual"]["ci_width"] is not None

    mutual = results["results"]["mutual"]
    counts = {payload["metrics"]["n"] for payload in mutual.values()}
    assert len(counts) == 1
    unique_pairs = results["dataset"]["n_mutual_observations"]
    # test-set mutual rows must not exceed the number of mutual observations
    assert counts.pop() <= unique_pairs
    for case in results["failure_cases"]:
        assert case["scored_by"] == "structured"
