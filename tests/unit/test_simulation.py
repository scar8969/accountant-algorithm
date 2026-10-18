"""Simulation reproducibility and label-honesty tests."""

from __future__ import annotations

import pytest

from backend.experiments.simulation import (generate_profiles, simulate_pairs,
                                            simulation_manifest)
from backend.features import schema as S
from backend.features.structured import validate_profile


def test_profiles_are_reproducible_for_a_seed():
    a = generate_profiles(12, seed=7)
    b = generate_profiles(12, seed=7)
    c = generate_profiles(12, seed=8)
    assert [(p["participant_id"], p["traits"], p["text"]) for p in a] == \
           [(p["participant_id"], p["traits"], p["text"]) for p in b]
    assert a[0]["traits"] != c[0]["traits"] or a[0]["text"] != c[0]["text"]


def test_generated_profiles_pass_validation():
    for profile in generate_profiles(40, seed=11):
        assert validate_profile(profile) == []


def test_missing_answers_exist_but_are_not_fatal():
    profiles = generate_profiles(60, seed=3, missing_rate=0.35)
    missing = sum(1 for p in profiles for t in S.TRAITS if t not in p["traits"])
    assert missing > 0
    assert all(validate_profile(p) == [] for p in profiles)


def test_interest_labels_only_exist_where_exposed():
    profiles = generate_profiles(30, seed=5)
    pairs = simulate_pairs(profiles, seed=5, exposure_k=6)
    assert pairs
    for pair in pairs:
        if not pair["exposed_a"]:
            assert pair["interest_a"] is None
        if not pair["exposed_b"]:
            assert pair["interest_b"] is None
        if pair["mutual"] is not None:
            assert pair["exposed_a"] and pair["exposed_b"]


def test_mutual_requires_both_sides():
    profiles = generate_profiles(30, seed=5)
    pairs = simulate_pairs(profiles, seed=5, exposure_k=6)
    for pair in pairs:
        if pair["mutual"] is True:
            assert pair["interest_a"] is True and pair["interest_b"] is True
        if pair["mutual"] is False:
            assert pair["interest_a"] is False or pair["interest_b"] is False


def test_quality_labels_only_exist_for_mutual_pairs():
    profiles = generate_profiles(40, seed=9)
    pairs = simulate_pairs(profiles, seed=9, exposure_k=8)
    for pair in pairs:
        if pair["quality_a"] is not None:
            assert pair["mutual"] is True
            assert 1 <= pair["quality_a"] <= 5


def test_exposure_is_bounded_and_symmetric_in_both_directions():
    profiles = generate_profiles(25, seed=2)
    pairs = simulate_pairs(profiles, seed=2, exposure_k=5)
    per_profile: dict[str, int] = {}
    for pair in pairs:
        for profile, exposed in ((pair["a"], pair["exposed_a"]),
                                 (pair["b"], pair["exposed_b"])):
            per_profile[profile["participant_id"]] = \
                per_profile.get(profile["participant_id"], 0) + int(exposed)
    assert all(count <= 5 for count in per_profile.values())
    assert all(count >= 1 for count in per_profile.values())


def test_manifest_documents_the_generative_process():
    manifest = simulation_manifest()
    for key in ("version", "interest_model", "exposure_model", "interaction_model",
                "warning"):
        assert key in manifest
    assert "not" in manifest["warning"].lower() or "synthetic" in manifest["warning"].lower()
