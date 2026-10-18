"""Leakage tests: the failure mode that quietly invalidates matching research."""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.algorithms.registry import build_model
from backend.algorithms.supervised import BASE_FEATURES, row_features
from backend.experiments.runner import _rows_for, make_rows, run_experiment, load_config
from backend.experiments.simulation import generate_profiles, simulate_pairs
from backend.experiments.splits import (candidate_sets, grouped_pair_split,
                                        leakage_report, pair_key, participant_ids,
                                        temporal_pair_split)
from backend.features.structured import score_pair


@pytest.fixture(scope="module")
def dataset():
    profiles = generate_profiles(40, seed=21)
    pairs = simulate_pairs(profiles, seed=21, exposure_k=8)
    return profiles, pairs


def test_grouped_split_never_shares_participants(dataset):
    _, pairs = dataset
    train, test, dropped = grouped_pair_split(pairs, test_frac=0.3, seed=1)
    report = leakage_report(train, test)
    assert report["disjoint_participants"] is True
    assert report["disjoint_pairs"] is True
    assert report["shared_participants"] == []
    assert report["shared_pairs"] == 0
    assert len(train) + len(test) + len(dropped) == len(pairs)


def test_grouped_split_is_deterministic(dataset):
    _, pairs = dataset
    first = grouped_pair_split(pairs, 0.3, seed=4)
    second = grouped_pair_split(pairs, 0.3, seed=4)
    assert [pair_key(p) for p in first[0]] == [pair_key(p) for p in second[0]]
    assert [pair_key(p) for p in first[1]] == [pair_key(p) for p in second[1]]


def test_temporal_split_is_ordered_and_disjoint(dataset):
    from backend.experiments.splits import pair_time
    _, pairs = dataset
    train, test = temporal_pair_split(pairs, 0.7)
    assert train and test
    assert max(pair_time(p) for p in train) <= min(pair_time(p) for p in test)
    assert not ({pair_key(p) for p in train} & {pair_key(p) for p in test})
    assert len(train) + len(test) == len(pairs)


def test_candidate_sets_only_use_observed_directions(dataset):
    _, pairs = dataset
    groups = candidate_sets(pairs)
    assert groups
    for query_id, items in groups.items():
        for item in items:
            assert item["interest"] is not None
            assert item["query"]["participant_id"] == query_id
            assert item["candidate"]["participant_id"] != query_id


def test_feature_builder_ignores_labels(dataset):
    _, pairs = dataset
    pair = next(p for p in pairs if p["interest_a"] is not None)
    row_pos = {"a": pair["a"], "b": pair["b"], "interest_a": True, "mutual": True,
               "quality_a": 5}
    row_neg = {"a": pair["a"], "b": pair["b"], "interest_a": False, "mutual": False,
               "quality_a": 1}
    assert row_features(row_pos) == row_features(row_neg)
    assert len(row_features(row_pos)) == len(BASE_FEATURES)
    assert not any("interest" in name or "quality" in name for name in BASE_FEATURES)


def test_fitted_model_scores_do_not_depend_on_the_label(dataset):
    _, pairs = dataset
    train, test, _ = grouped_pair_split(pairs, 0.3, seed=2)
    train_rows = _rows_for(make_rows(train), "directional")
    test_rows = _rows_for(make_rows(test), "directional")
    model = build_model("directional_logistic", seed=0)
    model.fit(train_rows, "directional")

    row = test_rows[0]
    clean = {"a": row["a"], "b": row["b"]}
    leaked = {"a": row["a"], "b": row["b"], "interest_a": not row["interest_a"]}
    assert model.score(clean) == pytest.approx(model.score(leaked))


def test_score_pair_ignores_outcome_fields(dataset):
    _, pairs = dataset
    pair = pairs[0]
    a = dict(pair["a"])
    b = dict(pair["b"])
    a["interest"] = True
    b["mutual"] = True
    assert score_pair(a, b).score == score_pair(pair["a"], pair["b"]).score


def test_run_experiment_protocol_is_clean(dataset, tmp_path):
    cfg = load_config("configs/experiments/h1_structured_vs_baselines.yaml")
    cfg["synthetic"]["n_profiles"] = 40
    cfg["protocol"]["bootstrap_iters"] = 40
    results = run_experiment(cfg, out_dir=tmp_path)

    leakage = results["protocol"]["leakage"]
    assert leakage["disjoint_participants"] and leakage["disjoint_pairs"]
    assert results["protocol"]["temporal"]["leakage"]["disjoint_pairs"]

    # every reported prediction records its model and feature-schema versions
    for task, models in results["results"].items():
        assert models, f"no models evaluated for {task}"
        for payload in models.values():
            assert payload["config"]["version"]
            assert results["experiment"]["feature_schema_version"]

    # baselines are compared on identical held-out rows
    counts = {name: payload["metrics"]["n"]
              for name, payload in results["results"]["mutual"].items()}
    assert len(set(counts.values())) == 1

    csv = (tmp_path / "results.csv").read_text()
    assert "mutual,mutual_logistic" in csv
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["experiment"]["primary_metric"] == "pr_auc"
