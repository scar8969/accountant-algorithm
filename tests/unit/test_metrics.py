"""Metric tests, including known-answer ranking cases."""

from __future__ import annotations

import numpy as np
import pytest

from backend.evaluation import ranking as rk
from backend.evaluation.classification import (brier, ece, log_loss, pr_auc,
                                               reliability, roc_auc, summarise)
from backend.evaluation.stats import cluster_bootstrap


def test_ndcg_perfect_and_reversed():
    rel = [3, 2, 1, 0]
    assert rk.ndcg_at_k(rel, 4) == pytest.approx(1.0)
    assert rk.ndcg_at_k([0, 1, 2, 3], 4) < 1.0
    assert rk.ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_known_value():
    # binary relevance, 2 relevant items; DCG = 1/log2(2) + 1/log2(4) = 1.5
    # ideal ordering gives 1.5 as well, so NDCG = 1.0
    assert rk.ndcg_at_k([1, 1, 0], 3) == pytest.approx(1.0)
    # exactly one relevant item, ranked second: DCG = 1/log2(3), ideal = 1
    assert rk.ndcg_at_k([0, 1, 0], 3) == pytest.approx(1 / np.log2(3))


def test_recall_at_k_and_mrr():
    assert rk.recall_at_k([1, 0, 1], 2) == pytest.approx(0.5)
    assert rk.recall_at_k([1, 0, 1], 3) == pytest.approx(1.0)
    assert rk.mrr([0, 0, 1]) == pytest.approx(1 / 3)
    assert rk.mrr([0, 0, 0]) == 0.0


def test_evaluate_rankings_skips_single_candidate_queries():
    block = rk.evaluate_rankings([[(0.9, 1), (0.1, 0)], [(0.5, 1)]], k=5)
    assert block["n_queries"] == 1
    assert block["skipped_queries"] == 1
    assert block["ndcg@5"] == 1.0


def test_pr_auc_perfect_separation_and_prevalence():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert pr_auc(y, p) == pytest.approx(1.0)
    assert roc_auc(y, p) == pytest.approx(1.0)
    block = summarise(y, p)
    assert block["prevalence"] == 0.5
    assert block["positives"] == 2
    assert block["probabilistic"] is True


def test_pr_auc_returns_none_for_single_class():
    y = np.zeros(5)
    p = np.linspace(0.1, 0.9, 5)
    assert pr_auc(y, p) is None
    assert roc_auc(y, p) is None


def test_log_loss_and_brier():
    y = np.array([1.0, 0.0])
    p = np.array([0.5, 0.5])
    assert log_loss(y, p) == pytest.approx(np.log(2))
    assert brier(y, p) == pytest.approx(0.25)


def test_reliability_and_ece_on_calibrated_bins():
    # two bins, both perfectly calibrated
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    bins = reliability(y, p, bins=2)
    assert sum(b["n"] for b in bins) == 6
    assert ece(y, p, bins=2) == pytest.approx(0.0, abs=1e-9)


def test_ece_positive_when_overconfident():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.9, 0.9, 0.9, 0.9])
    assert ece(y, p, bins=5) > 0.5


def test_cluster_bootstrap_ci_brackets_estimate():
    items = [{"y": float(i % 2), "s": float(i)} for i in range(60)]
    groups = [f"p{i % 20}" for i in range(60)]

    def metric(sample):
        y = np.array([i["y"] for i in sample])
        s = np.array([i["s"] for i in sample])
        return pr_auc(y, s)

    block = cluster_bootstrap(items, groups, metric, iters=100, seed=0)
    assert block["estimate"] is not None
    assert block["lo"] is not None and block["hi"] is not None
    assert block["lo"] <= block["hi"]
    assert block["n_valid"] > 0


def test_summarise_score_only_models_report_no_probability_metrics():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    block = summarise(y, None, scores=scores)
    assert block["pr_auc"] == pytest.approx(1.0)
    assert block["probabilistic"] is False
    assert block.get("brier") is None
    assert block.get("log_loss") is None
