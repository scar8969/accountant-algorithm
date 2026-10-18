"""Ranking metrics: NDCG@k, Recall@k, MRR, plus coverage and diversity."""

from __future__ import annotations

import math


def dcg(relevances: list[float], k: int | None = None) -> float:
    rel = relevances if k is None else relevances[:k]
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at_k(relevances: list[float], k: int = 5) -> float:
    ideal = dcg(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg(relevances, k) / ideal


def recall_at_k(relevances: list[float], k: int = 5) -> float:
    total = sum(1 for r in relevances if r > 0)
    if total == 0:
        return 0.0
    return sum(1 for r in relevances[:k] if r > 0) / total


def mrr(relevances: list[float]) -> float:
    for i, r in enumerate(relevances):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_rankings(ranked: list[list[tuple[float, float]]], k: int = 5) -> dict:
    """`ranked` = one list of (score, relevance) per query."""
    ndcgs, recalls, rr, hit_rates = [], [], [], []
    skipped = 0
    for items in ranked:
        if len(items) < 2:            # a single candidate cannot be ranked
            skipped += 1
            continue
        ordered = [rel for _, rel in sorted(items, key=lambda t: (-t[0], t[1]))]
        ndcgs.append(ndcg_at_k(ordered, k))
        recalls.append(recall_at_k(ordered, k))
        rr.append(mrr(ordered))
        hit_rates.append(1.0 if sum(1 for r in ordered[:k] if r > 0) else 0.0)

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "k": k,
        "n_queries": len(ndcgs),
        "skipped_queries": skipped,
        f"ndcg@{k}": mean(ndcgs),
        f"recall@{k}": mean(recalls),
        "mrr": mean(rr),
        f"hit_rate@{k}": mean(hit_rates),
    }


def coverage(scores: list[float | None]) -> float:
    if not scores:
        return 0.0
    return round(sum(1 for s in scores if s is not None) / len(scores), 4)


def diversity_at_k(top_items: list[dict], key: str, query_value, k: int = 5) -> float:
    """1 - share of top-k candidates sharing the query's value for `key`."""
    top = top_items[:k]
    if not top:
        return 0.0
    same = sum(1 for item in top if item.get(key) == query_value)
    return round(1.0 - same / len(top), 4)
