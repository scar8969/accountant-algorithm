"""Classification metrics used by the benchmark runner."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def prevalence(y: np.ndarray) -> float:
    return float(np.mean(y)) if len(y) else float("nan")


def pr_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(set(y.tolist())) < 2:
        return None
    return float(average_precision_score(y, p))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, p))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float | None:
    if len(y) == 0:
        return None
    q = np.clip(p.astype(float), eps, 1 - eps)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def brier(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) == 0:
        return None
    return float(np.mean((p.astype(float) - y) ** 2))


def prf(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[dict] = []
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        n = int(mask.sum())
        out.append({
            "lo": round(lo, 2), "hi": round(hi, 2), "n": n,
            "mean_pred": round(float(p[mask].mean()), 4) if n else None,
            "frac_pos": round(float(y[mask].mean()), 4) if n else None,
        })
    return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float | None:
    if len(y) == 0:
        return None
    total = len(y)
    err = 0.0
    for row in reliability(y, p, bins):
        if not row["n"]:
            continue
        err += row["n"] / total * abs(row["mean_pred"] - row["frac_pos"])
    return float(err)


def summarise(y: np.ndarray, p: np.ndarray | None = None, *,
              scores: np.ndarray | None = None, threshold: float = 0.5,
              bins: int = 10) -> dict:
    """Metric block. Ranking metrics are added by the caller.

    Models without calibrated probabilities are scored on discrimination only
    (PR-AUC / ROC-AUC from their ranking score); log loss, Brier and ECE are
    reported as null rather than pretending a 0-1 score is a probability.
    """
    out: dict = {"n": int(len(y)), "positives": int(np.sum(y)),
                 "prevalence": round(prevalence(y), 4),
                 "probabilistic": p is not None}
    if p is not None:
        out.update({
            "pr_auc": _round(pr_auc(y, p)),
            "roc_auc": _round(roc_auc(y, p)),
            "log_loss": _round(log_loss(y, p)),
            "brier": _round(brier(y, p)),
            "ece": _round(ece(y, p, bins)),
            "at_threshold": prf(y, p, threshold),
        })
        out["reliability"] = reliability(y, p, bins)
    elif scores is not None:
        out.update({"pr_auc": _round(pr_auc(y, scores)),
                    "roc_auc": _round(roc_auc(y, scores))})
    return out


def _round(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)
