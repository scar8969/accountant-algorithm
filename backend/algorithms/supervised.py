"""Learned models (B3/B4) on top of the structured feature vector.

Nothing here reads post-interaction feedback as an input feature: the feature
builder only touches profile fields and the pre-computed compatibility score.
tests/leakage/test_splits.py asserts this.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.structured import preference_mean, score_pair
from .base import TaskModel

# Trait differences were dropped after an ablation: they are linear
# re-encodings of compat_score/dim_* and only added variance at this sample size.
BASE_FEATURES = (
    ["compat_score", "coverage", "goal_sim", "text_sim",
     "dim_values", "dim_communication", "dim_lifestyle", "dim_goals",
     "pref_a_to_b", "pref_b_to_a", "confidence_a", "confidence_b",
     "gate_a", "gate_b"]
)


def row_features(row: dict, embed=None) -> list[float]:
    """Pre-outcome features for one (a, b) row. NaN marks missing information."""
    a, b = row["a"], row["b"]
    ps = score_pair(a, b)  # text leaf intentionally excluded from B3 features
    leaves = ps.leaves
    pref_ab = preference_mean(a, b)
    pref_ba = preference_mean(b, a)

    dims = [ps.dimensions.get(d, np.nan) for d in
            ("values", "communication", "lifestyle", "relationship_goals")]

    text_sim = np.nan
    if embed is not None and a.get("text") and b.get("text"):
        from ..features.embeddings import cosine
        text_sim = max(0.0, cosine(embed.embed(a["text"]), embed.embed(b["text"])))

    return [
        np.nan if ps.score is None else ps.score / 100.0,
        ps.coverage,
        np.nan if leaves["goal"]["s"] is None else leaves["goal"]["s"],
        text_sim,
        *dims,
        np.nan if pref_ab is None else pref_ab,
        np.nan if pref_ba is None else pref_ba,
        ps.confidence_a,
        ps.confidence_b,
        1.0 if ps.gate_a else 0.0,
        1.0 if ps.gate_b else 0.0,
    ]


def _fit_matrix(rows: list[dict], task: str, embed=None):
    X = np.array([row_features(r, embed) for r in rows], dtype=float)
    return X


def _unique_pairs(rows: list[dict]) -> list[dict]:
    """One row per unordered pair, keeping the first orientation.

    Pair-level labels (mutual interest, conversation quality) must not be counted
    twice just because both directions of a pair are present in the training set.
    """
    seen: set[frozenset] = set()
    out: list[dict] = []
    for row in rows:
        key = frozenset((row["a"].get("participant_id"), row["b"].get("participant_id")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


class _SklearnClassifier(TaskModel):
    outputs_probability = True
    uses_labels = True
    extra_features = 0

    def __init__(self, seed: int = 0, embed=None, C: float = 0.1) -> None:
        super().__init__(seed=seed, embed=embed)
        self.C = C
        self.model = None
        self.constant_prob: float | None = None
        self.n_train = 0
        self.feature_names = list(BASE_FEATURES)

    # subclass hooks -------------------------------------------------------
    def _make_estimator(self):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, max_iter=2000,
                                       random_state=self.seed)),
        ])

    def _matrix(self, rows: list[dict]) -> np.ndarray:
        return _fit_matrix(rows, "directional", self.embed)

    def _labels(self, rows: list[dict]) -> np.ndarray:
        return np.array([1.0 if r.get("interest_a") else 0.0 for r in rows])

    # --------------------------------------------------------------------- #
    def fit(self, rows: list[dict], task: str = "directional") -> "_SklearnClassifier":
        rows = [r for r in rows if r.get("interest_a") is not None]
        self.n_train = len(rows)
        y = self._labels(rows)
        if len(rows) == 0 or y.min() == y.max():
            self.constant_prob = float(y.mean()) if len(y) else 0.5
            self.model = None
            return self
        X = self._matrix(rows)
        self.model = self._make_estimator()
        self.model.fit(X, y)
        self.constant_prob = None
        return self

    def probability(self, row: dict, task: str = "directional") -> float | None:
        if self.constant_prob is not None:
            return self.constant_prob
        if self.model is None:
            return None
        X = self._matrix([row])
        return float(self.model.predict_proba(X)[0, 1])

    def score(self, row: dict, task: str = "directional") -> float | None:
        return self.probability(row, task)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"C": self.C, "n_train": self.n_train,
                    "features": self.feature_names,
                    "estimator": "logistic_regression"})
        return cfg


class DirectionalLogistic(_SklearnClassifier):
    """B3a — P(A is interested in B) from profile features."""

    name = "directional_logistic"
    version = "1.0.0"


class MutualLogistic(_SklearnClassifier):
    """B4 — direct P(mutual interest), using both directional predictions.

    The independence baseline (p_AB * p_BA) is a *separate* model; this one can
    learn the dependence between the two directions.
    """

    name = "mutual_logistic"
    version = "1.0.0"

    def __init__(self, seed: int = 0, embed=None, C: float = 0.1) -> None:
        super().__init__(seed=seed, embed=embed, C=C)
        self.inner = DirectionalLogistic(seed=seed, embed=embed, C=C)
        self.feature_names = list(BASE_FEATURES) + ["p_a_to_b", "p_b_to_a"]

    def _matrix(self, rows: list[dict]) -> np.ndarray:
        base = _fit_matrix(rows, "mutual", self.embed)
        extra = np.array([[
            self.inner.probability({"a": r["a"], "b": r["b"]}) or 0.0,
            self.inner.probability({"a": r["b"], "b": r["a"]}) or 0.0,
        ] for r in rows], dtype=float)
        return np.hstack([base, extra])

    def _labels(self, rows: list[dict]) -> np.ndarray:
        return np.array([1.0 if r.get("mutual") else 0.0 for r in rows])

    def fit(self, rows: list[dict], task: str = "mutual") -> "MutualLogistic":
        directional_rows = [r for r in rows if r.get("interest_a") is not None]
        self.inner.fit(directional_rows, "directional")
        labelled = _unique_pairs([r for r in rows if r.get("mutual") is not None])
        self.n_train = len(labelled)
        y = self._labels(labelled)
        if len(labelled) == 0 or y.min() == y.max():
            self.constant_prob = float(y.mean()) if len(y) else 0.5
            self.model = None
            return self
        self.model = self._make_estimator()
        self.model.fit(self._matrix(labelled), y)
        self.constant_prob = None
        return self

    def probability(self, row: dict, task: str = "mutual") -> float | None:
        if self.constant_prob is not None:
            return self.constant_prob
        if self.model is None:
            return None
        return float(self.model.predict_proba(self._matrix([row]))[0, 1])

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["inner_model"] = self.inner.get_config()
        return cfg


class MutualIndependence(TaskModel):
    """B3b — p_mutual = p_AB * p_BA. A baseline, not an assumption of truth."""

    name = "mutual_independence"
    version = "1.0.0"
    outputs_probability = True
    uses_labels = True

    def __init__(self, seed: int = 0, embed=None, C: float = 0.1) -> None:
        super().__init__(seed=seed, embed=embed)
        self.inner = DirectionalLogistic(seed=seed, embed=embed, C=C)

    def fit(self, rows: list[dict], task: str = "mutual") -> "MutualIndependence":
        self.inner.fit([r for r in rows if r.get("interest_a") is not None],
                       "directional")
        return self

    def probability(self, row: dict, task: str = "mutual") -> float:
        p_ab = self.inner.probability({"a": row["a"], "b": row["b"]}) or 0.0
        p_ba = self.inner.probability({"a": row["b"], "b": row["a"]}) or 0.0
        return float(p_ab * p_ba)

    def score(self, row: dict, task: str = "mutual") -> float:
        return self.probability(row, task)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["inner_model"] = self.inner.get_config()
        return cfg


class MutualGBM(MutualLogistic):
    """B4b — gradient-boosted pairwise model; tests nonlinear structure."""

    name = "mutual_gbm"
    version = "1.0.0"

    def _make_estimator(self):
        return HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.05, max_leaf_nodes=8,
            min_samples_leaf=20, l2_regularization=5.0, random_state=self.seed)


class InteractionMean(TaskModel):
    """Task C baseline — predict the training mean rating."""

    name = "interaction_mean"
    version = "1.0.0"
    uses_labels = True

    def __init__(self, seed: int = 0, embed=None) -> None:
        super().__init__(seed=seed, embed=embed)
        self.mean: float | None = None

    def fit(self, rows: list[dict], task: str = "interaction") -> "InteractionMean":
        vals = [float(r["quality_a"]) for r in rows if r.get("quality_a") is not None]
        self.mean = float(np.mean(vals)) if vals else None
        return self

    def score(self, row: dict, task: str = "interaction") -> float | None:
        return self.mean


class InteractionRidge(TaskModel):
    """Task C — conversation-quality regression on pre-interaction features."""

    name = "interaction_ridge"
    version = "1.0.0"
    uses_labels = True

    def __init__(self, seed: int = 0, embed=None) -> None:
        super().__init__(seed=seed, embed=embed)
        self.model = None
        self.mean = None
        self.feature_names = list(BASE_FEATURES)

    def fit(self, rows: list[dict], task: str = "interaction") -> "InteractionRidge":
        rows = _unique_pairs([r for r in rows if r.get("quality_a") is not None])
        y = np.array([float(r["quality_a"]) for r in rows])
        self.mean = float(y.mean()) if len(y) else None
        if len(rows) >= 20:
            self.model = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("reg", HistGradientBoostingRegressor(max_iter=200, random_state=self.seed)),
            ])
            self.model.fit(_fit_matrix(rows, task, self.embed), y)
        return self

    def predict(self, row: dict) -> float | None:
        if self.model is None:
            return self.mean
        return float(self.model.predict(_fit_matrix([row], "interaction", self.embed))[0])

    def score(self, row: dict, task: str = "interaction") -> float | None:
        return self.predict(row)

    def probability(self, row: dict, task: str = "interaction") -> float | None:
        return None
