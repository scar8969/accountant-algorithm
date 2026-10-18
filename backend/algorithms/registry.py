"""Model registry — the only place experiment configs resolve model names."""

from __future__ import annotations

from .base import TaskModel
from .baseline import RandomRanking, StructuredBaseline, TextSimilarity
from .supervised import (DirectionalLogistic, InteractionMean, InteractionRidge,
                         MutualGBM, MutualIndependence, MutualLogistic)

_BUILDERS = {
    "random": lambda seed, embed: RandomRanking(seed=seed),
    "structured": lambda seed, embed: StructuredBaseline(seed=seed),
    # accepted alias, matching the model_name used in the blueprint's API example
    "structured_baseline": lambda seed, embed: StructuredBaseline(seed=seed),
    "structured_text": lambda seed, embed: StructuredBaseline(seed=seed, embed=embed,
                                                              use_text=True),
    "text": lambda seed, embed: TextSimilarity(seed=seed, embed=embed),
    "directional_logistic": lambda seed, embed: DirectionalLogistic(seed=seed, embed=embed),
    "mutual_independence": lambda seed, embed: MutualIndependence(seed=seed, embed=embed),
    "mutual_logistic": lambda seed, embed: MutualLogistic(seed=seed, embed=embed),
    "mutual_gbm": lambda seed, embed: MutualGBM(seed=seed, embed=embed),
    "interaction_ridge": lambda seed, embed: InteractionRidge(seed=seed, embed=embed),
    "interaction_mean": lambda seed, embed: InteractionMean(seed=seed, embed=embed),
}

# Which models are benchmarked for which task by default.
DEFAULT_MODELS = {
    "directional": ["random", "structured", "text", "structured_text",
                    "directional_logistic"],
    "mutual": ["random", "structured", "text", "structured_text",
               "mutual_independence", "mutual_logistic", "mutual_gbm"],
    "interaction": ["interaction_mean", "interaction_ridge"],
}


def build_model(name: str, seed: int = 0, embed=None) -> TaskModel:
    if name not in _BUILDERS:
        raise KeyError(f"unknown model '{name}'. Known: {sorted(_BUILDERS)}")
    return _BUILDERS[name](seed, embed)


def registered_models() -> list[str]:
    return sorted(_BUILDERS)
