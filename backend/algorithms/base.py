"""Common interface for every ranker in the research platform."""

from __future__ import annotations

from typing import Any


class TaskModel:
    """A model that scores candidate pairs for one or more prediction tasks.

    Rows are dicts of the form::

        {"a": <profile dict>, "b": <profile dict>,
         "interest_a": bool|None, "interest_b": bool|None, ...}

    Conventions:
      * `score(row, task)` returns an unbounded score where higher means more
        likely to be a positive outcome. Used for ranking.
      * `probability(row, task)` returns a value in [0, 1] or None. Only models
        with `outputs_probability = True` are scored on log loss / Brier.
      * Models must never read label fields at scoring time.
    """

    name = "base"
    version = "0.1.0"
    outputs_probability = False
    uses_labels = False

    def __init__(self, seed: int = 0, embed: Any | None = None) -> None:
        self.seed = int(seed)
        if embed is None:
            from ..features.embeddings import get_default_provider
            embed = get_default_provider()
        self.embed = embed

    # -- lifecycle ---------------------------------------------------------- #
    def fit(self, rows: list[dict], task: str) -> "TaskModel":
        return self

    # -- inference ---------------------------------------------------------- #
    def score(self, row: dict, task: str) -> float | None:
        raise NotImplementedError

    def probability(self, row: dict, task: str) -> float | None:
        return None

    # -- provenance --------------------------------------------------------- #
    def get_config(self) -> dict:
        return {"name": self.name, "version": self.version, "seed": self.seed,
                "outputs_probability": self.outputs_probability,
                "class": type(self).__name__}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name}@{self.version}>"


def label_for(row: dict, task: str):
    if task == "directional":
        return row.get("interest_a")
    if task == "mutual":
        return row.get("mutual")
    if task == "interaction":
        return row.get("quality_a")
    raise ValueError(f"unknown task: {task}")
