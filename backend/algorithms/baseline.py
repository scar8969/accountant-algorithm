"""Baselines: random, hand-weighted structured, text-only."""

from __future__ import annotations

import hashlib

from ..features.embeddings import cosine
from ..features.structured import directional_score, score_pair
from .base import TaskModel


def _unit_hash(*parts: object) -> float:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2 ** 64


class RandomRanking(TaskModel):
    """B0 — seeded uniform noise. A sanity check, not a model."""

    name = "random"
    version = "1.0.0"

    def score(self, row: dict, task: str) -> float:
        a = row["a"].get("participant_id", "")
        b = row["b"].get("participant_id", "")
        return _unit_hash(self.seed, a, b, task)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["note"] = "Uniform pseudo-random score keyed by (seed, participant ids)."
        return cfg


class StructuredBaseline(TaskModel):
    """B1 — hand-weighted, pre-registered compatibility score with hard gates.

    `use_text` adds the experimental embedding leaf (that variant is reported as
    B1+text so the ablation stays visible).
    """

    name = "structured"
    version = "1.0.0"

    def __init__(self, seed: int = 0, embed=None, use_text: bool = False) -> None:
        super().__init__(seed=seed, embed=embed)
        self.use_text = use_text
        if use_text:
            self.name = "structured_text"

    def _pair(self, row: dict):
        cache = row.setdefault("_cache", {})
        key = "ps_text" if self.use_text else "ps"
        if key not in cache:
            cache[key] = score_pair(row["a"], row["b"],
                                    embed=self.embed if self.use_text else None)
        return cache[key]

    def score(self, row: dict, task: str) -> float | None:
        ps = self._pair(row)
        if ps.score is None:
            return None
        if task == "directional":
            gated = bool(ps.gate_a)
            if gated:
                return 0.0
            return directional_score(row["a"], row["b"], ps=ps,
                                    embed=self.embed if self.use_text else None)
        return ps.score

    def probability(self, row: dict, task: str) -> float | None:
        s = self.score(row, task)
        return None if s is None else max(0.0, min(1.0, s / 100.0))

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"use_text": self.use_text,
                    "note": "Pre-registered weights from features/schema.py; "
                            "deal-breakers applied as hard gates."})
        return cfg


class TextSimilarity(TaskModel):
    """B2 — embedding-only baseline over optional self-descriptions."""

    name = "text"
    version = "1.0.0"

    def score(self, row: dict, task: str) -> float | None:
        a_text, b_text = row["a"].get("text"), row["b"].get("text")
        if not a_text or not b_text or self.embed is None:
            return None
        return max(0.0, cosine(self.embed.embed(a_text), self.embed.embed(b_text)))

    def get_config(self) -> dict:
        cfg = super().get_config()
        provider = self.embed
        cfg["provider"] = getattr(provider, "name", None)
        cfg["provider_version"] = getattr(provider, "version", None)
        cfg["dim"] = getattr(provider, "dim", None)
        cfg["note"] = ("Lexical hashing embedding: shared wording, not semantics. "
                       "Swap the provider for a semantic encoder to test H2 properly.")
        return cfg
