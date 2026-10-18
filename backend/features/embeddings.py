"""Provider-independent text embeddings.

The default provider is a deterministic signed-hashing TF/IDF-style encoder.
It is a *lexical* baseline: it captures shared wording, not meaning. It exists
so the pipeline runs offline and reproducibly. Any provider implementing
`EmbeddingProvider` (an API model, a sentence-transformer, ...) can be swapped
in — bump `version` when you do, because stored experiments record it.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

import numpy as np

_WORD = re.compile(r"[a-z']+")
_STOP = {
    "the", "and", "a", "an", "to", "of", "in", "i", "im", "is", "it", "that",
    "for", "on", "with", "my", "me", "you", "am", "as", "at", "be", "are",
    "like", "love", "really", "very", "so", "but", "not", "about", "some",
    "get", "go", "going", "have", "has", "do", "does", "enjoy",
}


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    version: str
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedding:
    """Signed feature hashing over word unigrams + bigrams. Deterministic."""

    name = "hashing_tfidf"
    version = "1.0.0"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def _tokens(self, text: str) -> list[str]:
        words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
        unigrams = words
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return unigrams + bigrams

    def embed(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vec = np.zeros(self.dim, dtype=np.float64)
        counts: dict[str, int] = {}
        for tok in self._tokens(text):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, count in counts.items():
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(count))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        self._cache[text] = vec
        return vec


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


_DEFAULT = HashingEmbedding()


def get_default_provider() -> HashingEmbedding:
    return _DEFAULT
