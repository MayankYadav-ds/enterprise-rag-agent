"""Deterministic fake embedder for tests and offline runs."""

from __future__ import annotations

import hashlib

from rag_agent.embeddings.protocol import Embedder

_FAKE_DIMENSION = 32


class FakeEmbedder(Embedder):
    """Deterministic hash-based embedder that never downloads a model.

    Each text is mapped to a fixed-length vector of float32 values
    derived from its SHA-256 hash, so the same text always produces
    the same vector and different texts almost always produce different
    ones. ``embed_counter`` records how many texts have been embedded
    so tests can assert on call counts without a real model.
    """

    def __init__(self, dimension: int = _FAKE_DIMENSION) -> None:
        self._dimension = dimension
        self.embed_counter = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "fake"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_counter += len(texts)
        return [_hash_vector(t, self._dimension) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_counter += 1
        return _hash_vector(text, self._dimension)


def _hash_vector(text: str, dimension: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    for i in range(dimension):
        byte = digest[i % len(digest)]
        values.append(byte / 255.0)
    return values
