"""Embedding interface shared by real and test embedders."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Common interface for text embedders.

    Every embedder exposes a fixed output dimension and a model name, and
    produces one vector per input text. Queries may carry extra instructions
    (for example the BGE search prefix) that documents do not.
    """

    @property
    def dimension(self) -> int:
        """Number of floats in every vector produced by this embedder."""
        ...

    @property
    def model_name(self) -> str:
        """Human-readable identifier of the underlying model."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts, one vector per text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text, applying any query-specific prefix."""
        ...
