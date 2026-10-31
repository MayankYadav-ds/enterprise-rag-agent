"""Retrieval protocol.

The retriever contract: embed a query, search the vector store, and return
ranked ``RetrievedChunk`` hits. BM25 and any future retriever implement the
same interface so the answer generator and citation layer are retriever-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_agent.retrieval.models import RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """Interface every retriever implements.

    Args:
        query: Natural-language query text. Must be non-empty.
        top_k: Maximum number of hits to return. Must be in 1..100.
        doc_id: Restrict hits to one document, or ``None`` for all documents.
        chunk_type: Restrict hits to one chunk type (``"text"``/``"table"``),
            or ``None`` for both.
        min_score: Drop hits below this score, or ``None`` to keep every hit
            the store returns. No default threshold is chosen yet: a value
            here is an explicit caller decision, not a silent filter.

    Returns:
        Hits ranked 1..N, best first. Empty when the collection has no hits.
    """

    def retrieve(
        self,
        query: str,
        /,
        *,
        top_k: int = 5,
        doc_id: str | None = None,
        chunk_type: str | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return ranked hits for ``query``.

        ``query`` is positional-only so a caller cannot pass it by name and
        accidentally match a filter argument; the filters and ``min_score``
        are keyword-only because they are easy to confuse positionally —
        ``min_score`` in the ``chunk_type`` slot would silently filter by chunk
        type instead of thresholding the score.
        """
        ...
