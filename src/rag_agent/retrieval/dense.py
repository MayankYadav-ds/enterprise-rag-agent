"""Dense-vector retrieval.

``DenseRetriever`` embeds a query with the same embedder that indexed the
collection and asks Qdrant for the nearest neighbours. It is the baseline
retriever: BM25 and hybrid fusion arrive in later phases and implement the same
``Retriever`` protocol, so the answer generator never knows which one produced
a hit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_agent.embeddings.protocol import Embedder
from rag_agent.retrieval.models import RetrievedChunk
from rag_agent.retrieval.protocol import Retriever

if TYPE_CHECKING:
    from rag_agent.embeddings.store import QdrantStore


_MIN_TOP_K = 1
_MAX_TOP_K = 100


def _validate_top_k(top_k: int) -> int:
    """Reject out-of-range ``top_k`` early so a misconfiguration is loud."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError(f"top_k must be an int, got {type(top_k).__name__}")
    if top_k < _MIN_TOP_K or top_k > _MAX_TOP_K:
        raise ValueError(f"top_k must be in [{_MIN_TOP_K}, {_MAX_TOP_K}], got {top_k}")
    return top_k


def _validate_query(query: str) -> str:
    """Reject empty/whitespace queries before they reach the embedder."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return query.strip()


def _required(payload: dict, key: str, field: str) -> object:
    """Read a required payload field, raising if it is missing or empty.

    A hit whose payload lacks a field that ``RetrievedChunk`` requires is a
    corrupt index, not a hit to rank. ``or 0`` on a page number would silently
    render ``p. 0`` in a citation; ``or ""`` on the source file would silently
    render ``[, p. 22]``. Both are failures that should surface here.
    """
    value = payload.get(key)
    if value is None or value == "":
        raise KeyError(f"hit payload is missing required field {field!r} ({key!r})")
    return value


class DenseRetriever(Retriever):
    """Retrieve hits by embedding the query and searching Qdrant.

    Args:
        embedder: The embedder used both for indexing and for query embedding.
            Queries are prefixed with the embedder's instruction prefix (the
            bge search prefix), so the query vector lives in the same space as
            the document vectors.
        store: The Qdrant collection to search.
    """

    def __init__(self, embedder: Embedder, store: QdrantStore) -> None:
        self.embedder = embedder
        self.store = store

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
        query = _validate_query(query)
        top_k = _validate_top_k(top_k)

        query_vector = self.embedder.embed_query(query)
        hits = self.store.search(
            query_vector=query_vector,
            top_k=top_k,
            doc_id=doc_id,
            chunk_type=chunk_type,
        )

        # Apply min_score before ranking, so the returned list is numbered 1..N
        # with no gaps: a filtered-out hit does not leave a hole in the ranks.
        kept: list[object] = []
        for hit in hits:
            if min_score is not None and float(hit.score) < min_score:
                continue
            kept.append(hit)

        results: list[RetrievedChunk] = []
        for rank, hit in enumerate(kept, start=1):
            payload = hit.payload or {}
            score = float(hit.score)
            results.append(
                RetrievedChunk(
                    chunk_id=str(_required(payload, "chunk_id", "chunk_id")),
                    doc_id=str(_required(payload, "doc_id", "doc_id")),
                    source_file=str(_required(payload, "source_file", "source_file")),
                    page_start=int(_required(payload, "page_start", "page_start")),
                    page_end=int(_required(payload, "page_end", "page_end")),
                    section_heading=payload.get("section_heading"),
                    subheading=payload.get("subheading"),
                    chunk_type=str(_required(payload, "chunk_type", "chunk_type")),
                    text=str(_required(payload, "text", "text")),
                    score=score,
                    rank=rank,
                    retriever="dense",
                )
            )
        return results
