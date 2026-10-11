"""Smoke-test helpers for the retrieval path.

The real BAAI/bge-small-en-v1.5 model cannot be loaded on machines that block
torch (e.g. Windows with an Application Control policy, WinError 4551), so the
shared logic lives here in the package and is exercised by unit tests with a
FakeEmbedder and an in-memory Qdrant collection. scripts/smoke_queries.py is a
thin CLI wrapper around this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_agent.embeddings.protocol import Embedder
    from rag_agent.embeddings.store import QdrantStore


def run_smoke_queries(
    embedder: Embedder,
    store: QdrantStore,
    queries: list[str],
    top_k: int = 3,
) -> list[dict]:
    """Embed each query with the same embedder that indexed the collection.

    Calls the store's search with top_k and returns one dict per hit with the
    score, page range, section heading and a 100-character snippet of the
    stored text. The snippet is truncated in place so callers never receive
    more than 100 characters of payload text.
    """
    results: list[dict] = []
    for query in queries:
        query_vector = embedder.embed_query(query)
        hits = store.search(query_vector, top_k=top_k)
        for hit in hits:
            payload = hit.payload or {}
            snippet = (payload.get("text") or "").replace("\n", " ")[:100]
            results.append(
                {
                    "query": query,
                    "score": hit.score,
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                    "section_heading": payload.get("section_heading"),
                    "snippet": snippet,
                }
            )
    return results
