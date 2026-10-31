"""Unit tests for BM25Retriever.

The BM25 implementation is pure‑Python and operates over the exact payload
structure that the Qdrant store emits. All tests use an in‑memory Qdrant
collection with a :class:`FakeEmbedder` so no external resources are
required.
"""

from __future__ import annotations

import uuid

import pytest

from rag_agent.chunking.models import Chunk
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.store import QdrantStore
from rag_agent.retrieval.bm25 import BM25Retriever


def _setup_store_and_bm25() -> tuple[QdrantStore, list[Chunk], BM25Retriever]:
    """Create a sync-collected Qdrant collection and return the BM25 retriever.

    The store is local (``:memory:``) so no external daemon is required. The
    embedder used is deterministic; the returned BM25Retriever is built from
    the collection's payload dictionary.
    """
    embedder = FakeEmbedder(dimension=32)
    store = QdrantStore(
        embedder=embedder,
        collection_name=f"bm25_test_{uuid.uuid4().hex}",
        url=":memory:",
    )
    # Build a handful of chunks spanning two documents so we can test
    # filtering. We'll keep IDs deterministic by manually building them.
    c1 = Chunk.create(
        doc_id="doc_a",
        text="red apple",
        chunk_type="text",
        page_start=1,
        page_end=1,
        chunk_index=0,
    )
    c2 = Chunk.create(
        doc_id="doc_a",
        text="blue berry",
        chunk_type="text",
        page_start=1,
        page_end=1,
        chunk_index=1,
    )
    c3 = Chunk.create(
        doc_id="doc_b",
        text="lemon orange",
        chunk_type="text",
        page_start=2,
        page_end=2,
        chunk_index=0,
    )
    chunks = [c1, c2, c3]
    store.sync("doc_a", [c1, c2], store.embedder, source_file="x.pdf")
    store.sync("doc_b", [c3], store.embedder, source_file="y.pdf")
    # Build the payload index for BM25 using the same dict format the store
    # emits (QdrantStore._payload), so the BM25 retriever sees the exact
    # payload structure it will encounter in production.
    index = {
        c.chunk_id: QdrantStore._payload(c, store.embedder.model_name, "x.pdf") for c in [c1, c2]
    }
    index.update(
        {c.chunk_id: QdrantStore._payload(c, store.embedder.model_name, "y.pdf") for c in [c3]}
    )
    retriever = BM25Retriever(index=index)
    return store, chunks, retriever


def test_bm25_ranking_returns_best_first():
    _, _, bm25 = _setup_store_and_bm25()
    hits = bm25.retrieve("red apple", top_k=3)
    # Text has unique match only for 'red apple' -> should return that first
    assert hits[0].chunk_id.startswith("chunk_doc_a")
    assert hits[0].score > 0
    # Duplicates should not be present
    ids = [h.chunk_id for h in hits]
    assert len(ids) == len(set(ids))


def test_bm25_uses_logarithmic_idf():
    # Three equal-length documents; apple occurs once in exactly one document.
    # TF normalization is 1, so the score is ln(1 + 2.5 / 1.5).
    _, _, bm25 = _setup_store_and_bm25()
    assert bm25.retrieve("apple")[0].score == pytest.approx(0.9808292530117263)


def test_bm25_min_score_filters_weak_hits():
    _, _, bm25 = _setup_store_and_bm25()
    # Query "apple" matches only the first chunk; "berry" matches only the
    # second, so a multi-term query "apple berry" matches two chunks with
    # different scores.
    hits_all = bm25.retrieve("apple berry", top_k=3)
    assert len(hits_all) == 2
    # A threshold above the weaker score drops it.
    hits_filtered = bm25.retrieve("apple berry", top_k=3, min_score=2.0)
    # With a threshold higher than both scores, the list should be empty.
    assert hits_filtered == []


def test_bm25_doc_id_filter():
    _, _, bm25 = _setup_store_and_bm25()
    hits = bm25.retrieve("orange", top_k=5, doc_id="doc_b")
    # Only doc_b hit is relevant
    assert all(h.doc_id == "doc_b" for h in hits)


def test_bm25_chunk_type_filter():
    _, _, bm25 = _setup_store_and_bm25()
    # All chunks are type 'text', but let's simulate type filter
    hits = bm25.retrieve("apple", top_k=5, chunk_type="text")
    assert len(hits) == 1


def test_bm25_no_hits_returned():
    _, _, bm25 = _setup_store_and_bm25()
    hits = bm25.retrieve("unexistent", top_k=3)
    assert hits == []


def test_bm25_tie_breaking_determinism():
    # Build two identical chunks matching the query exactly but from
    # two different docs; after tokenisation both have identical scores.
    embedder = FakeEmbedder(dimension=32)
    store = QdrantStore(
        embedder=embedder,
        collection_name=f"bm25_tie_{uuid.uuid4().hex}",
        url=":memory:",
    )
    c1 = Chunk.create(
        doc_id="a",
        text="same text",
        chunk_type="text",
        page_start=1,
        page_end=1,
        chunk_index=0,
    )
    c2 = Chunk.create(
        doc_id="b",
        text="same text",
        chunk_type="text",
        page_start=1,
        page_end=1,
        chunk_index=0,
    )
    store.sync("a", [c1], embedder, source_file="a.pdf")
    store.sync("b", [c2], embedder, source_file="b.pdf")
    points = store.search(store.embedder.embed_query(""), top_k=100)
    bm25 = BM25Retriever(index={p.payload["chunk_id"]: p.payload for p in points})
    hits = bm25.retrieve("text", top_k=2)
    # Ensure deterministic order: sorted by score desc then chunk_id
    ids = [h.chunk_id for h in hits]
    assert ids == sorted(ids)
