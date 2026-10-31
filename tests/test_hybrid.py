"""Tests for HybridRetriever (RRF fusion of dense + BM25).

The tests construct a tiny in‑memory Qdrant collection, run a ``DenseRetriever``
against it, and build a ``BM25Retriever`` from the same payloads. The fusion
logic is exercised without hitting external services.
"""

from __future__ import annotations

import uuid

from rag_agent.chunking.models import Chunk
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.store import QdrantStore
from rag_agent.retrieval.bm25 import BM25Retriever
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.hybrid import HybridRetriever


def _setup_dense_and_bm25():
    """Create a Qdrant store with three chunks and return both retrievers.

    The dense retriever uses the store directly. The BM25 retriever is built
    from the exact payload dicts that the store emitted, guaranteeing identical
    metadata.
    """
    embedder = FakeEmbedder(dimension=32)
    store = QdrantStore(
        embedder=embedder,
        collection_name=f"hybrid_test_{uuid.uuid4().hex}",
        url=":memory:",
    )
    # Two chunks with overlapping terms, one unique.
    c1 = Chunk.create(
        doc_id="doc",
        text="apple banana",
        chunk_type="text",
        page_start=1,
        page_end=1,
        chunk_index=0,
    )
    c2 = Chunk.create(
        doc_id="doc",
        text="banana cherry",
        chunk_type="text",
        page_start=2,
        page_end=2,
        chunk_index=1,
    )
    c3 = Chunk.create(
        doc_id="doc",
        text="durian",
        chunk_type="text",
        page_start=3,
        page_end=3,
        chunk_index=2,
    )
    store.sync("doc", [c1, c2, c3], embedder, source_file="file.pdf")
    # Dense retriever uses the store directly.
    dense = DenseRetriever(embedder, store)
    # Build BM25 index from the store payloads (the same dict format used
    # by QdrantStore._payload).
    points = store.search(store.embedder.embed_query(""), top_k=100)
    bm25 = BM25Retriever(index={p.payload["chunk_id"]: p.payload for p in points})
    return dense, bm25


def test_hybrid_fusion_combines_scores_and_dedupes():
    dense, bm25 = _setup_dense_and_bm25()
    hybrid = HybridRetriever(dense, bm25, rrf_k=60)
    # Query "banana" matches both c1 and c2 in BM25; the dense retriever
    # returns every chunk with a non-zero cosine score. The hybrid must
    # deduplicate by chunk_id (no duplicate IDs) and rank the BM25-matched
    # chunks above the one that BM25 did not match.
    hits = hybrid.retrieve("banana", top_k=5)
    ids = [h.chunk_id for h in hits]
    assert "chunk_doc_" in ids[0]  # sanity check that IDs are chunk IDs
    # No duplicate IDs: deduplication by chunk_id works.
    assert len(ids) == len(set(ids))
    # The BM25-matched chunks (containing "banana") must rank above the
    # unmatched one ("durian"), because their RRF contribution from the BM25
    # list is strictly positive.
    banana_hits = [h for h in hits if "banana" in h.text]
    durian_hits = [h for h in hits if "durian" in h.text]
    assert banana_hits and durian_hits
    assert all(b.score > durian_hits[0].score for b in banana_hits)
    # The top hit must have higher score than the second.
    assert hits[0].score > hits[1].score


def test_hybrid_respects_top_k_and_filters():
    dense, bm25 = _setup_dense_and_bm25()
    hybrid = HybridRetriever(dense, bm25, rrf_k=30)
    # Use doc_id filter (all chunks belong to same doc, but we check passing through)
    hits = hybrid.retrieve("apple", top_k=1, doc_id="doc", chunk_type="text")
    assert len(hits) == 1
    # The single hit should be the chunk containing "apple"
    assert "apple" in hits[0].text


def test_hybrid_empty_result_when_no_hits():
    # Use an empty collection so neither retriever returns anything.
    embedder = FakeEmbedder(dimension=32)
    store = QdrantStore(
        embedder=embedder,
        collection_name=f"hybrid_empty_{uuid.uuid4().hex}",
        url=":memory:",
    )
    dense = DenseRetriever(embedder, store)
    bm25 = BM25Retriever(index=store.scroll_payloads())
    hybrid = HybridRetriever(dense, bm25)
    hits = hybrid.retrieve("nonexistent", top_k=5)
    assert hits == []


def test_hybrid_deterministic_tie_breaking():
    # Two chunks with identical text get identical RRF scores; the tie must
    # be broken deterministically, not by insertion order. We assert that
    # two consecutive calls return the same ordering.
    embedder = FakeEmbedder(dimension=32)
    store = QdrantStore(
        embedder=embedder,
        collection_name=f"hybrid_tie_{uuid.uuid4().hex}",
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
    dense = DenseRetriever(embedder, store)
    points = store.search(store.embedder.embed_query(""), top_k=100)
    bm25 = BM25Retriever(index={p.payload["chunk_id"]: p.payload for p in points})
    hybrid = HybridRetriever(dense, bm25)
    hits1 = hybrid.retrieve("text", top_k=2)
    hits2 = hybrid.retrieve("text", top_k=2)
    assert [h.chunk_id for h in hits1] == [h.chunk_id for h in hits2]


def test_hybrid_configurable_rrf_k_changes_scores():
    dense, bm25 = _setup_dense_and_bm25()
    # With a smaller k, the RRF contribution is smaller, so the hybrid score
    # is lower than with the default k=60.
    hybrid_default = HybridRetriever(dense, bm25, rrf_k=60)
    hybrid_small = HybridRetriever(dense, bm25, rrf_k=10)
    hits_default = hybrid_default.retrieve("banana", top_k=5)
    hits_small = hybrid_small.retrieve("banana", top_k=5)
    assert hits_default[0].score > hits_small[0].score


def test_hybrid_min_score_passes_through_to_components():
    dense, bm25 = _setup_dense_and_bm25()
    hybrid = HybridRetriever(dense, bm25, rrf_k=60)
    # A very high min_score drops every hit in both components, so the hybrid
    # returns nothing. This exercises the min_score passthrough path.
    hits = hybrid.retrieve("banana", top_k=5, min_score=100.0)
    assert hits == []
