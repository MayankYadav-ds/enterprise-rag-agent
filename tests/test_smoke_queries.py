"""Smoke-test the retrieval path with a FakeEmbedder and in-memory Qdrant.

The real BAAI/bge-small-en-v1.5 model cannot be loaded on machines that block
torch, so these tests exercise the exact code path that scripts/smoke_queries.py
uses — embed each query with the same embedder that indexed the collection, call
the store's search with top_k=3, and inspect score, page range, section heading
and snippet — without downloading a model.
"""

from __future__ import annotations

import uuid

from rag_agent.chunking.models import Chunk
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.smoke import run_smoke_queries
from rag_agent.embeddings.store import QdrantStore


def _make_chunk(**overrides) -> Chunk:
    defaults = dict(
        doc_id="doc_smoke",
        text="The company's total revenue was 1.2 billion dollars in 2024.",
        chunk_type="text",
        page_start=5,
        page_end=5,
        section_heading="Financial Results",
        chunk_index=0,
    )
    defaults.update(overrides)
    return Chunk.create(**defaults)


def _store_with(
    chunks, source_file="N-able_2024_Annual_Report.pdf"
) -> tuple[QdrantStore, FakeEmbedder]:
    embedder = FakeEmbedder(dimension=16)
    store = QdrantStore(
        embedder=embedder,
        collection_name=f"smoke_{uuid.uuid4().hex}",
        url=":memory:",
    )
    if chunks:
        store.sync(chunks[0].doc_id, chunks, embedder, source_file=source_file)
    return store, embedder


class TestRunSmokeQueries:
    def test_returns_hits_with_payload_fields(self):
        chunks = [_make_chunk()]
        store, embedder = _store_with(chunks)
        results = run_smoke_queries(embedder, store, ["What was total revenue?"], top_k=3)

        assert len(results) == 1
        hit = results[0]
        assert hit["query"] == "What was total revenue?"
        assert 0.0 <= hit["score"] <= 1.0
        assert hit["page_start"] == 5
        assert hit["page_end"] == 5
        assert hit["section_heading"] == "Financial Results"
        assert hit["source_file"] == "N-able_2024_Annual_Report.pdf"
        assert hit["doc_id_prefix"] == "doc_smoke"[:8]
        assert len(hit["snippet"]) <= 100

    def test_snippet_truncated_to_100_chars(self):
        long_text = "x" * 500
        chunks = [_make_chunk(text=long_text)]
        store, embedder = _store_with(chunks)
        results = run_smoke_queries(embedder, store, ["x"], top_k=1)

        assert results[0]["snippet"] == long_text[:100]

    def test_empty_collection_returns_no_hits(self):
        store, embedder = _store_with([])
        results = run_smoke_queries(embedder, store, ["anything"], top_k=3)
        assert results == []

    def test_multiple_queries_and_top_k(self):
        chunks = [
            _make_chunk(text="Risk factors include market volatility.", chunk_index=0),
            _make_chunk(text="Royalties are earned from licensing agreements.", chunk_index=1),
            _make_chunk(text="Total revenue was 1.2 billion.", chunk_index=2),
        ]
        store, embedder = _store_with(chunks)
        results = run_smoke_queries(
            embedder,
            store,
            ["What are the main risk factors?", "How does the company earn royalties?"],
            top_k=2,
        )

        # Each query can produce up to top_k hits; both queries must appear.
        queries = {r["query"] for r in results}
        assert queries == {
            "What are the main risk factors?",
            "How does the company earn royalties?",
        }
        assert len(results) <= 4
