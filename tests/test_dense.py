"""Tests for ``DenseRetriever`` against an in-memory Qdrant collection.

The real BAAI/bge-small-en-v1.5 model cannot be loaded on machines that block
torch, so every test drives ``DenseRetriever`` with a ``FakeEmbedder``. Querying
with the exact text of an indexed chunk returns that chunk at rank 1 with a
cosine score of ~1.0, which is enough to assert ordering, filtering and
thresholding without a model download.
"""

from __future__ import annotations

import uuid

import pytest

from rag_agent.chunking.models import Chunk
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.store import QdrantStore
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.models import RetrievedChunk


def _chunk(**overrides) -> Chunk:
    defaults = dict(
        doc_id="doc_retrieval",
        text="sample chunk text",
        chunk_type="text",
        page_start=1,
        page_end=1,
        chunk_index=0,
    )
    defaults.update(overrides)
    return Chunk.create(**defaults)


@pytest.fixture
def store() -> QdrantStore:
    return QdrantStore(
        embedder=FakeEmbedder(dimension=32),
        collection_name=f"retrieval_test_{uuid.uuid4().hex}",
        url=":memory:",
    )


def _index(store: QdrantStore, chunks: list[Chunk], source_file: str = "nps-2013-10k.pdf") -> None:
    """Insert chunks into the store, grouped by document id."""
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    for doc_id, doc_chunks in by_doc.items():
        store.sync(doc_id, doc_chunks, store.embedder, source_file=source_file)


class TestDenseRetriever:
    """Dense retrieval through an in-memory Qdrant collection."""

    def test_returns_ranked_hits_best_first(self, store):
        _index(
            store,
            [
                _chunk(chunk_index=0, text="the company earned one point two billion in revenue"),
                _chunk(chunk_index=1, text="risk factors include competition and regulation"),
            ],
        )
        retriever = DenseRetriever(store.embedder, store)
        hits = retriever.retrieve("What was total revenue?", top_k=2)

        assert [h.rank for h in hits] == [1, 2]
        assert hits[0].rank == 1 and hits[0].score > hits[1].score
        assert hits[0].retriever == "dense"
        assert hits[0].chunk_type == "text"
        assert hits[0].source_file == "nps-2013-10k.pdf"
        assert hits[0].doc_id == "doc_retrieval"

    def test_top_k_limits_results(self, store):
        chunks = [
            _chunk(chunk_index=i, text=f"unique chunk number {i} about cats") for i in range(5)
        ]
        _index(store, chunks)
        retriever = DenseRetriever(store.embedder, store)
        hits = retriever.retrieve("about cats", top_k=2)
        assert len(hits) == 2

    def test_doc_id_filter_restricts_hits(self, store):
        _index(
            store,
            [
                _chunk(doc_id="doc_a", chunk_index=0, text="alpha report revenue"),
                _chunk(doc_id="doc_b", chunk_index=0, text="beta report revenue"),
            ],
        )
        retriever = DenseRetriever(store.embedder, store)
        hits = retriever.retrieve("report revenue", top_k=5, doc_id="doc_a")
        assert hits and all(h.doc_id == "doc_a" for h in hits)

    def test_chunk_type_filter_restricts_hits(self, store):
        _index(
            store,
            [
                _chunk(chunk_index=0, chunk_type="text", text="prose about revenue"),
                _chunk(chunk_index=1, chunk_type="table", text="revenue table row"),
            ],
        )
        retriever = DenseRetriever(store.embedder, store)
        text_hits = retriever.retrieve("revenue", top_k=5, chunk_type="text")
        table_hits = retriever.retrieve("revenue", top_k=5, chunk_type="table")
        assert text_hits and all(h.chunk_type == "text" for h in text_hits)
        assert table_hits and all(h.chunk_type == "table" for h in table_hits)

    def test_min_score_drops_weak_hits(self, store):
        _index(
            store,
            [
                _chunk(chunk_index=0, text="the company earned one point two billion in revenue"),
                _chunk(chunk_index=1, text="unrelated text about gardening and soil"),
            ],
        )
        retriever = DenseRetriever(store.embedder, store)
        # Querying "total revenue" scores the revenue chunk ~0.83 and the
        # unrelated one ~0.72, so a threshold of 0.75 keeps only the strong hit.
        hits = retriever.retrieve("total revenue", top_k=5, min_score=0.75)
        assert [h.text for h in hits] == ["the company earned one point two billion in revenue"]
        assert all(h.score >= 0.75 for h in hits)

    def test_min_score_none_keeps_everything(self, store):
        _index(store, [_chunk(chunk_index=0, text="alpha"), _chunk(chunk_index=1, text="beta")])
        retriever = DenseRetriever(store.embedder, store)
        hits = retriever.retrieve("alpha", top_k=5)
        assert len(hits) == 2

    def test_empty_collection_returns_nothing(self, store):
        retriever = DenseRetriever(store.embedder, store)
        assert retriever.retrieve("anything", top_k=5) == []

    def test_top_k_zero_raises(self, store):
        retriever = DenseRetriever(store.embedder, store)
        with pytest.raises(ValueError):
            retriever.retrieve("q", top_k=0)

    def test_top_k_over_max_raises(self, store):
        retriever = DenseRetriever(store.embedder, store)
        with pytest.raises(ValueError):
            retriever.retrieve("q", top_k=101)

    def test_top_k_non_int_raises(self, store):
        retriever = DenseRetriever(store.embedder, store)
        with pytest.raises(TypeError):
            retriever.retrieve("q", top_k="five")  # type: ignore[arg-type]

    def test_top_k_bool_rejected(self, store):
        retriever = DenseRetriever(store.embedder, store)
        with pytest.raises(TypeError):
            retriever.retrieve("q", top_k=True)  # type: ignore[arg-type]

    def test_empty_query_raises(self, store):
        retriever = DenseRetriever(store.embedder, store)
        with pytest.raises(ValueError):
            retriever.retrieve("   ", top_k=5)

    def test_min_score_keyword_only(self, store):
        # min_score is keyword-only: passing it positionally would land on
        # chunk_type and silently filter by chunk type instead of thresholding.
        _index(store, [_chunk(chunk_index=0, text="alpha revenue")])
        retriever = DenseRetriever(store.embedder, store)
        with pytest.raises(TypeError):
            retriever.retrieve("alpha", 5, None, None, 0.9)  # type: ignore[misc]

    def test_returns_retrieved_chunk_instances(self, store):
        _index(store, [_chunk(chunk_index=0, text="alpha")])
        retriever = DenseRetriever(store.embedder, store)
        hits = retriever.retrieve("alpha", top_k=1)
        assert hits and isinstance(hits[0], RetrievedChunk)

    def test_ranks_are_contiguous_from_one(self, store):
        _index(
            store,
            [_chunk(chunk_index=i, text=f"chunk about topic {i}") for i in range(4)],
        )
        retriever = DenseRetriever(store.embedder, store)
        hits = retriever.retrieve("chunk about topic", top_k=4)
        assert [h.rank for h in hits] == [1, 2, 3, 4]

    def test_filtering_does_not_leave_rank_holes(self, store):
        # min_score is applied BEFORE ranking, so a dropped hit does not leave
        # a gap in the returned ranks.
        _index(
            store,
            [
                _chunk(chunk_index=0, text="the company earned one point two billion in revenue"),
                _chunk(chunk_index=1, text="unrelated text about gardening and soil"),
                _chunk(chunk_index=2, text="risk factors include competition and regulation"),
            ],
        )
        retriever = DenseRetriever(store.embedder, store)
        hits = retriever.retrieve("total revenue", top_k=5, min_score=0.9)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
