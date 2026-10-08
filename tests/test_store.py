"""Tests for QdrantStore (in-memory Qdrant)."""

from __future__ import annotations

import uuid

import pytest

from rag_agent.chunking.models import Chunk
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.protocol import Embedder
from rag_agent.embeddings.store import PROJECT_NAMESPACE, QdrantStore, point_id


class _StubEmbedder(Embedder):
    """Embedder with a custom model_name for model-mismatch tests."""

    def __init__(self, model_name: str, dimension: int) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self._dimension


def make_chunk(**overrides) -> Chunk:
    defaults = dict(
        doc_id="doc_abc123",
        text="sample text for embedding",
        chunk_type="text",
        page_start=1,
        page_end=1,
        section_heading=None,
        chunk_index=0,
    )
    defaults.update(overrides)
    return Chunk.create(**defaults)


@pytest.fixture
def store() -> QdrantStore:
    return QdrantStore(
        embedder=FakeEmbedder(dimension=16),
        collection_name=f"test_chunks_{uuid.uuid4().hex}",
        url=":memory:",
    )


class TestPointId:
    def test_deterministic(self):
        assert point_id("chunk_doc_abc_1") == point_id("chunk_doc_abc_1")

    def test_returns_uuid5(self):
        pid = point_id("chunk_doc_abc_1")
        assert isinstance(pid, uuid.UUID)
        assert pid.version == 5
        assert pid.hex == uuid.uuid5(PROJECT_NAMESPACE, "chunk_doc_abc_1").hex

    def test_different_chunks_differ(self):
        assert point_id("chunk_a") != point_id("chunk_b")


class TestQdrantStore:
    def test_upsert_and_count(self, store):
        chunks = [make_chunk()]
        result = store.sync(chunks[0].doc_id, chunks, store.embedder)
        assert result["upserted"] == 1
        assert result["deleted"] == 0
        assert result["skipped"] == 0
        assert store.count() == 1

    def test_idempotent_second_run(self, store):
        chunks = [make_chunk()]
        store.sync(chunks[0].doc_id, chunks, store.embedder)
        result = store.sync(chunks[0].doc_id, chunks, store.embedder)
        assert result["upserted"] == 0
        assert result["skipped"] == 1
        assert store.count() == 1

    def test_stale_points_removed(self, store):
        c1 = make_chunk(text="original text")
        c2 = make_chunk(text="revised text")
        assert c1.chunk_id != c2.chunk_id
        store.sync(c1.doc_id, [c1], store.embedder)
        assert store.count() == 1
        store.sync(c2.doc_id, [c2], store.embedder)
        assert store.count() == 1

    def test_payload_contents(self, store):
        c = make_chunk(
            page_start=3,
            page_end=5,
            section_heading="Risk Factors",
            chunk_type="text",
        )
        store.sync(c.doc_id, [c], store.embedder)
        hits = store.search(store.embedder.embed_query("x"), top_k=10)
        assert len(hits) == 1
        p = hits[0].payload
        assert p["page_start"] == 3
        assert p["page_end"] == 5
        assert p["section_heading"] == "Risk Factors"
        assert p["chunk_type"] == "text"
        assert p["text"] == "sample text for embedding"
        assert p["embedding_model"] == "fake"

    def test_doc_id_filter(self, store):
        c1 = make_chunk(doc_id="doc_1")
        c2 = make_chunk(doc_id="doc_2")
        store.sync(c1.doc_id, [c1], store.embedder)
        store.sync(c2.doc_id, [c2], store.embedder)
        hits = store.search(
            store.embedder.embed_query("x"),
            top_k=10,
            doc_id="doc_1",
        )
        assert len(hits) == 1
        assert hits[0].payload["doc_id"] == "doc_1"

    def test_chunk_type_filter(self, store):
        c1 = make_chunk(chunk_type="text", text="alpha table text")
        c2 = make_chunk(chunk_type="table", text="beta table text")
        assert c1.chunk_id != c2.chunk_id
        store.sync(c1.doc_id, [c1], store.embedder)
        store.sync(c2.doc_id, [c2], store.embedder)
        hits = store.search(
            store.embedder.embed_query("x"),
            top_k=10,
            chunk_type="table",
        )
        assert len(hits) == 1
        assert hits[0].payload["chunk_type"] == "table"

    def test_sync_does_not_touch_other_documents(self, store):
        """Syncing one document must not delete another document's points."""
        c1 = make_chunk(doc_id="doc_1", text="first chunk text")
        c2 = make_chunk(doc_id="doc_2", text="second chunk text")
        store.sync(c1.doc_id, [c1], store.embedder)
        store.sync(c2.doc_id, [c2], store.embedder)
        assert store.count() == 2
        assert len(store.scroll_ids(doc_id="doc_1")) == 1
        assert len(store.scroll_ids(doc_id="doc_2")) == 1

    def test_dimension_mismatch_raises(self):
        """A collection created with one dimension rejects a different embedder."""
        name = f"dim_mismatch_{uuid.uuid4().hex}"
        QdrantStore(
            embedder=FakeEmbedder(dimension=16),
            collection_name=name,
            url=":memory:",
        )
        with pytest.raises(ValueError, match="dimension"):
            QdrantStore(
                embedder=FakeEmbedder(dimension=32),
                collection_name=name,
                url=":memory:",
            )

    def test_wrong_vector_dimension_upsert_raises(self, store):
        c = make_chunk()
        with pytest.raises(ValueError, match="dimension"):
            store.sync(c.doc_id, [c], FakeEmbedder(dimension=32))

    def test_model_mismatch_raises(self):
        """A collection built with one model rejects a different embedder."""
        name = f"model_mismatch_{uuid.uuid4().hex}"
        store = QdrantStore(
            embedder=FakeEmbedder(dimension=16),
            collection_name=name,
            url=":memory:",
        )
        store.sync("doc_1", [make_chunk()], FakeEmbedder(dimension=16))
        other = _StubEmbedder(model_name="other-model", dimension=16)
        with pytest.raises(ValueError, match="embedding_model"):
            QdrantStore(
                embedder=other,
                collection_name=name,
                url=":memory:",
            )

    def test_scroll_ids(self, store):
        c1 = make_chunk(doc_id="doc_1", text="first chunk text")
        c2 = make_chunk(doc_id="doc_2", text="second chunk text")
        store.sync(c1.doc_id, [c1], store.embedder)
        store.sync(c2.doc_id, [c2], store.embedder)
        ids = store.scroll_ids()
        assert len(ids) == 2
        assert point_id(c1.chunk_id) in ids
        assert point_id(c2.chunk_id) in ids

    def test_scroll_ids_doc_scoped(self, store):
        c1 = make_chunk(doc_id="doc_1", text="first chunk text")
        c2 = make_chunk(doc_id="doc_2", text="second chunk text")
        store.sync(c1.doc_id, [c1], store.embedder)
        store.sync(c2.doc_id, [c2], store.embedder)
        assert len(store.scroll_ids(doc_id="doc_1")) == 1
        assert len(store.scroll_ids(doc_id="doc_2")) == 1
        assert len(store.scroll_ids()) == 2
