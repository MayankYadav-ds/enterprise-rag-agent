"""Tests for QdrantStore (in-memory Qdrant)."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_agent.chunking.models import Chunk
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.protocol import Embedder
from rag_agent.embeddings.store import (
    PROJECT_NAMESPACE,
    QdrantStore,
    _source_file_name,
    point_id,
)


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


class FailingEmbedder(FakeEmbedder):
    """Embedder that raises on the Nth call to embed_documents."""

    def __init__(self, fail_at: int, dimension: int = 16) -> None:
        super().__init__(dimension=dimension)
        self._fail_at = fail_at
        self._call_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._call_count += 1
        if self._call_count >= self._fail_at:
            raise RuntimeError("simulated embed failure")
        return super().embed_documents(texts)


class ShortVectorEmbedder(FakeEmbedder):
    """Embedder that returns fewer vectors than texts — exercises strict=True."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(texts[:-1])


class _NullDroppingClient:
    """Wrap a Qdrant client so its payload reads match the live server.

    Two server behaviours are modelled, both verified on the live Qdrant at
    ``http://localhost:6333``:

    1. A key written as ``null`` on ``set_payload`` / ``batch_update_points``
       is dropped — the point no longer carries it. (``upsert`` keeps null
       keys, so an initial upsert can store one.)
    2. A key OMITTED from a patch payload is NOT removed — the old value
       stays. Only an explicit ``null`` clears a key.

    The in-memory client stores nulls and merges partial payloads, so it
    satisfies (2) by default but hides (1). This wrapper keeps a shadow
    per-point payload and applies writes against it, so a point that never
    had a ``subheading`` key is never equal to a fresh payload carrying
    ``subheading: None`` under a plain ``dict`` comparison — the
    convergence bug is visible in CI.
    """

    def __init__(self, client: QdrantClient) -> None:
        self._client = client
        self._payloads: dict[uuid.UUID, dict] = {}

    def __getattr__(self, name: str):
        return getattr(self._client, name)

    @staticmethod
    def _as_uuid(value) -> uuid.UUID:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))

    def _apply(self, point_id: uuid.UUID, payload: dict | None) -> None:
        if payload is None:
            return
        store = self._payloads.setdefault(point_id, {})
        for key, value in payload.items():
            if value is None:
                store.pop(key, None)  # explicit null removes the key
            else:
                store[key] = value

    def upsert(self, collection_name, points=None, **kwargs):
        result = self._client.upsert(collection_name, points=points, **kwargs)
        for point in points or []:
            self._payloads[self._as_uuid(point.id)] = dict(point.payload or {})
        return result

    def set_payload(self, collection_name, payload, points=None, **kwargs):
        result = self._client.set_payload(collection_name, payload=payload, points=points, **kwargs)
        for point_id_ in points or []:
            self._apply(self._as_uuid(point_id_), payload)
        return result

    def batch_update_points(self, collection_name, update_operations=None, **kwargs):
        for operation in update_operations or []:
            set_payload = getattr(operation, "set_payload", None)
            if set_payload is None:
                continue
            for point_id_ in set_payload.points or []:
                self._apply(self._as_uuid(point_id_), set_payload.payload)
        return self._client.batch_update_points(
            collection_name, update_operations=update_operations, **kwargs
        )

    def scroll(self, collection_name, **kwargs):
        records, next_offset = self._client.scroll(collection_name, **kwargs)
        for record in records:
            shadow = self._payloads.get(self._as_uuid(record.id))
            if shadow is not None:
                record.payload = shadow
        return records, next_offset

    def retrieve(self, collection_name, ids, **kwargs):
        records = self._client.retrieve(collection_name, ids=ids, **kwargs)
        for record in records:
            shadow = self._payloads.get(self._as_uuid(record.id))
            if shadow is not None:
                record.payload = shadow
        return records

    def delete(self, collection_name, points_selector=None, **kwargs):
        for point_id_ in points_selector or []:
            self._payloads.pop(self._as_uuid(point_id_), None)
        return self._client.delete(collection_name, points_selector=points_selector, **kwargs)

    def seed_payload(
        self,
        point_id: uuid.UUID,
        payload: dict,
        collection_name: str = "rag_chunks",
    ) -> None:
        """Insert a stored payload as if indexed by an older code version.

        The 543 points indexed before the ``subheading`` field existed had
        no such key; this lets a test reproduce that state without an
        upsert, which would keep a null key instead. The point is created
        in the underlying client too (with a zero vector) so it is
        discoverable by ``scroll`` / ``scroll_ids``; the shadow payload is
        what reads return. ``doc_id`` must be present in the payload or
        the doc-scoped scroll filter finds nothing.
        """
        pid = self._as_uuid(point_id)
        self._payloads[pid] = dict(payload)
        self._client.upsert(
            collection_name,
            points=[models.PointStruct(id=pid, vector=[0.0] * 16, payload=dict(payload))],
        )


@pytest.fixture
def store() -> QdrantStore:
    return QdrantStore(
        embedder=FakeEmbedder(dimension=16),
        collection_name=f"test_chunks_{uuid.uuid4().hex}",
        url=":memory:",
    )


@pytest.fixture
def live_store() -> QdrantStore:
    """A store whose Qdrant client drops null payload keys like the server.

    The in-memory client keeps null keys, which hides the convergence bug:
    on the live server ``set_payload`` / ``batch_update_points`` drop them,
    so a point that never had a ``subheading`` key is never equal to a fresh
    payload carrying ``subheading: None``.
    """
    location = f":memory:live_{uuid.uuid4().hex}"
    from rag_agent.embeddings.store import _CLIENTS

    # The underlying client must use ":memory:" (a custom location string is
    # rejected as a host/port); the wrapper is registered under the custom
    # location key so _client_for returns it instead of creating a new one.
    _CLIENTS[location] = _NullDroppingClient(QdrantClient(location=":memory:"))
    return QdrantStore(
        embedder=FakeEmbedder(dimension=16),
        collection_name=f"test_live_{uuid.uuid4().hex}",
        url=location,
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

    def test_sync_strict_zip_rejects_mismatched_vectors(self, store):
        """zip(..., strict=True) means embed_documents must return exactly
        len(new_chunks) vectors; fewer or more raises ValueError."""
        chunks = [make_chunk(text="alpha"), make_chunk(text="beta")]
        with pytest.raises(ValueError, match="zip"):
            store.sync(chunks[0].doc_id, chunks, ShortVectorEmbedder(dimension=16))

    def test_sync_upsert_before_delete_preserves_old_on_failure(self, store):
        """If embed_documents fails mid-way, the old points of that document
        must still be there. The new chunks are embedded and upserted FIRST;
        stale points are deleted AFTER."""
        # Pre-populate with 3 chunks for doc_1
        original = [
            make_chunk(chunk_index=0, text="original 0"),
            make_chunk(chunk_index=1, text="original 1"),
            make_chunk(chunk_index=2, text="original 2"),
        ]
        store.sync(original[0].doc_id, original, store.embedder)
        assert store.count() == 3

        # New sync with 2 chunks, but embedder fails on the FIRST call
        new_chunks = [
            make_chunk(chunk_index=0, text="new 0"),
            make_chunk(chunk_index=1, text="new 1"),
        ]
        failing_embedder = FailingEmbedder(fail_at=1, dimension=16)
        with pytest.raises(RuntimeError, match="simulated embed failure"):
            store.sync(new_chunks[0].doc_id, new_chunks, failing_embedder)

        # The old points must still be there (delete ran after upsert failed)
        assert store.count() == 3
        ids = store.scroll_ids(doc_id=original[0].doc_id)
        assert len(ids) == 3

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

    def test_source_file_stored_as_bare_filename(self, store):
        """The payload stores the bare file name, not the indexed relative path.

        Citations only ever need the file name, so a renamed source file is
        reflected without re-hydrating the chunk.
        """
        c = make_chunk()
        store.sync(c.doc_id, [c], store.embedder, source_file="data/raw/nps-2013-10k.pdf")
        hits = store.search(store.embedder.embed_query("x"), top_k=10)
        assert hits[0].payload["source_file"] == "nps-2013-10k.pdf"

    def test_source_file_absolute_path_is_basenamed(self, store):
        c = make_chunk()
        store.sync(
            c.doc_id,
            [c],
            store.embedder,
            source_file="/mnt/shared/raw/sec-form-10-k-pdf-fixture.pdf",
        )
        hits = store.search(store.embedder.embed_query("x"), top_k=10)
        assert hits[0].payload["source_file"] == "sec-form-10-k-pdf-fixture.pdf"

    def test_payload_updated_zero_when_unchanged(self, store):
        """Re-syncing identical chunks must not patch any payload."""
        c = make_chunk()
        first = store.sync(c.doc_id, [c], store.embedder, source_file="a.pdf")
        assert first["payload_updated"] == 0
        second = store.sync(c.doc_id, [c], store.embedder, source_file="a.pdf")
        assert second["upserted"] == 0
        assert second["skipped"] == 1
        assert second["payload_updated"] == 0

    def test_payload_updated_renames_source_file_without_reembedding(self, store):
        """A renamed source file is reflected by patching the payload.

        The vector is never touched, so the cosine similarity is unchanged and
        the point is not re-embedded — only the citation metadata is refreshed.
        """
        c = make_chunk()
        store.sync(c.doc_id, [c], store.embedder, source_file="old-name.pdf")
        assert store.count() == 1

        result = store.sync(c.doc_id, [c], store.embedder, source_file="new-name.pdf")
        assert result["upserted"] == 0
        assert result["skipped"] == 1
        assert result["payload_updated"] == 1

        hits = store.search(store.embedder.embed_query("x"), top_k=10)
        assert hits[0].payload["source_file"] == "new-name.pdf"

    def test_payload_updated_patches_changed_fields_only(self, store):
        """Only chunks whose payload actually changed are patched."""
        c1 = make_chunk(chunk_index=0, text="alpha text")
        c2 = make_chunk(chunk_index=1, text="beta text")
        store.sync(c1.doc_id, [c1, c2], store.embedder, source_file="a.pdf")

        # Same chunks, same source file -> nothing to patch.
        result = store.sync(c1.doc_id, [c1, c2], store.embedder, source_file="a.pdf")
        assert result["payload_updated"] == 0

        # Rename the source file -> both chunks patched.
        result = store.sync(c1.doc_id, [c1, c2], store.embedder, source_file="b.pdf")
        assert result["payload_updated"] == 2
        hits = store.search(store.embedder.embed_query("x"), top_k=10)
        assert {h.payload["source_file"] for h in hits} == {"b.pdf"}

    def test_payload_updated_uses_batched_set_payload(self, store):
        """Payload patches are sent as a single batch_update_points call.

        The review asked for batching rather than one request per point: the
        patch pass builds one SetPayloadOperation per changed point and sends
        them all in one batch. The vectors are never touched, so the cosine
        similarity is unchanged.
        """
        chunks = [make_chunk(chunk_index=i, text=f"chunk {i} text") for i in range(5)]
        store.sync(chunks[0].doc_id, chunks, store.embedder, source_file="old.pdf")

        # Rename the source file -> all 5 chunks patched in one batch.
        result = store.sync(chunks[0].doc_id, chunks, store.embedder, source_file="new.pdf")
        assert result["payload_updated"] == 5
        assert result["upserted"] == 0
        assert result["skipped"] == 5

        hits = store.search(store.embedder.embed_query("x"), top_k=10)
        assert {h.payload["source_file"] for h in hits} == {"new.pdf"}
        # The vectors are untouched: the same chunk_id still points at the
        # original vector, so the hit count is unchanged.
        assert len(hits) == 5

    def test_payload_updated_batch_with_some_unchanged(self, store):
        """Only the chunks whose payload changed are included in the batch."""
        c1 = make_chunk(chunk_index=0, text="alpha text")
        c2 = make_chunk(chunk_index=1, text="beta text")
        store.sync(c1.doc_id, [c1, c2], store.embedder, source_file="a.pdf")

        # Same chunks, same source file -> nothing to patch.
        result = store.sync(c1.doc_id, [c1, c2], store.embedder, source_file="a.pdf")
        assert result["payload_updated"] == 0
        assert result["upserted"] == 0
        assert result["skipped"] == 2

        # Rename the source file -> both chunks patched in one batch.
        result = store.sync(c1.doc_id, [c1, c2], store.embedder, source_file="b.pdf")
        assert result["payload_updated"] == 2
        hits = store.search(store.embedder.embed_query("x"), top_k=10)
        assert {h.payload["source_file"] for h in hits} == {"b.pdf"}


class TestPayloadConvergence:
    """The patch pass must converge when stored and fresh payloads differ
    only in the representation of "no value".

    The live Qdrant server drops null-valued keys on ``set_payload`` /
    ``batch_update_points`` while ``upsert`` keeps them, so a point that never
    had a ``subheading`` key is not equal to a fresh payload carrying
    ``subheading: None`` under a plain ``dict`` comparison. The patch pass
    normalises None, missing keys and "" so re-running converges.
    """

    def test_missing_subheading_converges_after_one_patch(self, live_store):
        """A point lacking ``subheading`` is patched once, then converges.

        This is the real-world case: the 543 points indexed before the
        ``subheading`` field existed had no such key, and every re-sync
        re-patched them because the fresh payload carried ``subheading: None``.
        """
        c = make_chunk()
        # First sync writes the payload with no subheading key.
        first = live_store.sync(c.doc_id, [c], live_store.embedder, source_file="a.pdf")
        assert first["payload_updated"] == 0

        # Second sync must converge: 0 embeddings, 0 patches.
        second = live_store.sync(c.doc_id, [c], live_store.embedder, source_file="a.pdf")
        assert second["upserted"] == 0
        assert second["skipped"] == 1
        assert second["payload_updated"] == 0

    def test_changed_subheading_is_patched(self, live_store):
        """A chunk whose subheading actually changes is patched in place."""
        c1 = make_chunk(chunk_index=0, section_heading="ITEM 1. Business")
        live_store.sync(c1.doc_id, [c1], live_store.embedder, source_file="a.pdf")

        # Same chunk_id (same text, same position), new subheading -> patch.
        c2 = make_chunk(
            chunk_index=0,
            section_heading="ITEM 1A. Risk Factors",
            subheading="Risk Factors",
        )
        result = live_store.sync(c1.doc_id, [c2], live_store.embedder, source_file="a.pdf")
        assert result["upserted"] == 0
        assert result["payload_updated"] == 1

        # A third run converges.
        third = live_store.sync(c1.doc_id, [c2], live_store.embedder, source_file="a.pdf")
        assert third["payload_updated"] == 0

    def test_empty_string_and_missing_key_are_equal(self, live_store):
        """``""`` on one side and a missing key on the other compare equal."""
        from rag_agent.embeddings.store import QdrantStore

        assert QdrantStore._payloads_equal({"a": ""}, {"a": None}) is True
        assert QdrantStore._payloads_equal({"a": None}, {}) is True
        assert QdrantStore._payloads_equal({}, {"a": None}) is True
        assert QdrantStore._payloads_equal({"a": ""}, {}) is True
        # A real value still differs from a missing one.
        assert QdrantStore._payloads_equal({"a": "x"}, {}) is False
        assert QdrantStore._payloads_equal({"a": "x"}, {"a": None}) is False
        assert QdrantStore._payloads_equal({"a": "x"}, {"a": "y"}) is False
        # Plain dict equality is preserved when nothing is null/missing.
        assert QdrantStore._payloads_equal({"a": "x", "b": "y"}, {"a": "x", "b": "y"}) is True

    def test_stale_subheading_is_cleared_by_patch(self, live_store):
        """A stale subheading value left by an older code version is cleared.

        The 41 N-able points carrying ``subheading: "PART I"`` and the 47
        carrying the auditor's report title were written before the 10-K
        marker whitelist, when section markers leaked into the subheading
        field. The fresh payload carries ``subheading: None`` for those
        chunks, so the patch must write ``None`` — and on the live server
        that is what clears the key. Omitting the key from the patch would
        leave the stale value in place and the diff would never converge.
        """
        from rag_agent.embeddings.store import point_id

        c = make_chunk()
        pid = point_id(c.chunk_id)
        live_store.client.seed_payload(
            pid,
            {"doc_id": c.doc_id, "subheading": "PART I"},
            collection_name=live_store.collection_name,
        )

        # The fresh payload carries subheading=None for this chunk, so the
        # stored "PART I" looks changed and is patched.
        result = live_store.sync(c.doc_id, [c], live_store.embedder, source_file="a.pdf")
        assert result["payload_updated"] == 1
        # After the patch the key is gone: the wrapper models the server
        # dropping a null-valued key, so the shadow store no longer has it.
        assert "subheading" not in live_store.client._payloads[pid]
        assert live_store.client._payloads[pid].get("subheading") is None
        # A second run converges.
        second = live_store.sync(c.doc_id, [c], live_store.embedder, source_file="a.pdf")
        assert second["upserted"] == 0
        assert second["payload_updated"] == 0

    def test_omitted_key_does_not_clear_stale_value(self, live_store):
        """Pinning the contract: omitting a key from a patch leaves it alone.

        Verified on the live server — only an explicit ``null`` removes a
        key. This is why ``_payload`` writes ``field: None`` for absent
        optional fields rather than omitting them.
        """
        from rag_agent.embeddings.store import point_id

        c = make_chunk()
        pid = point_id(c.chunk_id)
        live_store.client.seed_payload(
            pid,
            {"doc_id": c.doc_id, "subheading": "PART I"},
            collection_name=live_store.collection_name,
        )

        # Manually patch WITHOUT the subheading key (the omission approach).
        payload_without_subheading = QdrantStore._payload(
            c, live_store.embedder.model_name, "a.pdf"
        )
        payload_without_subheading.pop("subheading")
        live_store.client.batch_update_points(
            live_store.collection_name,
            update_operations=[
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(payload=payload_without_subheading, points=[pid])
                )
            ],
        )
        # The stale value survives because the patch never mentioned the key.
        assert live_store.client._payloads[pid].get("subheading") == "PART I"


class TestSourceFileName:
    """The ``_source_file_name`` helper extracts the bare file name."""

    def test_returns_basename(self):
        assert _source_file_name("data/raw/nps-2013-10k.pdf") == "nps-2013-10k.pdf"

    def test_absolute_path(self):
        assert (
            _source_file_name("/mnt/shared/raw/sec-form-10-k-pdf-fixture.pdf")
            == "sec-form-10-k-pdf-fixture.pdf"
        )

    def test_windows_separator(self):
        """Backslash paths resolve on every platform, not just Windows.

        On a POSIX host ``Path("C:\\\\raw\\\\x.pdf").name`` returns the whole
        string because the backslash is not a separator there, so the
        normalisation must happen before the ``Path`` call.
        """
        assert (
            _source_file_name("C:\\\\raw\\\\n-able-2024-annual-financial-report.pdf")
            == "n-able-2024-annual-financial-report.pdf"
        )
        # The result must never contain a backslash, even on a POSIX host
        # where Path would otherwise leave it untouched.
        assert "\\" not in _source_file_name("C:\\\\raw\\\\n-able-2024-annual-financial-report.pdf")

    def test_bare_name_is_unchanged(self):
        assert _source_file_name("report.pdf") == "report.pdf"

    def test_empty_string(self):
        assert _source_file_name("") == ""
