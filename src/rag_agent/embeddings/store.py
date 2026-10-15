"""Qdrant-backed vector store for embedding chunks."""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_agent.chunking.models import Chunk
from rag_agent.embeddings.protocol import Embedder

PROJECT_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _source_file_name(source_file: str) -> str:
    """Return the bare file name from a source path.

    Citations only ever need the file name, not the relative path it was
    indexed from, so a renamed source file is reflected without re-hydrating
    the chunk. Backslashes are normalised to forward slashes first: on a
    POSIX host ``Path("C:\\\\raw\\\\x.pdf").name`` returns the whole string,
    because the backslash is not a separator there.
    """
    if not source_file:
        return ""
    return Path(source_file.replace("\\", "/")).name


def point_id(chunk_id: str) -> uuid.UUID:
    """Deterministic UUID5 for a chunk ID so re-runs never duplicate points."""
    return uuid.uuid5(PROJECT_NAMESPACE, chunk_id)


def _to_uuid(value: str | uuid.UUID) -> uuid.UUID:
    """Normalise a Qdrant point ID to a ``uuid.UUID``.

    Qdrant returns point IDs as plain strings in scroll responses even
    when they were stored as UUIDs, so they must be re-parsed for set
    comparisons to work.
    """
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


# Qdrant's local mode is not a singleton: each ``QdrantClient(location=...)``
# call creates an independent SQLite store, so re-opening a collection in
# memory would silently lose every point. Keep one client per location so
# ``:memory:`` behaves like a shared in-process database.
_CLIENTS: dict[str, QdrantClient] = {}


def _client_for(location: str) -> QdrantClient:
    return _CLIENTS.setdefault(location, QdrantClient(location=location))


class QdrantStore:
    """Wraps a Qdrant collection for chunk embeddings."""

    def __init__(
        self,
        embedder: Embedder,
        collection_name: str = "rag_chunks",
        url: str = ":memory:",
        recreate: bool = False,
    ) -> None:
        # ":memory:" selects Qdrant's local in-memory mode; any other value is
        # passed as a remote location (URL or host).
        self.embedder = embedder
        self.collection_name = collection_name
        self._client = _client_for(url)
        self._ensure_collection(recreate)

    def _ensure_collection(self, recreate: bool) -> None:
        exists = self._client.collection_exists(self.collection_name)
        if exists and recreate:
            self._client.recreate_collection(
                self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedder.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            self._create_indexes()
            return
        if exists:
            info = self._client.get_collection(self.collection_name)
            actual = info.config.params.vectors.size
            if actual != self.embedder.dimension:
                raise ValueError(
                    f"Collection {self.collection_name!r} has dimension "
                    f"{actual} but embedder produced {self.embedder.dimension}. "
                    "Use recreate=True to reset the collection."
                )
            existing_model = self._existing_model_name()
            if existing_model is not None and existing_model != self.embedder.model_name:
                raise ValueError(
                    f"Collection {self.collection_name!r} was built with "
                    f"embedding_model={existing_model!r} but this embedder is "
                    f"{self.embedder.model_name!r}. Mixing vectors from different "
                    "models is not supported. Use recreate=True to reset the collection."
                )
            return
        self._client.create_collection(
            self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension,
                distance=models.Distance.COSINE,
            ),
        )
        self._create_indexes()

    def _create_indexes(self) -> None:
        """Create payload keyword indexes; no-op for in-memory mode."""
        for field in ("doc_id", "chunk_type"):
            try:
                self._client.create_payload_index(
                    self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:  # noqa: BLE001
                pass

    @property
    def client(self) -> QdrantClient:
        return self._client

    def _existing_model_name(self, doc_id: str | None = None) -> str | None:
        """Return the embedding_model stored on the first point, if any."""
        scroll_filter = None
        if doc_id is not None:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
        records, _ = self._client.scroll(
            self.collection_name,
            scroll_filter=scroll_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        return payload.get("embedding_model")

    def count(self) -> int:
        return self._client.count(self.collection_name).count

    def scroll_ids(self, doc_id: str | None = None) -> list[uuid.UUID]:
        """Return stored point IDs without vectors or payload.

        If ``doc_id`` is given, only points belonging to that document are
        returned — this is required so a sync never deletes another
        document's points.
        """
        ids: list[uuid.UUID] = []
        offset = None
        scroll_filter = None
        if doc_id is not None:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
        while True:
            records, next_offset = self._client.scroll(
                self.collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for record in records:
                ids.append(_to_uuid(record.id))  # type: ignore[arg-type]
            if next_offset is None:
                break
            offset = next_offset
        return ids

    def sync(
        self,
        doc_id: str,
        chunks: list[Chunk],
        embedder: Embedder,
        source_file: str = "",
    ) -> dict:
        """Idempotent diff-based sync for one document.

        Returns ``{"upserted": int, "deleted": int, "skipped": int,
        "payload_updated": int}``. ``payload_updated`` counts chunks that were
        already present but whose payload (source file name, model name, page
        range, heading, text quality or text) changed — they are patched in
        place rather than re-embedded, so a renamed source file is reflected
        without re-running the embedder.
        """
        existing = set(self.scroll_ids(doc_id=doc_id))
        target_ids = {point_id(c.chunk_id) for c in chunks}

        # Upsert the new chunks FIRST. If the embedder fails mid-way the old
        # points of this document are still in the collection; deleting stale
        # points before embedding would leave the document half-gone on a
        # failure. The delete runs last so a failed sync is a no-op.
        new_chunks = [c for c in chunks if point_id(c.chunk_id) not in existing]
        upserted = 0
        if new_chunks:
            vectors = embedder.embed_documents([c.text for c in new_chunks])
            points = []
            for chunk, vector in zip(new_chunks, vectors, strict=True):
                points.append(
                    models.PointStruct(
                        id=point_id(chunk.chunk_id),
                        vector=vector,
                        payload=self._payload(chunk, embedder.model_name, source_file),
                    )
                )
            self._client.upsert(self.collection_name, points=points)
            upserted = len(new_chunks)

        # Patch payloads of chunks that already exist but whose stored payload
        # differs from the fresh one. This is what lets a renamed source file
        # show its new name without re-embedding the vector.
        payload_updated = self._patch_existing_payloads(
            doc_id=doc_id,
            chunks=[c for c in chunks if point_id(c.chunk_id) in existing],
            source_file=source_file,
            model_name=embedder.model_name,
        )

        stale = existing - target_ids
        deleted = 0
        if stale:
            self._client.delete(
                self.collection_name,
                points_selector=list(stale),
            )
            deleted = len(stale)

        return {
            "upserted": upserted,
            "deleted": deleted,
            "skipped": len(chunks) - upserted,
            "payload_updated": payload_updated,
        }

    def _patch_existing_payloads(
        self,
        doc_id: str,
        chunks: list[Chunk],
        source_file: str,
        model_name: str,
    ) -> int:
        """Patch the payload of existing points whose stored payload changed.

        Returns the number of points patched. The vector is never touched, so
        the cosine similarity is unchanged; only the citation metadata is
        refreshed. This is what lets a renamed source file show its new name
        without re-running the embedder.

        ``set_payload`` replaces the whole payload with the given dict, so the
        full fresh payload is written — not a partial merge. It is used rather
        than ``upsert`` because Qdrant's local mode rejects points with an
        empty vector, and re-embedding here would defeat the purpose.
        """
        if not chunks:
            return 0
        target = {point_id(c.chunk_id): self._payload(c, model_name, source_file) for c in chunks}
        existing_payloads = self._scroll_payloads(doc_id=doc_id, ids=list(target))
        to_patch = [
            (point_id_, payload)
            for point_id_, payload in target.items()
            if existing_payloads.get(point_id_) is not None
            and existing_payloads[point_id_] != payload
        ]
        if not to_patch:
            return 0
        # Batch the patches into a single batch_update_points call rather than
        # one set_payload request per point. Qdrant's local mode rejects
        # upserts with an empty vector, so the payload is patched in place
        # with SetPayloadOperation — the vector is never touched.
        operations = [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(payload=payload, points=[point_id_]),
            )
            for point_id_, payload in to_patch
        ]
        self._client.batch_update_points(
            self.collection_name,
            update_operations=operations,
        )
        return len(to_patch)

    def _scroll_payloads(self, doc_id: str, ids: list[uuid.UUID]) -> dict[uuid.UUID, dict]:
        """Return ``{point_id: payload}`` for the given points of a document.

        Qdrant's local mode scrolls by offset, not by point id, so the points
        are fetched in pages and filtered by id here. Empty vectors are
        requested so the patch pass never re-embeds anything.
        """
        result: dict[uuid.UUID, dict] = {}
        if not ids:
            return result
        wanted = set(ids)
        offset = None
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id),
                )
            ]
        )
        while True:
            records, next_offset = self._client.scroll(
                self.collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                pid = _to_uuid(record.id)
                if pid in wanted:
                    result[pid] = record.payload or {}
            if next_offset is None:
                break
            offset = next_offset
        return result

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        doc_id: str | None = None,
        chunk_type: str | None = None,
    ) -> list[models.ScoredPoint]:
        """Return top_k hits with score and payload."""
        query_filter: models.Filter | None = None
        if doc_id is not None or chunk_type is not None:
            must: list[models.FieldCondition] = []
            if doc_id is not None:
                must.append(
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                )
            if chunk_type is not None:
                must.append(
                    models.FieldCondition(
                        key="chunk_type",
                        match=models.MatchValue(value=chunk_type),
                    )
                )
            query_filter = models.Filter(must=must)

        result = self._client.query_points(
            self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return result.points  # type: ignore[return-value]

    @staticmethod
    def _payload(
        chunk: Chunk,
        model_name: str,
        source_file: str,
    ) -> dict:
        """Build the point payload for a chunk.

        ``source_file`` is stored as the bare file name (``Path(source_file).name``)
        so citations are stable across renames and re-indexes.
        """
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "source_file": _source_file_name(source_file),
            "chunk_type": chunk.chunk_type,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_heading": chunk.section_heading,
            "text_quality": chunk.text_quality,
            "token_count": chunk.token_count,
            "text": chunk.text,
            "embedding_model": model_name,
        }
