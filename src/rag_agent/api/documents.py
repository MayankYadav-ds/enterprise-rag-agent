"""Durable, single-writer document lifecycle using the existing indexing pipeline.

SQLite records intent before vector mutations. Uncertain writes are compensated
by exact document-ID deletion; unresolved compensation blocks API retrieval.
Only newly registered, API-owned documents can be mutated. Never adopt CLI data.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import fitz
from filelock import FileLock, Timeout
from pydantic import BaseModel
from qdrant_client.http import models

from rag_agent.chunking.pipeline import chunk_document
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.store import QdrantStore
from rag_agent.ingestion.models import document_id_from_hash, sha256_for_file
from rag_agent.ingestion.pipeline import ingest_pdf

if TYPE_CHECKING:
    from rag_agent.api.runtime import Runtime

logger = logging.getLogger("rag_agent.api.documents")
State = Literal[
    "queued",
    "processing",
    "indexing",
    "ready",
    "failed",
    "cleanup_required",
    "deleting",
    "deleted",
    "external",
]


class DocumentError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code = status, code


class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    status: State
    managed: bool = True
    owns_vectors: bool = False
    size_bytes: int | None = None
    page_count: int | None = None
    chunk_count: int = 0
    updated_at: str | None = None
    error: str | None = None


def safe_filename(value: str) -> str:
    """Reject path syntax rather than silently renaming untrusted input."""
    if (
        not value
        or len(value) > 120
        or value != value.strip()
        or not value.lower().endswith(".pdf")
        or any(c in value for c in '/\\:<>"|?*[]')
        or any(unicodedata.category(c).startswith("C") for c in value)
        or value.startswith(".")
        or value.split(".")[0].upper()
        in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *[f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)],
        }
    ):
        raise DocumentError(422, "invalid_filename", "Use a plain PDF filename without path syntax")
    return value


class DocumentManager:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.root = runtime.settings.document_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.guard = FileLock(str(self.root / "writer.lock"), thread_local=False)
        try:
            self.guard.acquire(timeout=0)
        except Timeout as exc:
            raise DocumentError(
                503, "writer_busy", "Document storage is owned by another worker"
            ) from exc
        self.db = self.root / "inventory.sqlite3"
        try:
            with self.connection() as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS documents "
                    "(id TEXT PRIMARY KEY, record TEXT, file TEXT)"
                )
                db.execute("CREATE TABLE IF NOT EXISTS binding (target TEXT PRIMARY KEY)")
                target = json.dumps([runtime.settings.qdrant_url, runtime.settings.collection])
                bindings = db.execute("SELECT target FROM binding").fetchall()
                if bindings and bindings[0][0] != target:
                    raise DocumentError(
                        503, "storage_mismatch", "Document storage belongs to another collection"
                    )
                db.execute("INSERT OR IGNORE INTO binding VALUES (?)", (target,))
            self.recover()
        except Exception:
            self.guard.release()
            raise

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def records(self) -> list[DocumentRecord]:
        with self.connection() as db:
            return [
                DocumentRecord.model_validate_json(row[0])
                for row in db.execute("SELECT record FROM documents")
            ]

    def get(self, doc_id: str) -> DocumentRecord:
        if not re.fullmatch(r"doc_[0-9a-f]{64}", doc_id):
            raise DocumentError(404, "not_found", "Managed document not found")
        with self.connection() as db:
            row = db.execute("SELECT record FROM documents WHERE id=?", (doc_id,)).fetchone()
        if row is None:
            raise DocumentError(404, "not_found", "Managed document not found")
        return DocumentRecord.model_validate_json(row[0])

    def _save(self, record: DocumentRecord, file: Path | None = None) -> None:
        record.updated_at = datetime.now(UTC).isoformat()
        with self.connection() as db:
            if file is not None:
                db.execute(
                    "INSERT OR REPLACE INTO documents VALUES (?,?,?)",
                    (record.doc_id, record.model_dump_json(), file.name),
                )
            else:
                db.execute(
                    "UPDATE documents SET record=? WHERE id=?",
                    (record.model_dump_json(), record.doc_id),
                )
        logger.info(
            json.dumps(
                {"event": "document_state", "doc_id": record.doc_id, "status": record.status}
            )
        )

    def _file(self, doc_id: str) -> Path:
        with self.connection() as db:
            name = db.execute("SELECT file FROM documents WHERE id=?", (doc_id,)).fetchone()[0]
        # Even a corrupt registry must not address arbitrary filesystem paths.
        if not re.fullmatch(r"[0-9a-f]{32}\.pdf", name):
            raise DocumentError(503, "invalid_registry", "Document storage requires inspection")
        file = self.root / name
        if file.resolve().parent != self.root or file.is_symlink():
            raise DocumentError(503, "invalid_registry", "Document storage requires inspection")
        return file

    def _store(self) -> QdrantStore:
        return self.runtime._store(FakeEmbedder(), validate=False)

    def ensure_readable(self) -> None:
        if any(r.status in {"indexing", "cleanup_required", "deleting"} for r in self.records()):
            raise DocumentError(
                503, "recovery_required", "Document cleanup must complete before retrieval"
            )

    def accept(self, file: Path, filename: str) -> DocumentRecord:
        with self.runtime.lock:
            self.ensure_readable()
            doc_id = document_id_from_hash(sha256_for_file(str(file)))
            previous = next((r for r in self.records() if r.doc_id == doc_id), None)
            if previous and previous.status not in {"failed", "deleted"}:
                raise DocumentError(409, "duplicate_document", "This PDF is already registered")
            if self._store().scroll_ids(doc_id):
                raise DocumentError(409, "duplicate_document", "This PDF is already indexed")
            if previous:
                # A retry must not orphan the previous API-owned copy of the same bytes.
                self._file(doc_id).unlink(missing_ok=True)
            record = DocumentRecord(
                doc_id=doc_id, filename=filename, status="queued", size_bytes=file.stat().st_size
            )
            self._save(record, file)
            return record

    def process(self, doc_id: str) -> None:
        with self.runtime.lock:
            record = self.get(doc_id)
            if record.status != "queued":
                return
            mutation_started = False
            try:
                self.ensure_readable()
                record.status = "processing"
                self._save(record)
                file = self._file(doc_id)
                # Inspect page count/encryption before full extraction, without duplicating parsing.
                with fitz.open(file) as pdf:
                    if pdf.needs_pass or pdf.page_count > self.runtime.settings.document_max_pages:
                        raise ValueError("Unsupported PDF limits")
                document, pages = ingest_pdf(file, self.root / doc_id)
                if document.doc_id != doc_id:
                    raise ValueError("Content identity changed")
                chunks = chunk_document(
                    document, pages, strategy="recursive", chunk_size=400, chunk_overlap=39
                )
                if not chunks or len(chunks) > self.runtime.settings.document_max_chunks:
                    raise ValueError("No usable chunks or chunk limit exceeded")
                record.page_count, record.chunk_count = document.num_pages, len(chunks)
                embedder = self.runtime.embedding()
                store = self.runtime._store(embedder, validate=True)
                # Never adopt or patch data introduced by an external writer.
                if store.scroll_ids(doc_id):
                    raise ValueError("Document already indexed externally")
                record.status = "indexing"
                record.owns_vectors = True
                self._save(record)
                mutation_started = True
                store.sync(doc_id, chunks, embedder, source_file=record.filename)
                if len(store.scroll_ids(doc_id)) != len(chunks):
                    raise ValueError("Incomplete vector write")
                record.status, record.error = "ready", None
                self._save(record)
            except Exception:
                if mutation_started:
                    self._compensate(record)
                else:
                    record.status, record.error = (
                        "failed",
                        "PDF processing failed; check format, limits, and services",
                    )
                    self._save(record)
            finally:
                self.runtime.invalidate_retrieval()

    def _erase_vectors(self, doc_id: str) -> None:
        store = self._store()
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        )
        store.client.delete(store.collection_name, points_selector=selector, wait=True)
        if store.scroll_ids(doc_id):
            raise RuntimeError("Vector deletion not confirmed")

    def _compensate(self, record: DocumentRecord) -> None:
        try:
            self._erase_vectors(record.doc_id)
            record.owns_vectors = False
            record.chunk_count = 0
            record.status, record.error = (
                "failed",
                "Ingestion failed; partial vector writes were removed. Re-upload to retry.",
            )
        except Exception:
            record.status, record.error = (
                "cleanup_required",
                "Cleanup could not be confirmed. Retry recovery before searching.",
            )
        self._save(record)

    def delete(self, doc_id: str) -> DocumentRecord:
        with self.runtime.lock:
            record = self.get(doc_id)
            if record.status == "deleted":
                return record
            record.status, record.error = "deleting", None
            self._save(record)
            try:
                if record.owns_vectors:
                    self._erase_vectors(doc_id)
                record.owns_vectors = False
                # Remove only known API-owned artifacts; never recursively delete directories.
                self._file(doc_id).unlink(missing_ok=True)
                for suffix in (".json", ".json.tmp"):
                    parsed = self.root / doc_id / (doc_id + suffix)
                    if parsed.resolve().parent != self.root / doc_id or parsed.is_symlink():
                        raise ValueError("Unsafe parsed artifact path")
                    parsed.unlink(missing_ok=True)
                record.status, record.chunk_count = "deleted", 0
                self._save(record)
            except Exception as exc:
                record.error = "Deletion incomplete; retry deletion to finish safely"
                self._save(record)
                raise DocumentError(503, "deletion_incomplete", record.error) from exc
            finally:
                self.runtime.invalidate_retrieval()
            return record

    def recover(self) -> None:
        with self.runtime.lock:
            for record in self.records():
                if record.status in {"indexing", "cleanup_required"}:
                    self._compensate(record)
                elif record.status == "deleting":
                    try:
                        self.delete(record.doc_id)
                    except DocumentError:
                        pass  # Durable deleting state remains visible and blocks retrieval.
                elif record.status in {"queued", "processing"}:
                    record.status, record.error = (
                        "failed",
                        "Processing was interrupted. Re-upload to retry.",
                    )
                    self._save(record)
            self.runtime.invalidate_retrieval()

    def inventory(self) -> list[DocumentRecord]:
        records = self.records()
        managed = {r.doc_id for r in records if r.status != "deleted"}
        external: dict[str, DocumentRecord] = {}
        for payload in self._store().scroll_payloads().values():
            doc_id = payload.get("doc_id")
            if not isinstance(doc_id, str) or doc_id in managed:
                continue
            if doc_id not in external:
                filename = Path(str(payload.get("source_file", "")).replace("\\", "/")).name
                external[doc_id] = DocumentRecord(
                    doc_id=doc_id, filename=filename, status="external", managed=False
                )
            external[doc_id].chunk_count += 1
        return [r for r in records if r.status != "deleted"] + list(external.values())

    def close(self) -> None:
        self.guard.release()
