"""Document lifecycle tests using temporary storage and isolated in-memory Qdrant."""

from unittest.mock import Mock

import fitz
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from rag_agent.api.app import create_app
from rag_agent.api.documents import DocumentError, DocumentManager, safe_filename
from rag_agent.api.schemas import RetrievalRequest
from rag_agent.api.settings import APISettings
from rag_agent.auth.models import create_user, ensure_auth_schema
from rag_agent.auth.password import hash_password
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.store import QdrantStore


def pdf_bytes(text="Revenue increased in 2024. The company reported growth in all regions."):
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 500, 700), (text + "\n") * 4)
        return pdf.tobytes()


@pytest.fixture
def system(tmp_path, monkeypatch):
    settings = APISettings(document_dir=tmp_path / "managed", upload_max_bytes=2048)
    app = create_app(settings)
    runtime = app.state.runtime
    ensure_auth_schema(runtime)
    with runtime.connection() as db:
        create_user(db, "doc-admin", hash_password("document-test-password-42"), is_admin=True)
    qdrant = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    QdrantStore(embedder, client=qdrant)
    runtime._client = qdrant
    monkeypatch.setattr(runtime, "embedding", lambda: embedder)
    with TestClient(app) as client:
        login = client.post(
            "/auth/login",
            headers={"Origin": "http://localhost:5173"},
            json={"username": "doc-admin", "password": "document-test-password-42"},
        )
        assert login.status_code == 200
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        yield client, runtime, qdrant


def upload(client, data=None, filename="report.pdf"):
    return client.post(
        "/api/v1/documents",
        params={"filename": filename},
        content=data if data is not None else pdf_bytes(),
        headers={"content-type": "application/pdf"},
    )


def test_valid_upload_inventory_duplicate_and_scoped_delete(system):
    client, runtime, qdrant = system
    data = pdf_bytes()
    response = upload(client, data)
    assert response.status_code == 202
    doc_id = response.json()["doc_id"]
    ready = client.get(f"/api/v1/documents/{doc_id}").json()
    assert ready["status"] == "ready" and ready["page_count"] == 1 and ready["chunk_count"] > 0
    assert "file" not in ready and "source_path" not in ready
    assert upload(client, data, "renamed.pdf").status_code == 409
    second = upload(client, pdf_bytes("Other unrelated content about markets."), "other.pdf").json()
    assert len(client.get("/api/v1/documents").json()) == 2
    # Warm the BM25 cache, then ensure deletion invalidates it.
    assert runtime.retrieve(RetrievalRequest(question="Revenue", strategy="bm25"))
    assert client.delete(f"/api/v1/documents/{doc_id}").json()["status"] == "deleted"
    assert runtime._retrievers == {}
    assert not runtime._store(FakeEmbedder(), validate=False).scroll_ids(doc_id)
    assert runtime._store(FakeEmbedder(), validate=False).scroll_ids(second["doc_id"])
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 200
    assert len(client.get("/api/v1/documents").json()) == 1
    assert not runtime.documents()._file(doc_id).exists()
    assert not (runtime.documents().root / doc_id / (doc_id + ".json")).exists()
    assert upload(client, data).status_code == 202


def test_failed_retry_does_not_orphan_uploaded_bytes(system, monkeypatch):
    client, runtime, _ = system
    data = pdf_bytes()
    original = runtime.embedding
    monkeypatch.setattr(runtime, "embedding", Mock(side_effect=RuntimeError("model missing")))
    doc_id = upload(client, data).json()["doc_id"]
    assert client.get(f"/api/v1/documents/{doc_id}").json()["status"] == "failed"
    old = runtime.documents()._file(doc_id)
    monkeypatch.setattr(runtime, "embedding", original)
    assert upload(client, data).status_code == 202
    assert not old.exists()
    assert len(list(runtime.documents().root.glob("*.pdf"))) == 1


def test_malformed_pdf_and_page_limits_are_failed_jobs(system):
    client, runtime, _ = system
    response = upload(client, b"%PDF-not-really-a-pdf")
    assert client.get("/api/v1/documents/" + response.json()["doc_id"]).json()["status"] == "failed"
    runtime.settings = runtime.settings.model_copy(update={"document_max_pages": 1})
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.new_page()
        response = upload(client, pdf.tobytes())
    assert client.get("/api/v1/documents/" + response.json()["doc_id"]).json()["status"] == "failed"


def test_interrupted_queue_is_recoverable_without_vector_writes(system):
    client, runtime, _ = system
    doc_id = upload(client).json()["doc_id"]
    service = runtime.documents()
    record = service.get(doc_id)
    service._erase_vectors(doc_id)
    record.status, record.owns_vectors = "queued", False
    service._save(record)
    service.recover()
    assert service.get(doc_id).status == "failed"


@pytest.mark.parametrize(
    "name",
    ["../x.pdf", "C:\\x.pdf", "a/b.pdf", "CON.pdf", "x\n.pdf", "a[1].pdf", ".pdf", "report.exe"],
)
def test_unsafe_filenames_rejected(name):
    with pytest.raises(DocumentError):
        safe_filename(name)


def test_upload_validation(system):
    client, runtime, _ = system
    assert upload(client, b"not a PDF").status_code == 415
    assert upload(client, b"%PDF-" + b"x" * 3000).status_code == 413
    assert upload(client, b"").status_code == 415
    assert client.post("/api/v1/documents?filename=a.pdf", content=b"x").status_code == 415
    assert not list(runtime.settings.document_dir.glob("*.pdf"))


def test_stream_size_limit_without_content_length(system):
    client, runtime, _ = system
    result = client.post(
        "/api/v1/documents?filename=a.pdf",
        content=iter([b"%PDF-", b"x" * 3000]),
        headers={"content-type": "application/pdf"},
    )
    assert result.status_code == 413
    assert not list(runtime.settings.document_dir.glob("*.pdf"))


@pytest.mark.parametrize("phase", ["parse", "embed", "write"])
def test_failures_never_report_ready_and_remove_partial_points(system, monkeypatch, phase):
    client, runtime, qdrant = system
    if phase == "parse":
        monkeypatch.setattr(
            "rag_agent.api.documents.ingest_pdf", Mock(side_effect=RuntimeError("private path"))
        )
    elif phase == "embed":
        monkeypatch.setattr(
            runtime.embedding(), "embed_documents", Mock(side_effect=RuntimeError("private key"))
        )
        # FakeEmbedder instance supplied by the fixture is stable.
    else:
        original = qdrant.upsert

        def partial(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("response lost after write")

        monkeypatch.setattr(qdrant, "upsert", partial)
    response = upload(client)
    doc_id = response.json()["doc_id"]
    record = client.get(f"/api/v1/documents/{doc_id}").json()
    assert record["status"] == "failed"
    assert "private" not in str(record)
    assert runtime._store(FakeEmbedder(), validate=False).scroll_ids(doc_id) == []


def test_cleanup_failure_blocks_retrieval_and_recovery_retries(system, monkeypatch):
    client, runtime, qdrant = system
    original_upsert, original_delete = qdrant.upsert, qdrant.delete

    def partial(*args, **kwargs):
        original_upsert(*args, **kwargs)
        raise RuntimeError("uncertain write")

    monkeypatch.setattr(qdrant, "upsert", partial)
    monkeypatch.setattr(qdrant, "delete", Mock(side_effect=ConnectionError("offline")))
    doc_id = upload(client).json()["doc_id"]
    assert client.get(f"/api/v1/documents/{doc_id}").json()["status"] == "cleanup_required"
    assert (
        client.post(
            "/api/v1/retrieve", json={"question": "Revenue", "strategy": "bm25"}
        ).status_code
        == 503
    )
    assert client.post("/api/v1/documents/recover").status_code == 503
    monkeypatch.setattr(qdrant, "delete", original_delete)
    assert client.post("/api/v1/documents/recover").status_code == 204
    assert client.get(f"/api/v1/documents/{doc_id}").json()["status"] == "failed"
    assert runtime._store(FakeEmbedder(), validate=False).scroll_ids(doc_id) == []


def test_registration_qdrant_failure_returns_503_without_queued_record(system, monkeypatch):
    client, runtime, qdrant = system
    monkeypatch.setattr(qdrant, "scroll", Mock(side_effect=ConnectionError("secret-host")))
    response = upload(client)
    assert response.status_code == 503 and "secret" not in response.text
    assert runtime.documents().records() == []


def test_deletion_failure_is_retriable_and_clears_cache(system, monkeypatch):
    client, runtime, qdrant = system
    doc_id = upload(client).json()["doc_id"]
    original = qdrant.delete
    monkeypatch.setattr(qdrant, "delete", Mock(side_effect=ConnectionError()))
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 503
    assert client.get(f"/api/v1/documents/{doc_id}").json()["status"] == "deleting"
    monkeypatch.setattr(qdrant, "delete", original)
    assert client.delete(f"/api/v1/documents/{doc_id}").json()["status"] == "deleted"


def test_external_documents_are_read_only_and_duplicates_not_adopted(system):
    client, runtime, qdrant = system
    data = pdf_bytes()
    doc_id = upload(client, data).json()["doc_id"]
    manager = runtime.documents()
    with manager.connection() as db:
        db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    inventory = client.get("/api/v1/documents").json()
    assert inventory[0]["status"] == "external" and not inventory[0]["managed"]
    assert upload(client, data).status_code == 409
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 404
    assert runtime._store(FakeEmbedder(), validate=False).scroll_ids(doc_id)


def test_restart_recovers_uncertain_index_and_preserves_other_documents(system):
    client, runtime, _ = system
    doc_id = upload(client).json()["doc_id"]
    other = upload(client, pdf_bytes("Other evidence."), "other.pdf").json()["doc_id"]
    service = runtime.documents()
    record = service.get(doc_id)
    record.status = "indexing"
    service._save(record)
    service.close()
    runtime._documents = None
    recovered = runtime.documents()
    assert recovered.get(doc_id).status == "failed"
    assert recovered._store().scroll_ids(other)
    assert not recovered._store().scroll_ids(doc_id)


def test_storage_has_exclusive_worker_ownership(system):
    _, runtime, _ = system
    runtime.documents()
    with pytest.raises(DocumentError, match="another worker"):
        DocumentManager(runtime)


def test_corrupt_registry_cannot_delete_outside_storage(system, tmp_path):
    client, runtime, _ = system
    doc_id = upload(client).json()["doc_id"]
    protected = tmp_path / "protected.pdf"
    protected.write_bytes(b"important")
    with runtime.documents().connection() as db:
        db.execute("UPDATE documents SET file=? WHERE id=?", (str(protected), doc_id))
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 503
    assert protected.read_bytes() == b"important"
