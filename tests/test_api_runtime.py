"""Real runtime composition using an isolated in-memory Qdrant and fake models."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from rag_agent.api.runtime import DependencyUnavailable, Runtime
from rag_agent.api.schemas import RetrievalRequest
from rag_agent.api.settings import APISettings
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.store import QdrantStore
from rag_agent.retrieval.reranker import FakeCrossEncoder


def test_read_only_missing_collection_and_recreate_do_not_write():
    client = Mock()
    client.collection_exists.return_value = False
    with pytest.raises(ValueError, match="does not exist"):
        QdrantStore(FakeEmbedder(), client=client, create_if_missing=False)
    with pytest.raises(ValueError, match="recreate"):
        QdrantStore(FakeEmbedder(), client=client, create_if_missing=False, recreate=True)
    client.create_collection.assert_not_called()
    client.recreate_collection.assert_not_called()
    client.create_payload_index.assert_not_called()


def test_bm25_runtime_uses_payloads_without_loading_models(monkeypatch):
    client = Mock()
    monkeypatch.setattr("rag_agent.api.runtime.QdrantClient", Mock(return_value=client))
    store = Mock()
    store.scroll_payloads.return_value = {}
    store_class = Mock(return_value=store)
    embedder_class = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr("rag_agent.api.runtime.QdrantStore", store_class)
    monkeypatch.setattr("rag_agent.api.runtime.SentenceTransformerEmbedder", embedder_class)
    runtime = Runtime(APISettings())
    assert runtime.retrieve(RetrievalRequest(question="x", strategy="bm25")) == []
    assert runtime.retrieve(RetrievalRequest(question="x", strategy="bm25")) == []
    store.scroll_payloads.assert_called_once()
    assert store_class.call_args.kwargs["create_if_missing"] is False
    embedder_class.assert_not_called()
    runtime.close()
    client.close.assert_called_once()


def test_unavailable_service_retries_without_loading_dense_model(monkeypatch):
    store = Mock(side_effect=ConnectionError("private host"))
    model = Mock()
    monkeypatch.setattr("rag_agent.api.runtime.QdrantStore", store)
    monkeypatch.setattr("rag_agent.api.runtime.QdrantClient", Mock())
    monkeypatch.setattr("rag_agent.api.runtime.SentenceTransformerEmbedder", model)
    runtime = Runtime(APISettings())
    for _ in range(2):
        with pytest.raises(DependencyUnavailable):
            runtime.retrieve(RetrievalRequest(question="x", strategy="dense"))
    assert store.call_count == 2
    model.assert_not_called()
    runtime.close()


def test_generator_is_lazy_cached_and_configuration_failure_is_retriable(monkeypatch):
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
    runtime = Runtime(APISettings())
    with pytest.raises(DependencyUnavailable):
        runtime.generator()
    monkeypatch.setenv("RAG_LLM_MODEL", "installed-model")
    provider = Mock()
    builder = Mock(return_value=provider)
    monkeypatch.setattr("rag_agent.api.runtime.build_provider", builder)
    assert runtime.generator() is runtime.generator()
    builder.assert_called_once()
    provider.complete.assert_not_called()
    runtime.close()


@pytest.mark.parametrize(
    "values",
    [
        {"qdrant_url": "file:///private"},
        {"qdrant_url": "http://user:secret@localhost"},
        {"qdrant_url": "http://localhost/path"},
        {"rerank_candidates": 101},
        {"qdrant_timeout": 0},
        {"qdrant_timeout": float("inf")},
    ],
)
def test_invalid_settings(values):
    with pytest.raises(ValidationError):
        APISettings(**values)


def test_environment_settings_hide_credentials(monkeypatch):
    monkeypatch.setenv("RAG_ENV", "test")
    monkeypatch.delenv("RAG_TRUSTED_ORIGINS", raising=False)
    monkeypatch.setenv("QDRANT_API_KEY", "private-key")
    monkeypatch.setenv("RAG_API_COLLECTION", "test_chunks")
    monkeypatch.setenv("RERANKER_CANDIDATE_COUNT", "30")
    settings = APISettings.from_env()
    assert settings.collection == "test_chunks" and settings.rerank_candidates == 30
    assert "private-key" not in repr(settings) + settings.model_dump_json()


def test_environment_configuration_requires_explicit_environment(monkeypatch):
    monkeypatch.delenv("RAG_ENV", raising=False)
    with pytest.raises(ValueError, match="RAG_ENV must be explicitly set"):
        APISettings.from_env()


def test_production_requires_explicit_https_trusted_origins(monkeypatch):
    monkeypatch.setenv("RAG_ENV", "production")
    monkeypatch.delenv("RAG_TRUSTED_ORIGINS", raising=False)
    with pytest.raises(ValueError, match="production requires RAG_TRUSTED_ORIGINS"):
        APISettings.from_env()

    monkeypatch.setenv("RAG_TRUSTED_ORIGINS", "http://localhost:5173")
    with pytest.raises(ValidationError, match="production requires one or more HTTPS"):
        APISettings.from_env()

    monkeypatch.setenv("RAG_TRUSTED_ORIGINS", "https://app.example.com")
    settings = APISettings.from_env()
    assert settings.environment == "production"
    assert settings.trusted_origins == ("https://app.example.com",)


def test_production_settings_reject_default_development_origins():
    with pytest.raises(ValidationError, match="production requires one or more HTTPS"):
        APISettings(environment="production")


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("rerank", [False, True])
def test_real_retrievers_and_optional_reranker_compose(monkeypatch, strategy, rerank):
    # Isolated test data only; never accesses the user's running Qdrant.
    from qdrant_client.http import models

    from rag_agent.embeddings.store import point_id

    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    QdrantStore(embedder, client=client)
    text = "Revenue increased in 2024."
    client.upsert(
        "rag_chunks",
        points=[
            models.PointStruct(
                id=str(point_id("c1")),
                vector=embedder.embed_query(text),
                payload={
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "source_file": "report.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "section_heading": None,
                    "subheading": None,
                    "chunk_type": "text",
                    "text": text,
                    "embedding_model": embedder.model_name,
                },
            )
        ],
    )
    monkeypatch.setattr("rag_agent.api.runtime.QdrantClient", Mock(return_value=client))
    model = Mock(return_value=embedder)
    monkeypatch.setattr("rag_agent.api.runtime.SentenceTransformerEmbedder", model)
    encoder = Mock(return_value=FakeCrossEncoder())
    monkeypatch.setattr("rag_agent.retrieval.reranker.CrossEncoderReranker._load_model", encoder)
    runtime = Runtime(APISettings())
    request = RetrievalRequest(question="Revenue", strategy=strategy, rerank=rerank, top_k=1)
    first = runtime.retrieve(request)
    assert first[0].chunk_id == "c1"
    assert first[0].source_file == "report.pdf"
    assert runtime.retrieve(request)[0].chunk_id == "c1"
    assert model.call_count == (0 if strategy == "bm25" else 1)
    if not rerank:
        encoder.assert_not_called()
    runtime.close()
