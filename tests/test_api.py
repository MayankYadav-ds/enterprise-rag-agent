"""HTTP contracts tested with real grounding logic and offline dependencies."""

import json
import logging
from dataclasses import replace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app
from rag_agent.api.dependencies import get_generator_factory, get_retriever
from rag_agent.api.runtime import DependencyUnavailable
from rag_agent.api.settings import APISettings
from rag_agent.generation import AnswerGenerator, ProviderError
from rag_agent.retrieval.models import RetrievedChunk


def hit(**changes):
    return replace(
        RetrievedChunk(
            "c1",
            "d1",
            "report.pdf",
            7,
            8,
            "Revenue",
            None,
            "text",
            "Revenue was $10 million.",
            0.9,
            1,
            "dense",
        ),
        **changes,
    )


@pytest.fixture
def api():
    app = create_app(APISettings())
    retriever = Mock(return_value=[hit()])
    provider = Mock()
    provider.complete.return_value = json.dumps(
        {
            "status": "answered",
            "selections": [{"evidence_id": "E1", "quote": hit().text}],
        }
    )
    generator = AnswerGenerator(provider)
    factory = Mock(return_value=generator)
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator_factory] = lambda: factory
    with TestClient(app) as client:
        yield client, retriever, provider, factory


def test_health_docs_and_openapi_do_not_initialize_services(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("health/docs must not initialize external dependencies")

    monkeypatch.setattr("rag_agent.api.runtime.QdrantClient", forbidden)
    monkeypatch.setattr("rag_agent.api.runtime.SentenceTransformerEmbedder", forbidden)
    monkeypatch.setattr("rag_agent.api.runtime.build_provider", forbidden)
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
        assert set(schema["paths"]) == {
            "/health",
            "/api/v1/query",
            "/api/v1/retrieve",
            "/api/v1/documents",
            "/api/v1/documents/{doc_id}",
            "/api/v1/documents/recover",
        }
        assert "503" in schema["paths"]["/api/v1/query"]["post"]["responses"]


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("rerank", [False, True])
def test_query_returns_actual_citations_and_forwards_options(api, strategy, rerank):
    client, retriever, provider, _ = api
    response = client.post(
        "/api/v1/query",
        json={
            "question": "  What was revenue?  ",
            "strategy": strategy,
            "rerank": rerank,
            "top_k": 3,
            "doc_id": "d1",
            "chunk_type": "text",
            "min_score": 0.2,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "answered"
    assert result["strategy"] == strategy and result["reranked"] == rerank
    assert result["citations"][0]["citation"] == "[report.pdf, pp. 7-8, Revenue]"
    assert result["citations"][0]["quote"] == hit().text
    assert result["evidence"][0]["chunk_id"] == "c1"
    request = retriever.call_args.args[0]
    assert (
        request.question,
        request.top_k,
        request.doc_id,
        request.chunk_type,
        request.min_score,
    ) == (
        "What was revenue?",
        3,
        "d1",
        "text",
        0.2,
    )
    retriever.assert_called_once()
    provider.complete.assert_called_once()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": " "},
        {"question": "x", "top_k": 0},
        {"question": "x", "top_k": 101},
        {"question": "x", "top_k": True},
        {"question": "x", "top_k": "5"},
        {"question": "x", "strategy": "invalid"},
        {"question": "x", "rerank": "yes"},
        {"question": "x", "model": "secret"},
        {"question": "x", "chunk_type": "unknown"},
        {"question": "x", "doc_id": " "},
        {"question": "x", "rerank": True, "top_k": 21},
    ],
    ids=[
        "missing",
        "blank",
        "zero",
        "large",
        "bool",
        "string",
        "strategy",
        "rerank",
        "extra",
        "type",
        "doc",
        "candidates",
    ],
)
@pytest.mark.parametrize("endpoint", ["query", "retrieve"])
def test_invalid_requests_do_not_call_dependencies(api, body, endpoint):
    client, retriever, provider, factory = api
    response = client.post(f"/api/v1/{endpoint}", json=body)
    assert response.status_code == 422
    assert "request_id" in response.json()
    retriever.assert_not_called()
    factory.assert_not_called()
    provider.complete.assert_not_called()


def test_oversized_question_and_nonfinite_score(api):
    client, retriever, _, _ = api
    assert client.post("/api/v1/query", json={"question": "x" * 4001}).status_code == 422
    for value in ("NaN", "Infinity", "-Infinity"):
        response = client.post(
            "/api/v1/retrieve",
            content='{"question":"x","min_score":' + value + "}",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
    retriever.assert_not_called()


def test_retrieval_never_initializes_generator(api):
    client, _, provider, factory = api
    result = client.post("/api/v1/retrieve", json={"question": "Revenue"})
    assert result.status_code == 200
    assert result.json()["evidence"][0]["text"] == hit().text
    factory.assert_not_called()
    provider.complete.assert_not_called()


def test_no_evidence_abstains_without_provider_call(api):
    client, retriever, provider, _ = api
    retriever.return_value = []
    result = client.post("/api/v1/query", json={"question": "Revenue"}).json()
    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == result["evidence"] == []
    provider.complete.assert_not_called()


def test_model_can_explicitly_abstain(api):
    client, _, provider, _ = api
    provider.complete.return_value = '{"status":"insufficient_evidence","selections":[]}'
    result = client.post("/api/v1/query", json={"question": "Other fact?"}).json()
    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []
    assert len(result["evidence"]) == 1


def test_conflicts_preserve_both_sources(api):
    client, retriever, provider, _ = api
    second = hit(chunk_id="c2", source_file="other.pdf", text="Revenue was $20 million.")
    retriever.return_value = [hit(), second]
    provider.complete.return_value = json.dumps(
        {
            "status": "conflicting_evidence",
            "selections": [
                {"evidence_id": "E1", "quote": hit().text},
                {"evidence_id": "E2", "quote": second.text},
            ],
        }
    )
    response = client.post("/api/v1/query", json={"question": "Revenue?"})
    assert response.status_code == 200
    assert response.json()["status"] == "conflicting_evidence"
    assert {c["source_file"] for c in response.json()["citations"]} == {"report.pdf", "other.pdf"}


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"status":"answered","selections":[{"evidence_id":"E99","quote":"fake"}]}',
        '{"status":"answered","selections":[{"evidence_id":"E1","quote":"fake"}]}',
    ],
)
def test_invalid_generation_is_502_not_success(api, raw):
    client, _, provider, _ = api
    provider.complete.return_value = raw
    response = client.post("/api/v1/query", json={"question": "Revenue?"})
    assert response.status_code == 502
    assert response.json()["code"] == "generation_failed"
    assert "answer" not in response.json()


@pytest.mark.parametrize("failure,status", [(ProviderError, 503), (RuntimeError, 500)])
def test_provider_errors_and_internal_failures_are_sanitized(api, failure, status, caplog):
    client, _, provider, _ = api
    provider.complete.side_effect = failure("PRIVATE_API_KEY=secret")
    with caplog.at_level(logging.INFO, logger="rag_agent.api"):
        response = client.post("/api/v1/query", json={"question": "private question"})
    assert response.status_code == status
    assert "secret" not in response.text + caplog.text
    assert "private question" not in caplog.text
    record = json.loads(next(r.message for r in caplog.records if r.name == "rag_agent.api"))
    assert record["status"] == status
    assert record["request_id"] == response.headers["X-Request-ID"]


def test_retrieval_unavailable_is_503(api):
    client, retriever, _, factory = api
    retriever.side_effect = DependencyUnavailable("secret endpoint")
    response = client.post("/api/v1/query", json={"question": "Revenue?"})
    assert response.status_code == 503
    assert "secret" not in response.text
    factory.assert_not_called()


def test_malformed_json_and_missing_routes(api):
    client, _, _, _ = api
    response = client.post(
        "/api/v1/query", content="secret-invalid-json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422 and "secret" not in response.text
    assert client.get("/missing").status_code == 404
    assert client.get("/api/v1/query").status_code == 405


def test_default_dependencies_call_app_runtime_and_close_it(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333/")
    app = create_app()
    runtime = app.state.runtime
    retriever = Mock(return_value=[hit()])
    provider = Mock()
    provider.complete.return_value = json.dumps(
        {"status": "answered", "selections": [{"evidence_id": "E1", "quote": hit().text}]}
    )
    factory = Mock(return_value=AnswerGenerator(provider))
    monkeypatch.setattr(runtime, "retrieve", retriever)
    monkeypatch.setattr(runtime, "generator", factory)
    closer = Mock()
    monkeypatch.setattr(runtime, "close", closer)
    with TestClient(app) as client:
        response = client.post("/api/v1/query", json={"question": "Revenue?"})
        assert response.status_code == 200
        assert response.json()["status"] == "answered"
    retriever.assert_called_once()
    factory.assert_called_once()
    closer.assert_called_once()


def test_unavailable_qdrant_through_default_runtime_is_503(monkeypatch):
    monkeypatch.setattr(
        "rag_agent.api.runtime.QdrantClient", Mock(side_effect=ConnectionError("secret-host"))
    )
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/retrieve", json={"question": "Revenue?"})
    assert response.status_code == 503
    assert "secret-host" not in response.text
