"""Mocked HTTP tests: no models, sockets, services or downloads required."""

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from rag_agent.generation import GenerationConfig, OllamaProvider, ProviderError, build_provider
from rag_agent.generation.provider import _NoRedirect


@pytest.fixture(autouse=True)
def clear_config(monkeypatch):
    import os

    for key in os.environ:
        if key.startswith(("RAG_LLM_", "RAG_MAX_EVIDENCE_")):
            monkeypatch.delenv(key)


def test_config_requires_explicit_model():
    with pytest.raises(ValueError):
        GenerationConfig.from_env()


def test_environment_configuration(monkeypatch):
    for key, value in {
        "RAG_LLM_PROVIDER": "local",
        "RAG_LLM_MODEL": "installed-model",
        "RAG_LLM_BASE_URL": "http://localhost:11435",
        "RAG_LLM_TIMEOUT_SECONDS": "12.5",
        "RAG_MAX_EVIDENCE_CHARS": "9000",
        "RAG_MAX_EVIDENCE_CHUNKS": "4",
        "RAG_LLM_MAX_OUTPUT_TOKENS": "512",
        "RAG_LLM_MAX_RESPONSE_BYTES": "4096",
    }.items():
        monkeypatch.setenv(key, value)
    config = GenerationConfig.from_env()
    assert config.model == "installed-model" and config.timeout_seconds == 12.5
    assert config.max_chunks == 4 and config.max_evidence_chars == 9000
    assert config.max_output_tokens == 512 and config.max_response_bytes == 4096
    assert isinstance(build_provider(), OllamaProvider)


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "unknown"},
        {"model": " "},
        {"model": " model"},
        {"base_url": "file:///etc/passwd"},
        {"base_url": "http://user:secret@localhost"},
        {"base_url": "http://localhost/api?secret=yes"},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": -1.0},
        {"max_chunks": True},
        {"max_output_tokens": 0},
    ],
)
def test_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        GenerationConfig(**{"model": "test", **overrides})


def adapter(monkeypatch, raw=None, error=None):
    provider = OllamaProvider(GenerationConfig(model="installed-model"))
    calls = []

    def open_request(request, timeout):
        calls.append((request, timeout))
        if error:
            raise error
        return io.BytesIO(
            raw
            if raw is not None
            else json.dumps(
                {
                    "done": True,
                    "message": {"role": "assistant", "content": '{"ok":true}'},
                }
            ).encode()
        )

    monkeypatch.setattr(provider._opener, "open", open_request)
    return provider, calls


def test_request_contract_and_complete_response(monkeypatch):
    provider, calls = adapter(monkeypatch)
    assert provider.complete([{"role": "user", "content": "evidence"}], schema={}) == '{"ok":true}'
    request, timeout = calls[0]
    assert request.full_url == "http://localhost:11434/api/chat"
    assert request.method == "POST" and timeout == 60.0
    body = json.loads(request.data)
    assert body == {
        "model": "installed-model",
        "stream": False,
        "format": {},
        "messages": [{"role": "user", "content": "evidence"}],
        "options": {"temperature": 0, "num_predict": 2048},
    }


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"\xff",
        b"[]",
        b"{}",
        b'{"done":true,"done_reason":"length"}',
        b'{"done":true,"message":null}',
        b'{"done":true,"message":{"role":"user","content":"bad"}}',
        b'{"done":true,"message":{"role":"assistant","content":""}}',
        b"x" * 131073,
    ],
    ids=[
        "json",
        "encoding",
        "list",
        "incomplete",
        "truncated",
        "message",
        "role",
        "empty",
        "oversized",
    ],
)
def test_invalid_envelopes_and_truncation(monkeypatch, raw):
    provider, _ = adapter(monkeypatch, raw)
    with pytest.raises(ProviderError):
        provider.complete([], schema={})


@pytest.mark.parametrize(
    "error",
    [
        URLError("private response"),
        TimeoutError("private response"),
        HTTPError("http://local", 500, "private response", {}, io.BytesIO()),
    ],
)
def test_http_and_timeout_errors_do_not_leak_server_details(monkeypatch, error):
    provider, _ = adapter(monkeypatch, error=error)
    with pytest.raises(ProviderError) as caught:
        provider.complete([], schema={})
    assert "private response" not in str(caught.value)


def test_redirects_are_not_followed():
    assert _NoRedirect().redirect_request(None, None, 302, "", {}, "http://other") is None
