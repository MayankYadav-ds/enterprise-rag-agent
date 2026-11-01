"""Injectable LLM interface and dependency-free Ollama chat adapter."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from rag_agent.generation.config import GenerationConfig
from rag_agent.generation.models import ProviderError


class LLMProvider(Protocol):
    def complete(self, messages: Sequence[dict[str, str]], *, schema: dict) -> str:
        """Return structured response text or raise ProviderError. No side effects on import."""
        ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Evidence must not be silently forwarded to a different endpoint.
        return None


class OllamaProvider:
    """Call an already-running Ollama service; never pull models or start services."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._opener = build_opener(_NoRedirect())

    def complete(self, messages: Sequence[dict[str, str]], *, schema: dict) -> str:
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": self.config.max_output_tokens},
        }
        request = Request(
            self.config.base_url.rstrip("/") + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read(self.config.max_response_bytes + 1)
            if len(raw) > self.config.max_response_bytes:
                raise ProviderError("LLM response exceeded configured byte limit")
            envelope = json.loads(raw)
            if not isinstance(envelope, dict) or envelope.get("done") is not True:
                raise ProviderError("LLM response is incomplete")
            if envelope.get("done_reason") == "length":
                raise ProviderError(
                    "LLM output reached its token limit; no partial answer returned"
                )
            message = envelope.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ProviderError("LLM response has no assistant message")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("LLM response has no content")
            return content
        except (URLError, OSError, TimeoutError):
            # Do not expose request contents, server response bodies, or credentials.
            raise ProviderError(
                "LLM request failed; check the configured service and model"
            ) from None
        except (ValueError, UnicodeError):
            raise ProviderError("LLM returned an invalid JSON envelope") from None


def build_provider(config: GenerationConfig | None = None) -> LLMProvider:
    return OllamaProvider(config or GenerationConfig.from_env())
