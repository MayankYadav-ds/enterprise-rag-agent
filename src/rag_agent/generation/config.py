"""Explicit environment-backed provider and evidence-budget configuration."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    provider: str = "ollama"
    model: str = Field(min_length=1)
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = Field(default=60.0, gt=0, le=600, allow_inf_nan=False)
    max_evidence_chars: int = Field(default=24000, ge=1, le=200000)
    max_chunks: int = Field(default=12, ge=1, le=100)
    max_output_tokens: int = Field(default=2048, ge=64, le=16384)
    max_response_bytes: int = Field(default=131072, ge=1024, le=1048576)

    @model_validator(mode="after")
    def validate_settings(self) -> GenerationConfig:
        if self.provider not in {"ollama", "local"}:
            raise ValueError("RAG_LLM_PROVIDER must be ollama or local (Ollama alias)")
        if not self.model.strip() or self.model != self.model.strip():
            raise ValueError("RAG_LLM_MODEL must be a nonblank model name without edge whitespace")
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("RAG_LLM_BASE_URL must be an HTTP(S) origin without credentials")
        return self

    @classmethod
    def from_env(cls) -> GenerationConfig:
        values: dict = {
            "provider": os.getenv("RAG_LLM_PROVIDER", "ollama"),
            "model": os.getenv("RAG_LLM_MODEL", ""),
            "base_url": os.getenv("RAG_LLM_BASE_URL", "http://localhost:11434"),
        }
        for field, env in (
            ("max_evidence_chars", "RAG_MAX_EVIDENCE_CHARS"),
            ("max_chunks", "RAG_MAX_EVIDENCE_CHUNKS"),
            ("max_output_tokens", "RAG_LLM_MAX_OUTPUT_TOKENS"),
            ("max_response_bytes", "RAG_LLM_MAX_RESPONSE_BYTES"),
        ):
            if env in os.environ:
                values[field] = int(os.environ[env])
        if "RAG_LLM_TIMEOUT_SECONDS" in os.environ:
            values["timeout_seconds"] = float(os.environ["RAG_LLM_TIMEOUT_SECONDS"])
        return cls(**values)
