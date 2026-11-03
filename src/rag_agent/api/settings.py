"""Server-owned configuration; no network or model loading at import time."""

import ipaddress
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class APISettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = Field(default=None, repr=False, exclude=True)
    collection: str = Field(default="rag_chunks", min_length=1, max_length=200)
    qdrant_timeout: float = Field(default=10, gt=0, le=120, allow_inf_nan=False)
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    document_dir: Path = Path("data/managed_documents")
    upload_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)
    document_max_pages: int = Field(default=500, ge=1, le=5000)
    document_max_chunks: int = Field(default=10000, ge=1, le=100000)
    login_rate_limit_attempts: int = Field(default=10, ge=1, le=100)
    login_rate_limit_window_seconds: int = Field(default=900, ge=60, le=86400)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    environment: Literal["development", "test", "production"] = "development"
    trusted_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    )

    @model_validator(mode="after")
    def validate_trusted_origins(self) -> "APISettings":
        for origin in self.trusted_origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path
                or parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
            ):
                raise ValueError("trusted_origins must contain HTTP(S) origins only")
        if self.environment == "production" and (
            not self.trusted_origins
            or any(urlsplit(origin).scheme != "https" for origin in self.trusted_origins)
        ):
            raise ValueError("production requires one or more HTTPS trusted origins")
        return self

    @field_validator("qdrant_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("Qdrant URL must be an HTTP(S) origin without credentials")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_remote_qdrant_transport(self) -> "APISettings":
        host = urlsplit(self.qdrant_url).hostname
        if host is None:
            raise ValueError("Qdrant URL requires a host")
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.casefold() == "localhost"
        if not loopback and urlsplit(self.qdrant_url).scheme != "https":
            raise ValueError("remote Qdrant connections require HTTPS")
        if not loopback and (
            self.qdrant_api_key is None or not self.qdrant_api_key.get_secret_value().strip()
        ):
            raise ValueError("remote Qdrant connections require QDRANT_API_KEY")
        return self

    @classmethod
    def from_env(cls) -> "APISettings":
        environment = os.environ.get("RAG_ENV")
        if environment is None:
            raise ValueError("RAG_ENV must be explicitly set to development, test, or production")
        if environment == "production" and "RAG_TRUSTED_ORIGINS" not in os.environ:
            raise ValueError("production requires RAG_TRUSTED_ORIGINS with HTTPS frontend origins")
        names = {
            "qdrant_url": "QDRANT_URL",
            "qdrant_api_key": "QDRANT_API_KEY",
            "collection": "RAG_API_COLLECTION",
            "qdrant_timeout": "RAG_API_QDRANT_TIMEOUT",
            "rerank_candidates": "RERANKER_CANDIDATE_COUNT",
            "document_dir": "RAG_DOCUMENT_DIR",
            "upload_max_bytes": "RAG_UPLOAD_MAX_BYTES",
            "document_max_pages": "RAG_DOCUMENT_MAX_PAGES",
            "document_max_chunks": "RAG_DOCUMENT_MAX_CHUNKS",
            "login_rate_limit_attempts": "RAG_LOGIN_RATE_LIMIT_ATTEMPTS",
            "login_rate_limit_window_seconds": "RAG_LOGIN_RATE_LIMIT_WINDOW_SECONDS",
            "log_level": "RAG_LOG_LEVEL",
            "environment": "RAG_ENV",
        }
        values = {key: os.environ[name] for key, name in names.items() if name in os.environ}
        if "RAG_TRUSTED_ORIGINS" in os.environ:
            values["trusted_origins"] = tuple(
                origin.strip().rstrip("/")
                for origin in os.environ["RAG_TRUSTED_ORIGINS"].split(",")
                if origin.strip()
            )
        return cls(**values)
