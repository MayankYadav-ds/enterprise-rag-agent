"""Server-owned configuration; no network or model loading at import time."""

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


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
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

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

    @classmethod
    def from_env(cls) -> "APISettings":
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
            "log_level": "RAG_LOG_LEVEL",
        }
        return cls(**{key: os.environ[name] for key, name in names.items() if name in os.environ})
