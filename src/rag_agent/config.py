"""Environment-backed application configuration."""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by future pipeline components.

    Secrets are read only from environment variables and are never logged.
    """

    environment: str = "development"
    log_level: str = "INFO"
    llm_provider: str = "local"
    llm_model: str | None = None
    qdrant_url: str = "http://localhost:6333"

    @classmethod
    def from_environment(cls) -> Settings:
        """Build settings from optional ``RAG_*`` environment variables."""
        return cls(
            environment=getenv("RAG_ENV", "development"),
            log_level=getenv("RAG_LOG_LEVEL", "INFO"),
            llm_provider=getenv("RAG_LLM_PROVIDER", "local"),
            llm_model=getenv("RAG_LLM_MODEL") or None,
            qdrant_url=getenv("QDRANT_URL", "http://localhost:6333"),
        )
