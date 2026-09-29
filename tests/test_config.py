"""Tests for application configuration."""

from rag_agent.config import Settings


def test_settings_have_safe_defaults(monkeypatch: object) -> None:
    """Settings work without a local .env file or credentials."""
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)  # type: ignore[attr-defined]
    settings = Settings.from_environment()

    assert settings.environment == "development"
    assert settings.llm_provider == "local"
    assert settings.llm_model is None
    assert settings.qdrant_url == "http://localhost:6333"
