"""Tests for SentenceTransformerEmbedder using a stub model (no real download)."""

from __future__ import annotations

import sys

import pytest

from rag_agent.embeddings.sentence_transformer import SentenceTransformerEmbedder


class StubModel:
    """Minimal stand-in for sentence_transformers.SentenceTransformer.

    Records every ``encode`` call so tests can assert on batching, the
    query instruction prefix, and dimension without touching the network.
    """

    def __init__(self, name: str = "stub", device: str = "cpu") -> None:
        self.name = name
        self.device = device
        self.calls: list[tuple[list[str], dict]] = []
        self._dimension = 8

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        return [[float(i)] * self._dimension for i in range(len(texts))]


@pytest.fixture
def stub_module(monkeypatch):
    """Inject a fake ``sentence_transformers`` module into sys.modules."""
    import types

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


class TestSentenceTransformerEmbedder:
    def test_constructs_with_stub(self, stub_module):
        e = SentenceTransformerEmbedder()
        assert e.model_name == "BAAI/bge-small-en-v1.5"
        assert e.dimension == 8

    def test_dimension_from_model(self, stub_module):
        e = SentenceTransformerEmbedder()
        assert e.dimension == e._model.get_sentence_embedding_dimension()

    def test_query_instruction_logic(self, stub_module):
        e = SentenceTransformerEmbedder(instruction="SEARCH: ")
        e.embed_query("hello")
        e.embed_documents(["hello"])

        query_call = e._model.calls[0][0]
        doc_call = e._model.calls[1][0]
        assert query_call == ["SEARCH: hello"]
        assert doc_call == ["hello"]

    def test_query_instruction_default(self, stub_module):
        e = SentenceTransformerEmbedder()
        e.embed_query("hello")
        assert e._model.calls[0][0] == [
            "Represent this sentence for searching relevant passages: hello"
        ]

    def test_no_instruction(self, stub_module):
        e = SentenceTransformerEmbedder(instruction="")
        e.embed_query("hello")
        assert e._model.calls[0][0] == ["hello"]

    def test_batching(self, stub_module):
        e = SentenceTransformerEmbedder(batch_size=2)
        e.embed_documents(["a", "b", "c", "d", "e"])
        assert len(e._model.calls) == 3  # 2 + 2 + 1
        assert e._model.calls[0][0] == ["a", "b"]
        assert e._model.calls[2][0] == ["e"]

    def test_embed_documents_empty(self, stub_module):
        e = SentenceTransformerEmbedder()
        assert e.embed_documents([]) == []

    def test_model_name_override(self, stub_module):
        e = SentenceTransformerEmbedder(model_name="other-model")
        assert e.model_name == "other-model"
        assert e._model.name == "other-model"

    def test_missing_extra_raises_clear_error(self, monkeypatch):
        """Without sentence-transformers the constructor must fail with guidance."""
        blocker = ModuleBlocker("sentence_transformers")
        monkeypatch.setitem(sys.modules, "sentence_transformers", blocker)
        with pytest.raises(RuntimeError, match="sentence-transformers is required"):
            SentenceTransformerEmbedder()

    def test_normalise_embeddings_passed_through(self, stub_module):
        e = SentenceTransformerEmbedder(normalize_embeddings=True)
        e.embed_query("x")
        assert e._model.calls[0][1]["normalize_embeddings"] is True


class ModuleBlocker:
    def __init__(self, name: str):
        self.__name__ = name

    def __getattr__(self, item: str):
        raise ImportError(
            "sentence-transformers is required for the real embedder; "
            'install with: python -m pip install -e ".[embeddings]"'
        )
