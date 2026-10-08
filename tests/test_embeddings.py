"""Tests for the embeddings package (FakeEmbedder, protocol, lazy import)."""

from __future__ import annotations

import sys

from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.protocol import Embedder


class TestFakeEmbedder:
    def test_deterministic(self):
        e = FakeEmbedder(dimension=32)
        v1 = e.embed_documents(["hello world"])[0]
        v2 = e.embed_documents(["hello world"])[0]
        assert v1 == v2

    def test_different_texts_differ(self):
        e = FakeEmbedder(dimension=32)
        v1 = e.embed_documents(["alpha"])[0]
        v2 = e.embed_documents(["beta"])[0]
        assert v1 != v2

    def test_dimension(self):
        e = FakeEmbedder(dimension=64)
        v = e.embed_documents(["x"])[0]
        assert len(v) == 64

    def test_embed_documents_counter(self):
        e = FakeEmbedder()
        e.embed_documents(["a", "b", "c"])
        assert e.embed_counter == 3

    def test_embed_query_counter(self):
        e = FakeEmbedder()
        e.embed_query("search text")
        assert e.embed_counter == 1

    def test_query_returns_single_vector(self):
        e = FakeEmbedder()
        v = e.embed_query("search text")
        assert len(v) == e.dimension

    def test_model_name(self):
        e = FakeEmbedder()
        assert e.model_name == "fake"

    def test_protocol_runtime_check(self):
        assert isinstance(FakeEmbedder(), Embedder)
        assert not isinstance("not an embedder", Embedder)


class TestLazyImport:
    def test_embeddings_package_loads_without_sentence_transformers(self):
        """Loading the embeddings package must not import sentence_transformers."""
        # sentence-transformers is not a dependency of the base package
        assert "sentence_transformers" not in sys.modules
        import rag_agent.embeddings  # noqa: F401

        assert "sentence_transformers" not in sys.modules

    def test_missing_sentence_transformers_raises_clear_error(self):
        """If sentence_transformers is removed from sys.modules then a
        real embedder would raise — FakeEmbedder still works without it."""
        blocker = ModuleBlocker("sentence_transformers")
        sys.modules["sentence_transformers"] = blocker
        try:
            from rag_agent.embeddings.fake import FakeEmbedder  # noqa: F811

            e = FakeEmbedder()
            assert e.model_name == "fake"
        finally:
            sys.modules.pop("sentence_transformers", None)


class ModuleBlocker:
    def __init__(self, name: str):
        self.__name__ = name

    def __getattr__(self, item: str):
        raise ImportError(
            "sentence-transformers is required for the real embedder; "
            'install with: python -m pip install -e ".[embeddings]"'
        )
