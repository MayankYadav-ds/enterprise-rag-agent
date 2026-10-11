"""Tests for the embeddings package (FakeEmbedder, protocol, lazy import)."""

from __future__ import annotations

import subprocess
import sys

import pytest

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
        """Loading the embeddings package must not import sentence_transformers.

        Checked in a subprocess so the test is order-independent: it does not
        depend on whether another test imported sentence-transformers first.
        """
        code = (
            "import rag_agent.embeddings; "
            "import sys; "
            "assert 'sentence_transformers' not in sys.modules; "
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_missing_sentence_transformers_raises_clear_error(self, monkeypatch):
        """Constructing the real embedder without sentence_transformers raises
        the clear install error. monkeypatch.setitem blocks the import without
        touching sys.modules directly."""
        from rag_agent.embeddings.sentence_transformer import SentenceTransformerEmbedder

        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(RuntimeError, match="sentence-transformers is required"):
            SentenceTransformerEmbedder()
