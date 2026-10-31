"""Tests for the batch query path in ``scripts/search.py``.

``run_queries`` runs a whole queries file through one retriever and prints
``===== <query>`` before each query's hits. It is unit-tested with a
FakeEmbedder and an in-memory Qdrant collection, so the batch path is
covered without a model download — the same way the single-query path is
covered by ``tests/test_dense.py``.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import pytest

from rag_agent.chunking.models import Chunk
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.store import QdrantStore
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.hybrid import HybridRetriever
from rag_agent.retrieval.reranker import CrossEncoderReranker

# scripts/ is not a package; load search.py by path so the batch query path is
# unit-tested the same way the single-query path is.
_SEARCH_PATH = Path(__file__).resolve().parent.parent / "scripts" / "search.py"
_spec = importlib.util.spec_from_file_location("rag_agent_scripts_search", _SEARCH_PATH)
assert _spec is not None and _spec.loader is not None
search = importlib.util.module_from_spec(_spec)
sys.modules["rag_agent_scripts_search"] = search
_spec.loader.exec_module(search)

_read_queries_file = search._read_queries_file
run_queries = search.run_queries


def _chunk(**overrides) -> Chunk:
    defaults = dict(
        doc_id="doc_cli",
        text="sample chunk text",
        chunk_type="text",
        page_start=1,
        page_end=1,
        chunk_index=0,
    )
    defaults.update(overrides)
    return Chunk.create(**defaults)


@pytest.fixture
def retriever() -> DenseRetriever:
    store = QdrantStore(
        embedder=FakeEmbedder(dimension=32),
        collection_name=f"cli_test_{uuid.uuid4().hex}",
        url=":memory:",
    )
    chunks = [
        _chunk(chunk_index=0, text="the company earned one point two billion in revenue"),
        _chunk(chunk_index=1, text="risk factors include competition and regulation"),
        _chunk(chunk_index=2, text="unrelated text about gardening and soil"),
    ]
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    for doc_id, doc_chunks in by_doc.items():
        store.sync(doc_id, doc_chunks, store.embedder, source_file="nps-2013-10k.pdf")
    return DenseRetriever(store.embedder, store)


def _write_queries(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "queries.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestReadQueriesFile:
    def test_reads_one_query_per_line(self, tmp_path):
        path = _write_queries(tmp_path, ["What was revenue?", "What are risks?"])
        assert _read_queries_file(path) == ["What was revenue?", "What are risks?"]

    def test_ignores_blank_lines(self, tmp_path):
        path = _write_queries(tmp_path, ["What was revenue?", "", "   ", "What are risks?"])
        assert _read_queries_file(path) == ["What was revenue?", "What are risks?"]

    def test_ignores_comment_lines(self, tmp_path):
        path = _write_queries(
            tmp_path,
            ["# smoke queries", "What was revenue?", "# trailing note", "What are risks?"],
        )
        assert _read_queries_file(path) == ["What was revenue?", "What are risks?"]

    def test_strips_whitespace(self, tmp_path):
        path = _write_queries(tmp_path, ["   What was revenue?   "])
        assert _read_queries_file(path) == ["What was revenue?"]

    def test_empty_file_raises(self, tmp_path):
        path = _write_queries(tmp_path, ["", "  ", "# only comments"])
        with pytest.raises(ValueError):
            _read_queries_file(path)


class TestRunQueries:
    def test_prints_header_per_query(self, retriever, capsys):
        exit_code = run_queries(retriever, ["What was revenue?", "What are risks?"])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert out.count("=====") == 2
        assert "What was revenue?" in out
        assert "What are risks?" in out

    def test_one_embedder_for_the_whole_file(self, retriever, capsys, monkeypatch):
        # The embedder is constructed once, so embed_query is called once per
        # query, not once per container run.
        calls = {"embed": 0}
        original = retriever.embedder.embed_query

        def counting_embed_query(text: str) -> list[float]:
            calls["embed"] += 1
            return original(text)

        monkeypatch.setattr(retriever.embedder, "embed_query", counting_embed_query)
        run_queries(retriever, ["What was revenue?", "What are risks?"])
        assert calls["embed"] == 2

    def test_no_hits_returns_exit_one(self, retriever, capsys):
        # An empty in-memory collection returns no hits for any query, so the
        # batch path must surface a non-zero exit code.
        empty_store = QdrantStore(
            embedder=FakeEmbedder(dimension=32),
            collection_name=f"empty_{uuid.uuid4().hex}",
            url=":memory:",
        )
        empty_retriever = DenseRetriever(empty_store.embedder, empty_store)
        exit_code = run_queries(empty_retriever, ["nothing here"])
        assert exit_code == 1

    def test_min_score_above_all_hits_returns_exit_one(self, retriever, capsys):
        # Every chunk on the fake collection scores ~0.7 against a query; a
        # threshold of 0.99 drops every hit and must surface exit code 1.
        exit_code = run_queries(
            retriever,
            ["What was revenue?"],
            min_score=0.99,
        )
        assert exit_code == 1

    def test_top_k_applies_to_every_query(self, retriever, capsys):
        run_queries(retriever, ["What was revenue?"], top_k=1)
        out = capsys.readouterr().out
        assert out.count("rank=1") == 1

    def test_filters_apply_to_every_query(self, retriever, capsys):
        run_queries(
            retriever,
            ["What was revenue?", "What are risks?"],
            chunk_type="text",
        )
        out = capsys.readouterr().out
        assert "rank=" in out

    def test_min_score_applies_to_every_query(self, retriever, capsys):
        run_queries(
            retriever,
            ["What was revenue?", "What are risks?"],
            min_score=0.7,
        )
        out = capsys.readouterr().out
        assert "score=" in out

    def test_empty_query_list_prints_nothing_returns_zero(self, retriever, capsys):
        exit_code = run_queries(retriever, [])
        assert exit_code == 0
        assert capsys.readouterr().out == ""

    def test_returns_worst_exit_code(self, retriever, capsys):
        # One query finds hits (exit 0), the thresholded one finds none (exit 1):
        # the batch returns the worst code, 1.
        exit_code = run_queries(
            retriever,
            ["What was revenue?", "What was revenue?"],
            min_score=0.0,
        )
        assert exit_code == 0


class TestRerankerCLI:
    def test_qdrant_url_environment_and_explicit_precedence(self, monkeypatch):
        calls = []

        class Store:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def scroll_payloads(self):
                return {}

        monkeypatch.setattr(search, "QdrantStore", Store)
        monkeypatch.setenv("QDRANT_URL", "http://configured.invalid:6333")
        monkeypatch.setattr(sys, "argv", ["search.py", "query", "--strategy", "bm25"])
        args = search._build_args()
        search._build_retriever(args, FakeEmbedder(dimension=32))
        assert calls[0]["url"] == "http://configured.invalid:6333"
        args.qdrant_url = ":memory:"
        search._build_retriever(args, FakeEmbedder(dimension=32))
        assert calls[1]["url"] == ":memory:"

    @pytest.mark.parametrize(
        ("cli_model", "env_model", "expected"),
        [
            (None, None, "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            (None, "environment/model", "environment/model"),
            ("explicit/model", "environment/model", "explicit/model"),
            ("explicit/model", None, "explicit/model"),
        ],
    )
    def test_model_precedence_and_lazy_construction(
        self, monkeypatch, cli_model, env_model, expected
    ):
        monkeypatch.delenv("RERANKER_MODEL", raising=False)
        if env_model is not None:
            monkeypatch.setenv("RERANKER_MODEL", env_model)
        monkeypatch.setenv("RERANKER_DEVICE", "cpu")
        monkeypatch.setenv("RERANKER_BATCH_SIZE", "7")
        argv = [
            "search.py",
            "query",
            "--strategy",
            "hybrid",
            "--rerank",
            "--top-k",
            "10",
            "--rerank-candidates",
            "25",
            "--qdrant-url",
            ":memory:",
            "--collection",
            f"rerank_cli_{uuid.uuid4().hex}",
        ]
        if cli_model is not None:
            argv.extend(["--rerank-model", cli_model])
        monkeypatch.setattr(sys, "argv", argv)

        def forbidden_load(self):
            pytest.fail("Building a search retriever must not load the cross-encoder")

        monkeypatch.setattr(CrossEncoderReranker, "_load_model", forbidden_load)
        result = search._build_retriever(search._build_args(), FakeEmbedder(dimension=32))
        assert isinstance(result, CrossEncoderReranker)
        assert isinstance(result.base_retriever, HybridRetriever)
        assert result.config.model_name == expected
        assert result.config.top_k == 10
        assert result.config.candidate_count == 25
        assert result.config.batch_size == 7

    def test_disabled_reranking_ignores_reranker_configuration(self, monkeypatch):
        monkeypatch.setenv("RERANKER_TOP_K", "invalid")
        monkeypatch.setenv("RERANKER_BATCH_SIZE", "invalid")
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "search.py",
                "query",
                "--strategy",
                "hybrid",
                "--qdrant-url",
                ":memory:",
                "--collection",
                f"disabled_cli_{uuid.uuid4().hex}",
            ],
        )
        result = search._build_retriever(search._build_args(), FakeEmbedder(dimension=32))
        assert isinstance(result, HybridRetriever)
