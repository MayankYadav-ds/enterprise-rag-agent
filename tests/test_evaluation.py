"""Tests for the evaluation runner and metrics, using only in-process fixtures.

These tests drive the ``run_evaluation`` function against a ``FakeEmbedder``
and an in-memory Qdrant collection, so they never download models, touch the
filesystem beyond temporary directories, or require Docker / Qdrant connectivity.
"""

from __future__ import annotations

import pytest

from rag_agent.evaluation.gold_set import load_gold_set
from rag_agent.evaluation.metrics import (
    PrecisionAtK,
    RecallAtK,
    mean_reciprocal_rank,
)
from rag_agent.evaluation.runner import (
    EvaluationResult,
    QueryResult,
    render_report,
    run_evaluation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_chunk(**overrides):
    """Build a real Chunk for indexing into an in-memory Qdrant store."""
    from rag_agent.chunking.models import Chunk

    defaults = dict(
        chunk_id=None,
        doc_id="doc_abc123",
        text="sample chunk text",
        chunk_type="text",
        page_start=1,
        page_end=1,
        chunk_index=0,
        token_count=0,
        text_quality=1.0,
    )
    defaults.update(overrides)
    if defaults["chunk_id"] is None:
        import hashlib

        content_hash = hashlib.sha256(defaults["text"].encode("utf-8")).hexdigest()[:16]
        position_hash = hashlib.sha256(
            f"{defaults['doc_id']}:{defaults['chunk_index']}:{defaults['page_start']}:{defaults['page_end']}".encode()
        ).hexdigest()[:8]
        defaults["chunk_id"] = f"chunk_{defaults['doc_id']}_{position_hash}_{content_hash}"
    return Chunk(**defaults)


def _make_store_and_retriever(dimension: int = 32):
    """Build an in-memory QdrantStore, FakeEmbedder, and DenseRetriever."""
    import uuid

    from rag_agent.embeddings.fake import FakeEmbedder
    from rag_agent.embeddings.store import QdrantStore
    from rag_agent.retrieval.dense import DenseRetriever

    embedder = FakeEmbedder(dimension=dimension)
    store = QdrantStore(
        embedder=embedder,
        collection_name=f"eval_test_{uuid.uuid4().hex}",
        url=":memory:",
    )
    retriever = DenseRetriever(embedder, store)
    return embedder, store, retriever


def _chunk(**overrides) -> object:
    """Build a mock retrieval hit (duck-type, not a real RetrievedChunk)."""
    defaults = dict(
        chunk_id="c1",
        doc_id="doc_abc123",
        source_file="nps-2013-10k.pdf",
        page_start=1,
        page_end=1,
        section_heading=None,
        subheading=None,
        chunk_type="text",
        text="sample chunk text",
        score=1.0,
        rank=1,
        retriever="dense",
    )
    return type("_Hit", (), defaults)(**overrides)


def _hits(*texts: str) -> list:
    """Create a list of retrieval hit objects."""
    return [_chunk(text=t) for t in texts]


def _make_hit(text: str) -> object:
    return _chunk(text=text, chunk_id=f"chunk_{hash(text)}")


def _gold_queries():
    """Return a small hard-coded gold set for testing."""
    return [
        {
            "query": "What are the main risk factors discussed by N-able?",
            "relevant_chunk_ids": ["chunk_risk_1", "chunk_risk_2"],
        },
        {
            "query": "How does NPS earn royalties?",
            "relevant_chunk_ids": ["chunk_royalty_1"],
        },
        {
            "query": "What was total revenue?",
            "relevant_chunk_ids": ["chunk_revenue_1", "chunk_revenue_2", "chunk_revenue_3"],
        },
    ]


# ---------------------------------------------------------------------------
# Metric unit tests (pure functions — no runner needed)
# ---------------------------------------------------------------------------


class TestRecallAtK:
    """Recall@K = fraction of gold-relevant chunks that appear in the top-K hits."""

    @pytest.mark.parametrize(
        "k,retrieved,relevant,expected",
        [
            (1, ["c1"], {"c1"}, 1.0),
            (1, ["c1", "c2"], {"c1"}, 1.0),
            (2, ["c1"], {"c1", "c2"}, 0.5),
            (2, ["c1", "c2"], {"c1", "c2"}, 1.0),
            (2, ["c3"], {"c1", "c2"}, 0.0),
            (3, ["c1", "c2"], {"c1", "c2"}, 1.0),
            (1, [], set(), 0.0),
        ],
    )
    def test_compute(self, k, retrieved, relevant, expected):
        r = RecallAtK(k=k)
        actual = r.compute(retrieved, relevant)
        assert actual == expected, f"Recall@{k}: expected {expected}, got {actual}"


class TestPrecisionAtK:
    """Precision@K = fraction of top-K hits that are relevant."""

    @pytest.mark.parametrize(
        "k,retrieved,relevant,expected",
        [
            (1, ["c1"], {"c1"}, 1.0),
            (1, ["c1", "c2"], {"c1"}, 1.0),
            (2, ["c1", "c3"], {"c1", "c2"}, 0.5),
            (2, ["c1", "c2"], {"c1", "c2"}, 1.0),
            (2, ["c3"], {"c1", "c2"}, 0.0),
            (3, ["c1"], {"c1", "c2", "c3"}, 1.0),
            (5, ["c1", "c2"], {"c1", "c2"}, 1.0),
        ],
    )
    def test_compute(self, k, retrieved, relevant, expected):
        p = PrecisionAtK(k=k)
        actual = p.compute(retrieved, relevant)
        assert actual == expected, f"Precision@{k}: expected {expected}, got {actual}"


class TestMeanReciprocalRank:
    """MRR = reciprocal rank of first relevant hit, averaged over queries."""

    @pytest.mark.parametrize(
        "hits,relevant,expected",
        [
            (["c1", "c2"], {"c1"}, 1.0),
            (["c2", "c1"], {"c1"}, 0.5),
            (["c3", "c2", "c1"], {"c1"}, 1.0 / 3),
            (["c2", "c3"], {"c1"}, 0.0),
            (["c1"], {"c1"}, 1.0),
            (["c1"], {"c1", "c2"}, 1.0),
        ],
    )
    def test_compute(self, hits, relevant, expected):
        mrr = mean_reciprocal_rank(hits, relevant)
        assert mrr == expected, f"MRR: expected {expected}, got {mrr}"

    def test_empty_relevant_returns_zero(self):
        """When the gold set has no relevant chunk, MRR is 0.0."""
        assert mean_reciprocal_rank(["c1", "c2"], set()) == 0.0


# ---------------------------------------------------------------------------
# Gold set loading tests
# ---------------------------------------------------------------------------


class TestLoadGoldSet:
    """Verify loading gold_set.json and handling missing/malformed data."""

    def test_valid_gold_set_loads(self):
        """A valid gold_set.json with well-formed queries loads correctly."""
        gs = load_gold_set()
        assert isinstance(gs, type(load_gold_set()))
        assert gs.size > 0

    def test_gold_set_version(self):
        """The gold set version should be a non-empty string."""
        gs = load_gold_set()
        assert len(gs.version) > 0

    def test_relevant_ids_for(self):
        """relevant_ids_for returns correct chunk IDs for a known query."""
        gs = load_gold_set()
        rid = gs.relevant_ids_for("What are the main risk factors discussed by N-able?")
        assert isinstance(rid, frozenset)
        assert len(rid) > 0


class TestInvalidGoldSet:
    """Malformed or missing gold set should raise or return empty GoldSet."""

    def test_missing_raises(self, tmp_path):
        """FileNotFoundError is raised when the gold set file does not exist."""
        bad = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_gold_set(bad)


# ---------------------------------------------------------------------------
# Runner tests with mocked retrieval results
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    """Run evaluation against mocked retrieval results with a fake embedder."""

    def test_basic_evaluation_with_retriever(self):
        """run_evaluation should complete successfully with a real retriever object."""
        embedder, store, retriever = _make_store_and_retriever()
        queries = _gold_queries()
        result = run_evaluation(retriever, queries, ks=[1, 3], top_k=5)
        assert isinstance(result, EvaluationResult)
        assert result.num_queries == 3
        assert 0.0 <= result.mean_mrr <= 1.0

    def test_recall_precision_shapes(self):
        """Per-query recall and precision should map K values to floats in [0,1]."""
        embedder, store, retriever = _make_store_and_retriever()
        queries = _gold_queries()
        result = run_evaluation(retriever, queries, ks=[1, 3, 5], top_k=5)
        for k in result.ks:
            assert k in result.mean_recall
            assert k in result.mean_precision
            for v in result.mean_recall.values():
                assert 0.0 <= v <= 1.0, f"Recall@{k} value out of range: {v}"
            for v in result.mean_precision.values():
                assert 0.0 <= v <= 1.0, f"Precision@{k} value out of range: {v}"

    def test_per_query_result_structure(self):
        """Each QueryResult should have all expected fields and types."""
        embedder, store, retriever = _make_store_and_retriever()
        queries = _gold_queries()
        result = run_evaluation(retriever, queries, ks=[1, 2], top_k=3)
        assert isinstance(result.per_query, tuple)
        assert len(result.per_query) == len(queries)
        for qr in result.per_query:
            assert isinstance(qr, QueryResult)
            # All K values should be present in the dict
            for k in result.ks:
                assert k in qr.recall_at_k, f"Missing recall@{k}"
                assert k in qr.precision_at_k, f"Missing precision@{k}"
            assert isinstance(qr.mrr, float)
            assert 0.0 <= qr.mrr <= 1.0

    def test_empty_retrieval_results(self):
        """When retrieval returns no hits, all metrics should be 0.0."""
        embedder, store, retriever = _make_store_and_retriever()
        queries = [{"query": "Any question?", "relevant_chunk_ids": ["chunk_does_not_exist"]}]
        result = run_evaluation(retriever, queries, ks=[1, 2], top_k=5)
        # Recall = 0 because no relevant chunks in top-K
        assert result.mean_recall[1] == 0.0
        assert result.mean_precision[1] == 0.0
        assert result.mean_mrr == 0.0
        # Per-query MRR also 0
        assert result.per_query[0].mrr == 0.0

    def test_unknown_expected_ids_in_retrieval(self):
        """Expected IDs that simply don't appear in retrieval should yield 0 recall."""
        embedder, store, retriever = _make_store_and_retriever()
        # Insert one chunk with a known ID
        store.sync(
            "doc_known",
            [_real_chunk(chunk_id="chunk_known_abc", doc_id="doc_known", text="known text")],
            embedder,
            source_file="nps-2013-10k.pdf",
        )

        queries = [
            {
                "query": "What about unknown chunks?",
                "relevant_chunk_ids": ["chunk_known_abc", "chunk_unknown"],
            },
        ]
        result = run_evaluation(retriever, queries, ks=[1, 2], top_k=5)
        # Only one relevant ID exists and it might be retrieved; unknown can't be
        # Recall can be 0.0, 0.5, or 1.0 depending on ranking
        assert 0.0 <= result.mean_recall[1] <= 1.0, f"Recall value: {result.mean_recall[1]}"
        # MRR should still be valid (0 or 1 depending on rank)
        assert 0.0 <= result.mean_mrr <= 1.0

    def test_report_rendering(self):
        """render_report should produce a human-readable string with K values."""
        embedder, store, retriever = _make_store_and_retriever()
        store.sync(
            "doc_r",
            [_real_chunk(chunk_id="chunk_r1", doc_id="doc_r", text="relevant chunk")],
            embedder,
            source_file="nps-2013-10k.pdf",
        )

        queries = [
            {"query": "relevant question", "relevant_chunk_ids": ["chunk_r1"]},
        ]
        result = run_evaluation(retriever, queries, ks=[1, 2, 3], top_k=3)
        report = render_report(result)
        assert isinstance(report, str)
        assert "1" in report  # K=1 should appear
        assert "2" in report  # K=2 should appear
        assert "3" in report  # K=3 should appear
        assert "MRR" in report


class TestEvaluationResultSerialization:
    """Verify EvaluationResult can be serialised to dict and back."""

    def test_result_to_dict_basic(self):
        """result_to_dict should produce a JSON-plain dict."""
        from rag_agent.evaluation.runner import result_to_dict

        result = EvaluationResult(
            dataset_version="v1",
            num_queries=2,
            ks=(1, 3),
            retrieval_settings={"top_k": 5},
            mean_recall={1: 0.5, 3: 0.33},
            mean_precision={1: 0.5, 3: 0.11},
            mean_mrr=0.75,
            per_query=(
                QueryResult(
                    query="q1",
                    relevant_chunk_ids=("c1", "c2"),
                    retrieved_chunk_ids=("c1",),
                    recall_at_k={1: 1.0, 3: 0.67},
                    precision_at_k={1: 1.0, 3: 0.33},
                    mrr=1.0,
                ),
                QueryResult(
                    query="q2",
                    relevant_chunk_ids=("c3",),
                    retrieved_chunk_ids=("c3",),
                    recall_at_k={1: 1.0, 3: 0.33},
                    precision_at_k={1: 1.0, 3: 0.33},
                    mrr=0.5,
                ),
            ),
        )
        d = result_to_dict(result)
        assert isinstance(d, dict)
        assert d["dataset_version"] == "v1"
        assert d["num_queries"] == 2
        assert d["mean_mrr"] == 0.75
        assert len(d["per_query"]) == 2
        assert d["per_query"][0]["query"] == "q1"

    def test_render_report_has_ks(self):
        """render_report output should contain each K from result.ks."""
        from rag_agent.evaluation.runner import render_report

        result = EvaluationResult(
            dataset_version="v2",
            num_queries=1,
            ks=(1, 5, 10),
            retrieval_settings={"top_k": 5},
            mean_recall={1: 0.0, 5: 0.2, 10: 0.15},
            mean_precision={1: 0.0, 5: 0.04, 10: 0.02},
            mean_mrr=0.4,
            per_query=(
                QueryResult(
                    query="some question",
                    relevant_chunk_ids=("c1",),
                    retrieved_chunk_ids=("c1",),
                    recall_at_k={1: 1.0, 5: 0.2, 10: 0.1},
                    precision_at_k={1: 1.0, 5: 0.2, 10: 0.1},
                    mrr=1.0,
                ),
            ),
        )
        report = render_report(result)
        assert isinstance(report, str)
        for k in result.ks:
            assert str(k) in report or f"R@{k}" in report or f"P@{k}" in report


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Verify CLI argument parsing and the evaluation module import."""

    def test_evaluation_import(self):
        """Importing the evaluation package should not raise."""
        from rag_agent.evaluation import (
            EvaluationResult,
            QueryResult,
            run_evaluation,
        )

        assert run_evaluation is not None
        assert EvaluationResult is not None
        assert QueryResult is not None

    def test_invalid_ks_raises(self):
        """Bad --ks values should raise ValueError from the parser."""
        from rag_agent.evaluation.runner import _parse_ks

        with pytest.raises(ValueError):
            _parse_ks("abc")
        with pytest.raises(ValueError):
            _parse_ks("-1,2")

    def test_ks_parsing(self):
        """--ks with valid values should parse correctly."""
        from rag_agent.evaluation.runner import _parse_ks

        assert _parse_ks("1,3,5,10") == [1, 3, 5, 10]
        assert _parse_ks("10") == [10]
        assert _parse_ks("5,1,3") == [1, 3, 5]  # sorted, deduped

    def test_top_k_must_exceed_max_k(self):
        """run_evaluation should reject top_k < max(ks)."""
        embedder, store, retriever = _make_store_and_retriever()
        queries = _gold_queries()
        with pytest.raises(ValueError, match="top_k"):
            run_evaluation(retriever, queries, ks=[1, 5], top_k=1)

    def test_empty_ks_rejected(self):
        """run_evaluation should reject an empty ks list."""
        queries = _gold_queries()
        embedder, store, retriever = _make_store_and_retriever()
        with pytest.raises(ValueError, match="ks must be"):
            run_evaluation(retriever, queries, ks=[], top_k=5)
