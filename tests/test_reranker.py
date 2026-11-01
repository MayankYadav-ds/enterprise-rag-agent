"""Unit tests for the cross‑encoder reranker.

These tests focus on the behaviour expected from the reranker implementation.
"""

from __future__ import annotations

import builtins
import os
import sys
import types

import pytest

from rag_agent.retrieval.models import RetrievedChunk
from rag_agent.retrieval.protocol import Retriever
from rag_agent.retrieval.reranker import (
    CrossEncoderReranker,
    CrossEncoderUnavailable,
    FakeCrossEncoder,
    RerankerConfig,
    build_reranked_retriever,
)


@pytest.fixture(autouse=True)
def _clear_reranker_environment(monkeypatch):
    for key in os.environ:
        if key.startswith("RERANKER_"):
            monkeypatch.delenv(key)


class MockRetriever(Retriever):
    def __init__(self, hits):
        self._hits = hits

    def retrieve(
        self,
        query: str,
        /,
        *,
        top_k: int = 5,
        doc_id: str | None = None,
        chunk_type: str | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        return self._hits[:top_k]


def make_hits(scores):
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            doc_id="doc",
            source_file="file",
            page_start=1,
            page_end=2,
            section_heading=None,
            subheading=None,
            chunk_type="text",
            text=f"txt_{i}",
            score=s,
            rank=i + 1,
            retriever="dense",
        )
        for i, s in enumerate(scores)
    ]


# ---------- tests ----------


def test_disabled_reranking_returns_original_retriever_without_model_import(monkeypatch):
    hits = make_hits([0.1, 0.2])
    retriever = MockRetriever(hits)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            pytest.fail("Disabled reranking must not import the model dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setenv("RERANKER_TOP_K", "invalid-unused-environment")
    wrapped = build_reranked_retriever(retriever, enable=False)
    assert wrapped is retriever
    assert wrapped.retrieve("test", top_k=2) == hits


def test_injected_model_no_download():
    fake = FakeCrossEncoder()
    hits = make_hits([1, 2])
    retriever = MockRetriever(hits)
    reranker = CrossEncoderReranker(retriever, cross_encoder=fake)
    results = reranker.retrieve("test")
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_scores_ordering():
    fake = FakeCrossEncoder()
    hits = make_hits([1, 5])
    retriever = MockRetriever(hits)
    reranker = CrossEncoderReranker(retriever, cross_encoder=fake)
    ordered = reranker.retrieve("q")
    assert len(ordered) == 2
    assert set(h.score for h in ordered) == {142.85714285714286}


def test_tie_breaker_deterministic():
    hits = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="doc",
            source_file="file",
            page_start=1,
            page_end=1,
            section_heading=None,
            subheading=None,
            chunk_type="text",
            text="abc",
            score=0.0,
            rank=1,
            retriever="dense",
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="doc",
            source_file="file",
            page_start=1,
            page_end=1,
            section_heading=None,
            subheading=None,
            chunk_type="text",
            text="abc",
            score=0.0,
            rank=2,
            retriever="dense",
        ),
    ]
    retriever = MockRetriever(hits)
    reranker = CrossEncoderReranker(retriever, cross_encoder=FakeCrossEncoder())
    result = reranker.retrieve("q")
    assert result[0].chunk_id == "c1"


def test_empty_candidates_returns_empty():
    retriever = MockRetriever([])
    reranker = CrossEncoderReranker(retriever, cross_encoder=FakeCrossEncoder())
    assert reranker.retrieve("q") == []


def test_missing_text_is_treated_as_empty():
    hit = RetrievedChunk(
        chunk_id="c1",
        doc_id="doc",
        source_file="file",
        page_start=1,
        page_end=1,
        section_heading=None,
        subheading=None,
        chunk_type="text",
        text=None,
        score=0.0,
        rank=1,
        retriever="dense",
    )
    retriever = MockRetriever([hit])
    reranker = CrossEncoderReranker(retriever, cross_encoder=FakeCrossEncoder())
    res = reranker.retrieve("q")
    assert res[0].score is not None


def test_metadata_preserved():
    hit = RetrievedChunk(
        chunk_id="c1",
        doc_id="doc",
        source_file="file",
        page_start=10,
        page_end=20,
        section_heading="sec",
        subheading="subchunk",
        chunk_type="text",
        text="data",
        score=0.9,
        rank=1,
        retriever="dense",
    )
    retriever = MockRetriever([hit])
    reranker = CrossEncoderReranker(retriever, cross_encoder=FakeCrossEncoder())
    r = reranker.retrieve("q")[0]
    assert r.chunk_id == hit.chunk_id
    assert r.doc_id == hit.doc_id
    assert r.source_file == hit.source_file
    assert r.page_start == hit.page_start
    assert r.page_end == hit.page_end
    assert r.section_heading == hit.section_heading
    assert r.subheading == hit.subheading
    assert r.chunk_type == hit.chunk_type


def test_invalid_config_rejection():
    with pytest.raises(ValueError):
        RerankerConfig(top_k=0)
    with pytest.raises(ValueError):
        RerankerConfig(candidate_count=0)
    with pytest.raises(ValueError):
        RerankerConfig(batch_size=0)


def test_model_loading_failure_produces_error(monkeypatch):
    dummy = types.ModuleType("sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", dummy)
    with pytest.raises(CrossEncoderUnavailable):
        r = CrossEncoderReranker(MockRetriever([]))
        r._load_model()


def test_score_count_mismatch():
    class BadCrossEncoder(FakeCrossEncoder):
        def predict(self, pairs, **kwargs):
            return [1.0]

    fake = BadCrossEncoder()
    retriever = MockRetriever(make_hits([0.5, 0.6]))
    reranker = CrossEncoderReranker(retriever, cross_encoder=fake)
    with pytest.raises(ValueError):
        reranker.retrieve("q")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("min_score", [None, 9999.0])
def test_non_finite_scores_are_rejected_before_filtering(score, min_score):
    class BadCrossEncoder(FakeCrossEncoder):
        def predict(self, pairs, **kwargs):
            return [score]

    reranker = CrossEncoderReranker(
        MockRetriever(make_hits([0.0])),
        config=RerankerConfig(min_score=min_score),
        cross_encoder=BadCrossEncoder(),
    )
    with pytest.raises(ValueError, match="non-finite score .* for chunk 'c0'"):
        reranker.retrieve("q")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_later_batch_does_not_return_partial_top_k(score):
    class BadCrossEncoder(FakeCrossEncoder):
        def __init__(self):
            self.calls = 0

        def predict(self, pairs, **kwargs):
            self.calls += 1
            return [1.0 if self.calls == 1 else score]

    model = BadCrossEncoder()
    reranker = CrossEncoderReranker(
        MockRetriever(make_hits([0.0, 0.0])),
        config=RerankerConfig(top_k=1, candidate_count=2, batch_size=1),
        cross_encoder=model,
    )
    with pytest.raises(ValueError, match="non-finite score .* for chunk 'c1'"):
        reranker.retrieve("q")
    assert model.calls == 2


def test_min_score_filtering():
    fake = FakeCrossEncoder()
    hits = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="doc",
            source_file="file",
            page_start=1,
            page_end=1,
            section_heading=None,
            subheading=None,
            chunk_type="text",
            text="short",
            score=0.0,
            rank=1,
            retriever="dense",
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="doc",
            source_file="file",
            page_start=1,
            page_end=1,
            section_heading=None,
            subheading=None,
            chunk_type="text",
            text="longer for scoring",
            score=0.0,
            rank=2,
            retriever="dense",
        ),
    ]
    retriever = MockRetriever(hits)
    cfg = RerankerConfig(min_score=9999)
    reranker = CrossEncoderReranker(retriever, config=cfg, cross_encoder=fake)
    assert reranker.retrieve("q") == []


def test_model_name_resolution(monkeypatch):
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    cfg = RerankerConfig.from_env()
    assert cfg.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    monkeypatch.setenv("RERANKER_MODEL", "custom/model")
    cfg2 = RerankerConfig.from_env()
    assert cfg2.model_name == "custom/model"


def test_environment_only_configuration(monkeypatch):
    values = {
        "RERANKER_MODEL": "environment/model",
        "RERANKER_TOP_K": "3",
        "RERANKER_CANDIDATE_COUNT": "12",
        "RERANKER_DEVICE": "cuda",
        "RERANKER_BATCH_SIZE": "4",
        "RERANKER_MIN_SCORE": "-2.5",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    assert RerankerConfig.from_env() == RerankerConfig(
        model_name="environment/model",
        top_k=3,
        candidate_count=12,
        device="cuda",
        batch_size=4,
        min_score=-2.5,
    )


def test_explicit_overrides_take_precedence_over_invalid_environment(monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", "environment/model")
    monkeypatch.setenv("RERANKER_DEVICE", "cuda")
    for key in ("TOP_K", "CANDIDATE_COUNT", "BATCH_SIZE", "MIN_SCORE"):
        monkeypatch.setenv(f"RERANKER_{key}", "invalid-overridden-value")

    config = RerankerConfig.from_env(
        model_name="explicit/model",
        top_k=2,
        candidate_count=7,
        device="cpu",
        batch_size=3,
        min_score=None,
    )
    assert config == RerankerConfig(
        model_name="explicit/model",
        top_k=2,
        candidate_count=7,
        device="cpu",
        batch_size=3,
        min_score=None,
    )


@pytest.mark.parametrize("factory", [RerankerConfig, RerankerConfig.from_env])
@pytest.mark.parametrize(
    ("explicit", "environment", "expected"),
    [
        (None, None, "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        (None, "environment/model", "environment/model"),
        ("explicit/model", "environment/model", "explicit/model"),
        ("", "environment/model", "environment/model"),
        ("  ", " ", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ],
)
def test_model_name_precedence(factory, explicit, environment, expected, monkeypatch):
    if environment is not None:
        monkeypatch.setenv("RERANKER_MODEL", environment)
    config = factory(model_name=explicit)
    assert config.model_name == expected


@pytest.mark.parametrize("factory", [RerankerConfig, RerankerConfig.from_env])
@pytest.mark.parametrize(
    "settings",
    [
        {"top_k": 0},
        {"top_k": 1.5},
        {"top_k": "2"},
        {"candidate_count": 4},
        {"candidate_count": -1},
        {"batch_size": 0},
        {"batch_size": None},
        {"model_name": 123},
    ],
)
def test_invalid_direct_and_override_settings_remain_rejected(factory, settings):
    with pytest.raises(ValueError):
        factory(**settings)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("RERANKER_TOP_K", "not-an-integer"),
        ("RERANKER_TOP_K", "0"),
        ("RERANKER_CANDIDATE_COUNT", "4"),
        ("RERANKER_BATCH_SIZE", "0"),
        ("RERANKER_MIN_SCORE", "not-a-number"),
    ],
)
def test_invalid_environment_settings_remain_rejected(key, value, monkeypatch):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        RerankerConfig.from_env()


@pytest.mark.parametrize(
    ("explicit", "environment", "expected"),
    [
        (None, None, "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        (None, "environment/model", "environment/model"),
        ("explicit/model", "environment/model", "explicit/model"),
    ],
)
def test_resolved_model_is_loaded_lazily_once(explicit, environment, expected, monkeypatch):
    if environment is not None:
        monkeypatch.setenv("RERANKER_MODEL", environment)
    calls = []

    class StubCrossEncoder(FakeCrossEncoder):
        def __init__(self, model_name, **kwargs):
            super().__init__()
            calls.append((model_name, kwargs))

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = StubCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    base = MockRetriever(make_hits([0.1]))
    config = RerankerConfig(model_name=explicit)
    reranker = build_reranked_retriever(base, enable=True, config=config)

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.config is config
    assert reranker.base_retriever is base
    assert calls == []
    # The recorded name and actual loaded name must remain identical even if
    # the environment changes after the configuration was constructed.
    monkeypatch.setenv("RERANKER_MODEL", "later/model")
    reranker.retrieve("q")
    reranker.retrieve("q")
    assert calls == [(expected, {"device": "cpu", "max_length": 512})]


def test_empty_candidates_do_not_load_model(monkeypatch):
    def unexpected_load(self):
        pytest.fail("An empty candidate list must not load a model")

    monkeypatch.setattr(CrossEncoderReranker, "_load_model", unexpected_load)
    assert CrossEncoderReranker(MockRetriever([])).retrieve("q") == []
