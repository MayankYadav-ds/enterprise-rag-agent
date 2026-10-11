"""Tests for model-token counting with a fake tokenizer."""

from __future__ import annotations

import pytest

from rag_agent.embeddings.token_counter import count_model_tokens, model_token_stats


class FakeTokenizer:
    """Tokenizer that maps each ASCII character to one token."""

    def encode(self, text: str) -> FakeEncoding:
        return FakeEncoding(list(text.encode("utf-8")))


class FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class TestCountModelTokens:
    def test_counts(self):
        assert count_model_tokens(["abc", "de"], FakeTokenizer()) == [3, 2]

    def test_empty(self):
        assert count_model_tokens([], FakeTokenizer()) == []

    def test_hf_style_input_ids(self):
        """Falls back to encoding['input_ids'] when .ids is absent."""

        # A tokenizer whose encode() returns a dict with input_ids
        class DictTokenizer:
            def encode(self, text: str):
                return {"input_ids": list(text.encode("utf-8"))}

        assert count_model_tokens(["abc"], DictTokenizer()) == [3]


class TestModelTokenStats:
    def test_empty(self):
        s = model_token_stats([], FakeTokenizer())
        assert s["chunk_count"] == 0
        assert s["over_limit"] == 0
        assert s["over_limit_pct"] == 0.0
        assert s["max_tokens"] == 512

    def test_over_limit(self):
        s = model_token_stats(["abc", "de", "f" * 600], FakeTokenizer())
        assert s["chunk_count"] == 3
        assert s["over_limit"] == 1  # "f" * 600 > 512
        assert s["over_limit_pct"] == pytest.approx(100.0 / 3, abs=0.1)
        assert s["p99"] == 600

    def test_median_and_p95(self):
        texts = ["x" * i for i in range(1, 101)]
        s = model_token_stats(texts, FakeTokenizer())
        assert s["median"] == 51
        assert s["p95"] == 96
