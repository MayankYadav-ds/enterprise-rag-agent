"""Count tokens the way the embedding model counts them.

Phase 3 sizes chunks in tiktoken (cl100k_base) tokens, but the embedding
model counts with its own WordPiece tokenizer and silently truncates beyond
its maximum length. Financial text with many numbers usually produces MORE
model tokens than tiktoken tokens, so a chunk sized to "512" tiktoken tokens
can be cut off mid-way and part of the text never gets embedded.

This module exposes a tokenizer-agnostic helper so the limit can be measured
against any tokenizer that exposes ``encode(text)`` returning an object with
``ids`` (the ``tokenizers.Tokenizer`` interface used by both
``sentence_transformers`` and Hugging Face tokenizers).

The real model can't be loaded on machines that block torch (e.g. Windows
machines with an Application Control policy), but the tokenizer itself can be
downloaded without torch: ``scripts/measure_token_limits.py`` uses
``tokenizers`` directly to measure the real WordPiece token counts.
"""

from __future__ import annotations

from typing import Any

from rag_agent.embeddings.sentence_transformer import model_max_tokens


def count_model_tokens(texts: list[str], tokenizer: Any) -> list[int]:
    """Count tokens for each text using the model's tokenizer.

    The tokenizer is invoked without truncation so the true length is
    measured; special tokens (CLS/SEP) are included in the count because the
    model adds them at encode time.

    Args:
        texts: Texts to count.
        tokenizer: Any object with ``encode(text)`` returning a result with
            an ``ids`` attribute (e.g. ``tokenizers.Tokenizer`` or a
            Hugging Face ``PreTrainedTokenizer``).

    Returns:
        One integer token count per input text.
    """
    counts: list[int] = []
    for text in texts:
        encoding = tokenizer.encode(text)
        ids = getattr(encoding, "ids", None)
        if ids is None:
            # Hugging Face tokenizers return a BatchEncoding-like object.
            ids = encoding["input_ids"]
        counts.append(len(ids))
    return counts


def model_token_stats(texts: list[str], tokenizer: Any) -> dict:
    """Summarise model-token usage relative to the model's max length."""
    counts = count_model_tokens(texts, tokenizer)
    if not counts:
        return {
            "max_tokens": model_max_tokens(),
            "chunk_count": 0,
            "over_limit": 0,
            "over_limit_pct": 0.0,
            "median": 0,
            "p95": 0,
            "p99": 0,
            "min": 0,
            "max": 0,
        }
    sorted_counts = sorted(counts)
    limit = model_max_tokens()
    over = [c for c in counts if c > limit]
    return {
        "max_tokens": limit,
        "chunk_count": len(counts),
        "over_limit": len(over),
        "over_limit_pct": 100.0 * len(over) / len(counts),
        "median": sorted_counts[len(sorted_counts) // 2],
        "p95": sorted_counts[int(len(sorted_counts) * 0.95)],
        "p99": sorted_counts[int(len(sorted_counts) * 0.99)],
        "min": sorted_counts[0],
        "max": sorted_counts[-1],
    }
