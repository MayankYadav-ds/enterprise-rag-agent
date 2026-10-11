"""Lazy-loaded WordPiece tokenizer for the BGE embedding model.

This module provides the model's actual WordPiece tokenizer (from the tokenizers
library, >=0.20) without importing torch or sentence-transformers. The tokenizer
weights are downloaded once from Hugging Face and cached locally.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

_MODEL_ID = "BAAI/bge-small-en-v1.5"
_MAX_MODEL_TOKENS = 512  # As declared in config.json: max_position_embeddings=512


@dataclass(frozen=True, slots=True)
class WordPieceTokenizer:
    """Wrapper around a tokenizers.Tokenizer with model-specific metadata."""

    tokenizer: Tokenizer
    tokenizer_path: str
    model_max_tokens: int

    def encode(self, text: str):
        return self.tokenizer.encode(text)


_tokenizer: WordPieceTokenizer | None = None
_lock = threading.Lock()


def wordpiece_tokenizer() -> WordPieceTokenizer:
    """Return the cached WordPiece tokenizer, downloading on first use."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    with _lock:
        if _tokenizer is not None:
            return _tokenizer
        path = hf_hub_download(repo_id=_MODEL_ID, filename="tokenizer.json")
        tok = Tokenizer.from_file(path)
        _tokenizer = WordPieceTokenizer(
            tokenizer=tok,
            tokenizer_path=path,
            model_max_tokens=_MAX_MODEL_TOKENS,
        )
        return _tokenizer
