"""Unit tests for the lazy WordPiece tokenizer (no download).

Every test patches ``huggingface_hub.hf_hub_download`` and
``tokenizers.Tokenizer.from_file`` at the source modules, because the imports
inside ``wordpiece_tokenizer()`` are lazy. No network, no torch.
"""

from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from rag_agent.utils import lazy_tokenizer
from rag_agent.utils.lazy_tokenizer import WordPieceTokenizer, wordpiece_tokenizer


class _FakeEncoding:
    """Stand-in for a tokenizers Encoding: carries ``ids``."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    """Stub tokenizer that records what it was asked to encode."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str) -> _FakeEncoding:
        self.calls.append(text)
        return _FakeEncoding([ord(c) % 512 for c in text])


@pytest.fixture(autouse=True)
def _fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> FakeTokenizer:
    """Reset the module-level cache and inject stubs before every test."""
    monkeypatch.setattr(lazy_tokenizer, "_tokenizer", None)
    fake = FakeTokenizer()
    monkeypatch.setattr(lazy_tokenizer, "_lock", threading.Lock())
    # Patch at the source modules: the imports inside wordpiece_tokenizer()
    # are lazy, so the module-level attribute does not exist yet.
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename: f"/fake/{filename}",
    )
    monkeypatch.setattr(
        "tokenizers.Tokenizer.from_file",
        lambda path: fake,
    )
    return fake


def test_import_utils_without_huggingface_hub():
    """``import rag_agent.utils`` must work even if huggingface_hub is blocked.

    The lazy imports inside ``wordpiece_tokenizer()`` must not be pulled in at
    module load time, so the rest of the package stays importable on machines
    where the embeddings extra is not installed.
    """
    code = (
        "import sys; sys.modules['huggingface_hub'] = None; "
        "import rag_agent.utils; "
        "assert not hasattr(rag_agent.utils.lazy_tokenizer, 'hf_hub_download'); "
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


def test_first_call_downloads_and_caches(_fake_tokenizer: FakeTokenizer) -> None:
    first = wordpiece_tokenizer()
    second = wordpiece_tokenizer()

    assert isinstance(first, WordPieceTokenizer)
    assert first is second
    assert first.tokenizer is _fake_tokenizer
    assert first.tokenizer_path == "/fake/tokenizer.json"
    assert first.model_max_tokens == 512
    assert _fake_tokenizer.calls == []


def test_encode_delegates_to_underlying_tokenizer() -> None:
    tokenizer = wordpiece_tokenizer()
    encoding = tokenizer.encode("hello world")

    assert encoding.ids == [ord(c) % 512 for c in "hello world"]
    assert tokenizer.tokenizer.calls == ["hello world"]


def test_model_max_tokens_constant_matches_model_limit() -> None:
    """The wrapper must not silently drift from the 512-token model limit."""
    assert lazy_tokenizer._MAX_MODEL_TOKENS == 512
    assert wordpiece_tokenizer().model_max_tokens == 512


def test_concurrent_calls_return_same_singleton() -> None:
    """The lock must serialize first-time construction."""
    results: list[WordPieceTokenizer] = []

    def grab() -> None:
        results.append(wordpiece_tokenizer())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert all(r is results[0] for r in results)
