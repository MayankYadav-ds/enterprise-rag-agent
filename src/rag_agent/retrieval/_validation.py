"""Shared input validation for retrievers.

``retrieve`` takes a query and a set of filters. The validation rules
(non-empty query, ``top_k`` in ``1..100``, ``bool`` rejected as ``top_k``) are
the same for every retriever, so they live here once instead of being
duplicated in ``DenseRetriever``, ``BM25Retriever`` and ``HybridRetriever``.

The rules are part of the ``Retriever`` protocol's contract: a caller that
violates them gets a loud, typed error before any embedding or scoring happens,
rather than a silently wrong result.
"""

from __future__ import annotations

_MIN_TOP_K = 1
_MAX_TOP_K = 100


def validate_top_k(top_k: int) -> int:
    """Reject out-of-range ``top_k`` early so a misconfiguration is loud.

    ``bool`` is rejected explicitly because it is a subclass of ``int`` and
    would otherwise pass as ``top_k=1`` (True) or ``top_k=0`` (False).
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError(f"top_k must be an int, got {type(top_k).__name__}")
    if top_k < _MIN_TOP_K or top_k > _MAX_TOP_K:
        raise ValueError(f"top_k must be in [{_MIN_TOP_K}, {_MAX_TOP_K}], got {top_k}")
    return top_k


def validate_query(query: str) -> str:
    """Reject empty/whitespace queries before they reach the retriever."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return query.strip()
