"""Pure-Python BM25 retriever.

The implementation is deliberately lightweight - it does **not** depend on any
external IR library. It works over an in-memory ``index`` mapping a ``chunk_id``
to the raw payload dictionary as produced by :class:`QdrantStore`. The payload
must contain at least the fields required by :class:`RetrievedChunk` (see
``retrieval.models``). The retriever validates arguments using the shared
``_validation`` helpers and then scores each document with the classic BM25
formula (k1=1.2, b=0.75).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from rag_agent.retrieval._validation import validate_query, validate_top_k
from rag_agent.retrieval.models import RetrievedChunk
from rag_agent.retrieval.protocol import Retriever


class BM25Retriever(Retriever):
    """BM25 retrieval over a simple in-memory index.

    The ``index`` argument is a mapping ``chunk_id -> payload dict`` where the
    payload matches the shape produced by :meth:`QdrantStore._payload`. Only the
    ``text`` field is required for scoring; all other fields are propagated to
    the resulting :class:`RetrievedChunk` unchanged.
    """

    def __init__(self, index: dict[str, dict]):
        self._index = index

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Return a list of lower-cased word tokens.

        ``re.findall(r"\\w+", ...)`` mirrors the tokenisation used by the
        original prototype and provides a deterministic, language-agnostic
        split on alphanumerics. Non-string input (a ``None`` text field, a
        number, a list) is treated as an empty document rather than crashing,
        so a malformed payload cannot take the whole retriever down.
        """
        if not isinstance(text, str):
            return []
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _bm25_score(
        tf: int,
        doc_len: int,
        avg_len: float,
        idf: float,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> float:
        """Return the BM25 contribution for a single term.

        The parameters follow the standard BM25 definition. ``idf`` is pre-
        computed for the term across the whole collection.
        """
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_len))
        return idf * (numerator / denominator)

    # ---------------------------------------------------------------------
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
        # Validate arguments using the shared helpers.
        query = validate_query(query)
        top_k = validate_top_k(top_k)

        # Tokenise query once.
        query_terms = self._tokenise(query)

        # -----------------------------------------------------------------
        # Gather candidate documents respecting the optional filters.
        # -----------------------------------------------------------------
        candidates: list[dict] = []
        for _cid, payload in self._index.items():
            # Apply filters early - cheap string comparisons.
            if doc_id is not None and payload.get("doc_id") != doc_id:
                continue
            if chunk_type is not None and payload.get("chunk_type") != chunk_type:
                continue
            candidates.append(payload)

        if not candidates:
            return []

        # -----------------------------------------------------------------
        # Compute collection statistics needed for BM25.
        # -----------------------------------------------------------------
        doc_lengths: list[int] = []  # number of tokens per document
        term_doc_freq: Counter[str] = Counter()
        tokenised_docs: list[list[str]] = []
        for payload in candidates:
            tokens = self._tokenise(payload.get("text", ""))
            tokenised_docs.append(tokens)
            doc_lengths.append(len(tokens))
            # Update document frequency for each *unique* term.
            term_doc_freq.update(set(tokens))

        N = len(candidates)
        avg_len = sum(doc_lengths) / N if N else 0.0

        # Pre-compute IDF for query terms (ignore terms not present in any doc).
        idf: dict[str, float] = {}
        for term in query_terms:
            n_t = term_doc_freq.get(term, 0)
            if n_t == 0:
                continue
            idf[term] = math.log1p((N - n_t + 0.5) / (n_t + 0.5))

        # -----------------------------------------------------------------
        # Score each candidate.
        # -----------------------------------------------------------------
        scored: list[tuple[dict, float]] = []
        for payload, tokens, length in zip(candidates, tokenised_docs, doc_lengths, strict=True):
            tf_counter = Counter(tokens)
            score = 0.0
            for term, term_idf in idf.items():
                tf = tf_counter.get(term, 0)
                if tf:
                    score += self._bm25_score(tf, length, avg_len, term_idf)
            # A document with no matching query term scores 0.0 and is not a
            # hit: returning it would let an empty query match everything.
            if score <= 0.0:
                continue
            if min_score is not None and score < min_score:
                continue
            scored.append((payload, score))

        # Sort by BM25 score descending. Ties are broken by chunk_id so the
        # ranking is deterministic regardless of the index's insertion order.
        scored.sort(key=lambda pair: (-pair[1], str(pair[0].get("chunk_id", ""))))

        # Build ``RetrievedChunk`` results, honouring ``top_k``.
        results: list[RetrievedChunk] = []
        for rank, (payload, score) in enumerate(scored[:top_k], start=1):
            results.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", "")),
                    doc_id=str(payload.get("doc_id", "")),
                    source_file=str(payload.get("source_file", "")),
                    page_start=int(payload.get("page_start", 0)),
                    page_end=int(payload.get("page_end", 0)),
                    section_heading=payload.get("section_heading"),
                    subheading=payload.get("subheading"),
                    chunk_type=str(payload.get("chunk_type", "")),
                    text=str(payload.get("text", "")),
                    score=score,
                    rank=rank,
                    retriever="bm25",
                )
            )
        return results
