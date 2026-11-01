"""Hybrid retriever that fuses dense and BM25 results via Reciprocal Rank Fusion (RRF).

The class expects two already-configured retrievers that implement the same
``Retriever`` protocol - typically a :class:`DenseRetriever` and a
``BM25Retriever``. The fusion constant ``k`` defaults to ``60`` which matches the
original RRF paper and provides a sensible balance between the two rank lists.
"""

from __future__ import annotations

from rag_agent.retrieval.models import RetrievedChunk
from rag_agent.retrieval.protocol import Retriever


class HybridRetriever(Retriever):
    """Combine two retrievers using Reciprocal Rank Fusion.

    Each chunk's hybrid score is the sum of its RRF contributions from both
    ranked lists: ``k / (k + rank_dense) + k / (k + rank_bm25)``. This is a
    rank-based fusion score, **not** a similarity: it lives in ``(0, 2)`` and is
    not comparable to the dense retriever's cosine similarity (in ``[0, 1]``)
    or to the BM25 retriever's unbounded BM25 score. It must not be passed to
    a caller that expects a cosine similarity, and ``min_score`` does not have
    a single meaning here (see below).

    Parameters
    ----------
    dense:
        The dense-vector based retriever (e.g. :class:`DenseRetriever`).
    bm25:
        The BM25-based retriever.
    rrf_k:
        The RRF constant ``k`` - higher values give more weight to lower-rank
        positions. ``60`` is the conventional default.
    """

    def __init__(self, dense: Retriever, bm25: Retriever, rrf_k: int = 60) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k

    # ---------------------------------------------------------------------
    # Helper - compute the RRF contribution for a given rank.
    # ---------------------------------------------------------------------
    def _rrf_score(self, rank: int) -> float:
        return self.rrf_k / (self.rrf_k + rank)

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
        # Retrieve from each component - they each honour ``min_score`` and
        # ``top_k`` internally, so we can pass the same arguments.
        dense_hits = self.dense.retrieve(
            query,
            top_k=top_k,
            doc_id=doc_id,
            chunk_type=chunk_type,
            min_score=min_score,
        )
        bm25_hits = self.bm25.retrieve(
            query,
            top_k=top_k,
            doc_id=doc_id,
            chunk_type=chunk_type,
            min_score=min_score,
        )

        # Merge hits by ``chunk_id``. Track the rank each retriever gave the
        # chunk so ties can be broken deterministically.
        merged: dict[str, dict] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            entry = merged.setdefault(
                hit.chunk_id,
                {"hit": hit, "dense_rank": None, "bm25_rank": None, "score": 0.0},
            )
            entry["dense_rank"] = rank
            entry["score"] += self._rrf_score(rank)
        for rank, hit in enumerate(bm25_hits, start=1):
            entry = merged.setdefault(
                hit.chunk_id,
                {"hit": hit, "dense_rank": None, "bm25_rank": None, "score": 0.0},
            )
            entry["bm25_rank"] = rank
            entry["score"] += self._rrf_score(rank)

        # Sort by the accumulated RRF score (descending). Ties are resolved by
        # the original dense rank, then the BM25 rank, so the output is
        # deterministic even when two chunks accumulate identical RRF scores.
        sorted_items = sorted(
            merged.items(),
            key=lambda item: (
                item[1]["score"],
                -(item[1]["dense_rank"] or 0),
                -(item[1]["bm25_rank"] or 0),
            ),
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for idx, (_cid, data) in enumerate(sorted_items, start=1):
            hit: RetrievedChunk = data["hit"]
            results.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    source_file=hit.source_file,
                    page_start=hit.page_start,
                    page_end=hit.page_end,
                    section_heading=hit.section_heading,
                    subheading=hit.subheading,
                    chunk_type=hit.chunk_type,
                    text=hit.text,
                    score=data["score"],
                    rank=idx,
                    retriever="hybrid",
                )
            )
        # Finally limit to ``top_k`` to honour the caller's request.
        return results[:top_k]
