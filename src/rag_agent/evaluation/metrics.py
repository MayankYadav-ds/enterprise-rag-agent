"""Retrieval-quality metrics.

Each metric is a pure function over plain ranked hit lists, so the same
function serves any retriever implementation and any evaluation runner.
Scores are clipped to ``[0, 1]`` and never raise on empty input.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float) -> float:
    """Clamp a score to the valid ``[0, 1]`` range."""
    return max(0.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class PrecisionAtK:
    """Fraction of the top-K hits that are relevant.

    ``relevant`` is the set of gold chunk IDs for the query. A hit at rank r
    counts when its chunk id is in that set. Precision is 0 when K is 0 or
    no hits are returned.
    """

    k: int

    def compute(self, hits: list[str], relevant: set[str]) -> float:
        if self.k <= 0 or not hits:
            return 0.0
        top = hits[: self.k]
        if not top:
            return 0.0
        return _clip(sum(1 for chunk_id in top if chunk_id in relevant) / len(top))


@dataclass(frozen=True, slots=True)
class RecallAtK:
    """Fraction of the gold relevant chunks that appear in the top-K hits.

    Recall is 0 when the gold set is empty (there is nothing to recall) and
    1.0 when every relevant chunk is returned in the top K. It does not
    penalise returning extra non-relevant chunks.
    """

    k: int

    def compute(self, hits: list[str], relevant: set[str]) -> float:
        if self.k <= 0 or not relevant:
            return 0.0
        top = set(hits[: self.k])
        if not top:
            return 0.0
        return _clip(len(top & relevant) / len(relevant))


def mean_reciprocal_rank(hits: list[str], relevant: set[str]) -> float:
    """Return the reciprocal rank of the first relevant hit.

    MRR is 1.0 when the first hit is relevant, 0.5 when the second is, and
    0.0 when no relevant hit appears in the list. It is averaged over the
    queries that have at least one relevant chunk; queries with no relevant
    chunk contribute 0.
    """
    if not relevant:
        return 0.0
    for rank, chunk_id in enumerate(hits, start=1):
        if chunk_id in relevant:
            return _clip(1.0 / rank)
    return 0.0
