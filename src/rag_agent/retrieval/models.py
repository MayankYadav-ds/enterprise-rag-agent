"""Retrieval result model.

A ``RetrievedChunk`` is the retrieval layer's view of a hit: the same fields
that a citation needs (file, page range, section heading) plus the fields the
answer generator needs (text, score, rank, which retriever produced it).

It is deliberately a frozen dataclass so a hit cannot be silently mutated
between retrieval and citation formatting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single retrieval hit, ranked and scored.

    Attributes:
        chunk_id: Deterministic chunk identifier (``chunk_<doc_id>_<...>``).
        doc_id: Parent document identifier.
        source_file: Bare file name of the source PDF (e.g. ``nps-2013-10k.pdf``).
        page_start: First physical PDF page the chunk spans (1-indexed).
        page_end: Last physical PDF page the chunk spans.
        section_heading: Section heading the chunk belongs to, or ``None``.
        subheading: Optional subheading under the section heading, or ``None``.
        chunk_type: ``"text"`` or ``"table"``.
        text: Chunk text (raw Markdown for tables).
        score: Retriever score, higher is better.
        rank: 1-based position in the ranked result list.
        retriever: Name of the retriever that produced this hit.
    """

    chunk_id: str
    doc_id: str
    source_file: str
    page_start: int
    page_end: int
    section_heading: str | None
    subheading: str | None
    chunk_type: str
    text: str
    score: float
    rank: int
    retriever: str
