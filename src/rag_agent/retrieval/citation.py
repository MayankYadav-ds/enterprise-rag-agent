"""Citation formatting.

A citation is the shortest traceable reference a reader needs to check a claim:
the source file, the page range, and the section heading. It is formatted from
a ``RetrievedChunk`` so the same function serves the CLI, the future API and
the answer generator — there is one citation shape, not three.

Format
    ``[<file>, p. 22, <section_heading>]``

- A single page is ``p. 22``; a range is ``pp. 22-23``.
- The heading is omitted when the chunk has none, so a heading-less hit reads
  ``[nps-2013-10k.pdf, p. 22]``.
- The file name is the bare name stored in the payload (``Path(...).name``),
  never the indexed path, so a renamed source file is reflected without
  re-hydrating the chunk.
- The apostrophe in a heading such as ``MANAGEMENT'S DISCUSSION`` is preserved
  as-is; the citation is a literal rendering of the chunk's metadata, not a
  sanitised copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_agent.retrieval.models import RetrievedChunk


def _page_range(page_start: int, page_end: int) -> str:
    """Render a page span as ``p. 22`` or ``pp. 22-23``.

    A chunk always spans at least one page, so a single-page span is the common
    case and gets the singular form. ``page_end`` below ``page_start`` (a
    malformed chunk) collapses to the start page rather than producing a
    backwards range.
    """
    end = max(page_start, page_end)
    if end == page_start:
        return f"p. {page_start}"
    return f"pp. {page_start}-{end}"


def format_citation(hit: RetrievedChunk) -> str:
    """Render a hit as ``[<file>, p. 22, <section_heading>]``.

    The heading is the third element and is dropped entirely when the chunk has
    no section heading, so the bracket never contains an empty slot.
    """
    parts = [hit.source_file or "", _page_range(hit.page_start, hit.page_end)]
    if hit.section_heading:
        parts.append(hit.section_heading)
    return "[" + ", ".join(parts) + "]"


def to_dict(hit: RetrievedChunk) -> dict:
    """Serialise a hit for the future API response.

    The citation string is derived from these fields, so the API can render a
    citation client-side or let a caller reformat it. ``rank`` and ``retriever``
    are included so a hybrid result can show which retriever produced each hit.
    """
    return {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "source_file": hit.source_file,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "section_heading": hit.section_heading,
        "subheading": hit.subheading,
        "chunk_type": hit.chunk_type,
        "text": hit.text,
        "score": hit.score,
        "rank": hit.rank,
        "retriever": hit.retriever,
        "citation": format_citation(hit),
    }
