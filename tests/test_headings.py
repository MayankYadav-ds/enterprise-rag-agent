"""Tests for the heading attachment function."""

from __future__ import annotations

from rag_agent.chunking import (
    FixedSizeChunking,
    RecursiveChunking,
    StructureAwareChunking,
)
from rag_agent.chunking.headings import attach_headings
from rag_agent.chunking.models import Chunk
from rag_agent.ingestion.models import PageContent

DOC_ID = "doc_" + "a" * 64


def _page(number: int, text: str, headings: list[str] | None = None) -> PageContent:
    return PageContent(doc_id=DOC_ID, page_number=number, text=text, headings=headings or [])


def _chunk(
    chunk_index: int,
    page_start: int,
    page_end: int,
    text: str = "chunk text",
    section_heading: str | None = None,
) -> Chunk:
    return Chunk.create(
        doc_id=DOC_ID,
        text=text,
        chunk_type="text",
        page_start=page_start,
        page_end=page_end,
        chunk_index=chunk_index,
        section_heading=section_heading,
    )


def test_attach_headings_none_when_no_headings():
    """No valid headings anywhere → every chunk keeps section_heading=None."""
    pages = [_page(1, "page one text", headings=[]), _page(2, "page two text", headings=[])]
    chunks = [_chunk(0, 1, 1), _chunk(1, 2, 2)]
    result = attach_headings(chunks, pages)
    assert all(c.section_heading is None for c in result)
    # chunk_id must not change
    assert [c.chunk_id for c in result] == [c.chunk_id for c in chunks]


def test_attach_headings_none_before_first_heading():
    """Chunks before the first heading stay None."""
    pages = [
        _page(1, "preamble text", headings=[]),
        _page(2, "ITEM 1. Business", headings=["ITEM 1. Business"]),
    ]
    chunks = [_chunk(0, 1, 1), _chunk(1, 2, 2)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading is None
    assert result[1].section_heading == "ITEM 1. Business"


def test_attach_headings_carry_forward_across_pages():
    """A heading on page 1 names chunks on page 2 of the same document."""
    pages = [
        _page(1, "ITEM 1. Business", headings=["ITEM 1. Business"]),
        _page(2, "continued business text", headings=[]),
    ]
    chunks = [_chunk(0, 1, 1), _chunk(1, 2, 2)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 1. Business"
    assert result[1].section_heading == "ITEM 1. Business"


def test_attach_headings_table_label_ignored():
    """Table labels such as 'Consolidated Balance Sheets Data:' are not headings."""
    pages = [
        _page(
            1,
            "Consolidated Balance Sheets Data: table content",
            headings=["Consolidated Balance Sheets Data:"],
        ),
    ]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading is None


def test_attach_headings_document_boundary():
    """A heading from document A must not leak into document B's chunks.

    attach_headings is called per-document with that document's pages, so the
    boundary is enforced by the caller. This test verifies the function does
    not carry headings across documents when given separate page lists.
    """
    pages_a = [_page(1, "ITEM 1. Business", headings=["ITEM 1. Business"])]
    pages_b = [_page(1, "ITEM 2. Properties", headings=["ITEM 2. Properties"])]
    chunks_a = [_chunk(0, 1, 1)]
    chunks_b = [_chunk(0, 1, 1)]

    result_a = attach_headings(chunks_a, pages_a)
    result_b = attach_headings(chunks_b, pages_b)

    assert result_a[0].section_heading == "ITEM 1. Business"
    assert result_b[0].section_heading == "ITEM 2. Properties"


def test_attach_headings_chunk_id_unchanged():
    """The heading is metadata; chunk_id must not change."""
    pages = [_page(1, "ITEM 1. Business", headings=["ITEM 1. Business"])]
    chunk = _chunk(0, 1, 1)
    result = attach_headings([chunk], pages)
    assert result[0].chunk_id == chunk.chunk_id


def test_attach_headings_uses_last_heading_on_page():
    """When a page has multiple headings, the last one names the section."""
    pages = [
        _page(
            1,
            "ITEM 1. Business\n\nITEM 1A. Risk Factors",
            headings=["ITEM 1. Business", "ITEM 1A. Risk Factors"],
        ),
    ]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 1A. Risk Factors"


def test_attach_headings_max_length_rejected():
    """Headings longer than MAX_HEADING_LENGTH are rejected."""
    long_heading = "X" * 61
    pages = [_page(1, "text", headings=[long_heading])]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading is None


def test_attach_headings_empty_chunks():
    """attach_headings on an empty chunk list returns an empty list."""
    result = attach_headings([], [_page(1, "text", headings=["ITEM 1. Business"])])
    assert result == []


def test_all_strategies_get_headings():
    """All three chunking strategies receive headings through the pipeline."""
    pages = [
        _page(1, "ITEM 1. Business\n\nThis is the business text.", headings=["ITEM 1. Business"]),
        _page(2, "Continued business text.", headings=[]),
    ]

    for strategy in (FixedSizeChunking, RecursiveChunking, StructureAwareChunking):
        chunks = strategy(chunk_size=512, chunk_overlap=50).chunk(pages, DOC_ID)
        chunks = attach_headings(chunks, pages)
        heading_chunks = [c for c in chunks if c.section_heading is not None]
        assert heading_chunks, f"{strategy.__name__} produced no chunks with a heading"
        assert all(c.section_heading == "ITEM 1. Business" for c in heading_chunks)
