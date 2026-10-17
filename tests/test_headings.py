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


def test_attach_headings_parenthesised_table_label_ignored():
    """Parenthesised table labels such as "(a) Convertible Notes" are not headings."""
    pages = [_page(1, "note text", headings=["(a) Convertible Notes"])]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading is None


def test_attach_headings_in_thousands_label_ignored():
    """ "(In thousands)" is a table scale label, not a heading."""
    pages = [_page(1, "table text", headings=["(In thousands)"])]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading is None


def test_attach_headings_year_range_header_ignored():
    """ "Years ended December 31, 2013 and 2012" is a table column header."""
    pages = [_page(1, "table text", headings=["Years ended December 31, 2013 and 2012"])]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading is None


def test_attach_headings_as_of_label_ignored():
    """ "As of December 31, 2013: Level 1 Level 2 ..." is a table header."""
    pages = [
        _page(
            1,
            "table text",
            headings=["As of December 31, 2013: Level 1 Level 2 Level 3 Total"],
        )
    ]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading is None


def test_attach_headings_company_name_ignored():
    """The company's own legal name is a running header, not a section heading."""
    pages = [
        _page(1, "balance sheet", headings=["NPS PHARMACEUTICALS, INC."]),
        _page(2, "balance sheet", headings=["NPS PHARMACEUTICALS, INC. AND SUBSIDIARIES"]),
    ]
    chunks = [_chunk(0, 1, 1), _chunk(1, 2, 2)]
    result = attach_headings(chunks, pages)
    assert all(c.section_heading is None for c in result)


def test_attach_headings_company_suffix_does_not_kill_real_headings():
    """A heading ending in a legal suffix that is not the company name survives.

    "Report of Independent Registered Public Accounting Firm" ends with
    "Firm", not a legal suffix, so it is a real section heading.
    """
    pages = [
        _page(
            1, "report text", headings=["Report of Independent Registered Public Accounting Firm"]
        )
    ]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "Report of Independent Registered Public Accounting Firm"


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


def test_attach_headings_long_item_title_accepted():
    """SEC "Item NN. Title" lines are exempt from the 60-character cap.

    Their full titles can exceed it, e.g. "ITEM 7. MANAGEMENT'S DISCUSSION AND
    ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS".
    """
    long_title = (
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION "
        "AND RESULTS OF OPERATIONS"
    )
    pages = [_page(1, "md&a text", headings=[long_title])]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == long_title


def test_attach_headings_long_item_7a_title_accepted():
    pages = [
        _page(
            1,
            "market risk text",
            headings=["ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK"],
        )
    ]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == (
        "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK"
    )


def test_attach_headings_long_item_9_title_accepted():
    pages = [_page(1, "controls text", headings=["ITEM 9A. CONTROLS AND PROCEDURES"])]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 9A. CONTROLS AND PROCEDURES"


def test_attach_headings_long_item_5_title_accepted():
    pages = [_page(1, "properties text", headings=["ITEM 5. MINE SAFETY DISCLOSURES"])]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 5. MINE SAFETY DISCLOSURES"


def test_attach_headings_item_inside_chunk_span():
    """A chunk whose text span contains an ITEM heading takes that heading.

    This is the N-able p75 pattern: the chunk starts mid-page after the last
    preceding ITEM heading (so carry-forward would give the wrong answer) but
    its span includes a later ITEM heading on the same page.
    """
    pages = [
        _page(
            1,
            "ITEM 1. Business\n\nITEM 7A. Market Risk\n\nmore text here\n\n"
            "ITEM 8. Financial Statements\n\nbody text of item 8",
            headings=[
                "ITEM 1. Business",
                "ITEM 7A. Market Risk",
                "ITEM 8. Financial Statements",
            ],
        ),
    ]
    # Chunk starts after ITEM 7A but its span includes ITEM 8.
    chunks = [
        _chunk(
            0, 1, 1, text="more text here\n\nITEM 8. Financial Statements\n\nbody text of item 8"
        ),
    ]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 8. Financial Statements"


def test_attach_headings_item_inside_chunk_span_preferred_over_carry_forward():
    """An ITEM heading inside the chunk beats carry-forward from an earlier page.

    Even though an earlier page has a later ITEM heading, the heading inside
    the chunk's own span names its section.
    """
    pages = [
        _page(1, "ITEM 5. Properties", headings=["ITEM 5. Properties"]),
        _page(
            2,
            "ITEM 6. Business\n\nbody of item 6",
            headings=["ITEM 6. Business"],
        ),
    ]
    chunks = [
        _chunk(0, 1, 1, text="ITEM 5. Properties"),
        _chunk(1, 2, 2, text="ITEM 6. Business\n\nbody of item 6"),
    ]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 5. Properties"
    assert result[1].section_heading == "ITEM 6. Business"


def test_attach_headings_chunk_spanning_pages():
    """A chunk spanning two pages takes the heading from its first page."""
    pages = [
        _page(1, "ITEM 1. Business\n\npage one body", headings=["ITEM 1. Business"]),
        _page(2, "page two body", headings=[]),
    ]
    chunks = [
        _chunk(0, 1, 2, text="ITEM 1. Business\n\npage one body page two body"),
    ]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 1. Business"


def test_attach_headings_low_quality_page_updates_carry_forward():
    """A low text-quality page still updates carry-forward when readable.

    Headings from excluded pages (tq < 0.900) are not stored, but a readable
    page with valid headings still carries forward to later pages.
    """
    pages = [
        _page(1, "ITEM 1. Business", headings=["ITEM 1. Business"]),
        _page(2, "continued", headings=[]),
        _page(3, "ITEM 2. Properties", headings=["ITEM 2. Properties"]),
    ]
    chunks = [
        _chunk(0, 1, 1),
        _chunk(1, 2, 2),
        _chunk(2, 3, 3),
    ]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 1. Business"
    assert result[1].section_heading == "ITEM 1. Business"
    assert result[2].section_heading == "ITEM 2. Properties"


def test_attach_headings_subheading_non_item():
    """A non-ITEM heading that passes the subheading test is stored in subheading.

    In a 10-K-structured document, non-ITEM headings no longer set
    section_heading, but a real subheading (bold, short, not a table label)
    is captured in the subheading field.
    """
    pages = [
        _page(
            1,
            "ITEM 1. Business\n\nITEM 1A. Risk Factors\n\nRisk Factors\n\nbody text",
            headings=["ITEM 1. Business", "ITEM 1A. Risk Factors", "Risk Factors"],
        ),
        _page(
            2,
            "ITEM 2. Properties\n\ntext",
            headings=["ITEM 2. Properties"],
        ),
    ]
    chunks = [
        _chunk(0, 1, 1, text="Risk Factors\n\nbody text"),
    ]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 1A. Risk Factors"
    assert result[0].subheading == "Risk Factors"


def test_attach_headings_subheading_table_label_not_stored():
    """Table labels and statement titles are never subheadings."""
    pages = [
        _page(
            1,
            "ITEM 1. Business\n\nITEM 1A. Risk Factors\n\n"
            "Consolidated Balance Sheets Data:\n\nbody",
            headings=[
                "ITEM 1. Business",
                "ITEM 1A. Risk Factors",
                "Consolidated Balance Sheets Data:",
            ],
        ),
    ]
    chunks = [_chunk(0, 1, 1, text="Consolidated Balance Sheets Data:\n\nbody")]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 1A. Risk Factors"
    assert result[0].subheading is None


def test_attach_headings_long_non_item_line_still_rejected():
    """A long line without an "Item NN." prefix is still capped at 60 chars."""
    long_cover = "X" * 100
    pages = [_page(1, "cover text", headings=[long_cover])]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading is None


def test_attach_headings_item_title_too_long_rejected():
    """Even "Item NN." lines are capped at 160 characters."""
    too_long = "ITEM 7. " + "X" * 200
    pages = [_page(1, "text", headings=[too_long])]
    chunks = [_chunk(0, 1, 1)]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading is None
