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


def test_attach_headings_non_item_before_item_is_not_subheading():
    """A non-ITEM heading BEFORE an ITEM heading never becomes its subheading.

    "PART I" precedes "ITEM 1. Business" on N-able page 7 and names the
    enclosing part, not a subheading of ITEM 1. It must not be stored.
    """
    pages = [
        _page(
            1,
            "ITEM 1. Business\n\nITEM 1A. Risk Factors\n\nITEM 2. Properties\n\nbody",
            headings=[
                "ITEM 1. Business",
                "ITEM 1A. Risk Factors",
                "ITEM 2. Properties",
            ],
        ),
        _page(
            2,
            "PART I\n\nITEM 1. Business\n\nbody of business",
            headings=["PART I", "ITEM 1. Business"],
        ),
    ]
    chunks = [_chunk(0, 2, 2, text="body of business")]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 1. Business"
    assert result[0].subheading is None


def test_attach_headings_report_of_auditor_not_subheading():
    """The auditor's report title precedes the financial-statement ITEM headings.

    On N-able page 83 the title comes first and names the report, not a
    subheading of the ITEM headings that follow. It must not be captured.
    """
    pages = [
        _page(
            1,
            "Report of Independent Registered Public Accounting Firm\n\n"
            "ITEM 8. Financial Statements\n\n"
            "ITEM 14. Principal Accountant Fees and Services\n\n"
            "body",
            headings=[
                "Report of Independent Registered Public Accounting Firm",
                "ITEM 8. Financial Statements",
                "ITEM 14. Principal Accountant Fees and Services",
            ],
        ),
    ]
    chunks = [_chunk(0, 1, 1, text="body")]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 14. Principal Accountant Fees and Services"
    assert result[0].subheading is None


def test_is_subheading_rejects_control_characters():
    """Headings carrying private-use/format characters are not subheadings.

    The N-able report's broken-font-mapping pages produce headings such as
    "\x06,-70\x05\x01\x1b9.\x07" that pass every other text-pattern check.
    """
    from rag_agent.chunking.headings import _is_subheading

    assert _is_subheading("\x06,-70\x05\x01\x1b9.\x07") is False
    assert _is_subheading("\x1b9?0=0>?\x01\x17C;09>0") is False
    # A real subheading is unaffected.
    assert _is_subheading("Risk Factors") is True
    assert _is_subheading("Significant Developments") is True


def test_is_subheading_rejects_punctuation_token():
    """A heading that is pure punctuation/symbols is not a subheading.

    ":=" is the only such token on the real filings (detected on N-able
    page 2, tq=1.000, so it is not excluded by the quality threshold). It
    passes every other text-pattern check — it is short, printable, not a
    table label or company name — but it names nothing, so it must not be
    stored as a subheading. Without this check it would be carried forward
    to chunks on pages 3-6 and force a re-patch on every sync.
    """
    from rag_agent.chunking.headings import _is_subheading

    assert _is_subheading(":=") is False
    assert _is_subheading("---") is False
    assert _is_subheading("...") is False
    assert _is_subheading("()") is False
    # A real subheading is unaffected. "Risk Factors:" is rejected by the
    # existing statement-title rule (trailing colon), not by the new
    # alphanumeric check, so this assertion pins that the new check does
    # not fire on a name that merely ends in a colon.
    assert _is_subheading("Risk Factors:") is False
    assert _is_subheading("Risk Factors") is True


def test_is_subheading_rejects_symbol_soup():
    """Headings that are mostly symbols are not subheadings.

    The N-able report's broken-font-mapping pages produce tokens such as
    "$6%24)3).'" and "$0A09@0" that are printable and contain digits, so
    the non-printable and alphanumeric checks both pass them. A subheading
    is text: it must contain a run of at least two consecutive letters.
    """
    from rag_agent.chunking.headings import _is_subheading

    assert _is_subheading("$6%24)3).'") is False
    assert _is_subheading("$0A09@0") is False
    assert _is_subheading(":=") is False
    # A real subheading is unaffected.
    assert _is_subheading("Risk Factors") is True
    assert _is_subheading("Significant Developments") is True
    assert _is_subheading("UNITED STATES SECURITIES AND EXCHANGE COMMISSION") is True


def test_is_subheading_checks_text_patterns_only():
    """_is_subheading inspects text patterns, not bold or font size.

    The function has no access to the font signals that the ingestion
    detector used, so it cannot confirm a candidate is bold or larger than
    the body font. This test pins that contract: a heading that is merely
    short and well-formed passes regardless of any typography.
    """
    from rag_agent.chunking.headings import _is_subheading

    # "Risk Factors" passes on text alone; there is no bold/font-size input.
    assert _is_subheading("Risk Factors") is True
    # An ITEM heading never passes, even though it is well-formed.
    assert _is_subheading("ITEM 1A. Risk Factors") is False


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


# --- 10-K section markers ---------------------------------------------------

from rag_agent.chunking.headings import _is_section_marker  # noqa: E402


def test_is_section_marker_matches_case_insensitively():
    """The 10-K section markers are matched case-insensitively."""
    assert _is_section_marker("Report of Independent Registered Public Accounting Firm")
    assert _is_section_marker("report of independent registered public accounting firm")
    assert _is_section_marker("Index to Consolidated Financial Statements")
    assert _is_section_marker("Signatures")
    assert _is_section_marker("PART I")
    assert _is_section_marker("part ii")
    assert _is_section_marker("PART III")
    assert _is_section_marker("PART IV")


def test_is_section_marker_rejects_non_markers():
    """A heading that is not on the whitelist is not a section marker."""
    assert not _is_section_marker("ITEM 1. Business")
    assert not _is_section_marker("Risk Factors")
    assert not _is_section_marker("Consolidated Balance Sheets Data:")
    assert not _is_section_marker("NPS PHARMACEUTICALS, INC.")


def test_attach_headings_auditor_report_marker_sets_section_heading():
    """The auditor's report title sets section_heading in a 10-K document.

    On N-able page 83 the title is detected and kept (tq=1.000) but, before
    the whitelist, only ITEM headings set section_heading, so the last ITEM
    (ITEM 14 on p77) carried forward over the F-pages that follow it and all
    47 chunks on p83-125 were mislabelled.
    """
    pages = [
        _page(
            1,
            "ITEM 14. Principal Accountant Fees and Services\n\n"
            "Report of Independent Registered Public Accounting Firm\n\n"
            "financial statement body",
            headings=[
                "ITEM 14. Principal Accountant Fees and Services",
                "Report of Independent Registered Public Accounting Firm",
            ],
        ),
        _page(2, "audit opinion text", headings=[]),
    ]
    chunks = [_chunk(0, 2, 2, text="audit opinion text")]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "Report of Independent Registered Public Accounting Firm"
    assert result[0].subheading is None


def test_attach_headings_part_marker_sets_section_heading():
    """A PART divider sets section_heading and replaces the carry-forward."""
    pages = [
        _page(
            1,
            "ITEM 1. Business\n\nITEM 1A. Risk Factors\n\nITEM 2. Properties\n\nbody",
            headings=["ITEM 1. Business", "ITEM 1A. Risk Factors", "ITEM 2. Properties"],
        ),
        _page(
            2,
            "PART II\n\nITEM 5. Market for Registrant's Common Equity\n\nbody",
            headings=["PART II", "ITEM 5. Market for Registrant's Common Equity"],
        ),
    ]
    chunks = [_chunk(0, 2, 2, text="body")]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 5. Market for Registrant's Common Equity"


def test_attach_headings_marker_inside_chunk_span_takes_priority():
    """A section heading inside a chunk's span beats carry-forward.

    The chunk on N-able p77 starts at offset 1773 (after ITEM 13) and its
    span contains ITEM 14, so it takes the last section heading in its span
    (ITEM 14) rather than the PART III heading that carries forward. Also
    covers a non-10-K document (below _MIN_ITEM_HEADINGS): a heading inside
    the chunk's span still wins over carry-forward, which was previously
    broken when _contained_section_heading was gated on has_item_structure.
    """
    pages = [
        _page(
            1,
            "PART III\n\nITEM 13. Certain Relationships\n\n"
            "ITEM 14. Principal Accountant Fees and Services\n\nbody",
            headings=[
                "PART III",
                "ITEM 13. Certain Relationships",
                "ITEM 14. Principal Accountant Fees and Services",
            ],
        ),
    ]
    chunks = [
        _chunk(
            0,
            1,
            1,
            text="ITEM 13. Certain Relationships\n\n"
            "ITEM 14. Principal Accountant Fees and Services\n\nbody",
        )
    ]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "ITEM 14. Principal Accountant Fees and Services"


def test_attach_headings_non_10k_document_unchanged():
    """A document without 10-K structure is unaffected by the whitelist.

    Below _MIN_ITEM_HEADINGS the previous behaviour is kept: any valid
    heading sets section_heading, and the markers are not special.
    """
    pages = [
        _page(
            1,
            "Report of Independent Registered Public Accounting Firm\n\nbody",
            headings=["Report of Independent Registered Public Accounting Firm"],
        ),
    ]
    chunks = [_chunk(0, 1, 1, text="body")]
    result = attach_headings(chunks, pages)
    # Non-10-K: the marker is just another valid heading.
    assert result[0].section_heading == "Report of Independent Registered Public Accounting Firm"


def test_attach_headings_non_10k_heading_inside_chunk_span_takes_priority():
    """A non-10-K document: a heading inside a chunk's span beats carry-forward.

    The contained-heading rule is not gated on 10-K structure, so a plain
    document (below _MIN_ITEM_HEADINGS) also prefers the heading that the
    chunk literally contains over the last heading from the previous page.
    """
    pages = [
        _page(
            1,
            "First Heading\n\nbody",
            headings=["First Heading"],
        ),
        _page(
            2,
            "Second Heading\n\nbody",
            headings=["Second Heading"],
        ),
    ]
    chunks = [
        _chunk(0, 1, 1, text="First Heading\n\nbody"),
        _chunk(1, 2, 2, text="Second Heading\n\nbody"),
    ]
    result = attach_headings(chunks, pages)
    assert result[0].section_heading == "First Heading"
    assert result[1].section_heading == "Second Heading"
