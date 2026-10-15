"""Heading attachment for chunking pipeline.

This module provides the `attach_headings` function that assigns section headings
to chunks based on the nearest preceding heading on the same page or an earlier
page of the SAME document. The heading is carried forward across pages but never
across documents. This is independent of the chunking strategy used.
"""

from __future__ import annotations

import re

from rag_agent.chunking.models import Chunk
from rag_agent.ingestion.models import PageContent

# Table column headers ("Years Ended December 31,") and row labels
# ("2013 2012 2011 2010 2009 (in thousands)") are bold but are not
# section headings; they are table content. "Consolidated Balance Sheets
# Data:" is the same class of label: it names a table, not a section.
# The same class also covers parenthesised table labels ("(a) Convertible
# Notes", "(i) Research and Development Expenses", "(In thousands)",
# "(13) Income Taxes") and year-range column headers ("Years ended December
# 31, 2013 and 2012", "As of December 31, 2013: Level 1 Level 2 ...").
_HEADING_PATTERN_TABLE_LABEL = re.compile(
    r",$"  # trailing comma: table row labels
    r"|^\d{4}(\s+\d{4})+\s+\("  # "2013 2012 2011 2010 2009 (in thousands)"
    r"|Data:$"  # "Consolidated Balance Sheets Data:"
    r"|\d{4} and \d{4}$"  # "Years ended December 31, 2013 and 2012"
    r"|^\s*\("  # "(a) Convertible Notes", "(In thousands)"
    r"|^Years ended "  # "Years ended December 31, 2013 and 2012"
    r"|^As of "  # "As of December 31, 2013: Level 1 Level 2 Level 3 Total"
)

# The company name ("NPS PHARMACEUTICALS, INC.", "..., INC. AND
# SUBSIDIARIES") is a running header on financial statement pages, not a
# section heading. Only the trailing legal suffix is checked so that real
# section headings are not rejected (e.g. "Report of Independent Registered
# Public Accounting Firm" ends with "Firm", not a legal suffix).
_HEADING_PATTERN_COMPANY = re.compile(
    r"\b(INC\.?|CORP\.?|CORPORATION|LLC\.?|LTD\.?|LIMITED|CO\.?|COMPANY)\.?$"
    r"|\bAND\s+SUBSIDIARIES\.?$",
    re.IGNORECASE,
)

# Maximum heading length (from ingestion pdf_parser.MAX_HEADING_LENGTH)
_MAX_HEADING_LENGTH = 60


def _is_table_label(heading: str) -> bool:
    """Return True if the heading matches table label patterns that should be ignored."""
    return bool(_HEADING_PATTERN_TABLE_LABEL.search(heading.strip()))


def _is_company_name(heading: str) -> bool:
    """Return True if the heading is the company's own legal name.

    The company name is a running header on financial statement pages, not a
    section heading (e.g. "NPS PHARMACEUTICALS, INC.",
    "NPS PHARMACEUTICALS, INC. AND SUBSIDIARIES").
    """
    return bool(_HEADING_PATTERN_COMPANY.search(heading.strip()))


def _is_valid_heading(heading: str) -> bool:
    """Return True if the heading passes the validation rules.

    Rules (from Phase 3 / ingestion pdf_parser):
    - Must not be empty
    - Must not exceed MAX_HEADING_LENGTH
    - Must not be a table label (e.g., "Consolidated Balance Sheets Data:")
    - Must not be the company's own legal name (running header)
    """
    if not heading:
        return False
    stripped = heading.strip()
    if not stripped:
        return False
    if len(stripped) > _MAX_HEADING_LENGTH:
        return False
    if _is_table_label(stripped):
        return False
    if _is_company_name(stripped):
        return False
    return True


def attach_headings(chunks: list[Chunk], pages: list[PageContent]) -> list[Chunk]:
    """Attach section headings to chunks based on nearest preceding heading.

    Args:
        chunks: List of chunks to annotate with section headings
        pages: List of page content with headings metadata

    Returns:
        New list of chunks with section_heading populated. The chunk_id is NOT
        recomputed — headings are metadata, not content.
    """
    if not chunks:
        return chunks

    # Build a mapping from page number to valid headings on that page
    # Only keep valid headings (not table labels, not too long, etc.)
    page_headings: dict[int, list[str]] = {}
    for page in pages:
        valid = [h for h in page.headings if _is_valid_heading(h)]
        if valid:
            page_headings[page.page_number] = valid

    if not page_headings:
        # No valid headings anywhere; return chunks unchanged (all None)
        return chunks

    # Sort page numbers for carry-forward logic
    sorted_pages = sorted(page_headings.keys())

    # For each chunk, find the nearest preceding heading on the same page
    # or an earlier page of the same document
    new_chunks: list[Chunk] = []
    for chunk in chunks:
        section_heading: str | None = None

        # Find the latest page <= chunk.page_end that has headings
        # We search backwards from the chunk's end page
        for page_num in reversed(sorted_pages):
            if page_num <= chunk.page_end:
                # The last heading on that page is the one that names the section
                # continuing onto subsequent pages
                section_heading = page_headings[page_num][-1]
                break

        # Create a new chunk with the heading attached (chunk_id unchanged)
        if chunk.section_heading == section_heading:
            new_chunks.append(chunk)
        else:
            new_chunks.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    chunk_type=chunk.chunk_type,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_heading=section_heading,
                    token_count=chunk.token_count,
                    text_quality=chunk.text_quality,
                    chunk_index=chunk.chunk_index,
                )
            )

    return new_chunks
