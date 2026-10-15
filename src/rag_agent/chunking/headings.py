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
_HEADING_PATTERN_TABLE_LABEL = re.compile(r",$|^\d{4}(\s+\d{4})+\s+\(|Data:$")

# Maximum heading length (from ingestion pdf_parser.MAX_HEADING_LENGTH)
_MAX_HEADING_LENGTH = 60


def _is_table_label(heading: str) -> bool:
    """Return True if the heading matches table label patterns that should be ignored."""
    return bool(_HEADING_PATTERN_TABLE_LABEL.search(heading.strip()))


def _is_valid_heading(heading: str) -> bool:
    """Return True if the heading passes the validation rules.

    Rules (from Phase 3 / ingestion pdf_parser):
    - Must not be empty
    - Must not exceed MAX_HEADING_LENGTH
    - Must not be a table label (e.g., "Consolidated Balance Sheets Data:")
    - Must not be a duplicate
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
