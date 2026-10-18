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

# SEC filings label sections with "Item NN. Title" (e.g. "Item 1A. Risk
# Factors", "Item 7. Management's Discussion"). These are real section
# headings even when rendered at body font size, and their full titles can
# exceed the normal cap ("ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF
# FINANCIAL CONDITION AND RESULTS OF OPERATIONS"). Such lines are exempt
# from the 60-character cap and are allowed up to _MAX_ITEM_HEADING_LENGTH.
_HEADING_PATTERN_ITEM = re.compile(r"^ITEM\s+\d+[A-C]?\.\s*\S", re.IGNORECASE)

# Maximum heading length (from ingestion pdf_parser.MAX_HEADING_LENGTH)
_MAX_HEADING_LENGTH = 60
# Item titles are allowed to be longer; a bare cover-page line without an
# "Item NN." prefix is still capped at _MAX_HEADING_LENGTH.
_MAX_ITEM_HEADING_LENGTH = 160
# Item titles are exempt from the 60-character cap (see above).

# A document is treated as 10-K-structured when at least this many distinct
# "Item NN. Title" headings are detected. Below this, the previous behaviour
# (any valid heading can set section_heading) is kept.
_MIN_ITEM_HEADINGS = 3

# 10-K section markers that are not "Item NN." lines but name a section of the
# filing. In a 10-K-structured document only ITEM headings set
# section_heading, which is why the auditor's report title and the index to
# the financial statements do not — and the last ITEM then carries forward
# over the F-pages that follow it (N-able p83-125, 47 chunks, labelled
# "ITEM 14. PRINCIPAL ACCOUNTANT FEES AND SERVICES"). These markers also set
# section_heading, case-insensitively, and only when they are matched against
# a heading actually detected by the parser — a substring of page text is
# never promoted to a heading on its own.
_SECTION_MARKERS = (
    "Report of Independent Registered Public Accounting Firm",
    "Index to Consolidated Financial Statements",
    "Signatures",
    "PART I",
    "PART II",
    "PART III",
    "PART IV",
)

# Subheading candidates must pass this stricter test on top of
# _is_valid_heading: they must be bold, larger than the body font, and not a
# table header, statement title, or line label. Year-range and digits-only
# text is also rejected.
_HEADING_PATTERN_YEAR_RANGE = re.compile(
    r"^\d{4}\s*(?:[-–]\s*\d{4})?\s*$"  # "2013", "2013-2014"
    r"|^\d{4}\s+and\s+\d{4}$"  # "2013 and 2014"
    r"|Years ended "  # "Years ended December 31,"
    r"|As of "  # "As of December 31,"
    r"|\(.*\)"  # parenthesised labels: "(a)", "(In thousands)"
)
_HEADING_PATTERN_DIGITS_ONLY = re.compile(r"^[\d,\s().]+$")

# A subheading is text, so it must contain a run of at least two consecutive
# letters. This rejects garbage from broken font mappings such as
# "$6%24)3).'" and "$0A09@0" — printable, digit-bearing tokens that pass
# every other text-pattern check. Safe on the real filings: all nine NPS
# subheadings have such a run and the three junk values do not.
_HEADING_PATTERN_TWO_LETTERS = re.compile(r"[A-Za-z]{2,}")

# Normalise whitespace for heading/text comparison (clean_text collapses runs
# of whitespace, so a stored heading may not be a literal substring of the page
# text).
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _is_subheading(heading: str) -> bool:
    """Return True for a non-ITEM heading that is a real subheading.

    A subheading is bold, larger than the body font, short, and not a table
    header, statement title, line label, year range, or digits-only text.

    NOTE: this function checks text patterns only. It does not look at bold
    or font-size signals — those were available at ingestion time but are not
    carried through to the chunker, so a subheading candidate that passes is
    not guaranteed to be bold or larger than the body font. The stricter
    ingestion detector already rejected most table labels and statement
    titles; this check is the second, text-pattern-only filter.
    """
    stripped = heading.strip()
    if not stripped or _is_item_heading(stripped):
        return False
    if len(stripped) > _MAX_HEADING_LENGTH:
        return False
    if _is_table_label(stripped):
        return False
    if _is_company_name(stripped):
        return False
    if _HEADING_PATTERN_YEAR_RANGE.search(stripped):
        return False
    if _HEADING_PATTERN_DIGITS_ONLY.match(stripped):
        return False
    # Reject statement-title-like labels ending in ":" (e.g. "Consolidated
    # Balance Sheets Data:") and parenthesised labels.
    if stripped.endswith(":"):
        return False
    # Reject headings that still carry non-printable characters. Ingestion's
    # clean_text drops private-use area glyphs (Co) and format characters
    # (Cf), but it does not drop control characters (Cc), so a heading block
    # whose text was not fully cleaned — e.g. the N-able report's
    # broken-font-mapping pages — can reach here with raw control characters
    # such as "\x06,-70\x05\x01\x1b9.\x07". Such text is not a subheading in
    # any language. All of the garbage values observed on the real filings
    # live on low-text-quality pages that are excluded from chunking anyway,
    # but the check keeps the subheading field honest.
    if not stripped.isprintable():
        return False
    # Reject headings that are pure punctuation/symbols. Ingestion's
    # clean_text drops private-use glyphs and format characters but keeps
    # ordinary punctuation, so a stray token such as ":=" (the only one on
    # the real filings, detected on N-able page 2) passes every other text
    # check. A subheading names something; a token with no letter or digit
    # does not. (This is a second filter alongside the non-printable check
    # above: the control-character soup contains digits, so it is caught
    # only by isprintable().)
    if not any(ch.isalnum() for ch in stripped):
        return False
    # Reject headings that are mostly symbols — garbage from broken font
    # mappings such as "$6%24)3).'" and "$0A09@0", which are printable and
    # contain digits so the two checks above pass them. A subheading is
    # text: it must contain a run of at least two consecutive letters. This
    # is safe on the real filings — all nine NPS subheadings have such a
    # run, and the three junk values do not.
    if not _HEADING_PATTERN_TWO_LETTERS.search(stripped):
        return False
    return True


def _find_chunk_offset(chunk_text: str, page_text: str, search_from: int) -> int:
    """Return the offset of chunk_text in page_text, searching from search_from.

    Chunks are searched forward from the previous chunk's end so that
    overlapping or repeated text does not pick up an earlier occurrence.
    Returns -1 when the chunk text cannot be found in the page text.
    """
    if not chunk_text or not page_text:
        return -1
    start = max(0, search_from)
    idx = page_text.find(chunk_text, start)
    if idx >= 0:
        return idx
    # Fall back to searching from the beginning of the page text.
    return page_text.find(chunk_text)


def _chunk_starts_with_heading(chunk_text: str, offsets: list[tuple[int, str]]) -> bool:
    """Return True when the chunk text begins with a heading from offsets.

    A chunk whose text starts at a heading (e.g. the whole page was chunked
    starting from "PART II ITEM 5...") should treat headings inside the chunk
    as preceding its body text. A chunk that starts mid-paragraph (e.g. after
    the last heading on the page) should not.
    """
    if not chunk_text or not offsets:
        return False
    stripped = chunk_text.lstrip()
    for _, heading in offsets:
        if stripped.startswith(heading):
            return True
    return False


def _find_heading_offset(heading: str, page_text: str) -> int:
    """Return the offset of heading in page_text, or len(page_text) if absent.

    A heading is not always a literal substring of the page text: ingestion's
    clean_text may normalise whitespace or drop private-use characters, so the
    stored heading can differ slightly from the page text. In that case the
    heading is treated as the last thing on the page (it names the section
    continuing onto the next page), which is the conservative choice.
    """
    if not heading:
        return 0
    idx = page_text.find(heading)
    if idx >= 0:
        return idx
    # Try a normalised comparison: collapse whitespace and drop the
    # private-use / format characters that clean_text removes.
    normalised_heading = _WHITESPACE_PATTERN.sub(" ", heading).strip()
    normalised_text = _WHITESPACE_PATTERN.sub(" ", page_text).strip()
    idx = normalised_text.find(normalised_heading)
    if idx >= 0:
        return idx
    return len(page_text)


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


def _is_item_heading(heading: str) -> bool:
    """Return True if the heading is an SEC "Item NN. Title" line.

    Such lines are exempt from the 60-character cap (their full titles can
    exceed it, e.g. "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF
    FINANCIAL CONDITION AND RESULTS OF OPERATIONS").
    """
    return bool(_HEADING_PATTERN_ITEM.match(heading.strip()))


def _is_section_marker(heading: str) -> bool:
    """Return True if the heading is a 10-K section marker.

    These are the non-ITEM headings that name a section of the filing: the
    auditor's report, the index to the financial statements, the signature
    page and the Part dividers. They are matched case-insensitively against
    headings the parser actually detected, so a substring of page text is
    never promoted to a heading on its own.
    """
    return heading.strip().upper() in {marker.upper() for marker in _SECTION_MARKERS}


def _is_section_heading(heading: str) -> bool:
    """Return True if the heading sets section_heading in a 10-K document.

    An "Item NN. Title" line or one of the 10-K section markers. In a
    10-K-structured document these are the only headings that name a
    section; everything else (table labels, statement titles, company
    names) is skipped so the walk continues to the nearest ITEM or marker.
    """
    return _is_item_heading(heading) or _is_section_marker(heading)


def _is_valid_heading(heading: str) -> bool:
    """Return True if the heading passes the validation rules.

    Rules (from Phase 3 / ingestion pdf_parser):
    - Must not be empty
    - Must not exceed the length cap (60 chars, or 160 for "Item NN. Title"
      lines, which are exempt because their full titles are long)
    - Must not be a table label (e.g., "Consolidated Balance Sheets Data:")
    - Must not be the company's own legal name (running header)
    """
    if not heading:
        return False
    stripped = heading.strip()
    if not stripped:
        return False
    cap = _MAX_ITEM_HEADING_LENGTH if _is_item_heading(stripped) else _MAX_HEADING_LENGTH
    if len(stripped) > cap:
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

    # Determine whether the document has a 10-K structure: at least
    # _MIN_ITEM_HEADINGS distinct ITEM headings. When it does, section_heading
    # is the nearest preceding ITEM heading (carried forward across pages);
    # table headers, statement titles and line labels no longer set it.
    item_headings = {
        page_num: [h for h in headings if _is_item_heading(h)]
        for page_num, headings in page_headings.items()
    }
    has_item_structure = sum(len(h) for h in item_headings.values()) >= _MIN_ITEM_HEADINGS

    # For each page, compute the character offset of every valid heading in the
    # page text. This lets us assign headings by position within a page rather
    # than by page number alone (issue #33).
    page_text_by_number = {page.page_number: page.text for page in pages}
    heading_offsets: dict[int, list[tuple[int, str]]] = {}
    for page in pages:
        offsets: list[tuple[int, str]] = []
        text = page.text
        for h in page_headings.get(page.page_number, []):
            idx = _find_heading_offset(h, text)
            offsets.append((idx, h))
        heading_offsets[page.page_number] = sorted(offsets, key=lambda t: t[0])

    # For each chunk, compute its character offset in the page text of its
    # first page. Chunks are searched forward from the previous chunk's end so
    # that overlapping or repeated text does not pick up an earlier occurrence.
    # When the chunk text is not a literal substring of the page text (rare,
    # e.g. after hyphen-join repair), the chunk boundary is approximated by the
    # chunk's cumulative length from the last known offset.
    chunk_offsets: dict[int, tuple[int, int, bool]] = {}
    last_end = 0
    for index, chunk in enumerate(chunks):
        page_text = page_text_by_number.get(chunk.page_start, "")
        offset = _find_chunk_offset(chunk.text, page_text, last_end)
        if offset >= 0:
            end = offset + len(chunk.text)
            found = True
        else:
            # Fallback: approximate from the last known offset and chunk length.
            offset = last_end
            end = last_end + len(chunk.text)
            found = False
        chunk_offsets[index] = (offset, end, found)
        last_end = end

    # For each chunk, find the nearest preceding heading on the same page
    # or an earlier page of the same document
    new_chunks: list[Chunk] = []
    for index, chunk in enumerate(chunks):
        section_heading: str | None = None
        subheading: str | None = None

        chunk_start, chunk_end, found_in_text = chunk_offsets[index]

        # When the chunk's own text span contains a section heading (an ITEM line
        # or a 10-K section marker), that heading names the section the chunk
        # belongs to. This catches chunks that begin mid-page after a heading
        # (e.g. N-able p75 chunk starting at offset 2003 with ITEM 8 at offset
        # 2876) and chunks that begin at the heading itself. It takes priority
        # over carry-forward from earlier pages because the heading is
        # literally inside the chunk's content.
        if has_item_structure and chunk_start >= 0 and found_in_text:
            offsets = heading_offsets.get(chunk.page_start, [])
            contained_section = [
                h
                for off, h in offsets
                if chunk_start <= off <= chunk_end and _is_section_heading(h)
            ]
            if contained_section:
                section_heading = contained_section[0]

        # Walk backwards through pages <= chunk.page_start to find the nearest
        # preceding heading at or before the chunk's start offset.
        if section_heading is None:
            for page_num in reversed(sorted_pages):
                if page_num > chunk.page_start:
                    continue
                offsets = heading_offsets.get(page_num, [])
                if not offsets:
                    continue
                # On the chunk's own page, accept headings at or before the chunk
                # start. On earlier pages, accept all headings (the last one names
                # the section continuing onto the next page).
                page_text_len = len(page_text_by_number.get(page_num, ""))
                if page_num == chunk.page_start and chunk_start >= 0 and found_in_text:
                    # On the chunk's own page: headings at or before the chunk's
                    # start are definitely preceding. When the chunk text itself
                    # starts with heading text (the chunk begins at a heading),
                    # headings inside the chunk also precede the chunk's body text,
                    # so the limit extends to the chunk's end.
                    if _chunk_starts_with_heading(chunk.text, offsets):
                        limit = chunk_end
                    else:
                        limit = chunk_start
                else:
                    # When the chunk text was not found in the page text or on an
                    # earlier page, treat the whole page as preceding so the last
                    # heading on the page still applies.
                    limit = page_text_len
                preceding = [h for off, h in offsets if off <= limit]
                if preceding:
                    if has_item_structure:
                        section_preceding = [h for h in preceding if _is_section_heading(h)]
                        if section_preceding:
                            section_heading = section_preceding[-1]
                            # Even when a section heading is found, capture a
                            # non-section subheading that comes AFTER it on
                            # the same page (e.g. "Risk Factors" under ITEM
                            # 1A). A non-section heading that appears BEFORE
                            # the section heading (e.g. "PART I" on N-able
                            # p7, "Report of Independent Registered Public
                            # Accounting Firm" on N-able p83) names the
                            # enclosing part or the auditor's report, not a
                            # subheading of the section, so it is not
                            # captured here.
                            if subheading is None:
                                last_section_offset = max(
                                    off
                                    for off, h in offsets
                                    if off <= limit and _is_section_heading(h)
                                )
                                non_section_after = [
                                    h
                                    for off, h in offsets
                                    if last_section_offset < off <= limit
                                    and not _is_section_heading(h)
                                    and _is_subheading(h)
                                ]
                                if non_section_after:
                                    subheading = non_section_after[-1]
                            break
                        # Non-section headings (statement titles, etc.) do
                        # not set section_heading but may still inform
                        # subheading. Keep walking backwards to find a
                        # section heading from an earlier page; capture
                        # subheading candidates along the way.
                        non_section_preceding = [
                            h for h in preceding if not _is_section_heading(h) and _is_subheading(h)
                        ]
                        if non_section_preceding and subheading is None:
                            subheading = non_section_preceding[-1]
                    else:
                        section_heading = preceding[-1]
                        break

        # If no heading was found on the chunk's first page or earlier pages
        # (e.g. the chunk starts at offset 0 and contains the heading itself),
        # fall back to headings within the chunk's text span on the first page.
        if section_heading is None and chunk_start >= 0:
            offsets = heading_offsets.get(chunk.page_start, [])
            section_in_span = [
                h
                for off, h in offsets
                if chunk_start <= off <= chunk_end and _is_section_heading(h)
            ]
            if section_in_span:
                section_heading = section_in_span[-1]

        # Create a new chunk with the heading attached (chunk_id unchanged)
        if (
            chunk.section_heading == section_heading
            and getattr(chunk, "subheading", None) == subheading
        ):
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
                    subheading=subheading,
                    token_count=chunk.token_count,
                    text_quality=chunk.text_quality,
                    chunk_index=chunk.chunk_index,
                )
            )

    return new_chunks
