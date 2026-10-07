"""Page-aware PDF extraction with layout, table, and source-quality signals."""

from __future__ import annotations

import logging
import re
import statistics
import unicodedata
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf

from rag_agent.ingestion.exceptions import PdfCorruptError, PdfEncryptedError, PdfNotFoundError
from rag_agent.ingestion.models import PageContent

logger = logging.getLogger(__name__)

_PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_HYPHENATED_LINEBREAK_PATTERN = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])")
LOW_TEXT_QUALITY_THRESHOLD = 0.9
# Section headings are short, titled lines. Cover-page headings like
# "FORM 10-K ... ANNUAL REPORT PURSUANT TO ..." are page titles, not sections.
MAX_HEADING_LENGTH = 60


@dataclass(frozen=True, slots=True)
class ParseWarning:
    """A non-fatal condition encountered on one PDF page."""

    page_number: int
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class PageAudit:
    """Extraction details used to inspect repeated-edge removal decisions."""

    page_number: int
    raw_text: str
    cleaned_text: str
    removed_edge_text: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedPdf:
    """Parser output before the ingestion pipeline persists it."""

    title: str
    num_pages: int
    pages: list[PageContent]
    warnings: list[ParseWarning]
    audits: list[PageAudit]


@dataclass(frozen=True, slots=True)
class _TextBlock:
    """Text and typography information for one page block."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    font_sizes: tuple[float, ...]
    has_bold_span: bool

    @property
    def width(self) -> float:
        """Return the horizontal span of the block."""
        return self.x1 - self.x0


def parse_pdf(path: Path, doc_id: str) -> ParsedPdf:
    """Extract clean text, headings, tables, and diagnostics from a PDF.

    The parser deliberately keeps original PDF page indexes (starting at one),
    including when empty pages are excluded from the returned content list.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        msg = f"PDF file does not exist: {pdf_path}"
        raise PdfNotFoundError(msg)

    try:
        pdf_document = pymupdf.open(pdf_path)
    except (pymupdf.FileDataError, OSError, RuntimeError) as error:
        msg = f"Could not open PDF '{pdf_path}'; it may be corrupt or unsupported."
        raise PdfCorruptError(msg) from error

    try:
        if pdf_document.needs_pass:
            msg = f"PDF '{pdf_path}' is encrypted; password-protected files are not supported."
            raise PdfEncryptedError(msg)
        return _parse_open_document(pdf_document, pdf_path, doc_id)
    finally:
        pdf_document.close()


def _parse_open_document(pdf_document: pymupdf.Document, pdf_path: Path, doc_id: str) -> ParsedPdf:
    """Parse an already-open, unencrypted PyMuPDF document."""
    pages_and_blocks = [(page, _extract_text_blocks(page)) for page in pdf_document]
    repeated_edges = _find_repeated_edge_texts(pages_and_blocks)
    title = _document_title(pdf_document, pdf_path)
    pages: list[PageContent] = []
    audits: list[PageAudit] = []
    warnings: list[ParseWarning] = []

    try:
        with pdfplumber.open(pdf_path) as plumber_document:
            for page_index, (page, blocks) in enumerate(pages_and_blocks, start=1):
                ordered_blocks = _sort_in_reading_order(blocks, page.rect.width)
                raw_text = "\n".join(block.text for block in ordered_blocks)
                retained_blocks, removed_edges = _remove_repeated_edges(
                    ordered_blocks, page.rect.height, repeated_edges
                )
                cleaned_text = clean_text("\n".join(block.text for block in retained_blocks))
                tables, table_warning = _extract_tables(plumber_document.pages[page_index - 1])
                if table_warning is not None:
                    warnings.append(
                        ParseWarning(
                            page_number=page_index,
                            kind="table_extraction",
                            message=table_warning,
                        )
                    )
                headings = _detect_headings(retained_blocks)
                audits.append(
                    PageAudit(
                        page_number=page_index,
                        raw_text=raw_text,
                        cleaned_text=cleaned_text,
                        removed_edge_text=tuple(removed_edges),
                    )
                )
                if not cleaned_text and not tables:
                    warning = _empty_page_warning(page, page_index)
                    warnings.append(warning)
                    logger.warning("%s", warning.message)
                    continue
                page_content = PageContent(
                    doc_id=doc_id,
                    page_number=page_index,
                    text=cleaned_text,
                    tables=tables,
                    headings=headings,
                )
                if page_content.text_quality < LOW_TEXT_QUALITY_THRESHOLD:
                    warning = ParseWarning(
                        page_number=page_index,
                        kind="low_text_quality",
                        message=(
                            f"PDF page {page_index} has low text quality "
                            f"({page_content.text_quality:.3f} < "
                            f"{LOW_TEXT_QUALITY_THRESHOLD:.3f}); review the source page before use."
                        ),
                    )
                    warnings.append(warning)
                    logger.warning("%s", warning.message)
                pages.append(page_content)
    except PdfEncryptedError:
        raise
    except Exception as error:
        msg = f"Could not decode PDF '{pdf_path}'; it may be corrupt or unsupported."
        raise PdfCorruptError(msg) from error

    return ParsedPdf(
        title=title,
        num_pages=len(pdf_document),
        pages=pages,
        warnings=warnings,
        audits=audits,
    )


def _document_title(pdf_document: pymupdf.Document, pdf_path: Path) -> str:
    """Use embedded metadata when available, otherwise derive a readable title."""
    metadata_title = (pdf_document.metadata.get("title") or "").strip()
    return metadata_title or pdf_path.stem.replace("_", " ").replace("-", " ").title()


def _extract_text_blocks(page: pymupdf.Page) -> list[_TextBlock]:
    """Read text spans without trusting source order from complex page layouts."""
    page_dictionary: dict[str, Any] = page.get_text("dict", sort=False)
    blocks: list[_TextBlock] = []
    for raw_block in page_dictionary["blocks"]:
        if raw_block["type"] != 0:
            continue
        lines = raw_block.get("lines", [])
        line_texts: list[str] = []
        font_sizes: list[float] = []
        has_bold_span = False
        for line in lines:
            spans = line.get("spans", [])
            line_texts.append("".join(span.get("text", "") for span in spans))
            font_sizes.extend(float(span.get("size", 0)) for span in spans)
            has_bold_span = has_bold_span or any(int(span.get("flags", 0)) & 16 for span in spans)
        text = "\n".join(line_texts).strip()
        if not text:
            continue
        x0, y0, x1, y1 = raw_block["bbox"]
        blocks.append(
            _TextBlock(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                text=text,
                font_sizes=tuple(font_sizes),
                has_bold_span=has_bold_span,
            )
        )
    return blocks


def _sort_in_reading_order(blocks: list[_TextBlock], page_width: float) -> list[_TextBlock]:
    """Sort single and two-column pages in a predictable top-to-bottom order."""
    vertically_sorted = sorted(blocks, key=lambda block: (block.y0, block.x0))
    if len(blocks) < 2:
        return vertically_sorted

    middle = page_width / 2
    left_column = [block for block in blocks if block.x1 <= middle]
    right_column = [block for block in blocks if block.x0 >= middle]
    spanning = [block for block in blocks if block not in left_column + right_column]
    has_two_columns = bool(left_column and right_column)
    if not has_two_columns:
        return vertically_sorted

    first_column_y = min(block.y0 for block in left_column + right_column)
    top_spanning = sorted(
        (block for block in spanning if block.y1 <= first_column_y), key=lambda block: block.y0
    )
    other_spanning = sorted(
        (block for block in spanning if block.y1 > first_column_y), key=lambda block: block.y0
    )
    return (
        top_spanning
        + sorted(left_column, key=lambda block: block.y0)
        + sorted(right_column, key=lambda block: block.y0)
        + other_spanning
    )


def _find_repeated_edge_texts(
    pages_and_blocks: list[tuple[pymupdf.Page, list[_TextBlock]]],
) -> set[str]:
    """Find top/bottom block strings repeated across a majority of pages."""
    occurrences: dict[str, int] = {}
    for page, blocks in pages_and_blocks:
        edge_blocks = [
            block
            for block in blocks
            if block.y0 <= page.rect.height * 0.1 or block.y1 >= page.rect.height * 0.9
        ]
        for edge_block in edge_blocks:
            normalized = _frequency_key(edge_block.text)
            if normalized:
                occurrences[normalized] = occurrences.get(normalized, 0) + 1
    threshold = max(2, ceil(len(pages_and_blocks) * 0.6))
    return {text for text, count in occurrences.items() if count >= threshold}


def _remove_repeated_edges(
    blocks: list[_TextBlock], page_height: float, repeated_edges: set[str]
) -> tuple[list[_TextBlock], list[str]]:
    """Remove only frequent edge blocks and standalone PDF page numbers."""
    retained: list[_TextBlock] = []
    removed: list[str] = []
    for block in blocks:
        is_edge = block.y0 <= page_height * 0.1 or block.y1 >= page_height * 0.9
        frequency_key = _frequency_key(block.text)
        should_remove = is_edge and (
            frequency_key in repeated_edges or _is_page_number(frequency_key)
        )
        if should_remove:
            removed.append(block.text)
        else:
            retained.append(block)
    return retained, removed


def _frequency_key(text: str) -> str:
    """Normalise changing page-number digits before counting repeated edge text."""
    normalized = _WHITESPACE_PATTERN.sub(" ", text).strip().lower()
    return re.sub(r"\d+", "#", normalized)


def _is_page_number(frequency_key: str) -> bool:
    """Recognise standalone printed page-number blocks at document edges."""
    return bool(_PAGE_NUMBER_PATTERN.fullmatch(frequency_key.replace("#", "1")))


def clean_text(text: str) -> str:
    """Normalise Unicode and whitespace while repairing hyphenated line breaks.

    Private-use area glyphs (``\\uE000``-``\\uF8FF`` and supplementary planes) and
    non-printing format characters are dropped: they carry no visible text and
    would otherwise pollute retrieval tokens and quality scores.
    """
    normalized = unicodedata.normalize("NFKC", text)
    repaired = _HYPHENATED_LINEBREAK_PATTERN.sub("", normalized)
    return _WHITESPACE_PATTERN.sub(" ", _strip_non_printing_characters(repaired)).strip()


def _strip_non_printing_characters(text: str) -> str:
    """Remove private-use and format characters that cannot render as text."""
    return "".join(
        character for character in text if unicodedata.category(character) not in {"Co", "Cf"}
    )


def _detect_headings(blocks: list[_TextBlock]) -> list[str]:
    """Infer heading blocks from relative font size or bold typography.

    Long all-caps lines such as a cover page's ``FORM 10-K ...`` heading are
    rejected: they are page titles, not section headings, and keeping them
    would seed the structure-aware chunker with noisy section boundaries.
    """
    font_sizes = [size for block in blocks for size in block.font_sizes if size > 0]
    if not font_sizes:
        return []
    baseline_size = statistics.median(font_sizes)
    headings: list[str] = []
    for block in blocks:
        block_size = max(block.font_sizes, default=0)
        is_large = block_size >= baseline_size * 1.25
        is_emphasised = block.has_bold_span and block_size > baseline_size
        candidate = clean_text(block.text)
        if (
            candidate
            and (is_large or is_emphasised)
            and candidate not in headings
            and len(candidate) <= MAX_HEADING_LENGTH
        ):
            headings.append(candidate)
    return headings


def _extract_tables(plumber_page: pdfplumber.page.Page) -> tuple[list[str], str | None]:
    """Extract ruled tables and render each as pipe-delimited Markdown."""
    try:
        raw_tables = plumber_page.extract_tables(
            table_settings={"snap_tolerance": 3, "join_tolerance": 3}
        )
    except Exception as error:  # pdfplumber can fail on malformed layout objects.
        return [], f"Could not extract tables on PDF page: {error}"
    return [markdown_table(table) for table in raw_tables if _is_usable_table(table)], None


def _is_usable_table(table: list[list[str | None]]) -> bool:
    """Reject single-column prose regions that ruled-line detection mistakes for tables."""
    has_multiple_columns = max((len(row) for row in table), default=0) >= 2
    has_row_with_multiple_values = any(
        sum(bool(cell and cell.strip()) for cell in row) >= 2 for row in table
    )
    return has_multiple_columns and has_row_with_multiple_values


def markdown_table(table: list[list[str | None]]) -> str:
    """Convert a pdfplumber row matrix to readable GitHub-flavoured Markdown."""
    width = max(len(row) for row in table)
    rows = [_normalise_table_row(row, width) for row in table]
    header = rows[0]
    separator = ["---"] * width
    body = rows[1:]
    return "\n".join(_markdown_row(row) for row in [header, separator, *body])


def _normalise_table_row(row: list[str | None], width: int) -> list[str]:
    """Pad a table row and make inline cell content Markdown-safe."""
    normalized = [clean_text(cell or "").replace("|", "\\|") for cell in row]
    return [*normalized, *([""] * (width - len(normalized)))]


def _markdown_row(cells: list[str]) -> str:
    """Render one Markdown table row."""
    return "| " + " | ".join(cells) + " |"


def _empty_page_warning(page: pymupdf.Page, page_number: int) -> ParseWarning:
    """Classify an empty page as scanned/image-only when it contains images."""
    if page.get_images(full=True):
        return ParseWarning(
            page_number=page_number,
            kind="scanned_page",
            message=(
                f"PDF page {page_number} appears image-only; OCR is not implemented and the page "
                "was skipped."
            ),
        )
    return ParseWarning(
        page_number=page_number,
        kind="empty_page",
        message=f"PDF page {page_number} contains no extractable text or table and was skipped.",
    )
