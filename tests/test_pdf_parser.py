"""Tests for text, heading, table, and source-quality PDF parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_agent.ingestion.exceptions import PdfCorruptError
from rag_agent.ingestion.pdf_parser import clean_text, parse_pdf

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_report.pdf"
DOC_ID = "doc_" + "a" * 64


def test_parser_keeps_original_page_numbers_and_detects_headings() -> None:
    """Pages and typography metadata preserve citation-ready PDF page indexes."""
    parsed = parse_pdf(FIXTURE_PATH, DOC_ID)

    assert parsed.num_pages == 2
    assert [page.page_number for page in parsed.pages] == [1, 2]
    assert parsed.pages[0].headings == ["Financial Overview"]
    assert parsed.pages[1].headings == ["Operational Notes"]


def test_parser_extracts_markdown_table_and_cleans_repeated_edges() -> None:
    """A ruled table remains readable while repeated header/footer text is removed."""
    parsed = parse_pdf(FIXTURE_PATH, DOC_ID)
    first_page = parsed.pages[0]

    assert first_page.tables == [
        "| Metric | FY 2025 | FY 2024 |\n"
        "| --- | --- | --- |\n"
        "| Revenue | $125 | $100 |\n"
        "| Operating margin | 24% | 21% |"
    ]
    assert "ENTERPRISE RAG TEST REPORT" not in first_page.text
    assert "Page 1 of 2" not in first_page.text
    assert parsed.audits[0].removed_edge_text == (
        "ENTERPRISE RAG TEST REPORT",
        "Page 1 of 2",
    )


def test_parser_repairs_hyphenated_line_breaks() -> None:
    """A line-wrapped word does not become two misleading retrieval tokens."""
    parsed = parse_pdf(FIXTURE_PATH, DOC_ID)

    assert "This hyphenated phrase" in parsed.pages[0].text
    assert clean_text("cross-\nreference") == "crossreference"


def test_parser_rejects_corrupt_pdf(tmp_path: Path) -> None:
    """Invalid source bytes produce a clear domain exception rather than a library trace."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"this is not a PDF")

    with pytest.raises(PdfCorruptError, match="corrupt or unsupported"):
        parse_pdf(corrupt_file, DOC_ID)
