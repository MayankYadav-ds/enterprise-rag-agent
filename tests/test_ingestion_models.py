"""Tests for validated ingestion data models."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_agent.ingestion.models import (
    Document,
    PageContent,
    document_id_from_hash,
    sha256_for_file,
    text_quality_score,
)


def test_document_id_is_deterministic_from_file_hash() -> None:
    """Identical content digests must produce an identical stable document ID."""
    file_hash = "a" * 64
    document = Document(
        doc_id=document_id_from_hash(file_hash),
        source_path="data/raw/report.pdf",
        title="Report",
        num_pages=2,
        file_hash=file_hash.upper(),
        ingested_at=datetime(2026, 9, 30, tzinfo=UTC),
    )

    assert document.file_hash == file_hash
    assert document.doc_id == "doc_" + file_hash


def test_document_rejects_identifier_not_derived_from_hash() -> None:
    """A caller cannot accidentally make a non-idempotent document record."""
    with pytest.raises(ValidationError, match="doc_id must be derived"):
        Document(
            doc_id="doc_" + "b" * 64,
            source_path="report.pdf",
            title="Report",
            num_pages=1,
            file_hash="a" * 64,
        )


def test_page_content_derives_its_character_count() -> None:
    """Character count reflects normalized page text rather than caller input."""
    page = PageContent(
        doc_id="doc_" + "a" * 64,
        page_number=3,
        text="Clean page text",
        tables=["| Item | Value |\n| --- | --- |\n| A | 1 |"],
    )

    assert page.char_count == len("Clean page text")
    assert page.text_quality == 1.0


def test_page_content_rejects_an_incorrect_character_count() -> None:
    """An inaccurate text-size field cannot silently enter the index."""
    with pytest.raises(ValidationError, match="char_count must equal"):
        PageContent(
            doc_id="doc_" + "a" * 64,
            page_number=3,
            text="Clean page text",
            char_count=1,
        )


def test_sha256_for_file_is_stable(tmp_path: Path) -> None:
    """A content digest changes only when the source bytes change."""
    source = tmp_path / "source.pdf"
    source.write_bytes(b"same source bytes")

    assert sha256_for_file(str(source)) == sha256_for_file(str(source))


def test_document_identifier_rejects_non_sha256_hashes() -> None:
    """The file hash field cannot contain an arbitrary identifier."""
    with pytest.raises(ValueError, match="SHA-256"):
        document_id_from_hash("not-a-hash")


def test_text_quality_penalises_known_pdf_extraction_artifacts() -> None:
    """Broken font mappings cannot look as trustworthy as clean source text."""
    text = "Readable (cid:123) text\ufffd with a control\x01character."
    page = PageContent(doc_id="doc_" + "a" * 64, page_number=1, text=text)

    assert 0.0 < page.text_quality < 1.0
    assert page.text_quality == text_quality_score(text)
    assert text_quality_score("") == 1.0
