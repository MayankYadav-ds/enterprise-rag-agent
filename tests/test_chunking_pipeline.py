"""Tests for chunking pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from rag_agent.chunking.models import Chunk
from rag_agent.chunking.pipeline import chunk_and_save, chunk_document
from rag_agent.ingestion.models import Document, PageContent, document_id_from_hash

FILE_HASH = "a" * 64
DOC_ID = document_id_from_hash(FILE_HASH)


def create_test_pages() -> list[PageContent]:
    """Create test pages for chunking tests."""
    return [
        PageContent(
            doc_id=DOC_ID,
            page_number=1,
            text=(
                "This is the first sentence. This is the second sentence. "
                "This is a longer sentence that should be split appropriately. "
                "Another sentence here. And another one."
            ),
            tables=[],
            headings=[],
            char_count=0,
            text_quality=1.0,
        ),
        PageContent(
            doc_id=DOC_ID,
            page_number=2,
            text=(
                "This is page two. It has some content that should also be chunked. "
                "We need to test page boundaries correctly."
            ),
            tables=[],
            headings=[],
            char_count=0,
            text_quality=1.0,
        ),
    ]


def create_test_document() -> Document:
    """Create a test document."""
    return Document(
        doc_id=DOC_ID,
        source_path="data/raw/test.pdf",
        title="Test Document",
        num_pages=2,
        file_hash=FILE_HASH,
        ingested_at=datetime(2023, 1, 1, tzinfo=UTC),
    )


def test_chunk_document_fixed_size() -> None:
    """Test chunk_document with fixed_size strategy."""
    pages = create_test_pages()
    document = create_test_document()

    chunks = chunk_document(
        document=document,
        pages=pages,
        strategy="fixed_size",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.doc_id == document.doc_id
        assert chunk.chunk_type == "text"
        assert chunk.token_count > 0
        assert chunk.token_count <= 120


def test_chunk_document_recursive() -> None:
    """Test chunk_document with recursive strategy."""
    pages = create_test_pages()
    document = create_test_document()

    chunks = chunk_document(
        document=document,
        pages=pages,
        strategy="recursive",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.doc_id == document.doc_id
        assert chunk.chunk_type == "text"
        assert chunk.token_count > 0
        assert chunk.token_count <= 120


def test_chunk_document_structure_aware() -> None:
    """Test chunk_document with structure_aware strategy."""
    pages = create_test_pages()
    document = create_test_document()

    chunks = chunk_document(
        document=document,
        pages=pages,
        strategy="structure_aware",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.doc_id == document.doc_id
        assert chunk.chunk_type == "text"
        assert chunk.token_count > 0
        assert chunk.token_count <= 120


def test_chunk_document_invalid_strategy() -> None:
    """Test chunk_document with invalid strategy raises ValueError."""
    pages = create_test_pages()
    document = create_test_document()

    with pytest.raises(ValueError, match="Unknown strategy: invalid_strategy"):
        chunk_document(
            document=document,
            pages=pages,
            strategy="invalid_strategy",
            chunk_size=100,
            chunk_overlap=20,
        )


def test_chunk_document_empty_pages() -> None:
    """Test chunk_document with empty pages."""
    document = create_test_document()

    chunks = chunk_document(
        document=document,
        pages=[],
        strategy="fixed_size",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) == 0


@patch("rag_agent.chunking.pipeline.ingest_pdf")
def test_chunk_and_save(mock_ingest_pdf: Mock, tmp_path: Path) -> None:
    """Test chunk_and_save function."""
    mock_ingest_pdf.return_value = (create_test_document(), create_test_pages())

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("fake pdf content")
    output_dir = tmp_path / "output"

    chunk_and_save(
        pdf_path=pdf_path,
        output_dir=output_dir,
        strategy="fixed_size",
        chunk_size=100,
        chunk_overlap=20,
    )

    mock_ingest_pdf.assert_called_once_with(pdf_path, output_dir)

    expected_file = output_dir / f"{DOC_ID}_chunks.jsonl"
    assert expected_file.exists()

    lines = expected_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0
    for line in lines:
        chunk_data = json.loads(line)
        assert "chunk_id" in chunk_data
        assert chunk_data["doc_id"] == DOC_ID
        assert "text" in chunk_data


def test_chunk_and_save_creates_output_dir(tmp_path: Path) -> None:
    """Test that chunk_and_save creates output directory if it doesn't exist."""
    with patch("rag_agent.chunking.pipeline.ingest_pdf") as mock_ingest_pdf:
        mock_ingest_pdf.return_value = (create_test_document(), create_test_pages())

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("fake pdf content")
        output_dir = tmp_path / "new_output"

        chunk_and_save(
            pdf_path=pdf_path,
            output_dir=output_dir,
            strategy="fixed_size",
            chunk_size=100,
            chunk_overlap=20,
        )

        assert output_dir.is_dir()
        assert (output_dir / f"{DOC_ID}_chunks.jsonl").exists()


def test_chunk_document_propagates_page_text_quality(caplog: pytest.LogCaptureFixture) -> None:
    """Chunks inherit source-page quality; pages below 0.900 are excluded and counted."""
    good_text = ("This is good quality text with proper sentences and enough length. " * 4).strip()
    mixed_text = ("Readable sentence about operations and cash flow. " * 8) + ("\x01" * 20)
    bad_text = "\ufffd" * 40

    pages = [
        PageContent(doc_id=DOC_ID, page_number=1, text=good_text),
        PageContent(doc_id=DOC_ID, page_number=2, text=mixed_text),
        PageContent(doc_id=DOC_ID, page_number=3, text=bad_text),
    ]
    document = create_test_document()
    document = document.model_copy(update={"num_pages": 3})

    assert pages[0].text_quality == 1.0
    assert 0.900 <= pages[1].text_quality < 1.0
    assert pages[2].text_quality < 0.900

    with caplog.at_level("INFO", logger="rag_agent.chunking.pipeline"):
        chunks = chunk_document(
            document=document,
            pages=pages,
            strategy="fixed_size",
            chunk_size=512,
            chunk_overlap=50,
        )

    assert any(chunk.page_start == 1 for chunk in chunks)
    assert any(chunk.page_start == 2 for chunk in chunks)
    assert all(chunk.page_start != 3 and chunk.page_end != 3 for chunk in chunks)

    page_one_chunks = [chunk for chunk in chunks if chunk.page_start == 1]
    page_two_chunks = [chunk for chunk in chunks if chunk.page_start == 2]
    assert page_one_chunks
    assert page_two_chunks
    assert all(chunk.text_quality == pages[0].text_quality for chunk in page_one_chunks)
    assert all(chunk.text_quality == pages[1].text_quality for chunk in page_two_chunks)
    assert "Excluded 1 low-quality page" in caplog.text


def test_with_source_text_quality_no_overlap() -> None:
    """Cover the 'no overlapping pages' branch in _with_source_text_quality (line 86)."""
    from rag_agent.chunking.pipeline import _with_source_text_quality

    doc_id = DOC_ID
    # Create a chunk with page numbers not present in the pages list
    chunk = Chunk.create(
        doc_id=doc_id,
        text="Some text.",
        chunk_type="text",
        page_start=99,
        page_end=99,
        chunk_index=0,
    )

    # Empty pages list → no overlap → returns unchanged
    result = _with_source_text_quality(chunk, [])
    assert result is chunk
    assert result.text_quality == 1.0

    # Pages with different page numbers → no overlap → returns unchanged
    pages = [
        PageContent(
            doc_id=doc_id,
            page_number=1,
            text="Page one.",
            char_count=0,
            text_quality=0.9,
        )
    ]
    result = _with_source_text_quality(chunk, pages)
    assert result is chunk
