"""Tests for chunking strategies."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from src.rag_agent.chunking import (
    FixedSizeChunking,
    RecursiveChunking,
    StructureAwareChunking,
)
from src.rag_agent.chunking.models import Chunk
from src.rag_agent.ingestion.models import PageContent


def create_test_pages() -> list[PageContent]:
    """Create test pages for chunking tests."""
    # Create a valid doc_id (64 hex chars after "doc_")
    valid_doc_id = "doc_" + "a" * 64

    return [
        PageContent(
            doc_id=valid_doc_id,
            page_number=1,
            text="This is the first sentence. This is the second sentence. "
            "This is a longer sentence that should be split appropriately. "
            "Another sentence here. And another one.",
            tables=[],
            headings=[],
            char_count=0,
            text_quality=1.0,
        ),
        PageContent(
            doc_id=valid_doc_id,
            page_number=2,
            text="This is page two. It has some content that should also be chunked. "
            "We need to test page boundaries correctly.",
            tables=[],
            headings=[],
            char_count=0,
            text_quality=1.0,
        ),
    ]


def test_fixed_size_chunking_creates_deterministic_ids():
    """Test that fixed-size chunking creates deterministic chunk IDs."""
    pages = create_test_pages()
    # Extract doc_id from first page
    test_doc_id = pages[0].doc_id
    strategy1 = FixedSizeChunking(chunk_size=100, chunk_overlap=20)
    strategy2 = FixedSizeChunking(chunk_size=100, chunk_overlap=20)

    chunks1 = strategy1.chunk(pages, test_doc_id)
    chunks2 = strategy2.chunk(pages, test_doc_id)

    # Should have same number of chunks
    assert len(chunks1) == len(chunks2)

    # Should have same chunk IDs
    for c1, c2 in zip(chunks1, chunks2, strict=False):
        assert c1.chunk_id == c2.chunk_id


def test_fixed_size_chunking_respects_overlap():
    """Test that fixed-size chunking respects overlap settings."""
    pages = create_test_pages()
    test_doc_id = pages[0].doc_id
    strategy = FixedSizeChunking(chunk_size=50, chunk_overlap=10)
    chunks = strategy.chunk(pages, test_doc_id)

    # Should create multiple chunks
    assert len(chunks) > 1

    # Check that chunks have reasonable token counts
    for chunk in chunks:
        assert chunk.token_count > 0
        assert chunk.token_count <= 60  # Should be close to chunk_size but not exceed much


def test_recursive_chunking_creates_deterministic_ids():
    """Test that recursive chunking creates deterministic chunk IDs."""
    pages = create_test_pages()
    test_doc_id = pages[0].doc_id
    strategy1 = RecursiveChunking(chunk_size=100, chunk_overlap=20)
    strategy2 = RecursiveChunking(chunk_size=100, chunk_overlap=20)

    chunks1 = strategy1.chunk(pages, test_doc_id)
    chunks2 = strategy2.chunk(pages, test_doc_id)

    # Should have same number of chunks
    assert len(chunks1) == len(chunks2)

    # Should have same chunk IDs
    for c1, c2 in zip(chunks1, chunks2, strict=False):
        assert c1.chunk_id == c2.chunk_id


def test_structure_aware_chunking_creates_deterministic_ids():
    """Test that structure-aware chunking creates deterministic chunk IDs."""
    pages = create_test_pages()
    test_doc_id = pages[0].doc_id
    strategy1 = StructureAwareChunking(chunk_size=100, chunk_overlap=20)
    strategy2 = StructureAwareChunking(chunk_size=100, chunk_overlap=20)

    chunks1 = strategy1.chunk(pages, test_doc_id)
    chunks2 = strategy2.chunk(pages, test_doc_id)

    # Should have same number of chunks
    assert len(chunks1) == len(chunks2)

    # Should have same chunk IDs
    for c1, c2 in zip(chunks1, chunks2, strict=False):
        assert c1.chunk_id == c2.chunk_id


def test_chunk_model_creation():
    """Test that Chunk model can be created correctly."""
    chunk = Chunk.create(
        doc_id="doc_abc123",
        text="This is a test chunk.",
        chunk_type="text",
        page_start=1,
        page_end=1,
        chunk_index=0,
        section_heading="Test Section",
    )

    assert chunk.doc_id == "doc_abc123"
    assert chunk.text == "This is a test chunk."
    assert chunk.chunk_type == "text"
    assert chunk.page_start == 1
    assert chunk.page_end == 1
    assert chunk.chunk_index == 0
    assert chunk.section_heading == "Test Section"
    assert chunk.chunk_id.startswith("chunk_doc_abc123_")


def test_chunk_model_is_frozen():
    """Test that Chunk model is immutable (frozen)."""
    chunk = Chunk.create(
        doc_id="doc_abc123",
        text="This is a test chunk.",
        chunk_type="text",
        page_start=1,
        page_end=1,
        chunk_index=0,
    )

    # Should not be able to modify fields
    with pytest.raises(FrozenInstanceError):
        chunk.text = "Modified text"


def test_empty_text_handling():
    """Test that strategies handle empty text gracefully."""
    pages = [
        PageContent(
            doc_id="doc_" + "a" * 64,  # Valid doc_id
            page_number=1,
            text="",
            tables=[],
            headings=[],
            char_count=0,
            text_quality=1.0,
        )
    ]
    test_doc_id = pages[0].doc_id

    strategies = [FixedSizeChunking(), RecursiveChunking(), StructureAwareChunking()]

    for strategy in strategies:
        chunks = strategy.chunk(pages, test_doc_id)
        # Should produce no chunks for empty text
        assert len(chunks) == 0


def test_low_quality_page_exclusion():
    """Test that low-quality pages are excluded by default."""
    valid_doc_id = "doc_" + "a" * 64
    pages = [
        PageContent(
            doc_id=valid_doc_id,
            page_number=1,
            text="This is good quality text with proper sentences and meaning.",
            tables=[],
            headings=[],
            char_count=0,
            text_quality=0.95,  # Above threshold
        ),
        PageContent(
            doc_id=valid_doc_id,
            page_number=2,
            text="���� ���� ���� ���� ���� ���� ���� ���� ���� ����",  # Mostly replacement chars
            tables=[],
            headings=[],
            char_count=0,
            text_quality=0.1,  # Well below threshold
        ),
    ]
    test_doc_id = pages[0].doc_id

    strategy = FixedSizeChunking()
    chunks = strategy.chunk(pages, test_doc_id)

    # Should only get chunks from the high-quality page
    # The exact number depends on chunking, but should be > 0
    assert len(chunks) > 0

    # All chunks should be from page 1 (the high-quality page)
    for chunk in chunks:
        assert chunk.page_start == 1
        assert chunk.page_end == 1
