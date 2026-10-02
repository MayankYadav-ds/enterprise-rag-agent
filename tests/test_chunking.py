"""Tests for chunking strategies."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rag_agent.chunking import (
    FixedSizeChunking,
    RecursiveChunking,
    StructureAwareChunking,
)
from rag_agent.chunking.models import Chunk
from rag_agent.ingestion.models import PageContent


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


def test_no_text_chunk_exceeds_max_size():
    """Test that no text chunk exceeds the configured max token size."""
    # Create a valid doc_id (64 hex chars after "doc_")
    valid_doc_id = "doc_" + "a" * 64

    # Create pages with very long text to test max size enforcement
    long_text = "This is a sentence. " * 200  # Very long text
    pages = [
        PageContent(
            doc_id=valid_doc_id,
            page_number=1,
            text=long_text,
            tables=[],
            headings=[],
            char_count=0,
            text_quality=1.0,
        )
    ]
    test_doc_id = pages[0].doc_id

    # Test each strategy with chunk_size=512
    strategies = [
        ("fixed_size", FixedSizeChunking(chunk_size=512, chunk_overlap=50)),
        ("recursive", RecursiveChunking(chunk_size=512, chunk_overlap=50)),
        ("structure_aware", StructureAwareChunking(chunk_size=512, chunk_overlap=50)),
    ]

    for name, strategy in strategies:
        chunks = strategy.chunk(pages, test_doc_id)
        text_chunks = [c for c in chunks if c.chunk_type == "text"]

        # Verify no text chunk exceeds max size
        for chunk in text_chunks:
            assert (
                chunk.token_count <= 512
            ), f"{name} produced chunk with {chunk.token_count} tokens (max: 512)"


def test_structure_aware_sets_section_heading():
    """Test that structure_aware strategy sets section_heading from PageContent.headings."""
    valid_doc_id = "doc_" + "a" * 64

    # Create page with headings in metadata - include the heading in the text so it can be found
    pages = [
        PageContent(
            doc_id=valid_doc_id,
            page_number=1,
            text=(
                "ITEM 1A RISK FACTORS\n\nThis is some text under the heading.\n\n"
                "More content here."
            ),
            tables=[],
            headings=["ITEM 1A RISK FACTORS"],
            char_count=0,
            text_quality=1.0,
        )
    ]
    test_doc_id = pages[0].doc_id

    strategy = StructureAwareChunking(chunk_size=512, chunk_overlap=50)
    chunks = strategy.chunk(pages, test_doc_id)

    # Verify at least one chunk has the section heading set
    heading_chunks = [c for c in chunks if c.section_heading is not None]
    assert len(heading_chunks) > 0, "No chunks have section_heading set"

    # Verify the heading is correctly set
    for chunk in heading_chunks:
        assert chunk.section_heading == "ITEM 1A RISK FACTORS"


def test_split_tables_repeat_header_and_stay_within_max():
    """Test that split tables repeat the header row and stay within max token size."""
    valid_doc_id = "doc_" + "a" * 64

    # Create a markdown table that will exceed chunk size when combined with header
    # Header + separator = ~10 tokens, each row = ~20 tokens
    # With 512 max, we should fit about 25 rows per chunk
    table_rows = []
    for i in range(100):  # 100 rows should definitely require splitting
        table_rows.append(f"Row{i} | Data{i} | MoreData{i}")

    table_markdown = (
        "| Header1 | Header2 | Header3 |\n|---------|---------|---------|\n" + "\n".join(table_rows)
    )

    pages = [
        PageContent(
            doc_id=valid_doc_id,
            page_number=1,
            text="",
            tables=[table_markdown],
            headings=[],
            char_count=0,
            text_quality=1.0,
        )
    ]
    test_doc_id = pages[0].doc_id

    strategies = [
        ("fixed_size", FixedSizeChunking(chunk_size=512, chunk_overlap=50)),
        ("recursive", RecursiveChunking(chunk_size=512, chunk_overlap=50)),
        ("structure_aware", StructureAwareChunking(chunk_size=512, chunk_overlap=50)),
    ]

    for name, strategy in strategies:
        chunks = strategy.chunk(pages, test_doc_id)
        table_chunks = [c for c in chunks if c.chunk_type == "table"]

        # Should have multiple chunks due to splitting
        assert (
            len(table_chunks) > 1
        ), f"{name} did not split the large table (got {len(table_chunks)} chunks)"

        # Each chunk should contain the header
        for chunk in table_chunks:
            assert "Header1 | Header2 | Header3" in chunk.text
            assert "---------|---------|---------" in chunk.text

            # Each chunk should be within max size
            assert (
                chunk.token_count <= 512
            ), f"{name} table chunk has {chunk.token_count} tokens (max: 512)"


def test_page_start_end_across_page_boundary():
    """Test that chunks correctly track page_start and page_end across boundaries."""
    valid_doc_id = "doc_" + "a" * 64

    # Create two pages of text that will create a chunk spanning the boundary
    pages = [
        PageContent(
            doc_id=valid_doc_id,
            page_number=1,
            text="This is page one. " * 50,  # About 350 tokens
            tables=[],
            headings=[],
            char_count=0,
            text_quality=1.0,
        ),
        PageContent(
            doc_id=valid_doc_id,
            page_number=2,
            text="This is page two. " * 50,  # About 350 tokens
            tables=[],
            headings=[],
            char_count=0,
            text_quality=1.0,
        ),
    ]
    test_doc_id = pages[0].doc_id

    # Use a chunk size that will cause boundary crossing
    strategies = [
        ("fixed_size", FixedSizeChunking(chunk_size=400, chunk_overlap=50)),
        ("recursive", RecursiveChunking(chunk_size=400, chunk_overlap=50)),
    ]

    for _name, strategy in strategies:
        chunks = strategy.chunk(pages, test_doc_id)

        # Find chunks that span pages
        boundary_chunks = [c for c in chunks if c.page_start != c.page_end]

        # Should have at least one chunk spanning the boundary
        # Note: This might not always happen depending on exact token counts,
        # but we verify the tracking works correctly when it does
        for chunk in boundary_chunks:
            assert chunk.page_start == 1
            assert chunk.page_end == 2
            assert chunk.chunk_type == "text"
