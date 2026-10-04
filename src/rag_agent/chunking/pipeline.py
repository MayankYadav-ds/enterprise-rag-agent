"""Chunking pipeline for combining ingestion and chunking."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from rag_agent.chunking import (
    FixedSizeChunking,
    RecursiveChunking,
    StructureAwareChunking,
)
from rag_agent.chunking.models import Chunk
from rag_agent.ingestion.models import Document, PageContent
from rag_agent.ingestion.pdf_parser import LOW_TEXT_QUALITY_THRESHOLD
from rag_agent.ingestion.pipeline import ingest_pdf

logger = logging.getLogger(__name__)


def chunk_document(
    document: Document,
    pages: list[PageContent],
    strategy: str = "fixed_size",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Chunk a document using the specified strategy.

    Args:
        document: Source document metadata
        pages: List of page content to chunk
        strategy: Chunking strategy ("fixed_size", "recursive", or "structure_aware")
        chunk_size: Target chunk size in tokens
        chunk_overlap: Number of tokens to overlap between chunks

    Returns:
        List of Chunk objects
    """
    # Select strategy
    strategy_map = {
        "fixed_size": FixedSizeChunking,
        "recursive": RecursiveChunking,
        "structure_aware": StructureAwareChunking,
    }

    if strategy not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategy_map.keys())}")

    kept_pages, skipped = _exclude_low_quality_pages(pages)
    if skipped:
        logger.info(
            "Excluded %d low-quality page(s) below %.3f for doc %s",
            skipped,
            LOW_TEXT_QUALITY_THRESHOLD,
            document.doc_id,
        )

    chunking_strategy = strategy_map[strategy](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunking_strategy.chunk(kept_pages, document.doc_id)
    return [_with_source_text_quality(chunk, kept_pages) for chunk in chunks]


def _exclude_low_quality_pages(pages: list[PageContent]) -> tuple[list[PageContent], int]:
    """Drop pages whose ingestion quality is below the shared threshold."""
    kept: list[PageContent] = []
    skipped = 0
    for page in pages:
        if page.text_quality < LOW_TEXT_QUALITY_THRESHOLD:
            skipped += 1
            continue
        kept.append(page)
    return kept, skipped


def _with_source_text_quality(chunk: Chunk, pages: list[PageContent]) -> Chunk:
    """Copy a frozen chunk with the mean text_quality of overlapping source pages."""
    overlapping = [
        page.text_quality
        for page in pages
        if chunk.page_start <= page.page_number <= chunk.page_end
    ]
    if not overlapping:
        return chunk
    mean_quality = sum(overlapping) / len(overlapping)
    return replace(chunk, text_quality=mean_quality)


def chunk_and_save(
    pdf_path: Path,
    output_dir: Path = Path("data/processed"),
    strategy: str = "fixed_size",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """Ingest a PDF and chunk it, saving chunks as JSONL.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save chunked output
        strategy: Chunking strategy to use
        chunk_size: Target chunk size in tokens
        chunk_overlap: Number of tokens to overlap between chunks
    """
    # Ingest the PDF
    document, pages = ingest_pdf(pdf_path, output_dir)

    # Chunk the document
    chunks = chunk_document(document, pages, strategy, chunk_size, chunk_overlap)

    # Save chunks as JSONL
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = output_dir / f"{document.doc_id}_chunks.jsonl"

    with open(chunk_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # Convert chunk to dict for JSON serialization
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_heading": chunk.section_heading,
                "token_count": chunk.token_count,
                "text_quality": chunk.text_quality,
                "chunk_index": chunk.chunk_index,
            }
            f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")
