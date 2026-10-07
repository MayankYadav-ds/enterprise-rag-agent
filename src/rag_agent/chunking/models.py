"""Chunk-level metadata for the RAG pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """A processed chunk of text or table ready for embedding and retrieval."""

    chunk_id: str
    doc_id: str
    text: str
    chunk_type: str  # "text" or "table"
    page_start: int
    page_end: int
    section_heading: str | None = None
    token_count: int = 0
    text_quality: float = 1.0
    chunk_index: int = 0

    @classmethod
    def create(
        cls,
        doc_id: str,
        text: str,
        chunk_type: str,
        page_start: int,
        page_end: int,
        chunk_index: int,
        section_heading: str | None = None,
        tokenizer=None,
    ) -> Chunk:
        """Create a chunk with deterministic ID and computed token count.

        Args:
            doc_id: Parent document identifier
            text: Chunk text content
            chunk_type: Either "text" or "table"
            page_start: Starting page number (1-indexed)
            page_end: Ending page number (1-indexed)
            chunk_index: Sequential index within the document
            section_heading: Optional section heading for context
            tokenizer: Optional tokenizer for token counting

        Returns:
            Chunk instance with computed chunk_id and token_count
        """
        # Generate deterministic chunk ID from doc_id, position, and content hash
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        position_hash = hashlib.sha256(
            f"{doc_id}:{chunk_index}:{page_start}:{page_end}".encode()
        ).hexdigest()[:8]
        chunk_id = f"chunk_{doc_id}_{position_hash}_{content_hash}"

        # Count tokens if tokenizer provided, otherwise estimate
        token_count = 0
        if tokenizer:
            token_count = len(tokenizer.encode(text))
        else:
            # Rough estimation: ~4 characters per token for English text
            token_count = max(1, len(text) // 4)

        return cls(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            chunk_type=chunk_type,
            page_start=page_start,
            page_end=page_end,
            section_heading=section_heading,
            token_count=token_count,
            text_quality=1.0,  # Will be set later based on source page quality
            chunk_index=chunk_index,
        )
