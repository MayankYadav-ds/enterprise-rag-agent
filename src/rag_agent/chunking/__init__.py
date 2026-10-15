"""Document chunking strategies."""

from .headings import attach_headings
from .models import Chunk
from .strategies import (
    ChunkingStrategy,
    FixedSizeChunking,
    RecursiveChunking,
    StructureAwareChunking,
)

__all__ = [
    "Chunk",
    "ChunkingStrategy",
    "FixedSizeChunking",
    "RecursiveChunking",
    "StructureAwareChunking",
    "attach_headings",
]
