"""Embedding pipeline components."""

from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.protocol import Embedder

__all__ = ["Embedder", "FakeEmbedder"]
