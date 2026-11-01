"""Retrieval components.

Dense-vector retrieval (``DenseRetriever``), BM25 retrieval
(``BM25Retriever``), reciprocal-rank-fusion hybrid retrieval
(``HybridRetriever``), and optional cross-encoder reranking
(``CrossEncoderReranker``, ``RerankerConfig``) are implemented. Every
retriever implements the same ``Retriever`` protocol, so the answer generator
and citation layer are retriever-agnostic.
"""

from rag_agent.retrieval.bm25 import BM25Retriever
from rag_agent.retrieval.citation import format_citation, to_dict
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.hybrid import HybridRetriever
from rag_agent.retrieval.models import RetrievedChunk
from rag_agent.retrieval.protocol import Retriever
from rag_agent.retrieval.reranker import (
    CrossEncoderReranker,
    CrossEncoderUnavailable,
    FakeCrossEncoder,
    RerankerConfig,
    build_reranked_retriever,
)

__all__ = [
    "BM25Retriever",
    "CrossEncoderReranker",
    "CrossEncoderUnavailable",
    "DenseRetriever",
    "FakeCrossEncoder",
    "HybridRetriever",
    "RerankerConfig",
    "RetrievedChunk",
    "Retriever",
    "build_reranked_retriever",
    "format_citation",
    "to_dict",
]
