"""Retrieval components.

Dense-vector retrieval is implemented (``DenseRetriever``). BM25, reciprocal-rank
fusion and cross-encoder reranking are planned for later phases and implement
the same ``Retriever`` protocol, so the answer generator is retriever-agnostic.
"""

from rag_agent.retrieval.citation import format_citation, to_dict
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.models import RetrievedChunk
from rag_agent.retrieval.protocol import Retriever

__all__ = [
    "DenseRetriever",
    "RetrievedChunk",
    "Retriever",
    "format_citation",
    "to_dict",
]
