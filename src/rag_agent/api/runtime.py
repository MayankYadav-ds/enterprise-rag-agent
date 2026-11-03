"""Lazy per-application services and coordinated managed document mutations."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock

from qdrant_client import QdrantClient

from rag_agent.api.documents import DocumentManager
from rag_agent.api.schemas import RetrievalRequest
from rag_agent.api.settings import APISettings
from rag_agent.api.storage import private_directory, private_file
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.protocol import Embedder
from rag_agent.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_agent.embeddings.store import QdrantStore
from rag_agent.generation import AnswerGenerator, GenerationConfig, build_provider
from rag_agent.retrieval.bm25 import BM25Retriever
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.hybrid import HybridRetriever
from rag_agent.retrieval.models import RetrievedChunk
from rag_agent.retrieval.protocol import Retriever
from rag_agent.retrieval.reranker import CrossEncoderReranker, RerankerConfig


class DependencyUnavailable(RuntimeError):
    """A required service, model, or configuration is unavailable."""


class Runtime:
    """Cache models and BM25 snapshots; serialize retrieval to protect model use.

    Clients belong to this runtime and are closed by the application lifespan.
    Failed initialization is not cached, so a subsequent request can retry.
    """

    def __init__(self, settings: APISettings) -> None:
        self.settings = settings
        self.lock = RLock()
        self._client: QdrantClient | None = None
        self._retrievers: dict[str, Retriever] = {}
        self._generator: AnswerGenerator | None = None
        self._embedder: Embedder | None = None
        self._documents: DocumentManager | None = None

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived connection to the app's shared local registry DB."""
        path = self.settings.document_dir / "inventory.sqlite3"
        private_directory(path.parent)
        connection = sqlite3.connect(path, timeout=10)
        private_file(path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def documents(self) -> DocumentManager:
        if self._documents is not None:
            return self._documents
        with self.lock:
            if self._documents is None:
                self._documents = DocumentManager(self)
            return self._documents

    def embedding(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder()
        return self._embedder

    def invalidate_retrieval(self) -> None:
        self._retrievers.clear()

    def _store(self, embedder: Embedder, *, validate: bool) -> QdrantStore:
        if self._client is None:
            key = self.settings.qdrant_api_key
            self._client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=key.get_secret_value() if key else None,
                timeout=self.settings.qdrant_timeout,
            )
        return QdrantStore(
            embedder,
            collection_name=self.settings.collection,
            client=self._client,
            validate_dimension=validate,
            create_if_missing=False,
        )

    def _base(self, strategy: str) -> Retriever:
        if strategy not in self._retrievers:
            if strategy == "bm25":
                # The placeholder never embeds; payload-only access needs no ML model.
                store = self._store(FakeEmbedder(), validate=False)
                retriever = BM25Retriever(store.scroll_payloads())
            elif strategy == "dense":
                # Check service/collection before the expensive local model load.
                self._store(FakeEmbedder(), validate=False)
                embedder = self.embedding()
                retriever = DenseRetriever(embedder, self._store(embedder, validate=True))
            else:
                retriever = HybridRetriever(self._base("dense"), self._base("bm25"))
            self._retrievers[strategy] = retriever
        return self._retrievers[strategy]

    def retrieve(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        with self.lock:
            try:
                if (
                    self._documents is not None
                    or (self.settings.document_dir / "inventory.sqlite3").exists()
                ):
                    self.documents().ensure_readable()
                retriever = self._base(request.strategy)
                if request.rerank:
                    key = f"reranked_{request.strategy}"
                    if key not in self._retrievers:
                        config = RerankerConfig.from_env(
                            top_k=1, candidate_count=self.settings.rerank_candidates
                        )
                        self._retrievers[key] = CrossEncoderReranker(retriever, config)
                    retriever = self._retrievers[key]
                return retriever.retrieve(
                    request.question,
                    top_k=request.top_k,
                    doc_id=request.doc_id,
                    chunk_type=request.chunk_type,
                    min_score=request.min_score,
                )
            except Exception as exc:
                raise DependencyUnavailable("Retrieval service or model unavailable") from exc

    def generator(self) -> AnswerGenerator:
        with self.lock:
            if self._generator is None:
                try:
                    config = GenerationConfig.from_env()
                    self._generator = AnswerGenerator(
                        build_provider(config),
                        max_evidence_chars=config.max_evidence_chars,
                        max_chunks=config.max_chunks,
                    )
                except Exception as exc:
                    raise DependencyUnavailable("Generation configuration unavailable") from exc
            return self._generator

    def close(self) -> None:
        with self.lock:
            if self._client is not None:
                self._client.close()
                self._client = None
            self._retrievers.clear()
            self._generator = None
            self._embedder = None
            if self._documents is not None:
                self._documents.close()
                self._documents = None
