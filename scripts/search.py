"""Search the indexed Qdrant collection and print hits.

Embeds the query with the same embedder that indexed the collection, runs
``DenseRetriever.retrieve`` (or ``BM25Retriever`` / ``HybridRetriever``) and
prints one line per hit: rank, score, the citation
(``[file, p. 22, section heading]``) and a 120-character snippet of the chunk
text.

``--queries-file`` runs a whole file in ONE process: one model load, one
container run. Each query prints ``===== <query>`` and then the same hit lines
as single-query mode, so a batch output is still readable per query. Blank
lines and lines beginning with ``#`` are ignored, so the file can carry notes.

``--strategy`` selects the retrieval strategy: ``dense`` (default), ``bm25``,
or ``hybrid``. ``hybrid`` fuses dense and BM25 rankings with Reciprocal Rank
Fusion (see :class:`HybridRetriever`). The BM25 retriever is built from the
collection's payload dictionary, so it needs no model download and works on
machines that block torch.

``--rerank`` enables optional cross-encoder reranking on top of the selected
strategy. When enabled, the retriever fetches ``--rerank-candidates`` hits
from the base strategy, then re-scores them with a cross-encoder model and
returns the top ``--top-k`` reranked hits. The default reranker model is
``cross-encoder/ms-marco-MiniLM-L-6-v2`` (configurable via
``RERANKER_MODEL`` env var). Reranking requires the ``[embeddings]`` extra
and torch.

The real BAAI/bge-small-en-v1.5 model cannot be loaded on machines that block
torch (e.g. Windows with an Application Control policy, WinError 4551). On those
machines dense and hybrid search require a working torch environment.
BM25 does not need the embedder for scoring, but still opens the store.

The shared retrieval logic is unit-tested in ``tests/test_dense.py`` (the
``DenseRetriever``), ``tests/test_bm25.py``, ``tests/test_hybrid.py``,
``tests/test_reranker.py``, ``tests/test_citation.py`` and ``tests/test_retrieval.py``
with a FakeEmbedder and an in-memory Qdrant collection, so the code path is
covered without a model download.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_agent.embeddings.store import QdrantStore
from rag_agent.retrieval.bm25 import BM25Retriever
from rag_agent.retrieval.citation import format_citation
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.hybrid import HybridRetriever
from rag_agent.retrieval.protocol import Retriever
from rag_agent.retrieval.reranker import RerankerConfig, build_reranked_retriever

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
MAX_SNIPPET = 120
DEFAULT_STRATEGY = "dense"
STRATEGIES = ("dense", "bm25", "hybrid")
DEFAULT_RERANK_CANDIDATES = 20


def _read_queries_file(path: Path) -> list[str]:
    """Read one query per line from ``path``.

    Blank lines and lines beginning with ``#`` are ignored, so a queries
    file can carry notes and be edited without breaking the run. The model
    is embedded once for the whole file, so a batch of queries costs one
    model load, not one per query.
    """
    queries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
    if not queries:
        raise ValueError(f"queries file {path} contains no queries")
    return queries


def _print_hits(hits, collection: str) -> int:
    """Print the same hit lines as single-query mode; return 1 if no hits."""
    if not hits:
        logger.warning("No hits found in collection %r.", collection)
        return 1
    for hit in hits:
        snippet = (hit.text or "").replace("\n", " ")[:MAX_SNIPPET]
        print(
            f"rank={hit.rank} score={hit.score:.4f} "
            f"citation={format_citation(hit)} "
            f"snippet={snippet!r}"
        )
    return 0


def run_queries(
    retriever: Retriever,
    queries: list[str],
    *,
    top_k: int = DEFAULT_TOP_K,
    doc_id: str | None = None,
    chunk_type: str | None = None,
    min_score: float | None = None,
    collection: str = "rag_chunks",
) -> int:
    """Run a batch of queries through one retriever and print the output.

    The embedder and the store are constructed once by the caller, so the
    whole file shares one model load and one connection. Each query prints
    ``===== <query>`` and then the same hit lines as single-query mode.
    Returns the worst per-query exit code (1 if any query found no hits).
    """
    exit_code = 0
    for query in queries:
        print(f"===== {query}")
        exit_code = max(
            exit_code,
            _print_hits(
                retriever.retrieve(
                    query,
                    top_k=top_k,
                    doc_id=doc_id,
                    chunk_type=chunk_type,
                    min_score=min_score,
                ),
                collection,
            ),
        )
    return exit_code


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the Qdrant collection with dense, BM25 or hybrid retrieval, "
        "optionally reranked."
    )
    parser.add_argument("query", nargs="?", help="Natural-language query")
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="One query per line (blank lines and # comments ignored). "
        "The whole file runs in one process: one model load, one container run.",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=STRATEGIES,
        help="Retrieval strategy: dense (default), bm25, or hybrid (dense + BM25 RRF).",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Hits per query")
    parser.add_argument("--doc-id", default=None, help="Restrict to one document")
    parser.add_argument(
        "--chunk-type",
        default=None,
        choices=["text", "table"],
        help="Restrict to one chunk type",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Drop hits below this score (applied by upstream retriever)",
    )
    parser.add_argument("--collection", default="rag_chunks", help="Qdrant collection name")
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant URL (default: env QDRANT_URL, fallback http://localhost:6333)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking on top of the selected strategy.",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=DEFAULT_RERANK_CANDIDATES,
        help="Number of candidates to fetch for reranking (default: 20).",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="Cross-encoder model name (overrides RERANKER_MODEL; "
        "default: cross-encoder/ms-marco-MiniLM-L-6-v2).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def _build_retriever(args: argparse.Namespace, embedder) -> Retriever:
    """Build the retriever for the requested strategy, optionally with reranking.

    ``dense`` and ``hybrid`` need the embedder for scoring; BM25 scores text
    only and never calls the embedder, but the store still needs one to open the
    collection (the collection's vector dimension is checked on open). For BM25
    we skip the dimension check since it only reads payloads.

    If ``--rerank`` is enabled, the base retriever is wrapped with
    ``CrossEncoderReranker`` which fetches ``--rerank-candidates`` hits and
    re-scores them with a cross-encoder model.
    """
    validate_dimension = args.strategy != "bm25"
    store = QdrantStore(
        embedder=embedder,
        collection_name=args.collection,
        url=args.qdrant_url or os.environ.get("QDRANT_URL") or "http://localhost:6333",
        validate_dimension=validate_dimension,
    )
    if args.strategy == "dense":
        base_retriever = DenseRetriever(embedder, store)
    elif args.strategy == "bm25":
        base_retriever = BM25Retriever(index=store.scroll_payloads())
    else:  # hybrid
        base_retriever = HybridRetriever(
            DenseRetriever(embedder, store),
            BM25Retriever(index=store.scroll_payloads()),
        )

    # Optional cross-encoder reranking
    if args.rerank:
        rerank_config = RerankerConfig.from_env(
            model_name=args.rerank_model,
            top_k=args.top_k,
            candidate_count=args.rerank_candidates,
        )
        return build_reranked_retriever(base_retriever, enable=True, config=rerank_config)

    return base_retriever


def main() -> int:
    args = _build_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.queries_file is not None:
        queries = _read_queries_file(args.queries_file)
    elif args.query is not None:
        queries = [args.query]
    else:
        logger.error("Provide a query or --queries-file")
        return 1

    # Dense and hybrid need the embedder for scoring; BM25 scores text only and
    # never calls the embedder, but the store still needs one to open the
    # collection (the collection's vector dimension is checked on open), so a
    # deterministic FakeEmbedder is used for BM25 to avoid a model download.
    if args.strategy in {"dense", "hybrid"}:
        try:
            embedder = SentenceTransformerEmbedder()
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
    else:
        embedder = FakeEmbedder(dimension=32)

    retriever = _build_retriever(args, embedder)

    # The embedder and the store are constructed once, so the whole file shares
    # one model load and one connection. Each query prints its own header and
    # hit lines, so a batch output is still readable per query.
    return run_queries(
        retriever,
        queries,
        top_k=args.top_k,
        doc_id=args.doc_id,
        chunk_type=args.chunk_type,
        min_score=args.min_score,
        collection=args.collection,
    )


if __name__ == "__main__":
    raise SystemExit(main())
