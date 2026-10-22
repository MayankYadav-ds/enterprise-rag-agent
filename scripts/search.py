"""Search the indexed Qdrant collection with dense retrieval and print hits.

Embeds the query with the same embedder that indexed the collection, runs
``DenseRetriever.retrieve`` and prints one line per hit: rank, score, the
citation (``[file, p. 22, section heading]``) and a 120-character snippet of
the chunk text.

``--queries-file`` runs a whole file in ONE process: one model load, one
container run. Each query prints ``===== <query>`` and then the same hit lines
as single-query mode, so a batch output is still readable per query. Blank
lines and lines beginning with ``#`` are ignored, so the file can carry notes.

The real BAAI/bge-small-en-v1.5 model cannot be loaded on machines that block
torch (e.g. Windows with an Application Control policy, WinError 4551). On those
machines this script exits with a clear error before touching the store. On a
machine where torch works it searches the live collection.

The shared retrieval logic is unit-tested in ``tests/test_dense.py`` (the
``DenseRetriever``), ``tests/test_citation.py`` and ``tests/test_retrieval.py``
with a FakeEmbedder and an in-memory Qdrant collection, so the code path is
covered without a model download.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_agent.embeddings.store import QdrantStore
from rag_agent.retrieval.citation import format_citation
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.protocol import Retriever

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
MAX_SNIPPET = 120


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
    parser = argparse.ArgumentParser(description="Dense-search the Qdrant collection.")
    parser.add_argument("query", nargs="?", help="Natural-language query")
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="One query per line (blank lines and # comments ignored). "
        "The whole file runs in one process: one model load, one container run.",
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
        help="Drop hits below this score",
    )
    parser.add_argument("--collection", default="rag_chunks", help="Qdrant collection name")
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant URL (default: env QDRANT_URL, fallback http://localhost:6333)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


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

    try:
        embedder = SentenceTransformerEmbedder()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    store = QdrantStore(
        embedder=embedder,
        collection_name=args.collection,
        url=args.qdrant_url or "http://localhost:6333",
    )
    retriever = DenseRetriever(embedder, store)

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
