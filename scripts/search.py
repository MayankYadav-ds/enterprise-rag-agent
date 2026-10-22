"""Search the indexed Qdrant collection with dense retrieval and print hits.

Embeds the query with the same embedder that indexed the collection, runs
``DenseRetriever.retrieve`` and prints one line per hit: rank, score, the
citation (``[file, p. 22, section heading]``) and a 120-character snippet of
the chunk text.

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

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
MAX_SNIPPET = 120


def main() -> int:
    parser = argparse.ArgumentParser(description="Dense-search the Qdrant collection.")
    parser.add_argument("query", help="Natural-language query")
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
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

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

    hits = retriever.retrieve(
        args.query,
        top_k=args.top_k,
        doc_id=args.doc_id,
        chunk_type=args.chunk_type,
        min_score=args.min_score,
    )
    if not hits:
        logger.warning("No hits found in collection %r.", args.collection)
        return 1

    for hit in hits:
        snippet = (hit.text or "").replace("\n", " ")[:MAX_SNIPPET]
        print(
            f"rank={hit.rank} score={hit.score:.4f} "
            f"citation={format_citation(hit)} "
            f"snippet={snippet!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
