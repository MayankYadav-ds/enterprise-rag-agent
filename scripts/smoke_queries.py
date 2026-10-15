"""Smoke-test the retrieval path: embed queries and print hits.

The real BAAI/bge-small-en-v1.5 model cannot be loaded on machines that block
torch (e.g. Windows with an Application Control policy, WinError 4551). On those
machines this script exits with a clear error before touching the store. On a
machine where torch works it embeds each query with the same embedder that
indexed the collection, calls the store's search with top_k=3, and prints the
score, page range, section heading and a 100-character snippet for every hit.

The shared logic lives in ``rag_agent.embeddings.smoke`` and is unit-tested in
tests/test_smoke_queries.py with a FakeEmbedder and an in-memory Qdrant
collection, so the code path is covered without a model download.

``source_file`` is stored as the bare file name, so a renamed source file is
reflected without re-hydrating the chunk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_agent.embeddings.smoke import run_smoke_queries
from rag_agent.embeddings.store import QdrantStore

logger = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "What are the main risk factors?",
    "How does the company earn royalties?",
    "What was total revenue?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the retrieval path.")
    parser.add_argument("--collection", default="rag_chunks", help="Qdrant collection name")
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant URL (default: env QDRANT_URL, fallback :memory:)",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        default=DEFAULT_QUERIES,
        help="Queries to embed and search",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Hits per query")
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

    hits = run_smoke_queries(embedder, store, args.queries, top_k=args.top_k)
    if not hits:
        logger.warning("No hits found in collection %r.", args.collection)
        return 1

    for hit in hits:
        print(
            f"[{hit['query']}] score={hit['score']:.4f} "
            f"pages={hit['page_start']}-{hit['page_end']} "
            f"heading={hit['section_heading']!r} "
            f"source={hit['source_file']!r} doc_id={hit['doc_id_prefix']} "
            f"snippet={hit['snippet']!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
