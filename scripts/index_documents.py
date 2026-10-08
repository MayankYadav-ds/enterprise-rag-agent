"""Ingest → chunk → index pipeline with a Qdrant store."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

# Make the package importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.chunking import FixedSizeChunking, RecursiveChunking, StructureAwareChunking
from rag_agent.chunking.config import LOW_TEXT_QUALITY_THRESHOLD
from rag_agent.chunking.pipeline import chunk_document
from rag_agent.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_agent.embeddings.store import QdrantStore
from rag_agent.ingestion.pipeline import ingest_pdf

logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "fixed_size": FixedSizeChunking,
    "recursive": RecursiveChunking,
    "structure_aware": StructureAwareChunking,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs, chunk them, and index into Qdrant.")
    parser.add_argument("--input", type=Path, required=True, help="Directory of PDFs")
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed"), help="Chunk output dir"
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGY_MAP),
        default="recursive",
        help="Chunking strategy",
    )
    parser.add_argument("--chunk-size", type=int, default=400, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=39, help="Chunk overlap")
    parser.add_argument("--collection", default="rag_chunks", help="Qdrant collection name")
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", ":memory:"),
        help="Qdrant URL (default: env QDRANT_URL, fallback :memory:)",
    )
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args()


def build_embedder() -> SentenceTransformerEmbedder:
    """Build the real embedding provider; raises a clear error if unavailable."""
    try:
        return SentenceTransformerEmbedder()
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


def index_documents() -> int:
    args = parse_arguments()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started_at = perf_counter()

    input_dir = args.input
    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    embedder = build_embedder()
    store = QdrantStore(
        embedder=embedder,
        collection_name=args.collection,
        url=args.qdrant_url,
        recreate=args.recreate,
    )

    pdf_paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() == ".pdf")
    total_docs = 0
    total_chunks = 0
    total_skipped = 0
    total_deleted = 0

    for pdf_path in pdf_paths:
        doc_start = perf_counter()
        document, pages = ingest_pdf(pdf_path, args.output)
        total_docs += 1
        kept_pages, skipped = _exclude_low_quality_pages(pages)
        if skipped:
            logger.info(
                "Excluded %d low-quality page(s) below %.3f for doc %s",
                skipped,
                LOW_TEXT_QUALITY_THRESHOLD,
                document.doc_id,
            )
        chunks = chunk_document(
            document,
            pages,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        result = store.sync(document.doc_id, chunks, embedder, source_file=str(pdf_path))
        elapsed = perf_counter() - doc_start
        total_chunks += result["upserted"]
        total_skipped += result["skipped"]
        total_deleted += result["deleted"]
        logger.info(
            "Indexed %s: %d chunks (%d upserted, %d skipped, %d stale) in %.2fs",
            pdf_path.name,
            len(chunks),
            result["upserted"],
            result["skipped"],
            result["deleted"],
            elapsed,
        )

    elapsed = perf_counter() - started_at
    logger.info(
        "Summary: %d documents, %d chunks indexed, %d skipped unchanged, "
        "%d stale deleted, %.2fs total.",
        total_docs,
        total_chunks,
        total_skipped,
        total_deleted,
        elapsed,
    )
    return 0


def _exclude_low_quality_pages(pages: list) -> tuple[list, int]:
    kept, skipped = [], 0
    for page in pages:
        if page.text_quality < LOW_TEXT_QUALITY_THRESHOLD:
            skipped += 1
            continue
        kept.append(page)
    return kept, skipped


if __name__ == "__main__":
    raise SystemExit(index_documents())
