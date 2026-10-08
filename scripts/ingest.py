"""Command-line interface for idempotent PDF-directory ingestion and chunking."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter

from rag_agent.chunking.pipeline import chunk_and_save
from rag_agent.ingestion.pipeline import ingest_directory

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the ingestion batch."""
    parser = argparse.ArgumentParser(
        description="Parse PDFs into page-aware JSON records and optionally chunk them."
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="PDF directory")
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed"), help="JSON output directory"
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument("--chunk", action="store_true", help="Also chunk the ingested documents")
    parser.add_argument(
        "--chunk-strategy",
        choices=["fixed_size", "recursive", "structure_aware"],
        default="fixed_size",
        help="Chunking strategy to use",
    )
    parser.add_argument("--chunk-size", type=int, default=400, help="Target chunk size in tokens")
    parser.add_argument(
        "--chunk-overlap", type=int, default=39, help="Overlap between chunks in tokens"
    )
    return parser.parse_args()


def main() -> int:
    """Run ingestion and log a concise progress summary."""
    arguments = parse_arguments()
    logging.basicConfig(
        level=arguments.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    started_at = perf_counter()

    if arguments.chunk:
        # Run ingestion with chunking
        from pathlib import Path

        input_dir = Path(arguments.input)
        output_dir = Path(arguments.output)

        if not input_dir.is_dir():
            logger.error("Input directory does not exist: %s", input_dir)
            return 1

        pdf_paths = sorted(
            candidate for candidate in input_dir.rglob("*") if candidate.suffix.lower() == ".pdf"
        )

        for pdf_path in pdf_paths:
            try:
                chunk_and_save(
                    pdf_path,
                    output_dir,
                    strategy=arguments.chunk_strategy,
                    chunk_size=arguments.chunk_size,
                    chunk_overlap=arguments.chunk_overlap,
                )
                logger.info("Chunked %s using %s strategy", pdf_path.name, arguments.chunk_strategy)
            except Exception as e:
                logger.error("Failed to chunk %s: %s", pdf_path.name, e)

        # Also run regular ingestion for summary
        result = ingest_directory(arguments.input, arguments.output)
    else:
        # Run regular ingestion only
        result = ingest_directory(arguments.input, arguments.output)

    elapsed = perf_counter() - started_at
    logger.info(
        "Summary: %d processed, %d skipped, %d failed; %d PDF pages, %d tables, "
        "%d low-quality pages, %d warnings in %.2fs.",
        result.processed_count,
        result.skipped_count,
        result.failed_count,
        result.page_count,
        result.table_count,
        result.low_quality_page_count,
        result.warning_count,
        elapsed,
    )
    return 1 if result.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
