"""Command-line interface for idempotent PDF-directory ingestion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter

from rag_agent.ingestion.pipeline import ingest_directory

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the ingestion batch."""
    parser = argparse.ArgumentParser(description="Parse PDFs into page-aware JSON records.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="PDF directory")
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed"), help="JSON output directory"
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args()


def main() -> int:
    """Run ingestion and log a concise progress summary."""
    arguments = parse_arguments()
    logging.basicConfig(
        level=arguments.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    started_at = perf_counter()
    result = ingest_directory(arguments.input, arguments.output)
    elapsed = perf_counter() - started_at
    logger.info(
        "Summary: %d processed, %d skipped, %d failed; %d PDF pages, %d tables, "
        "%d warnings in %.2fs.",
        result.processed_count,
        result.skipped_count,
        result.failed_count,
        result.page_count,
        result.table_count,
        result.warning_count,
        elapsed,
    )
    return 1 if result.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
