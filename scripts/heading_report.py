"""Report section_heading coverage for a Qdrant collection or a set of PDFs.

Two modes:

* ``--collection`` (default): read point payloads directly from Qdrant — no
  embedder, no model download.
* ``--from-pdf DIR``: ingest and chunk the PDFs in-process (no embedding),
  which is the only way to compare the chunker without a running Qdrant.

In both modes it prints, per document and per heading value: the number of
chunks, the first and last page those chunks span, and the top 20 headings by
chunk count. Chunks with a null section_heading are reported separately.

Usage:
    python scripts/heading_report.py --collection rag_chunks --qdrant-url http://localhost:6333
    python scripts/heading_report.py --from-pdf data/raw
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report section_heading coverage in a Qdrant collection or PDFs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--collection", default=None, help="Qdrant collection name")
    source.add_argument(
        "--from-pdf",
        type=Path,
        default=None,
        help="Directory of PDFs to ingest and chunk in-process (no embedding)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant URL (default: env QDRANT_URL, fallback :memory:)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Number of headings to print per document"
    )
    return parser.parse_args()


def _scroll_all_payloads(client: QdrantClient, collection: str) -> list[dict]:
    """Return every point payload in the collection, paged by offset."""
    payloads: list[dict] = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection,
            scroll_filter=None,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend(record.payload or {} for record in records)
        if next_offset is None:
            break
        offset = next_offset
    return payloads


def _chunk_payloads_from_pdf(pdf_path: Path) -> list[dict]:
    """Ingest and chunk one PDF in-process, returning Qdrant-shaped payloads."""
    from rag_agent.chunking.pipeline import chunk_document
    from rag_agent.ingestion.pipeline import ingest_pdf

    document, pages = ingest_pdf(pdf_path, Path("data/processed"))
    chunks = chunk_document(document, pages, strategy="recursive")
    return [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "source_file": pdf_path.name,
            "chunk_type": c.chunk_type,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "section_heading": c.section_heading,
            "token_count": c.token_count,
            "text_quality": c.text_quality,
            "text": c.text,
        }
        for c in chunks
    ]


def report(payloads: list[dict], top_k: int, source_label: str) -> None:
    """Print per-document heading stats."""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for payload in payloads:
        by_doc[payload.get("doc_id", "?")].append(payload)

    for doc_id in sorted(by_doc):
        chunks = by_doc[doc_id]
        source = chunks[0].get("source_file", "?")
        with_heading = [c for c in chunks if c.get("section_heading")]
        without_heading = [c for c in chunks if not c.get("section_heading")]

        counts: dict[str, list[int]] = defaultdict(list)
        for c in with_heading:
            counts[c["section_heading"]].append(c.get("page_start", 0))

        ranked = sorted(
            counts.items(),
            key=lambda kv: (-len(kv[1]), min(kv[1]), kv[0]),
        )[:top_k]

        print(f"\n=== {source}  doc_id={doc_id}  ({source_label}) ===")
        print(
            f"chunks={len(chunks)} with_heading={len(with_heading)} "
            f"without_heading={len(without_heading)}"
        )
        print(f"top {len(ranked)} section_heading values:")
        for heading, pages in ranked:
            print(f"  {len(pages):4d}  p{min(pages)}-{max(pages)}  {heading!r}")


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.collection:
        url = args.qdrant_url or "http://localhost:6333"
        client = QdrantClient(url=url)
        payloads = _scroll_all_payloads(client, args.collection)
        if not payloads:
            logger.warning("Collection %r is empty.", args.collection)
            return 1
        logger.info("Read %d points from %r", len(payloads), args.collection)
        report(payloads, args.top_k, "qdrant")
        return 0

    pdf_paths = sorted(p for p in args.from_pdf.rglob("*") if p.suffix.lower() == ".pdf")
    if not pdf_paths:
        logger.error("No PDFs found under %s", args.from_pdf)
        return 1
    payloads = []
    for pdf_path in pdf_paths:
        payloads.extend(_chunk_payloads_from_pdf(pdf_path))
    logger.info("Chunked %d PDFs -> %d chunks", len(pdf_paths), len(payloads))
    report(payloads, args.top_k, "in-process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
