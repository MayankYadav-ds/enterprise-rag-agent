"""Report section_heading coverage for a Qdrant collection.

Reads point payloads directly from Qdrant (no embedder, no model download)
and prints, per document and per heading value: the number of chunks, the
first and last page those chunks span, and the top 20 headings by chunk
count. Chunks with a null section_heading are reported separately.

Usage:
    python scripts/heading_report.py --collection rag_chunks --qdrant-url http://localhost:6333
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
        description="Report section_heading coverage in a Qdrant collection."
    )
    parser.add_argument("--collection", default="rag_chunks", help="Qdrant collection name")
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


def report(payloads: list[dict], top_k: int) -> int:
    """Print per-document heading stats. Returns the number of documents."""
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

        print(f"\n=== {source}  doc_id={doc_id} ===")
        print(
            f"chunks={len(chunks)} with_heading={len(with_heading)} "
            f"without_heading={len(without_heading)}"
        )
        print(f"top {len(ranked)} section_heading values:")
        for heading, pages in ranked:
            print(f"  {len(pages):4d}  p{min(pages)}-{max(pages)}  {heading!r}")

    return len(by_doc)


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    url = args.qdrant_url or "http://localhost:6333"
    client = QdrantClient(url=url)
    payloads = _scroll_all_payloads(client, args.collection)
    if not payloads:
        logger.warning("Collection %r is empty.", args.collection)
        return 1
    logger.info("Read %d points from %r", len(payloads), args.collection)
    report(payloads, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
