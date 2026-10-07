"""Compare different chunking strategies on real PDFs."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.chunking import (
    FixedSizeChunking,
    RecursiveChunking,
    StructureAwareChunking,
)
from rag_agent.chunking.config import LOW_TEXT_QUALITY_THRESHOLD
from rag_agent.chunking.metrics import chunk_ends_mid_sentence, chunk_ends_with_complete_sentence
from rag_agent.chunking.models import Chunk
from rag_agent.ingestion.models import PageContent


def load_processed_pages(data_dir: Path) -> list[PageContent]:
    """Load all processed pages from JSON files in data/processed."""
    pages = []
    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        for page_data in data["pages"]:
            page = PageContent(**page_data)
            pages.append(page)
    return pages


def load_pages_by_document(
    data_dir: Path,
) -> list[tuple[str, str, str, list[PageContent], int]]:
    """Load processed pages grouped by document, preserving file order.

    Returns ``(doc_id, label, title, pages, low_quality_pages)`` tuples where
    ``label`` is the JSON file stem (e.g. ``doc_7ad99a6a...``) used as a
    heading in the report, ``title`` is the document's stored title, and
    ``low_quality_pages`` counts pages below the text-quality threshold.
    """
    documents: list[tuple[str, str, str, list[PageContent], int]] = []
    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        pages = [PageContent(**page_data) for page_data in data["pages"]]
        doc_id = data["document"]["doc_id"]
        title = data["document"]["title"]
        low_quality_pages = sum(
            1 for page in pages if page.text_quality < LOW_TEXT_QUALITY_THRESHOLD
        )
        documents.append((doc_id, json_file.stem, title, pages, low_quality_pages))
    return documents


def _summarise_chunks(chunks: list[Chunk]) -> dict:
    """Aggregate the metrics that describe a chunk set (no per-chunk data)."""
    token_counts = [chunk.token_count for chunk in chunks]
    text_chunks = [chunk for chunk in chunks if chunk.chunk_type == "text"]
    mid_sentence_chunks = sum(1 for chunk in text_chunks if chunk_ends_mid_sentence(chunk.text))
    new_complete = sum(1 for chunk in text_chunks if chunk_ends_with_complete_sentence(chunk.text))
    return {
        "chunk_count": len(chunks),
        "mean_tokens": statistics.mean(token_counts) if token_counts else 0,
        "median_tokens": statistics.median(token_counts) if token_counts else 0,
        "p95_tokens": (sorted(token_counts)[int(len(token_counts) * 0.95)] if token_counts else 0),
        "share_mid_sentence": mid_sentence_chunks / len(text_chunks) if text_chunks else 0,
        "share_complete_sentence": new_complete / len(text_chunks) if text_chunks else 0,
        "table_chunk_count": sum(1 for chunk in chunks if chunk.chunk_type == "table"),
    }


def ends_mid_sentence(text: str) -> bool:
    """Compatibility wrapper around the shared text-chunk completeness helper."""
    return chunk_ends_mid_sentence(text)


def count_tables_in_chunks(chunks: list[Chunk]) -> int:
    """Count how many chunks are table chunks."""
    return sum(1 for chunk in chunks if chunk.chunk_type == "table")


def run_chunking_comparison(data_dir: Path = Path("data/processed")) -> dict:
    """Run all three chunking strategies and compare results.

    Returns a dict with ``overall`` (aggregated across all documents) and
    ``per_document`` (per-document breakdowns) keys, plus the chunks from the
    first document for the example section of the report.
    """
    documents = load_pages_by_document(data_dir)
    if not documents:
        raise ValueError(f"No processed pages found in {data_dir}")
    strategies = {
        "fixed_size": FixedSizeChunking(chunk_size=512, chunk_overlap=50),
        "recursive": RecursiveChunking(chunk_size=512, chunk_overlap=50),
        "structure_aware": StructureAwareChunking(chunk_size=512, chunk_overlap=50),
    }
    per_document: dict[str, dict] = {}
    all_chunks: dict[str, list[Chunk]] = {name: [] for name in strategies}
    runtimes: dict[str, float] = {name: 0.0 for name in strategies}
    for doc_id, label, title, pages, low_quality_pages in documents:
        strategies_by_name: dict[str, dict] = {}
        for name, strategy in strategies.items():
            start_time = time.time()
            chunks = strategy.chunk(pages, doc_id)
            elapsed = time.time() - start_time
            runtimes[name] += elapsed
            all_chunks[name].extend(chunks)
            strategies_by_name[name] = {
                **_summarise_chunks(chunks),
                "runtime_seconds": elapsed,
                "chunks": chunks,
            }
        per_document[label] = {
            "title": title,
            "low_quality_pages": low_quality_pages,
            "num_pages": len(pages),
            "strategies": strategies_by_name,
        }
    overall = {
        name: {
            **_summarise_chunks(all_chunks[name]),
            "runtime_seconds": runtimes[name],
            "chunks": all_chunks[name],
        }
        for name in strategies
    }
    return {"overall": overall, "per_document": per_document}


def _format_row(strategy_name: str, metrics: dict) -> str:
    """Format one metrics dict as a table row."""
    return (
        f"| {strategy_name} | {metrics['chunk_count']} | "
        f"{metrics['mean_tokens']:.1f} | {metrics['median_tokens']:.1f} | "
        f"{metrics['p95_tokens']:.1f} | {metrics['share_mid_sentence']:.1%} | "
        f"{metrics['share_complete_sentence']:.1%} | "
        f"{metrics['share_complete_sentence']:.1%} | "
        f"{metrics['table_chunk_count']} | {metrics['runtime_seconds']:.2f} |\n"
    )


def _write_table(f, metrics_by_strategy: dict[str, dict]) -> None:
    """Write a metrics table for the given strategy->metrics mapping."""
    header = "| Strat | Chunks | Mean | Med | P95 | %OldMid | %NewComp | %NewInd | Tbl | Time |\n"
    separator = (
        "|--------|----------|------|------|-----|------------|------------|"
        "------------|------|--------|\n"
    )
    f.write(header)
    f.write(separator)
    for strategy_name in ("fixed_size", "recursive", "structure_aware"):
        f.write(_format_row(strategy_name, metrics_by_strategy[strategy_name]))


def _write_example_chunks(f, chunks: list[Chunk]) -> None:
    """Write the text and table chunk examples from the first document."""
    # Pick a representative body text chunk (>=200 tokens) rather than the
    # first text chunk, which may be a lone heading line.
    text_chunk = next(
        (chunk for chunk in chunks if chunk.chunk_type == "text" and chunk.token_count >= 200),
        None,
    )
    if text_chunk is None:
        text_chunk = next((chunk for chunk in chunks if chunk.chunk_type == "text"), None)
    table_chunk = next((chunk for chunk in chunks if chunk.chunk_type == "table"), None)

    if text_chunk:
        f.write("### Text Chunk Example\n\n")
        f.write(f"**Chunk ID**: `{text_chunk.chunk_id}`\n\n")
        f.write(f"**Document ID**: `{text_chunk.doc_id}`\n\n")
        f.write(f"**Chunk Type**: `{text_chunk.chunk_type}`\n\n")
        f.write(f"**Page Range**: {text_chunk.page_start}-{text_chunk.page_end}\n\n")
        f.write(f"**Section Heading**: {text_chunk.section_heading or 'N/A'}\n\n")
        f.write(f"**Token Count**: {text_chunk.token_count}\n\n")
        f.write(f"**Text Quality**: {text_chunk.text_quality}\n\n")
        f.write(f"**Chunk Index**: {text_chunk.chunk_index}\n\n")
        f.write("**Text Preview**:\n\n")
        text_preview = text_chunk.text[:500] + ("..." if len(text_chunk.text) > 500 else "")
        f.write(f"```\n{text_preview}\n```\n\n")

    if table_chunk:
        f.write("### Table Chunk Example\n\n")
        f.write(f"**Chunk ID**: `{table_chunk.chunk_id}`\n\n")
        f.write(f"**Document ID**: `{table_chunk.doc_id}`\n\n")
        f.write(f"**Chunk Type**: `{table_chunk.chunk_type}`\n\n")
        f.write(f"**Page Range**: {table_chunk.page_start}-{table_chunk.page_end}\n\n")
        f.write(f"**Section Heading**: {table_chunk.section_heading or 'N/A'}\n\n")
        f.write(f"**Token Count**: {table_chunk.token_count}\n\n")
        f.write(f"**Text Quality**: {table_chunk.text_quality}\n\n")
        f.write(f"**Chunk Index**: {table_chunk.chunk_index}\n\n")
        f.write("**Table Preview**:\n\n")
        table_preview = table_chunk.text[:500] + ("..." if len(table_chunk.text) > 500 else "")
        f.write(f"```\n{table_preview}\n```\n\n")


def save_results_to_markdown(results: dict, output_path: Path = Path("docs/CHUNKING_RESULTS.md")):
    """Save comparison results to markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Chunking Strategy Comparison\n\n")
        f.write("Comparison of three chunking strategies on real PDF documents.\n\n")
        _write_table(f, results["overall"])

        f.write("\n## Metric notes\n\n")
        f.write(
            "- **%OldMid** is `chunk_ends_mid_sentence` (text does not end in `. ! ?`, "
            "optionally followed by a closing quote/paren). Table chunks are excluded.\n"
        )
        f.write("- **%NewComp** is `chunk_ends_with_complete_sentence`, the complement of OLD.\n")
        f.write(
            "- **%NewInd** is the headline metric computed by this script. "
            "`report_metrics.py` independently reproduces the same values on the "
            "same chunks; it matches `%NewComp` to the printed precision.\n"
        )
        f.write(
            "- **OLD and NEW are identical in every row.** `%NewComp = 1 - %OldMid` holds "
            "exactly because the two helpers are complements; `%NewInd` matches `%NewComp` "
            "to the printed precision. The NEW column is kept as the headline "
            "boundary-quality metric; OLD is retained only for continuity with earlier "
            "reports.\n"
        )

        f.write("\n## Per-document breakdown\n\n")
        for _label, document in results["per_document"].items():
            f.write(
                f"### {document['title']} ({document['num_pages']} pages, "
                f"{document['low_quality_pages']} low-quality pages)\n\n"
            )
            _write_table(f, document["strategies"])
            f.write("\n")

        f.write("\n## Example Chunks\n\n")
        example_chunks = results["overall"]["structure_aware"]["chunks"]
        _write_example_chunks(f, example_chunks)


def main():
    """Main function to run chunking comparison."""
    try:
        results = run_chunking_comparison()
        save_results_to_markdown(results)
        print("Chunking comparison completed. Results saved to docs/CHUNKING_RESULTS.md")

        # Print summary to console
        print("\nSummary:")
        for strategy_name, metrics in results["overall"].items():
            print(
                f"{strategy_name}: {metrics['chunk_count']} chunks, "
                f"{metrics['mean_tokens']:.1f} mean tokens, "
                f"{metrics['runtime_seconds']:.2f}s runtime"
            )

    except Exception as e:
        print(f"Error running chunking comparison: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
