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
from rag_agent.chunking.metrics import chunk_ends_mid_sentence
from rag_agent.chunking.models import Chunk
from rag_agent.ingestion.models import PageContent


def load_processed_pages(data_dir: Path) -> list[PageContent]:
    """Load all processed pages from JSON files in data/processed."""
    pages = []

    for json_file in data_dir.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        for page_data in data["pages"]:
            page = PageContent(**page_data)
            pages.append(page)

    return pages


def ends_mid_sentence(text: str) -> bool:
    """Compatibility wrapper around the shared text-chunk completeness helper."""
    return chunk_ends_mid_sentence(text)


def count_tables_in_chunks(chunks: list[Chunk]) -> int:
    """Count how many chunks are table chunks."""
    return sum(1 for chunk in chunks if chunk.chunk_type == "table")


def run_chunking_comparison(data_dir: Path = Path("data/processed")) -> dict:
    """Run all three chunking strategies and compare results."""

    # Load processed pages
    pages = load_processed_pages(data_dir)

    if not pages:
        raise ValueError(f"No processed pages found in {data_dir}")

    # Get document ID from first page
    doc_id = pages[0].doc_id if pages else "unknown"

    # Initialize strategies
    strategies = {
        "fixed_size": FixedSizeChunking(chunk_size=512, chunk_overlap=50),
        "recursive": RecursiveChunking(chunk_size=512, chunk_overlap=50),
        "structure_aware": StructureAwareChunking(chunk_size=512, chunk_overlap=50),
    }

    results = {}

    for name, strategy in strategies.items():
        start_time = time.time()
        chunks = strategy.chunk(pages, doc_id)
        end_time = time.time()

        # Calculate statistics
        token_counts = [chunk.token_count for chunk in chunks]
        table_chunks = count_tables_in_chunks(chunks)
        text_chunks = [chunk for chunk in chunks if chunk.chunk_type == "text"]
        mid_sentence_chunks = sum(1 for chunk in text_chunks if chunk_ends_mid_sentence(chunk.text))

        results[name] = {
            "chunk_count": len(chunks),
            "mean_tokens": statistics.mean(token_counts) if token_counts else 0,
            "median_tokens": statistics.median(token_counts) if token_counts else 0,
            "p95_tokens": (
                sorted(token_counts)[int(len(token_counts) * 0.95)] if token_counts else 0
            ),
            "share_mid_sentence": (mid_sentence_chunks / len(text_chunks) if text_chunks else 0),
            "table_chunk_count": table_chunks,
            "runtime_seconds": end_time - start_time,
            "chunks": chunks,  # Store for example output
        }

    return results


def save_results_to_markdown(results: dict, output_path: Path = Path("docs/CHUNKING_RESULTS.md")):
    """Save comparison results to markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Chunking Strategy Comparison\n\n")
        f.write("Comparison of three chunking strategies on real PDF documents.\n\n")

        # Create comparison table
        # Create header line (split to avoid line length issues)
        header_part1 = "| Strat | Chunks | Mean | Med | P95 | %Mid | Tbl | Time |\n"
        header_part2 = "|--------|----------|------|------|-----|-------|------|--------|\n"
        header = header_part1 + header_part2
        f.write(header)
        f.write("|--------|----------|------|------|-----|-------|------|--------|\n")

        for strategy_name, metrics in results.items():
            f.write(
                f"| {strategy_name} | {metrics['chunk_count']} | "
                f"{metrics['mean_tokens']:.1f} | {metrics['median_tokens']:.1f} | "
                f"{metrics['p95_tokens']:.1f} | {metrics['share_mid_sentence']:.1%} | "
                f"{metrics['table_chunk_count']} | {metrics['runtime_seconds']:.2f} |\n"
            )

        f.write("\n")

        # Add example chunks
        f.write("## Example Chunks\n\n")

        # Find a strategy with chunks to show examples
        example_strategy = None
        for strategy_name, metrics in results.items():
            if metrics["chunks"]:
                example_strategy = strategy_name
                break

        if example_strategy and results[example_strategy]["chunks"]:
            chunks = results[example_strategy]["chunks"]

            # Find a text chunk example
            text_chunk = None
            table_chunk = None

            for chunk in chunks:
                if chunk.chunk_type == "text" and text_chunk is None:
                    text_chunk = chunk
                elif chunk.chunk_type == "table" and table_chunk is None:
                    table_chunk = chunk

                if text_chunk and table_chunk:
                    break

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
                # Limit preview to 500 chars with ellipsis if longer
                if len(text_chunk.text) > 500:
                    text_preview = text_chunk.text[:500] + "..."
                else:
                    text_preview = text_chunk.text
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
                # Limit preview to 500 chars with ellipsis if longer
                if len(table_chunk.text) > 500:
                    table_preview = table_chunk.text[:500] + "..."
                else:
                    table_preview = table_chunk.text
                f.write(f"```\n{table_preview}\n```\n\n")


def main():
    """Main function to run chunking comparison."""
    try:
        results = run_chunking_comparison()
        save_results_to_markdown(results)
        print("Chunking comparison completed. Results saved to docs/CHUNKING_RESULTS.md")

        # Print summary to console
        print("\nSummary:")
        for strategy_name, metrics in results.items():
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
