"""Measure model-token limits over chunked PDF text using the embedding model's tokenizer.

The chunking package sizes chunks in tiktoken (cl100k_base) tokens, but the embedding
model (BAAI/bge-small-en-v1.5) counts with its own WordPiece tokenizer and silently
truncates text beyond its 512-token maximum. Because these tokenizers disagree, a
chunk sized to "400 tiktoken tokens" can still exceed the model limit and be cut off.

This script re-measures the true model-token length of every chunk produced by the
RECURSIVE chunking strategy, at both 512/50 and 400/39, using the bge model's actual
WordPiece tokenizer (tokenizers >=0.20) downloaded from Hugging Face. No torch is
required — only the tokenizer weights.

For each document it prints:
  - chunk count
  - % of chunks over the model limit
  - median / p95 / p99 model tokens
  - median ratio model_tokens / tiktoken_tokens
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import date
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.chunking import RecursiveChunking
from rag_agent.ingestion.models import PageContent
from rag_agent.utils import lazy_tokenizer


def load_pages_by_document(data_dir: Path):
    """Load processed pages grouped by document, preserving file order."""
    documents: list[tuple[str, str, str, list[PageContent]]] = []
    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        pages = [PageContent(**page_data) for page_data in data["pages"]]
        doc_id = data["document"]["doc_id"]
        title = data["document"]["title"]
        documents.append((doc_id, json_file.stem, title, pages))
    return documents


def run_model_token_measure(
    pages: list[PageContent],
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    """Chunk the pages with the recursive strategy and measure real model tokens.

    Returns per-chunk dicts with tiktoken_tokens and model_tokens.
    """
    chunker = RecursiveChunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk(pages, doc_id)

    tiktok_tok = tiktoken.get_encoding("cl100k_base")
    model_tok = lazy_tokenizer.wordpiece_tokenizer()

    results: list[dict] = []
    for chunk in chunks:
        if chunk.chunk_type != "text":
            continue  # Only measure prose chunks; tables are raw Markdown.
        tiktok_count = len(tiktok_tok.encode(chunk.text))
        model_count = len(model_tok.encode(chunk.text).ids)
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "tiktoken_tokens": tiktok_count,
                "model_tokens": model_count,
                "ratio": model_count / tiktok_count if tiktok_count else 0,
                "over_limit": model_count > model_tok.model_max_tokens,
            }
        )
    return results


def _quantile(sorted_values: list[int], q: float) -> int:
    """Return the element at position q*len (nearest-rank, consistent with metrics.py)."""
    if not sorted_values:
        return 0
    return sorted_values[int(len(sorted_values) * q)]


def summarize(results: list[dict], chunk_size: int, chunk_overlap: int) -> dict:
    model_limit = lazy_tokenizer.wordpiece_tokenizer().model_max_tokens
    counts = [r["model_tokens"] for r in results]
    over = [r for r in results if r["over_limit"]]
    sorted_counts = sorted(counts)
    ratios = [r["ratio"] for r in results if r["tiktoken_tokens"]]

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "model_limit": model_limit,
        "chunk_count": len(results),
        "over_limit": len(over),
        "over_limit_pct": 100.0 * len(over) / len(results) if results else 0.0,
        "median_model_tokens": statistics.median(counts) if counts else 0,
        "p95_model_tokens": _quantile(sorted_counts, 0.95) if sorted_counts else 0,
        "p99_model_tokens": _quantile(sorted_counts, 0.99) if sorted_counts else 0,
        "max_model_tokens": max(counts) if counts else 0,
        "median_tiktoken_tokens": statistics.median([r["tiktoken_tokens"] for r in results])
        if results
        else 0,
        "median_ratio_model_tiktoken": (statistics.median(ratios) if ratios else 0.0),
    }


def _write_doc_table(
    f, doc_title: str, pages: list[PageContent], doc_id: str, config: tuple[int, int]
) -> None:
    chunk_size, chunk_overlap = config
    results = run_model_token_measure(pages, doc_id, chunk_size, chunk_overlap)
    summary = summarize(results, chunk_size, chunk_overlap)

    f.write(f"### {doc_title}\n\n")
    f.write(
        f"- **Config**: recursive chunks, size={chunk_size}, overlap={chunk_overlap}\n"
        "- **Model limit (bge-small-en-v1.5, WordPiece incl. special tokens)**: 512\n"
        "- **Chunk set measured**: all prose (chunk_type=text) chunks produced by "
        f"`RecursiveChunking(chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap})` on the processed pages of this document. "
        "Table chunks are excluded.\n\n"
    )
    f.write(
        "| Metric | Value |\n"
        "|--------|------|\n"
        f"| Chunk count | {summary['chunk_count']} |\n"
        f"| Chunks over model limit | {summary['over_limit']} "
        f"({summary['over_limit_pct']:.1f}%) |\n"
        f"| Median model tokens | {summary['median_model_tokens']} |\n"
        f"| P95 model tokens | {summary['p95_model_tokens']} |\n"
        f"| P99 model tokens | {summary['p99_model_tokens']} |\n"
        f"| Max model tokens | {summary['max_model_tokens']} |\n"
        f"| Median tiktoken tokens | {summary['median_tiktoken_tokens']} |\n"
        f"| Median ratio model_tokens / tiktoken_tokens | "
        f"{summary['median_ratio_model_tiktoken']:.2f} |\n\n"
    )


def main() -> int:
    data_dir = Path("data/processed")
    tokenizer_file = lazy_tokenizer.wordpiece_tokenizer().tokenizer_path
    print(
        f"Token-limit measurement on RECURSIVE chunks\n"
        f"Embedder: BAAI/bge-small-en-v1.5 (WordPiece)\n"
        f"Tokenizer source: {tokenizer_file}\n"
    )

    documents = load_pages_by_document(data_dir)
    if not documents:
        raise ValueError(f"No processed pages found in {data_dir}")

    config_512 = (512, 50)
    config_400 = (400, 39)
    output = Path("docs/TOKEN_LIMIT_REPORT.md")

    with open(output, "w", encoding="utf-8") as f:
        f.write("# Model token-limit report: recursive chunks\n\n")
        f.write(
            f"**Measured on**: {date.today().isoformat()}\n"
            "**Embedding model**: BAAI/bge-small-en-v1.5 "
            "(max 512 WordPiece tokens incl. CLS/SEP)\n"
            "**Tokenizer used**: tokenizers >=0.20, the model's own "
            "`tokenizer.json` from Hugging Face\n"
            "**Strategy measured**: `RecursiveChunking` only (per request)\n\n"
        )

        for doc_id, _label, title, pages in documents:
            for config in (config_512, config_400):
                _write_doc_table(f, title, pages, doc_id, config)

    # Also print a compact summary to the console.
    print()
    for doc_id, _label, title, pages in documents:
        for config in (config_512, config_400):
            results = run_model_token_measure(pages, doc_id, config[0], config[1])
            summary = summarize(results, config[0], config[1])
            print(
                f"{title} [{config[0]}/{config[1]}]: "
                f"{summary['chunk_count']} text chunks, "
                f"over-limit {summary['over_limit_pct']:.1f}%, "
                f"median {summary['median_model_tokens']}, "
                f"p95 {summary['p95_model_tokens']}, "
                f"p99 {summary['p99_model_tokens']}, "
                f"max {summary['max_model_tokens']}, "
                f"median ratio {summary['median_ratio_model_tiktoken']:.2f}"
            )

    print(f"\nDetailed report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
