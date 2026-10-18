# Architecture

## Overview

The agent ingests public PDFs, chunks them, indexes the chunks, and answers
questions over the indexed evidence with page-level citations.

```mermaid
flowchart LR
    D[Public PDFs] --> I[Ingestion]
    I --> C[Chunking + metadata]
    C --> E[Embeddings]
    E --> Q[(Qdrant)]
```

## Ingestion

`src/rag_agent/ingestion/` parses each PDF into a hash-named JSON record:

- PyMuPDF extracts positioned text spans with font size and bold flags.
- pdfplumber extracts ruled tables as Markdown.
- `clean_text` normalises Unicode, repairs hyphenated line breaks, and drops
  private-use area glyphs (`Co`) and format characters (`Cf`).
- Repeated edge headers/footers and standalone printed page numbers are removed.
- Each page is scored 0-1; pages below 0.900 warn about extraction artifacts.

## Chunking

`src/rag_agent/chunking/` implements three strategies behind a common
interface. The chunking step is the focus of this phase.

```mermaid
flowchart TD
    P[PageContent] --> S{Chunking strategy}
    S --> F[FixedSizeChunking]
    S --> R[RecursiveChunking]
    S --> SA[StructureAwareChunking]
    F --> M[_merge_splits]
    R --> M
    SA --> S2[_sections_for_page]
    S2 --> M
    M --> C[Chunk.create]
    C --> O[chunk_*.jsonl]
```

### Fixed-size

Sentences are packed into chunks of approximately `chunk_size` tokens with
`chunk_overlap` tokens of carry-over. Simplest and fastest; boundaries fall at
arbitrary sentence positions.

### Recursive

Splits by paragraph, then sentence, then word, then merges with overlap. The
paragraph/sentence boundaries are the same as fixed-size, but oversized
paragraphs are split by sentence boundary before falling back to a
character-boundary cut, so chunks end at a natural sentence end more often.

### Structure-aware

Splits each page by detected headings first, then applies recursive chunking
within each section. A heading names the section that **follows** it; a
trailing heading with no body on its page is dropped rather than glued onto
the previous section's body. SEC "Item NN." headings are recognised by pattern
in both the ingestion detector and the chunker's fallback.

### Shared rules

- Tables are atomic units, split by rows with header repetition when they
  exceed chunk size.
- Pages below the text-quality threshold are excluded by default.
- No empty or near-empty chunks are produced (minimum size enforced).
- Chunk IDs are deterministic: `chunk_<doc_id>_<8-hex-position>_<16-hex-sha256>`.

## Indexing (embeddings + Qdrant)

`scripts/index_documents.py` runs ingest → chunk → embed → Qdrant in one pass.
Each chunk is embedded with the BAAI/bge-small-en-v1.5 model and stored as a
point in a Qdrant collection.

```mermaid
flowchart LR
    PDF[PDF in data/raw] --> I[Ingestion: page-aware JSON]
    I --> C[Chunking: recursive strategy]
    C --> E[Embeddings: bge-small-en-v1.5]
    E --> Q[(Qdrant collection)]
    Q --> S[Search: query vector, top_k]
    S --> H[Hits: score, page, heading, snippet]
```

- Vectors are L2-normalised so cosine similarity equals the dot product in Qdrant.
- Point IDs are deterministic UUID5s derived from the chunk ID, so re-runs never
  duplicate points and a diff-based sync only upserts what changed.
- Each point carries a payload with `chunk_id`, `doc_id`, `source_file`,
  `chunk_type`, `page_start`, `page_end`, `section_heading`, `text_quality`,
  `token_count`, `text` and `embedding_model`, so citations can be answered from
  Qdrant without re-hydrating chunks. `source_file` is stored as the bare file
  name, so a renamed source file is reflected without re-hydrating the chunk.
- The diff-based sync returns `{"upserted", "deleted", "skipped",
  "payload_updated"}`. Chunks that already exist but whose payload changed
  (a renamed source file, a different model name, or refreshed page/heading/
  quality/text metadata) are patched in place with `batch_update_points` and
  `SetPayloadOperation` — the vector is never touched, so the cosine similarity
  is unchanged and nothing is re-embedded.
- The model silently truncates text beyond 512 WordPiece tokens, so chunk size
  was lowered from 512/50 to 400/39. See [`docs/TOKEN_LIMIT_REPORT.md`](TOKEN_LIMIT_REPORT.md).

## Retrieval, generation, evaluation, API

`src/rag_agent/retrieval/`, `src/rag_agent/generation/`,
`src/rag_agent/evaluation/` and `src/rag_agent/api/` are stubs for later
phases. Their `__init__.py` files carry a docstring and nothing else.
Retrieval will combine BM25 and dense-vector retrieval with reciprocal-rank
fusion and a cross-encoder reranker; generation will stream grounded answers
over Server-Sent Events with document and page citations; evaluation will run
RAGAS and retrieval metrics against a reproducible gold set. These phases are
built incrementally and documented in [docs/DECISIONS.md](docs/DECISIONS.md).
