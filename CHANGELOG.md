# Changelog

## [Phase 4 — embeddings and Qdrant] — in progress on feat/embeddings-qdrant

- **feat(embeddings):** `SentenceTransformerEmbedder` backed by BAAI/bge-small-en-v1.5, with the bge instruction prefix on queries and L2-normalised output.
- **feat(embeddings):** `QdrantStore` with cosine vectors, UUID5 point IDs, payload keyword indexes, `search` with `doc_id`/`chunk_type` filters, and a diff-based `sync` that returns `{"upserted", "deleted", "skipped"}` and never touches another document's points.
- **feat(embeddings):** `count_model_tokens` / `model_token_stats` measure the real WordPiece token length of any chunk set against the model's 512-token limit.
- **fix(chunking):** default chunk size lowered from 512/50 to 400/39 after measuring the bge tokenizer on the real filings: at 512/50, 14.1% of NPS and 21.1% of N-able text chunks exceeded the model limit and were silently truncated; at 400/39, 0.3% and 0.0% respectively, with p99 of 456 and 429.
- **feat(scripts):** `index_documents.py` (ingest → chunk → embed → Qdrant), `docker-compose.yml` + `.env.example`, and `smoke_queries.py` (embeds three natural-language queries and prints score, page range, section heading and snippet for each top-3 hit).
- **feat(embeddings):** `rag_agent.embeddings.smoke` shared helper unit-tested with a `FakeEmbedder` and in-memory Qdrant, so the retrieval path is covered without torch.

## [Phase 3 — chunking] — in progress on feat/chunking

- **fix(ingestion):** `clean_text` now drops private-use area glyphs (Co category) and non-printing format characters (Cf), closing issue #17. Verified against the real NPS 10-K (4 distinct Co codepoints: 0xf020, 0xf078, 0xf0a8, 0xf0b7).
- **fix(ingestion):** `_detect_headings` rejects long all-caps cover lines (>60 chars) and table labels ("2013 2012 2011 2009 (in thousands)") with a table-label pattern.
- **feat(chunking):** three strategies (fixed_size, recursive, structure_aware) with deterministic chunk IDs, table splitting by rows with header repetition, and quality filtering.
- **feat(chunking):** `chunk_ends_with_complete_sentence` added as the complement of `chunk_ends_mid_sentence`; structure-aware now splits oversized paragraphs by sentence boundary before character fallback, reducing mid-sentence rate from 20.5% to 15.5% on NPS.
- **fix(chunking):** "Item NN." headings detected by pattern in both ingestion and chunker fallback; inline headings split off their own line; trailing orphan headings dropped.

## [Phase 2 — PDF ingestion]

- page-aware extraction with font-size and bold typography
- hash-named JSON persistence; idempotent directory ingestion
- repeated edge header/footer removal; standalone page-number removal
- text quality scoring (threshold 0.900)

## [Phase 1 — Foundation]

- package layout, pydantic v2 models, dataclasses with `slots=True`
- FastAPI + LangChain orchestration skeleton
- Ruff lint/format, pre-commit hooks, GitHub Actions CI
