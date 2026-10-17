# Changelog

## [Fix heading hierarchy] — in progress on fix/heading-hierarchy

- **fix(chunking):** position-based heading assignment — chunks now take the nearest preceding ITEM heading at or before their character offset in the page text, using per-page heading offset maps. Closes #34 and #36. Refs #4.
- **fix(chunking):** an ITEM heading *inside* a chunk's text span (not just before it) now names the chunk's section. This fixes N-able p75 where chunks starting after ITEM 7A at offset 2003 correctly resolve to ITEM 8 at offset 2876 instead of falling through to carry-forward.
- **fix(chunking):** carry-forward walk no longer `break`s on non-ITEM headings in 10-K-structured docs; only ITEM headings terminate the backwards walk.
- **feat(chunking):** added `subheading` field to `Chunk` model and `attach_headings` output. Non-ITEM headings that pass a stricter test (bold, larger than body, not a table label/statement title/year-range/digits-only) are captured as `subheading` when the document has 10-K structure.
- **feat(store):** `subheading` added to `_payload` so Qdrant point payloads carry it.
- **fix(store):** `_patch_existing_payloads` now correctly refreshes the `subheading` field alongside `section_heading` for existing points.
- **feat(scripts):** `heading_report.py` now reports `subheading` in its per-chunk output.

## [Phase 4 — embeddings and Qdrant] — in progress on feat/embeddings-qdrant

## [Phase 4 — embeddings and Qdrant] — in progress on feat/embeddings-qdrant

- **fix(store):** `source_file` is now stored as the bare file name (`Path(source_file).name`) instead of the indexed relative path, so a renamed source file is reflected without re-hydrating the chunk. Verified on the real run: renaming `sec-form-10-k-pdf-fixture.pdf` to `nps-pharmaceuticals-2013-10k.pdf` and re-indexing without `--recreate` patched all 300 of its chunks in place.
- **feat(store):** `sync` returns a fourth key, `payload_updated`, and patches the payload of chunks that already exist but whose stored payload changed (renamed source file, refreshed metadata) with `batch_update_points` + `SetPayloadOperation` — the vector is never touched, so the cosine similarity is unchanged and nothing is re-embedded. `scripts/index_documents.py` reports it in the per-document line and the summary.

## [Phase 3 — chunking] — in progress on feat/chunking

- **fix(ingestion):** `clean_text` now drops private-use area glyphs (Co category) and non-printing format characters (Cf), closing issue #17. Verified against the real NPS 10-K (4 distinct Co codepoints: 0xf020, 0xf078, 0xf0a8, 0xf0b7).
- **fix(ingestion):** `_detect_headings` rejects long all-caps cover lines (>60 chars) and table labels ("2013 2012 2011 2009 (in thousands)") with a table-label pattern.
- **feat(chunking):** three strategies (fixed_size, recursive, structure_aware) with deterministic chunk IDs, table splitting by rows with header repetition, and quality filtering.
- **feat(chunking):** `chunk_ends_with_complete_sentence` added as the complement of `chunk_ends_mid_sentence`; structure-aware now splits oversized paragraphs by sentence boundary before character fallback, reducing mid-sentence rate from 20.5% to 15.5% on NPS.
- **fix(chunking):** "Item NN." headings detected by pattern in both ingestion and chunker fallback; inline headings split off their own line; trailing orphan headings dropped.

## [Phase 2 — PDF ingestion] — Phase 2 follow-up

- **ITEM headings exempt from the heading length cap**: SEC "Item NN. Title" lines (e.g. `"ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS"` = 93 chars) were dropped at ingestion because the 60-character length cap was checked **before** the ITEM-pattern match. After the fix (commit `d634574` / `61352ab`), Item titles are capped at 160 chars in both ingestion (`pdf_parser._detect_headings`) and chunking (`chunking.headings._is_valid_heading`). Verified on real PDFs: N-able Item 7/7A/8/9A/9B/9C all now detected, `ITEM 6. RESERVED` dropped from 26 chunks to 1 (p57). No ITEM 8 heading detected in NPS (title wraps and each line fails separately).

## [Phase 1 — Foundation]

- **fix(store):** `source_file` is now stored as the bare file name (`Path(source_file).name`) instead of the indexed relative path, so a renamed source file is reflected without re-hydrating the chunk. Verified on the real run: renaming `sec-form-10-k-pdf-fixture.pdf` to `nps-pharmaceuticals-2013-10k.pdf` and re-indexing without `--recreate` patched all 300 of its chunks in place.
- **feat(store):** `sync` returns a fourth key, `payload_updated`, and patches the payload of chunks that already exist but whose stored payload changed (renamed source file, refreshed metadata) with `batch_update_points` + `SetPayloadOperation` — the vector is never touched, so the cosine similarity is unchanged and nothing is re-embedded. `scripts/index_documents.py` reports it in the per-document line and the summary.

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
