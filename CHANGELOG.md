# Changelog

## [Fix heading hierarchy] — fix/heading-hierarchy

- **fix(store):** `QdrantStore._payload` now omits optional fields with no value
  instead of storing them as `null`, and the patch pass normalises the
  stored-vs-fresh comparison so `None`, a missing key and `""` are all "no
  value". The 543 points indexed before the `subheading` field existed carried
  no such key, while the fresh payload carried `subheading: None`; Qdrant's
  `set_payload` / `batch_update_points` drop null keys (verified on the live
  server), so the two were never equal under a plain `dict ==` comparison and
  the patch pass re-applied the same 258 points on every run. A second run
  after the fix reports `0 payload updated`.
- **fix(chunking):** `attach_headings` no longer captures a non-ITEM heading
  that appears *before* an ITEM heading on the same page as that ITEM's
  subheading. `PART I` (N-able p7), `PART II` (p56), `PART III` (p77) and the
  auditor's report title (p83) name the enclosing part or the report, not a
  subheading of the ITEM headings that follow.
- **fix(chunking):** `_is_subheading` rejects headings containing non-printable
  characters. The N-able report's broken-font-mapping pages produce headings
  such as `"\x06,-70\x05\x01\x1b9.\x07"` that pass every other text-pattern
  check; all of the garbage values live on low-text-quality pages that are
  excluded from chunking, but the field is now honest regardless.
- **fix(chunking):** `attach_headings` docstring now states that `_is_subheading`
  checks text patterns only, not bold or font size.
- **test(store):** a `_NullDroppingClient` fixture reproduces the live server's
  null-dropping behaviour, so the convergence bug is visible in CI: a point
  lacking `subheading` is patched once and a second sync reports
  `payload_updated == 0`; a changed subheading is patched.

## [Phase 4 — embeddings and Qdrant] — feat/embeddings-qdrant

- **fix(store):** `source_file` is now stored as the bare file name (`Path(source_file).name`) instead of the indexed relative path, so a renamed source file is reflected without re-hydrating the chunk. Verified on the real run: renaming `sec-form-10-k-pdf-fixture.pdf` to `nps-pharmaceuticals-2013-10k.pdf` and re-indexing without `--recreate` patched all 300 of its chunks in place.
- **feat(store):** `sync` returns a fourth key, `payload_updated`, and patches the payload of chunks that already exist but whose stored payload changed (renamed source file, refreshed metadata) with `batch_update_points` + `SetPayloadOperation` — the vector is never touched, so the cosine similarity is unchanged and nothing is re-embedded. `scripts/index_documents.py` reports it in the per-document line and the summary.

## [Phase 3 — chunking] — feat/chunking

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
- **ITEM headings exempt from the heading length cap**: SEC "Item NN. Title" lines (e.g. `"ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS"` = 93 chars) were dropped at ingestion because the 60-character length cap was checked **before** the ITEM-pattern match. After the fix (commit `d634574` / `61352ab`), Item titles are capped at 160 chars in both ingestion (`pdf_parser._detect_headings`) and chunking (`chunking.headings._is_valid_heading`). Verified on real PDFs: N-able Item 7/7A/8/9A/9B/9C all now detected, `ITEM 6. RESERVED` dropped from 26 chunks to 1 (p57). No ITEM 8 heading detected in NPS (title wraps and each line fails separately).

## [Phase 1 — Foundation]

- package layout, pydantic v2 models, dataclasses with `slots=True`
- FastAPI + LangChain orchestration skeleton
- Ruff lint/format, pre-commit hooks, GitHub Actions CI
