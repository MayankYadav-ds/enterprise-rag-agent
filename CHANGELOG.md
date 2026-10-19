# Changelog

## [Fix heading hierarchy] — PR #41

Shipped by [PR #41](https://github.com/MayankYadav-ds/enterprise-rag-agent/pull/41).
PR #40 was closed unmerged (`mergedAt` null, branch `fix/heading-convergence`
since deleted). The whitelist commit `6f30f50` ("10-K section markers set
section_heading") reached `main` by a direct push of PR #40's rebased
commits — `git cherry -v origin/main origin/fix/heading-convergence` showed
three of its four commits as patch-equivalent to commits already on `main`
(including `6f30f50`) and only the comment-only commit `3a69f33` as extra —
not by the GitHub merge. PR #41 then merged the remaining three commits on
top (`36c7132`, `bebb5e1`, `8e0ba89`) at `8e0ba89`.

- **fix(store):** `QdrantStore._payload` now writes optional fields as `None`
  when absent rather than omitting them, and the patch pass normalises the
  stored-vs-fresh comparison so `None`, a missing key and `""` are all "no
  value". The 543 points indexed before the `subheading` field existed carried
  no such key, while the fresh payload carries `subheading: None`; Qdrant's
  `set_payload` / `batch_update_points` drop null keys (verified on the live
  server), so the two were never equal under a plain `dict ==` comparison and
  the patch pass re-applied the same 258 points on every run. A second run
  after the fix reports `0 payload updated`. Writing `None` (rather than
  omitting) also clears stale values left by older code — on the live server
  only an explicit `null` removes a key from a patch, which clears 41 points
  carrying `subheading: "PART I"` and 47 carrying the auditor's report title
  (written before the 10-K marker whitelist).
- **fix(chunking):** `attach_headings` no longer captures a non-ITEM heading
  that appears *before* an ITEM heading on the same page as that ITEM's
  subheading. `PART I` (N-able p7), `PART II` (p56), `PART III` (p77) and the
  auditor's report title (p83) name the enclosing part or the report, not a
  subheading of the ITEM headings that follow.
- **fix(chunking):** `_is_subheading` rejects headings containing non-printable
  characters, and headings that are pure punctuation/symbols or mostly symbols
  (a run of two consecutive letters is required). The N-able report's
  broken-font-mapping pages produce `"\x06,-70\x05\x01\x1b9.\x07"`, `:=`,
  `$6%24)3).'"` and `$0A09@0` — all printable with digits, so they passed the
  earlier checks. A real subheading is text; the new rule rejects the junk
  and is safe on the real filings (all nine NPS subheadings survive).
- **fix(chunking):** `attach_headings` docstring now states that `_is_subheading`
  checks text patterns only, not bold or font size.
- **test(store):** a `_NullDroppingClient` fixture reproduces the live server's
  null-dropping behaviour, so the convergence bug is visible in CI: a point
  lacking `subheading` is patched once and a second sync reports
  `payload_updated == 0`; a changed subheading is patched.

## [Audit extras: dead-check removal, non-10-K span test, delete_payload trade-off] — PR #42

- **fix(chunking):** `_is_subheading` dropped the `any(ch.isalnum() for ch in
  stripped)` check, which was provably dead — the two-letters regex
  `[A-Za-z]{2,}` that follows it is a strict superset, so no heading could
  reach the regex after failing the isalnum() check. The docstring now notes
  that the regex is ASCII-only, which is a deliberate trade-off: it rejects
  accented headings such as "Café", which the real filings do not contain.
- **test(chunking):** `test_attach_headings_non_10k_heading_inside_chunk_span_takes_priority`
  covers `attach_headings` for a document without 10-K structure — a heading
  inside a chunk's span takes priority over carry-forward, the same as for a
  10-K. The contained-heading rule was previously gated on `has_item_structure`.
- **docs:** `docs/DECISIONS.md` records why `delete_payload(keys=[...])` was
  rejected in favour of writing `None` for absent optional fields: it is a
  separate request per field rather than a batch operation, it does not
  converge without an extra guard, and it would diverge from what `upsert`
  produces. The one case where it is strictly better — removing a field from
  the schema entirely — is a one-time migration, not a per-sync operation.

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
