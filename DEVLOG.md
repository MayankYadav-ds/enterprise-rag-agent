# Development log

## Phase 1 — Foundation

I set up the package layout and a deliberately small configuration object so later components have one clear place to read non-secret settings. I chose LangChain for the eventual orchestration layer because its retriever and streaming interfaces work well with a FastAPI service, while keeping the project’s retrieval logic testable in plain Python.

The main snags at this stage are external GitHub automation and a Windows application-control restriction: the GitHub CLI is not installed, and its policy blocks pre-commit's bundled YAML checker. I kept the local gate dependable by retaining end-of-file, whitespace, merge-conflict, Ruff lint, and Ruff format hooks; GitHub Actions remains the server-side quality gate. The codebase itself is provider-agnostic and does not require an API key to run the current checks.

## Phase 2 — PDF ingestion

I built a page-aware PDF ingestion path around deterministic SHA-256 document IDs. It uses PyMuPDF for text positions and font signals, pdfplumber for ruled tables, and writes a hash-named JSON record so a repeated directory run skips unchanged content. The CLI reports files, pages, tables, warnings, and elapsed time rather than hiding its work.

### Parsing quality notes

**What worked well.** On the official NPS Pharmaceuticals 2013 Form 10-K PDF, the parser retained all 104 physical PDF pages as text pages, found 6 tables across 4 pages, and flagged 0 scanned or low-text-quality pages. The patent table on PDF page 21 was extracted as readable Markdown. On the official N-able 2024 annual report, it retained 125 text pages, found 11 tables across 8 pages, and flagged 0 scanned pages. Page metadata keeps the physical PDF index, so later citations remain reproducible even when a filing's printed numbering differs.

**Measured timing.** The NPS filing parsed in **11.05 seconds** inside the ingestion pipeline, or **0.106 seconds per PDF page**. The CLI process took 12.12 seconds including startup and one already-indexed-file skip. The N-able report's fresh quality-scoring run parsed in **15.77 seconds**, or **0.126 seconds per PDF page**; the CLI process took 15.83 seconds while also skipping the already-indexed NPS filing. Both figures were measured locally on 2026-09-30 with the repository CLI; the PDFs and generated JSON remain ignored.

**What is imperfect.** The N-able report exposed a font-mapping failure: 44 of 125 pages scored below the 0.900 quality threshold because of non-whitespace control characters. The fresh CLI run logged 44 low-text-quality-page warnings. The score also detects literal U+FFFD and `(cid:N)` tokens, but it does not repair them. On the NPS report, the unruled multi-year balance-sheet table on PDF page 80 was fragmented by pdfplumber; the parser now discards the surrounding one-column prose regions falsely reported as tables, but it does not reconstruct the five-column table. The page-21 table also fuses superscript footnotes into dates such as `2015¹` becoming `20151`.

**Trademark glyph check.** The `Sensipar®/Mimpara®` text on NPS PDF page 5 is not U+FFFD. PyMuPDF and pdfplumber both returned U+00AE (`REGISTERED SIGN`, code point 174); the `�` appearance came from the terminal/output font used during inspection. A synthetic regression test preserves U+00AE and checks that it is not converted to U+FFFD. No source-text mapping was added because altering a correct code point would make the parser worse.

**Header/footer and reading-order checks.** In the N-able validation audit, repeated edge text `Table of Contents` and printed page `2` were removed from PDF page 5 while the following `Safe Harbor Cautionary Statement` body paragraph remained. That is a positive spot check, not proof that the frequency heuristic never removes meaningful boundary text; a repeated running heading can be semantically useful. In the NPS report, 102 pages had a standalone printed page number removed. I inspected PDF page 80 because the layout detector found left/right block groups there: its narrative paragraphs stayed in reading order, but the groups were driven largely by a financial table rather than a conventional two-column article. I do not claim verified support for every sidebar or multi-column layout.

**Future work.** Track the known limitations in [#12](https://github.com/MayankYadav-ds/enterprise-rag-agent/issues/12) (superscript footnotes fused into values), [#13](https://github.com/MayankYadav-ds/enterprise-rag-agent/issues/13) (fragmented unruled financial tables), [#14](https://github.com/MayankYadav-ds/enterprise-rag-agent/issues/14) (alternate extractor or OCR fallback for broken font mappings), and [#15](https://github.com/MayankYadav-ds/enterprise-rag-agent/issues/15) (broader multi-column/sidebar validation). I also plan to render representative pages in automated validation and add safer exceptions for repeated headings that are meaningful page titles.

## Phase 2 follow-up: ITEM headings exempt from the heading length cap

The heading detector in `pdf_parser._detect_headings` rejected the 60-character length cap **before** it checked the "Item NN. Title" pattern, so long SEC item titles were dropped at ingestion and never reached the chunker. The worst visible symptom was `ITEM 6. RESERVED` (N-able page 57) inheriting 26 chunks: with Item 7 missing, the carry-forward stopped at page 57 and every later chunk was mislabelled.

After the fix (commits `d634574` and `61352ab`, applied at both layers), Item titles are capped at 160 characters in ingestion and in `chunking.headings._is_valid_heading`, while everything else stays at 60 — so long cover-page lines without an Item prefix are still rejected. On the real run:

- **N-able:** `ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS` (93 chars) now covers 21 chunks on p58-73; `ITEM 7A` 4 chunks p74; `ITEM 9A` 3 chunks p75; `ITEM 9C` 3 chunks p76; `ITEM 5` 2 chunks p56; `ITEM 14` 2 chunks p77. `ITEM 6. RESERVED` dropped from 26 chunks to 1 (p57). 236 of 243 chunks carry a heading; 7 do not.
- **NPS:** 300 of 300 chunks carry a heading, but several labels are table headers and statement titles rather than section headings — `Product and Development Programs` (67 chunks, p5-p25), `Consolidated Statements of Stockholders' Equity (Deficit)` (38, p68-p79), `Proceeds from Available for Sale Securities` (37, p82-p94). NPS has no `ITEM 5` or `ITEM 7` heading: p48-p60 are labelled `Results of Operations` (41 chunks), and the `ITEM 8` title wraps over two lines so each line fails separately.

The smoke query "How does the company earn royalties?" now returns a p72 hit cited under `Consolidated Statements of Stockholders' Equity (Deficit)`, which is wrong — the label is a statement title, not the section the answer lives in. That is tracked as a known limitation.

## Phase 4 — Fix heading hierarchy (fix/heading-hierarchy)

### Problem

Two chunks from real 10-K filings were misattributed:

1. **N-able 2024 10-K, page 75**: Chunk starting at offset 2003 with text "in U.S. dollars for foreign subsidiaries..." was assigned `ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK` (from page 74) instead of `ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA`.
2. **NPS 2013 10-K, page 44**: Chunk starting at offset 0 with text "PART II ITEM 5. Market for Registrant's Common Equity..." was assigned `ITEM 4. Mine Safety Disclosures` (from page 43) instead of `ITEM 5`.

### Root cause

Both cases share the same root cause: a chunk whose text span *contains* an ITEM heading on its own page was still being resolved via carry-forward from an earlier page. The original `attach_headings` logic only looked for headings *before* `chunk_start`; when the chunk started before a heading that was *inside* its span, the carry-forward from earlier pages won.

### Fix

1. **Contained-ITEM check** (runs first): if the chunk's text span on its first page contains an ITEM heading, that heading is the section. This is O(n) per chunk and takes priority over backwards walk.

2. **Subheading capture on ITEM match**: when an ITEM heading is found during the backwards walk, any non-ITEM subheading preceding it on the same page is now also captured in the `subheading` field.

3. **Carry-forward no longer breaks on non-ITEM headings**: in 10-K-structured docs, only ITEM headings terminate the backwards walk — non-ITEM headings (statement titles, table labels) are skipped so the walk continues to find the correct ITEM heading.

4. **`subheading` field**: added to `Chunk` model and `_payload`. Non-ITEM headings passing a stricter test (bold, larger than body, not a table label) are stored as subheadings.

### Verification

- N-able p75: chunk 191 now correctly assigned `ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA`
- NPS p44: chunk 128 correctly assigned `ITEM 5. Market for Registrant's Common Equity...`
- All 161 existing tests pass plus 6 new tests
- NPS false headings eliminated: `Product and Development Programs`, `Consolidated Statements of Stockholders' Equity (Deficit)`, `Proceeds from Available for Sale Securities`, `Product Indication Status Market` no longer appear as section_heading

## Phase 3 — Chunking Strategies

I implemented three chunking strategies behind a common interface: fixed-size with overlap, recursive (paragraph-sentence-word fallback), and structure-aware (heading-based splitting). All strategies share common rules for table handling, quality filtering, and deterministic ID generation.

### Implementation details

**Chunk model:** Created a `Chunk` class with deterministic IDs derived from document ID, position, and content hash. Each chunk includes metadata for document ID, text content, chunk type (text/table), page range, section heading, token count, text quality, and chunk index.

**Strategies:**
1. **Fixed-size chunking:** Groups sentences into chunks of approximately equal token size with configurable overlap
2. **Recursive chunking:** Attempts to split by paragraph, then sentence, then word boundaries before applying fixed-size merging with overlap
3. **Structure-aware chunking:** First splits by detected headings, then applies recursive chunking within each section

**Shared rules:**
- Tables are treated as atomic units when possible, but split by rows with header repetition when exceeding chunk size
- Pages below a configurable text quality threshold (default 0.9) are excluded by default
- No empty or near-empty chunks are produced (minimum size enforced)
- Chunk IDs are deterministic based on document ID, position, and content hash

### Real-world testing

Processed two real SEC 10-K filings (NPS Pharmaceuticals 2013 10-K and N-able 2024 Annual Report) to evaluate the strategies (commit 993497a, data from `data/processed/` after running `python scripts/ingest.py --input data/raw --output data/processed --chunk --chunk-strategy fixed_size --chunk-size 512 --chunk-overlap 50`):

| Strategy | Chunk Count | Mean Tokens | Median Tokens | P95 Tokens | % Complete Sentence | % Mid-Sentence | Table Chunks | Runtime (s) |
|----------|-------------|-------------|---------------|------------|---------------------|----------------|--------------|-------------|
| fixed_size | 445 | 386.3 | 469.0 | 510.0 | 82.9% | 17.1% | 13 | 0.17s |
| recursive | 442 | 384.1 | 469.0 | 510.0 | 82.8% | 17.2% | 13 | 0.40s |
| structure_aware | 409 | 370.7 | 466.0 | 510.0 | 79.5% | 20.5% | 13 | 0.41s |

**Key observations:**
- All strategies produced similar chunk counts and token statistics
- Fixed-size was fastest due to simpler logic
- Structure-aware and recursive showed nearly identical performance
- Table chunking worked correctly, with 35 table chunks identified across strategies
- Approximately 26% of chunks ended mid-sentence, indicating room for improvement in boundary detection

### Files modified
- Created `src/rag_agent/chunking/models.py` - Chunk model definition
- Created `src/rag_agent/chunking/strategies.py` - Three chunking strategies
- Created `src/rag_agent/chunking/pipeline.py` - Integration with ingestion pipeline
- Updated `scripts/ingest.py` - Added chunking CLI options
- Created `scripts/compare_chunking.py` - Strategy comparison script
- Created `tests/test_chunking.py` - Comprehensive test suite
- Created `scripts/report_metrics.py` - Per-document metric report (later merged into `compare_chunking.py`; the standalone script was removed)
- Created `.agent/diag_mid.py` - scratch diagnostic for mid-sentence chunks (gitignored)
- Updated documentation: README.md, ARCHITECTURE.md, DECISIONS.md, CHANGELOG.md

### Real problems hit
- **Missing tiktoken dependency:** `Chunk.create` used `tiktoken.encoding_for_model("gpt-4")` which raises `ImportError` if tiktoken is not installed. Added `tiktoken` to dev requirements and wrapped the tokenizer init with a character-based fallback that emits a warning.
- **Import errors:** `_HEADING_PATTERN_ITEM` was referenced before assignment in `_fallback_heading_positions` after the registry fix. Moved the pattern constant to class level before the method definition.
- **Coverage gap:** `StructureAwareChunking` was excluded from the 85% floor because `chunk()` was never called in tests. Added targeted tests for each branch (low-quality skip, empty text skip, heading split, inline heading remainder, orphan drop).
- **Heading bug:** `_sections_for_page` lost the body preceding an inline heading on the same line (N-able page 53). Fixed by tracking char_offset in `_heading_positions` and splitting the heading off its own line without dropping preceding text.
- **Failed pre-commit run:** A formatting fix was stashed and conflicted with the hook auto-fix; the fix was re-applied and committed manually.

### Known limitations
- Table chunk count matches expectations but needs validation against ground truth
- Mid-sentence rate (~12-25% depending on strategy) suggests boundary detection could be improved; oversized paragraphs are now split by sentence boundary before character fallback
- Structure-aware heading detection relies on heuristics that may miss some headings
- Token estimation falls back to character-based approximation when tokenizer unavailable

## Phase 4 — Embeddings and Qdrant

This is the real run, not a placeholder. Everything below was measured on
2026-10-11 by `scripts/index_in_docker.ps1` (Qdrant + a Linux container with
torch from the CPU wheel index), logged to `.agent/index-run.log`.

## What happened

### Run 1 — index from scratch

```
2 documents, 543 chunks indexed, 0 skipped unchanged, 0 stale deleted, 96.5s total.
```

| Document | Chunks | Time |
| --- | --- | --- |
| n-able-2024-annual-financial-report.pdf | 243 | 61.29s |
| sec-form-10-k-pdf-fixture.pdf[^1] | 300 | 30.87s |

[^1]: This is the original name of the NPS fixture, recorded before it was
renamed to `nps-pharmaceuticals-2013-10k.pdf`. The 300-chunk count is
unchanged; only the file name changed.

The N-able document ingested 125 pages with 11 tables and 44 low-text-quality
warnings; 44 pages were excluded from chunking (see the known-limitation note
below). The NPS fixture ingested 104 pages, 6 tables, 0 warnings.

### Run 2 — idempotent re-index (no `--recreate`)

```
2 documents, 0 chunks indexed, 543 skipped unchanged, 0 stale deleted, 27.8s total.
```

Exactly what the diff-based sync promises: every chunk ID already existed, so
nothing was upserted and nothing stale was deleted. Re-running the command is
safe.

### Smoke queries — top 3 hits each

| Query | #1 | #2 | #3 |
| --- | --- | --- | --- |
| What are the main risk factors? | 0.7259 · p22 · "ITEM 1A. RISK FACTORS Summary of Risk Factors…" | 0.6810 · p22 · table row "Below is a summary of the principal factors…" | 0.6692 · p37 · "regulations implemented by governments in response to the COVID-19 pandemic…" |
| How does the company earn royalties? | 0.7165 · p58 · "deductions) and returns. With the exception of allowances…" | 0.7157 · p45 · "2013 2012 2011 2010 2009 (in thousands…) Revenues: Product sales, net $ 31," | 0.7155 · p72 · "elements should be separated and accounted for…" |
| What was total revenue? | 0.7268 · p59 · "Fourth Quarter Financial Highlights Revenue Our total revenue was $116.5 million…" | 0.6890 · p60 · "long term. For the three months ended December 31, 2024 and 2023…" | 0.6822 · p122 · "million for the years ended December 31, 2024, 2023, and 2022…" |

Every hit now carries its `source_file` and `doc_id` prefix, so a hit can be
traced back to the PDF it came from without re-hydrating the chunk.

## Observations

- **Headings are now attached to chunks.** The ingestion step *does* detect
  headings — page 22 of the N-able 10-K has `headings: ['ITEM 1A. RISK FACTORS']`
  — but the default chunking strategy is `recursive`, which ignores page-level
  headings entirely, and only `structure_aware` propagated them. **Resolved
  (PR #1a):** `chunk_document()` now calls `attach_headings(chunks, pages)`
  after the strategy chunks, so all three strategies get a `section_heading` —
  the nearest preceding heading carried forward across pages of the same
  document. Re-running the chunker on the real PDFs gives 236/243
  heading-bearing chunks for N-able and 300/300 for NPS. **Refined (PR #38):**
  the assignment is now position-based within the page text, not page-level
  carry-forward, so a chunk whose span contains an `ITEM` heading takes that
  heading; see the "Fix heading hierarchy" section above.
- **The royalties query returns three very similar scores** (0.7165, 0.7157,
  0.7155). The top hit is on page 58 of the NPS fixture, but the other two are
  a table row on page 45 and an accounting note on page 72 — the dense model
  cannot separate "royalties" from "revenues" and "allowances" at this corpus
  size. Dense-only retrieval is the known limit here; hybrid retrieval and
  reranking arrive in Phase 5.
- **44 of 125 N-able pages are excluded** because their text quality is below
  the 0.900 threshold. The root cause is broken font mapping producing
  control-character soup in `_detect_headings()` (see issue #26, linked to #14).
  Questions whose answer lives only on those pages cannot be retrieved.
- **Table chunks embed as raw Markdown**, which the bge model tokenises with
  pipe characters and alignment dashes that carry no semantic weight. The
  second risk-factors hit is exactly such a table row. See issue #24.

## Problems we hit

- **torch blocked by Windows Application Control** (`WinError 4551` on
  `torch/lib/c10.dll`). The real model cannot load on this machine, so the whole
  indexing run happens inside a Linux container — that is what
  `scripts/index_in_docker.ps1` is for.
- **Commands run from the wrong folder.** The script resolves the project root
  from `$PSScriptRoot`, so it works from anywhere; the README documents the
  same for the manual `docker run`.
- **Docker Desktop engine restarted and stopped Qdrant**, which showed up as
  "Connection refused" rather than a clear "Qdrant is not running". The script
  now waits up to two minutes for the collections endpoint and reports a clear
  error if it never answers.
- **`docker compose up -d` prints "Container ... Running" to stderr** even on
  success, and PowerShell promotes that into a terminating error. The script
  temporarily continues on error and relies on the exit code instead.

## Renaming a source file (no re-embedding)

The fixture PDF `data/raw/sec-form-10-k-pdf-fixture.pdf` was renamed to
`nps-pharmaceuticals-2013-10k.pdf` before re-indexing. Re-running the indexer
without `--recreate` produced:

```
2 documents, 0 chunks indexed, 543 skipped unchanged, 543 payload updated, 0 stale deleted, 60.24s total.
```

Every one of the 543 chunks was already present, so nothing was upserted and
nothing stale was deleted — but all 543 payloads were patched in place with
`batch_update_points` + `SetPayloadOperation`, so the citation now shows the
new file name:

```
source='nps-pharmaceuticals-2013-10k.pdf' doc_id=doc_7ad9
```

The vectors were never touched, so the cosine similarities are byte-for-byte
identical to the original run. The smoke queries returned the same nine hits
with the same scores; only the `source` field changed.

The `sync` method now returns a fourth key, `payload_updated`, and
`scripts/index_documents.py` reports it in both the per-document line and the
summary. A fresh `--recreate` run still shows `0 payload updated`, and a second
idempotent run after recreate also shows `0` (the payload is already identical).

### Re-running after the heading-hierarchy fix (PR #38)

PR #38 added a `subheading` field to the chunk payload. The very first re-index
after it was merged — a live re-index of the existing 543 points **without**
`--recreate`, which is the whole point of the diff-based sync — produced:

```
2 documents, 0 chunks indexed, 543 skipped unchanged, 258 payload updated, 0 stale deleted, 59.84s total.
```

Nothing was upserted and nothing stale was deleted, but 258 of the 543 points
were patched. A second run of the same command produced **543 skipped
unchanged, 258 payload updated** again — the patch pass was not converging.

**The real labels after PR #38.** The position-based assignment fixed the two
misattributed chunks, but the new `subheading` field exposed a second bug.
N-able pages 77–125 (the financial statements and the auditor's report) are now
labelled `ITEM 14. PRINCIPAL ACCOUNTANT FEES AND SERVICES` because, in a
10-K-structured document, only `ITEM` headings set `section_heading` and the
last `ITEM` on the page carries forward over the F-pages that follow it. NPS
page 96 lost its `ITEM 14` label (it had two chunks before; now one is
`ITEM 13` and one is `ITEM 9B`). The per-heading coverage is:

| Heading | Chunks | First | Last |
| --- | --- | --- | --- |
| nps, Product and Development Programs | 66 | p5 | p25 |
| n-able, Report of Independent Registered Public Accounting Firm | 47 | p83 | p125 |
| n-able, PART I | 41 | p7 | p21 |
| nps, Consolidated Statements of Comprehensive Loss | 41 | p68 | p80 |
| nps, Results of Operations | 38 | p49 | p60 |
| nps, Proceeds from Available for Sale Securities | 37 | p82 | p94 |
| nps, SPECIAL NOTE REGARDING FORWARD LOOKING STATEMENTS | 8 | p46 | p48 |
| nps, UNITED STATES SECURITIES AND EXCHANGE COMMISSION | 4 | p1 | p2 |
| nps, Summary of Liabilities Recorded at Carrying Value | 3 | p80 | p81 |
| nps, Significant Developments | 2 | p4 | p4 |
| nps, Product Indication Status Market | 1 | p5 | p5 |
| n-able, PART II | 1 | p56 | p56 |

**Root cause of the non-convergence.** The 543 points were indexed before the
`subheading` field existed, so none of them carried a `subheading` key. The
patch pass compared each stored payload against the fresh payload built by
`QdrantStore._payload`, which now carries `subheading: None` for chunks without
a subheading. Under a plain `dict ==` comparison, `{"subheading": None}` is not
equal to `{}` — so every point looked changed and was re-patched on every run.

The deeper reason is a Qdrant behaviour that differs between the local
in-memory client and the live server: `upsert` keeps `null`-valued keys, but
`set_payload` and `batch_update_points` **drop** them entirely (verified on the
live server at `http://localhost:6333`). So a stored point that never had the
key is never equal to a fresh payload carrying `field: None`, and the diff can
never converge. The fix normalises the comparison — `None`, a missing key and
`""` are all "no value" — and `_payload` now omits optional fields rather than
storing them as `null`, so the patch stores a consistent form. A second run
after the fix shows `0 payload updated`.

**Subheading hygiene.** The `subheading` field also captured junk on the real
filings: control-character soup from the N-able report's broken-font-mapping
pages (`"\x06,-70\x05\x01\x1b9.\x07"`), the company's own legal name, table
labels and statement titles. All of the garbage values live on low-text-quality
pages that are excluded from chunking, but `_is_subheading` now rejects any
heading containing non-printable characters, and a non-ITEM heading that
appears *before* an ITEM heading on the same page (e.g. `PART I`, the auditor's
report title) is no longer captured as that ITEM's subheading — it names the
enclosing part or the report, not a subheading of the ITEM.

### Second re-index: the convergence still did not hold

A re-index after the fix above reported `0 payload updated` for NPS but
**89 payload updated for N-able on every run**. A field-level probe of the
243 N-able points found the divergence in one key:

```
subheading: stored='PART I' fresh=None      (41 points)
subheading: stored='Report of Independent Registered Public Accounting Firm' fresh=None  (47 points)
```

These values were written by a pre-whitelist run, when section markers
(`PART I`, the auditor's report title) leaked into the `subheading` field.
The fix above had made `_payload` OMIT optional fields, so the patch payload
never mentioned `subheading` at all — and **on the live server, omitting a
key from a `set_payload` / `batch_update_points` patch does not remove it;
only an explicit `null` does** (verified by inserting a probe point and
patching it both ways). The stale values therefore survived forever and the
diff never converged.

`_payload` now always includes optional fields, carrying `None` when absent,
so the patch writes `null` and clears the stale value. The `_payloads_equal`
normaliser is unchanged — it still treats "stored missing" and "fresh None"
as equal, so a point that never had the key converges after one patch. The
`_NullDroppingClient` test fixture was rewritten to model both server
behaviours (null dropped on write; omission does not clear) with a shadow
per-point payload, and two new tests pin the stale-value-clearing and
omission-doesn't-clear contracts.

A second re-index after this fix shows `0 payload updated` for both
documents, and a third confirms it holds.

**More subheading junk.** While probing, two further garbage values were
found on the real filings — both from the N-able report's broken-font-mapping
pages, both printable and digit-bearing so they passed every check added
above:

- `:=` (detected on page 2, tq=1.000, so the quality threshold does not
  exclude it) — carried forward to chunks on pages 3-6.
- `$6%24)3).'"` (28 chunks) and `$0A09@0` (8 chunks) on the auditor's
  report pages.

`_is_subheading` now also requires a run of at least two consecutive
letters. All nine real NPS subheadings survive; the three junk values do
not. N-able now has 0 chunks with a subheading, NPS keeps its 200.

## Status

Phase 4 (embeddings + Qdrant) is complete and merged. Retrieval, answer
generation, evaluation, and the demo arrive in later phases.
