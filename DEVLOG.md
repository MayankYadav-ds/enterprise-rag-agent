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
