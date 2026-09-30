# Development log

## Phase 1 — Foundation

I set up the package layout and a deliberately small configuration object so later components have one clear place to read non-secret settings. I chose LangChain for the eventual orchestration layer because its retriever and streaming interfaces work well with a FastAPI service, while keeping the project’s retrieval logic testable in plain Python.

The main snags at this stage are external GitHub automation and a Windows application-control restriction: the GitHub CLI is not installed, and its policy blocks pre-commit's bundled YAML checker. I kept the local gate dependable by retaining end-of-file, whitespace, merge-conflict, Ruff lint, and Ruff format hooks; GitHub Actions remains the server-side quality gate. The codebase itself is provider-agnostic and does not require an API key to run the current checks.

## Phase 2 — PDF ingestion

I built a page-aware PDF ingestion path around deterministic SHA-256 document IDs. It uses PyMuPDF for text positions and font signals, pdfplumber for ruled tables, and writes a hash-named JSON record so a repeated directory run skips unchanged content. The CLI reports files, pages, tables, warnings, and elapsed time rather than hiding its work.

### Parsing quality notes

**What worked well.** On the official NPS Pharmaceuticals 2013 Form 10-K PDF, the parser retained all 104 physical PDF pages as text pages, found 6 tables across 4 pages, and flagged 0 scanned pages. The patent table on PDF page 21 was extracted as readable Markdown. On the official N-able 2024 annual report, it retained 125 text pages, found 11 tables across 8 pages, and flagged 0 scanned pages. Page metadata keeps the physical PDF index, so later citations remain reproducible even when a filing's printed numbering differs.

**Measured timing.** The NPS filing parsed in **11.05 seconds** inside the ingestion pipeline, or **0.106 seconds per PDF page**. The CLI process took 12.12 seconds including startup and one already-indexed-file skip. The N-able report parsed in **19.30 seconds**, or **0.154 seconds per PDF page**. Both figures were measured locally on 2026-09-30 with the repository CLI; the PDFs and generated JSON remain ignored.

**What is imperfect.** The N-able report exposed a font-encoding failure: 44 of 125 pages contain control-character or `(cid:...)` text, and 4 of 11 retained tables contain the same artifact. This is not a scanned-page problem, so the current warning does not catch it. On the NPS report, the unruled multi-year balance-sheet table on PDF page 80 was fragmented by pdfplumber; the parser now discards the surrounding one-column prose regions falsely reported as tables, but it does not reconstruct the five-column table. This is a real limitation, not a passing result.

**Header/footer and reading-order checks.** In the N-able validation audit, repeated edge text `Table of Contents` and printed page `2` were removed from PDF page 5 while the following `Safe Harbor Cautionary Statement` body paragraph remained. That is a positive spot check, not proof that the frequency heuristic never removes meaningful boundary text; a repeated running heading can be semantically useful. In the NPS report, 102 pages had a standalone printed page number removed. I inspected PDF page 80 because the layout detector found left/right block groups there: its narrative paragraphs stayed in reading order, but the groups were driven largely by a financial table rather than a conventional two-column article. I do not claim verified support for every sidebar or multi-column layout.

**Future work.** Add OCR and an explicit low-quality-text warning for image-only or broken-font pages; use table-region detection and merging for unruled/multi-page financial tables; render representative pages in automated validation; and add safer exceptions for repeated headings that are meaningful page titles.
