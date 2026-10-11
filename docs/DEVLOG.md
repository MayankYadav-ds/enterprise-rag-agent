# DEVLOG — Phase 4 real run

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
| sec-form-10-k-pdf-fixture.pdf | 300 | 30.87s |

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

- **Headings are `None` on all 9 hits.** The ingestion step *does* detect
  headings — page 22 of the N-able 10-K has `headings: ['ITEM 1A. RISK FACTORS']`
  — but the default chunking strategy is `recursive`, which ignores page-level
  headings entirely. Only `structure_aware` propagates them. See issue #25.
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

## Status

Phase 4 (embeddings + Qdrant) is complete and merged. Retrieval, answer
generation, evaluation, and the demo arrive in later phases.
