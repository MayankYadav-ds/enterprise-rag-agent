# Data sources

The project domain is public SEC annual reports and Form 10-K filings. `data/raw/` and `data/processed/` are intentionally Git-ignored: download source PDFs locally, and keep derived JSON local. The only tracked PDF is the 2.7 KB synthetic test fixture in `tests/fixtures/`.

## Validated SEC documents

| Document | Publisher and source | Accessed | Reuse basis |
| --- | --- | --- | --- |
| NPS Pharmaceuticals, Inc. 2013 Form 10-K (PDF) | [SEC filing index](https://www.sec.gov/Archives/edgar/data/890465/0001136261-14-000082-index.html) and [direct PDF](https://www.sec.gov/Archives/edgar/data/890465/000113626114000082/form10-k.pdf) | 2026-09-30 | SEC states that EDGAR public filing content is free to access and reuse. |
| N-able, Inc. 2024 Annual Financial Report (PDF) | [Direct SEC PDF](https://www.sec.gov/Archives/edgar/data/0001834488/000183448825000091/a2024n-ablears.pdf) | 2026-09-30 | SEC EDGAR public filing content is free to access and reuse. |

The NPS filing is the primary readable validation sample in Phase 2. The N-able report is retained locally as a difficult modern-PDF stress case; its embedded font mapping produces unreadable control characters on some pages, as recorded in `DEVLOG.md`.

To download the NPS filing from PowerShell:

```powershell
curl.exe --fail --location --user-agent "enterprise-rag-agent/0.1 contact: you@example.com" `
  --output data/raw/sec-form-10-k.pdf `
  https://www.sec.gov/Archives/edgar/data/890465/000113626114000082/form10-k.pdf
```

Replace the contact address with yours and follow SEC access policies. Do not commit downloaded filings.

## Optional open technical-manual source

[NASA Systems Engineering Handbook, NASA/SP-2016-6105 Rev 2](https://ntrs.nasa.gov/citations/20170001761) is a useful contrasting document type. NASA's record marks it "Public Use Permitted." It has dense prose, callout boxes, figures, and an extensive contents structure, so it is suitable for later parser and chunking comparisons.

## Provenance rule

Record the title, publisher, source URL, access date, and reuse basis for every future source before ingestion. Use only freely available, legally usable documents.
