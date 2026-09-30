# Design decisions

This is a lightweight decision log. Decisions can change when measured evidence points to a better trade-off.

## ADR-001: Use LangChain as the orchestration layer

**Status:** accepted (Phase 1)

The project will use LangChain rather than LlamaIndex. The pipeline needs modular retrievers, provider-neutral model interfaces, and async streaming that fit a FastAPI service. LangChain provides those primitives without requiring the project to hide retrieval behind a framework-specific index abstraction.

**Trade-off:** LangChain changes quickly and adds abstraction. Core records, metrics, prompts, and retrieval-fusion logic will therefore remain in this repository and be covered by tests.

## ADR-002: Preserve source metadata as a first-class record

**Status:** accepted (Phase 1)

Document ID, source URL, page number, section, and extraction method will travel with every chunk. Citations are a product requirement, not a rendering step bolted on after generation.

**Trade-off:** Metadata-aware models and storage schemas take more care to design, but they enable filterable retrieval and auditable answers.

## ADR-003: Make providers configurable

**Status:** accepted (Phase 1)

The model provider and model name are environment settings. The eventual generation layer will support both a local/open model and an API provider.

**Trade-off:** A provider adapter adds surface area, but makes the demo usable without committing to one commercial API or embedding credentials in the codebase.

## ADR-004: Combine PyMuPDF text layout with pdfplumber table extraction

**Status:** accepted (Phase 2)

PyMuPDF provides fast page access, positioned text blocks, span font sizes, font flags, and page images. The parser uses those signals for reading order, heading heuristics, repeated edge-text removal, and image-only-page warnings. pdfplumber complements it with a focused ruled-table API that yields row matrices, which the project renders as Markdown.

**Trade-off:** PDF tables are presentation instructions, not a universal semantic structure. Ruled tables extract well, while unruled or highly merged financial tables can split or be missed. The parser filters single-column prose regions that ruled-line detection mistakes for tables, but it does not yet merge fragments or use an OCR/layout model.

## ADR-005: Cite physical PDF page indexes

**Status:** accepted (Phase 2)

Every `PageContent.page_number` is the one-based index of the page in the PDF file. This is deterministic, works for every source, and lets a reader navigate a downloaded source without relying on extracted text. SEC filings can have a cover and contents pages before the report's printed page numbering, so a citation's PDF index can differ from the page number printed on the page.

**Trade-off:** The future UI should make this distinction explicit, for example: "PDF p. 46 (printed p. 46)" when a printed number can be detected safely.
