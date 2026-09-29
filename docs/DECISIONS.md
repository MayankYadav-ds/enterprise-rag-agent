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
