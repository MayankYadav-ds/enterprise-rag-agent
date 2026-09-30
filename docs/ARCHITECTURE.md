# Architecture

## Current state

Phase 2 implements local, page-aware PDF ingestion. The pipeline persists deterministic JSON records in `data/processed/`, which remains outside version control. Vector storage, model calls, and HTTP endpoints are not implemented yet.

## Ingestion flow

```mermaid
flowchart LR
    S[PDF in data/raw] --> H[SHA-256 hash]
    H --> I[Deterministic doc_id]
    S --> M[PyMuPDF: text, blocks, fonts]
    M --> O[Reading-order and text cleaning]
    O --> E[Header/footer frequency filter]
    S --> T[pdfplumber: ruled table extraction]
    E --> P[PageContent: PDF page index, text, headings]
    T --> P
    P --> J[Hash-named JSON in data/processed]
    M --> W[Image-only page warning]
```

`PageContent.page_number` is the one-based physical PDF page index. It deliberately does not attempt to reconcile a filing's printed page number, which can start after cover and contents pages.

## Target request path

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Retriever as Hybrid retriever
    participant Store as Qdrant
    participant Model as Configured LLM
    Client->>API: POST /query
    API->>Retriever: question + filters
    Retriever->>Store: dense candidates
    Retriever->>Retriever: BM25 + RRF + rerank
    Retriever-->>API: evidence chunks with page metadata
    API->>Model: grounded prompt + evidence
    Model-->>Client: SSE answer tokens and citations
```

## Architectural invariants

- Each chunk keeps document identity and page number from ingestion through generation.
- Retrieval quality is measured independently from generative quality.
- Answer generation receives only retrieved evidence, and declines to answer when evidence does not meet a configured threshold.
- Provider credentials stay in environment variables and never enter version control or logs.

The details will be updated as each component is implemented.
