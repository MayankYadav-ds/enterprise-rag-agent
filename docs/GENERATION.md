# Phase 6: evidence-constrained answer generation

The generation layer composes with the existing `Retriever` protocol. It does
not alter dense, BM25, hybrid, reranker, gold-set, or retrieval-metric behavior.
This first implementation is deliberately **extractive**: an LLM chooses exact
passages and an answer status; the application validates those passages and
renders citations. It does not display unchecked, abstractive model prose.

## Components

| Component | Responsibility |
| --- | --- |
| `generation.models` | Frozen answer/excerpt dataclasses, strict response schema, typed errors |
| `generation.config` | Validated configuration from process environment |
| `generation.provider.LLMProvider` | Injectable `complete(messages, schema=...) -> str` contract |
| `generation.provider.OllamaProvider` | Non-streaming structured chat via standard-library HTTP |
| `generation.engine.AnswerGenerator` | Evidence validation, budget, prompt, output checks, citations |
| `generation.engine.AnswerPipeline` | Existing retriever followed by the generator |
| `python -m rag_agent.generation` | Saved-evidence CLI without Qdrant |
| `scripts/search.py --answer` | Existing search strategies followed by generation |

No dependency was added. Pydantic was already a required dependency. Importing or
constructing a provider does not contact a service. No code pulls models or starts
services. The adapter uses Ollama's [chat endpoint](https://docs.ollama.com/api/chat)
and [JSON-schema format](https://docs.ollama.com/capabilities/structured-outputs).

## Configuration and execution

Use an already-running Ollama service with an already-installed model that
supports structured output. Model selection is explicit: there is no hardcoded
model or silent fallback. The service's available models and context capacity are
the operator's responsibility. `.env` files are not automatically loaded.

PowerShell, from the repository root:

```powershell
$env:RAG_LLM_PROVIDER = 'ollama'
$env:RAG_LLM_MODEL = '<your-already-installed-model-name>'
$env:RAG_LLM_BASE_URL = 'http://localhost:11434'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

# Existing populated Qdrant collection; BM25 does not load an embedding model.
.venv\Scripts\python.exe scripts/search.py 'What risks does the report identify?' --strategy bm25 --top-k 5 --answer

# Requires cached embedding and cross-encoder models as well as the LLM service.
.venv\Scripts\python.exe scripts/search.py 'What risks does the report identify?' --strategy hybrid --rerank --rerank-candidates 20 --top-k 5 --answer
```

The existing QdrantStore constructor can create a missing collection; use the
correct existing collection/endpoint for search. Generation itself does not
write to Qdrant. No live search or LLM calls were used to validate Phase 6.

Without `--answer`, search output and behavior remain unchanged. `--queries-file`
also supports answer mode, using the same retriever and provider across questions.
`status=answered`, `status=insufficient_evidence`, or `status=conflicting_evidence`
precedes the answer. Operational or output-validation failures return exit code 2;
abstention is a valid result and returns 0. Search stops its batch on a generation
failure; previously printed answers remain on stdout.

| Environment variable | Default / meaning |
| --- | --- |
| `RAG_LLM_PROVIDER` | `ollama`; `local` is an alias for compatibility |
| `RAG_LLM_MODEL` | Required, nonblank installed model name |
| `RAG_LLM_BASE_URL` | `http://localhost:11434`; HTTP(S) origin, no embedded credentials |
| `RAG_LLM_TIMEOUT_SECONDS` | 60; positive request/socket timeout, at most 600 |
| `RAG_LLM_MAX_OUTPUT_TOKENS` | 2048; model output limit |
| `RAG_LLM_MAX_RESPONSE_BYTES` | 131072; maximum HTTP response size |
| `RAG_MAX_EVIDENCE_CHARS` | 24000; serialized evidence budget including metadata |
| `RAG_MAX_EVIDENCE_CHUNKS` | 12; maximum unique evidence chunks |

Configuration errors are explicit. Only Ollama is implemented; other integrations
can implement `LLMProvider` and be injected without changing the generator.
The adapter does not use `OPENAI_API_KEY` or any other credential. A remote
endpoint receives the question and selected document content; configure only an
endpoint appropriate for that data. HTTP redirects are not followed.

## Python integration

Given any already-configured `retriever` implementing the existing protocol:

```python
from rag_agent.generation import (
    AnswerGenerator, AnswerPipeline, GenerationConfig, build_provider,
)

config = GenerationConfig.from_env()
generator = AnswerGenerator(
    build_provider(config),
    max_evidence_chars=config.max_evidence_chars,
    max_chunks=config.max_chunks,
)
pipeline = AnswerPipeline(retriever, generator)
answer = pipeline.answer("What were the reported risks?", top_k=5)
print(answer.status)
print(answer.text)

# Or use chunks already retrieved by another application:
answer = generator.generate("What were the reported risks?", retrieved_chunks)
```

`top_k`, `doc_id`, `chunk_type`, and `min_score` are forwarded unchanged to the
retriever. No universal score threshold is assumed: the retrieval strategies have
different score semantics. Result `excerpts` include chunk IDs, exact quotations,
source files, and physical page ranges. Result `evidence_ids` records the local
IDs supplied in the prompt; it is not a corpus identifier or a quality score.

For the saved-evidence CLI, provide a JSON list of `dataclasses.asdict(hit)`
objects. Its fields match `RetrievedChunk` exactly. The canonical
`data/eval/all_chunks.json` has a different shape and is not valid CLI input.
Save new evidence snapshots to unused paths; do not overwrite evaluation files.

```powershell
.venv\Scripts\python.exe -m rag_agent.generation 'What were the reported risks?' --evidence C:\path\to\retrieved-evidence.json --json
```

The saved-evidence CLI reads at most 1 MiB / 100 chunks, writes only stdout/stderr,
and never opens Qdrant. The same LLM environment settings apply.

## Grounding policy

1. Reject malformed chunk IDs, filenames, text, or page ranges before model use.
   Exact duplicate chunks are deduplicated; inconsistent duplicate IDs fail.
2. Empty evidence abstains without calling the LLM. If the full evidence set
   exceeds the budget, abstain with `evidence_budget_exceeded`; never silently
   omit a potentially contradictory chunk or cut a paragraph mid-sentence.
3. Questions and evidence are serialized as data, separate from system rules.
   The model is instructed to ignore document instructions, use no outside facts,
   preserve qualifications, and compare entities, periods, units and scope.
4. Validate strict JSON: only status and evidence selections are accepted. Unknown
   IDs, invented/non-verbatim quotes, extra citation fields, empty answers, and
   duplicate selections raise `InvalidResponseError`. No partial answer escapes.
5. Construct citations using the existing `format_citation()` and original
   metadata. The model never provides filenames, page numbers or citation strings.
6. An insufficient-evidence selection emits a fixed uncertainty response. A
   conflicting-evidence selection requires at least two distinct chunks and shows
   both accounts with citations without choosing a winner. Conflicts confined to
   one chunk cannot satisfy this schema; the model should abstain instead.
7. `ProviderError` identifies transport, malformed-envelope, size-limit and
   truncation failures. These are operational failures, not evidence insufficiency.
   Raw service error bodies are not printed. There are no automatic retries.

Answers are plain text with quoted source material. A future HTML/Markdown UI
must escape untrusted excerpts and metadata before rendering. Stored metadata
is treated as provenance; the generator does not reopen PDFs to verify it.

## Limits and validation

Exact quotation verifies **provenance, not truth or entailment**. A model can
still select irrelevant or misleading fragments, overlook negation/context, miss
conflicts, or misclassify sufficiency. Prompt-injection resistance is layered, not
a guarantee. No deterministic validator can establish semantic correctness from
substring matching alone. Source errors and incomplete retrieval remain limits.

Character budgets are not model-token budgets; configure a model with adequate
context capacity. The service may truncate its input independently. No real-model
quality or conflict-detection benchmark is claimed. Tests cover mocked model/HTTP
responses, citation integrity, abstention, conflicts, injection attempts, bounded
output, errors, CLI behavior, and integration with all retrieval strategies.

Free-form synthesis, streaming/SSE, conversation memory, HTTP API, and learned
faithfulness evaluation are deferred. Buffering the whole response allows output
validation before exposing any answer. Existing retrieval evaluation artifacts
and canonical gold labels remain unchanged.
