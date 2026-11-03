# Phase 7: FastAPI backend

The REST API reuses the existing retrieval and evidence-constrained generation
components. It returns validated quotations with citations, not unrestricted
LLM prose. Phase 9 adds managed PDF ingestion and document lifecycle endpoints.

## Install and start

From the repository root, using the existing virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[api,dev]"
# Also install .[embeddings] for dense/hybrid retrieval or reranking.
# Models must already be cached when running with these offline settings:
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:QDRANT_URL = "http://localhost:6333"
$env:RAG_LLM_PROVIDER = "ollama"
$env:RAG_LLM_MODEL = "your-already-installed-model"
.venv\Scripts\python.exe -m uvicorn rag_agent.api.app:create_app --factory --host 127.0.0.1 --port 8000 --no-access-log
```

Replace the model placeholder with a model already installed in Ollama.
The API never starts services or pulls Ollama models. Use one API worker for document management. Existing embedding and
reranking loaders can download Hugging Face weights unless the offline variables
above are set. Health, Swagger, and OpenAPI need neither cached weights nor live
Qdrant/Ollama. Retrieval requires an existing indexed Qdrant collection. Query
also requires valid generation configuration; empty evidence skips the LLM call.
BM25 without reranking does not load any ML model and needs only payload access.

Browse `http://127.0.0.1:8000/docs` for Swagger or `/openapi.json` for the schema.
Swagger's browser assets use its upstream CDN; schema generation is fully offline.
Use one worker for the local demo. Each extra worker owns separate model caches.

## Configuration

Export environment variables before starting the process; `.env` is not loaded
automatically. Client requests cannot change service URLs, model names, or keys.

Authentication uses the same SQLite registry under `RAG_DOCUMENT_DIR`. Set
`RAG_ENV=development` for local HTTP operation, `RAG_ENV=test` for test
configuration, or `RAG_ENV=production` for production; startup through
`APISettings.from_env()` rejects an unset value. An
operator provisions users without exposing passwords in shell history with
`.venv\\Scripts\\python.exe scripts/create_user.py <username> --admin` before
any account exists. This creates the initial owner; there is no public
registration or subsequent account-creation command. The password is prompted
twice and must contain 12 to 1024 characters. For local-console recovery, run
`.venv\\Scripts\\python.exe scripts/reset_owner_password.py <username>`; it
prompts for the new password and revokes all sessions for the owner. There is no
web recovery endpoint. Login failures are throttled per direct client address;
the default is 10 failures per 15 minutes, configurable with the variables below.
The limiter reserves each in-flight login in a short SQLite `BEGIN IMMEDIATE`
transaction before password verification, then records failure or clears prior
failures in a second transaction. SQLite coordinates independent connections
and workers sharing this local database file; this is not distributed rate
limiting across hosts or separate databases. Database contention fails closed
with a temporary service error. Successful authentication clears the peer's
failure count.
`RAG_TRUSTED_ORIGINS` is a comma-separated
allowlist used for login and CSRF-token refresh. Production requires an explicit
`RAG_TRUSTED_ORIGINS` value containing one or more HTTPS origins; HTTP origins are
rejected. Production session cookies use Secure, HttpOnly, SameSite=Lax, and
Path=/; local development permits HTTP while keeping HttpOnly and SameSite=Lax.
The frontend proxy must forward `/health`, `/auth/*`, and `/api/v1/*` to FastAPI.
The frontend holds the CSRF token
in memory and refreshes it after restoring a valid session. The server retains a
bounded set of recent CSRF token hashes per session so other active tabs continue
to work after a refresh. Sessions expire after eight hours; logout revokes the
stored session and its CSRF verifiers. A protected API 401 expires the frontend
session and returns the user to login. Recovery requires an admin user.
Authentication does not yet provide per-user document ownership or tenant isolation:
authenticated users share the configured corpus and document-management scope.

| Variable | Default / meaning |
| --- | --- |
| `QDRANT_URL` | `http://localhost:6333`; remote Qdrant requires HTTPS |
| `QDRANT_API_KEY` | Required for non-loopback Qdrant; excluded from configuration serialization |
| `RAG_LOGIN_RATE_LIMIT_ATTEMPTS` | 10 failed attempts per direct client address |
| `RAG_LOGIN_RATE_LIMIT_WINDOW_SECONDS` | 900 seconds; range 60-86400 |
| `RAG_API_COLLECTION` | `rag_chunks`; must already exist |
| `RAG_API_QDRANT_TIMEOUT` | 10 seconds, positive and at most 120 |
| `RAG_LOG_LEVEL` | `INFO`; DEBUG/INFO/WARNING/ERROR/CRITICAL |
| `EMBEDDING_MODEL` | Existing embedder default `BAAI/bge-small-en-v1.5`; must match the collection |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANKER_CANDIDATE_COUNT` | 20, range 1-100 |
| `RERANKER_DEVICE` | `cpu` |
| `RERANKER_BATCH_SIZE` | 32 |
| `RERANKER_MIN_SCORE` | Optional post-reranking logit threshold |
| `RAG_LLM_PROVIDER` | `ollama` (`local` is an alias) |
| `RAG_LLM_MODEL` | Required for query; no default model is invented |
| `RAG_LLM_BASE_URL` | `http://localhost:11434` |
| `RAG_LLM_TIMEOUT_SECONDS` | 60 |
| `RAG_MAX_EVIDENCE_CHARS` | 24000 serialized characters |
| `RAG_MAX_EVIDENCE_CHUNKS` | 12 |

See [GENERATION.md](GENERATION.md) for output/response limits and grounding
guarantees. Leave `EMBEDDING_MODEL` unset to use its default; do not export an
empty value. Configuration is server-owned and caches last until process restart.

## Endpoints

`GET /health` returns HTTP 200 with `{"status":"ok"}`. This is process liveness,
not a claim that Qdrant, models, or Ollama are available.

Both POST endpoints accept the same strict JSON schema:

```json
{
  "question": "What was revenue?",
  "strategy": "hybrid",
  "top_k": 5,
  "rerank": false,
  "doc_id": null,
  "chunk_type": null,
  "min_score": null
}
```

- `question`: nonblank, trimmed, maximum 4000 characters.
- `strategy`: `dense`, `bm25`, or `hybrid` (default).
- `top_k`: integer 1-100, default 5. With reranking it must not exceed the
  configured candidate count.
- `rerank`: boolean, default false. Uses the real cross-encoder in production.
- `doc_id`: optional exact document filter; `chunk_type`: `text` or `table`.
- `min_score`: optional finite number passed to the existing base retriever.
  Dense similarity, BM25 scores, and RRF scores are not interchangeable.
  Hybrid forwards this threshold to both component retrievers. With reranking,
  it remains an upstream filter; `RERANKER_MIN_SCORE` filters final logits.

Unknown fields, coerced booleans/integers, invalid enums, and invalid ranges
return 422. Without reranking, hybrid uses `top_k` candidates from each component;
with reranking it uses `RERANKER_CANDIDATE_COUNT` before selecting final `top_k`.
This is existing pipeline behavior, not a controlled retrieval-quality benchmark.

### Retrieval only

```powershell
$body = @{ question="revenue"; strategy="bm25"; top_k=3 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/retrieve -Method Post -ContentType application/json -Body $body
```

Illustrative response (synthetic example, not an evaluation result):

```json
{
  "question": "revenue",
  "strategy": "bm25",
  "reranked": false,
  "evidence": [{
    "chunk_id": "c1", "doc_id": "d1", "source_file": "report.pdf",
    "page_start": 7, "page_end": 8, "section_heading": "Revenue",
    "subheading": null, "chunk_type": "text", "text": "Revenue was $10 million.",
    "score": 0.3, "rank": 1, "retriever": "bm25",
    "citation": "[report.pdf, pp. 7-8, Revenue]"
  }]
}
```

### Grounded query

```powershell
$body = @{ question="What was revenue?"; strategy="hybrid"; top_k=5; rerank=$true } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/query -Method Post -ContentType application/json -Body $body
```

The response includes the same retrieval fields and evidence, plus `status`,
`answer`, `citations`, and `reason`. An illustrative successful answer adds:

```json
{
  "status": "answered",
  "answer": "The retrieved documents state:\n\n\"Revenue was $10 million.\" [report.pdf, pp. 7-8, Revenue]",
  "citations": [{
    "evidence_id": "E1", "chunk_id": "c1", "source_file": "report.pdf",
    "page_start": 7, "page_end": 8, "quote": "Revenue was $10 million.",
    "citation": "[report.pdf, pp. 7-8, Revenue]"
  }],
  "reason": null
}
```

`evidence` contains the actual retrieved passages; `citations` contains only
validated excerpts selected for the answer. Empty retrieval or evidence exceeding
generation budgets produces `insufficient_evidence`, with no model call. Model
abstention also uses that status. `conflicting_evidence` displays supported quotes
from at least two distinct chunks without choosing a winner. These are successful
HTTP 200 outcomes, distinct from operational failures. Citation IDs, filenames,
pages, and quotes are checked by the existing generator, never invented by the API.
Semantic relevance and conflict detection still depend on the model and evidence.

## Failures and logging

| HTTP status | Meaning |
| --- | --- |
| 422 | Invalid request or reranking candidate-count constraint |
| 503 | Retrieval/model dependency unavailable, missing/mismatched collection, invalid generation configuration, or Ollama/provider failure |
| 502 | Generation failed its evidence/output contract; no answer is returned |
| 500 | Unexpected internal failure; no exception details are exposed |

Example error envelope:

```json
{"code":"dependency_unavailable","message":"Required service or model unavailable","request_id":"server-generated-id"}
```

Every response has `X-Request-ID`. JSON request logs contain the generated ID,
method, matched route template, status, and duration, excluding questions, evidence,
query strings, keys, and raw exception details. `--no-access-log` disables Uvicorn's
separate raw-URL access logging. Configure log collection according to your deployment.

## Lifecycle, tests, and limitations

Models initialize lazily on qualifying requests. Per-worker retrieval is serialized
to avoid concurrent model initialization/inference. Retrievers are cached by
strategy; reranked strategies each own a lazy cross-encoder, so using several
strategies can consume additional RAM. API-managed mutations invalidate cached
retrieval views automatically. Never run external indexing concurrently; restart
the API after offline CLI indexing to refresh BM25. Failed initialization
can be retried. The runtime closes its owned Qdrant client during shutdown.

Tests override `get_retriever` and `get_generator_factory` in
`app.dependency_overrides`. Runtime tests also exercise real retrieval classes with
isolated in-memory Qdrant, fake embeddings, and a fake cross-encoder solely to test
wiring. They establish no model-quality result and contact no live services.

```powershell
.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_api_runtime.py --no-cov
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

The health endpoint, login endpoint, and API documentation/OpenAPI endpoints are
public. Protected retrieval, query, and document endpoints require authentication;
state-changing protected requests also require CSRF validation. Document recovery
requires admin authorization. Admission control, rate limiting, and streaming are
not implemented. Do not expose the service publicly without deployment hardening.
Gold data, historical evaluations, and CLI interfaces are unchanged.
Managed document mutations are explicit API operations described below.
No live-LLM quality benchmark is claimed.

## Phase 9: PDF ingestion and management

Run one API worker with one durable storage directory for the collection. Keep
that directory across restarts. Do not run concurrent CLI writers, other indexing
applications, or API instances using different storage roots against the same
collection. The collection must already exist with the intended embedding model
and dimension. The API never creates, recreates, or clears a collection.

| Endpoint | Behavior |
| --- | --- |
| `POST /api/v1/documents?filename=report.pdf` | Raw `application/pdf` body; 202 with a queued record |
| `GET /api/v1/documents` | Managed inventory plus read-only legacy/vector documents |
| `GET /api/v1/documents/{doc_id}` | Durable processing status for a managed document |
| `DELETE /api/v1/documents/{doc_id}` | Confirmed, document-specific deletion; retryable |
| `POST /api/v1/documents/recover` | Retry cleanup; 204 only when retrieval can proceed |

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/documents?filename=report.pdf' -Method Post -ContentType application/pdf -InFile .\report.pdf
Invoke-RestMethod http://127.0.0.1:8000/api/v1/documents
```

A 202 response means acceptance, **not successful indexing**. Poll until `ready`,
`failed`, or `cleanup_required`. States are `queued`, `processing`, `indexing`,
`ready`, `failed`, `cleanup_required`, `deleting`, `deleted`, and `external`.
Records contain `doc_id`, `filename`, `managed`, `owns_vectors`, `size_bytes`,
`page_count`, `chunk_count`, `updated_at`, and a sanitized `error`. Unknown metadata
for external documents is null. During indexing, chunk count is the prepared count;
only `ready` confirms indexing.

Extension, media type, PDF signature, actual streamed size, filename, encryption,
page count, and usable chunks are checked. A PDF-looking file that fails full
parsing becomes failed, not ready. OCR is not added: scanned files with no usable
text fail explicitly. Client filenames are citation labels only; random UUIDs
determine disk paths. Path syntax, control characters, reserved device names, and
citation delimiters are rejected. Renaming identical bytes does not bypass the
full SHA-256 duplicate check. A duplicate returns 409; safely failed or deleted
documents may be re-uploaded.

Additional failures: 400 invalid length, 413 oversized content, 415 invalid
PDF/media type, 422 invalid filename, 404 unknown/unmanaged deletion target, 409
already registered/indexed, and 503 unavailable services or incomplete cleanup.
Responses omit internal paths and raw exceptions. State logs contain document IDs
and statuses, not contents.

| Environment setting | Default |
| --- | --- |
| `RAG_DOCUMENT_DIR` | `data/managed_documents` (ignored by Git) |
| `RAG_UPLOAD_MAX_BYTES` | 20971520 (20 MiB), range 1 KiB-100 MiB |
| `RAG_DOCUMENT_MAX_PAGES` | 500 |
| `RAG_DOCUMENT_MAX_CHUNKS` | 10000 |

The processing chain reuses PDF ingestion/cleaning, recursive chunking (400 tokens,
39 overlap), the configured embedder, and `QdrantStore.sync`. Keep the offline
environment settings to prohibit model downloads.

### Integrity and recovery contract

SQLite records ownership and mutation intent before Qdrant writes. The registry
is bound to its Qdrant URL/collection. A file lock prevents other processes using
the same registry; mutation and retrieval share the runtime lock. Status and
inventory remain readable during processing. A ready state requires confirmed
point count after indexing.

Uncertain writes trigger exact document-ID compensation. Unverified cleanup stays
`cleanup_required` and blocks API retrieval with 503. Reopening the registry
reconciles interrupted indexing/deletion. Interrupted queued/parsing jobs become
failed and can be re-uploaded. Recovery is also explicitly retryable. Explicit
recovery abandons queued work; do not invoke it while healthy jobs are waiting.
This is compensating recovery, not a shared SQLite/Qdrant transaction. Readers and
writers outside this API are not protected by its lock or recovery gate.

Deletion is restricted to registered API-owned documents. It journals intent,
deletes only exact document-ID points with acknowledgement, verifies absence, and
removes the owned PDF and known parsed JSON artifacts. It never recursively removes
directories. A metadata tombstone remains for idempotency. Failed artifacts remain
until deletion or operator-reviewed maintenance. Historical/in-flight answers keep
their original evidence snapshots. Other documents, datasets, gold labels, and
historical evaluation files are never deletion targets. Legacy documents are not
automatically adopted even if their content hashes match.

Protected document routes require authentication, state-changing routes require
CSRF validation, and recovery is admin-only. Login throttling uses the direct peer
address visible to the API, not untrusted forwarded headers; behind a proxy this
can aggregate clients unless deployment configures the trusted network boundary.
The implementation has no distributed job queue, upload-rate limit, malware scanner, parser process sandbox, or automatic
retention policy. Byte/page/chunk limits do not bound every possible PDF parser
resource cost.
Add deployment controls before accepting untrusted public uploads. Health and docs
do not initialize storage or contact services.
