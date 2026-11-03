# Phase 8: Document intelligence frontend

The `frontend/` application uses React, TypeScript, Vite, Tailwind CSS, and Lucide
icons. It connects to the Phase 7 query and Phase 9 document endpoints. There is no
demo-data mode in the application; synthetic responses exist only in tests.

## Start locally

Use Node.js 22.12+ (validated with Node 24) and the existing Python environment.
From the repository root, start the backend in one PowerShell terminal:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:RAG_ENV = "development"
$env:QDRANT_URL = "http://localhost:6333"
$env:RAG_LLM_MODEL = "your-already-installed-model"
.venv\Scripts\python.exe -m uvicorn rag_agent.api.app:create_app --factory --host 127.0.0.1 --port 8000 --no-access-log
```

Replace the model placeholder with an already installed Ollama model. See
[API.md](API.md) for backend installation and configuration. The frontend never
starts Qdrant/Ollama, indexes documents, downloads models, or supplies API keys.

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite, normally `http://127.0.0.1:5173`. The development
proxy forwards `/health`, `/auth/*`, and `/api/v1/*` to
`http://127.0.0.1:8000`. Backend CORS changes are unnecessary for this same-origin
browser connection. Health and the interface can be checked without a live LLM;
actual queries require the services and models described in the backend
documentation.

## Configuration and deployment

`frontend/.env.example` lists public configuration; Vite also reads process
environment variables. Restart the development server after changing settings.

| Setting | Purpose |
| --- | --- |
| `API_PROXY_TARGET` | Development/preview proxy target; defaults to `http://127.0.0.1:8000` |
| `VITE_API_BASE_URL` | Public build-time API base URL, empty by default for same-origin requests |

Example alternate local backend port:

```powershell
$env:API_PROXY_TARGET = "http://127.0.0.1:8010"
npm run dev
```

Do not put credentials in either URL. All `VITE_*` values are public browser
configuration, not a secret store. API responses and model output are rendered
as React text, never injected as HTML or interpreted as executable Markdown.

```powershell
npm run build
npm run preview
```

The production build is `frontend/dist/`. Vite preview supports the same proxy
for local build inspection; it is not a production server. For deployment, serve
the static build with an HTTPS reverse proxy forwarding `/health`, `/auth/*`,
and `/api/v1/*` to FastAPI on the same origin. Preserve the `Set-Cookie` and
`Cookie` headers and the `/` cookie path. Vite's development proxy does not exist
inside the static build.
An absolute, cross-origin `VITE_API_BASE_URL` requires an explicit allowed-origin
CORS policy at the backend/proxy; the current FastAPI application has none.
Set backend `RAG_ENV=production` and `RAG_TRUSTED_ORIGINS` to a comma-separated
list containing only the exact HTTPS frontend origins. Startup rejects missing
`RAG_ENV`, a missing production origin list, or HTTP production origins. This
enables Secure, HttpOnly, SameSite=Lax, Path=/ session cookies. Local HTTP
development requires the explicit `RAG_ENV=development` setting. Sessions expire
after eight hours; protected API 401 responses return the UI to login. The
configured document corpus remains shared among authenticated users; per-user
ownership, tenant isolation, and rate limiting are not implemented.

At application startup, `AuthProvider` attempts to restore the authenticated
session. A successful restoration displays the authenticated workspace. If
restoration fails for any reason—including an expired or invalid session, a
backend outage, a network error, or another transient error—the current frontend
treats the user as unauthenticated and displays the login screen. A temporary
backend or network failure can therefore look like a logged-out session; it does
not necessarily mean the server-side session expired or was revoked. A future
improvement could distinguish authentication failures such as HTTP 401 from
temporary service or network failures. That distinction is not implemented.

## Screens and supported behavior

- **Overview:** live backend liveness with refresh and 60-second polling;
  counts of completed requests and unique source document IDs encountered in this
  tab. These are session counts, not corpus totals. An online API does not imply
  Qdrant/model/Ollama readiness.
- **Documents:** PDF selection and drag-and-drop, actual byte-transfer progress,
  processing status, five-second inventory refresh, metadata, recovery controls,
  and explicit delete confirmation. Transfer completion is not indexing completion.
  Legacy CLI documents appear read-only. Leaving the screen aborts an active
  transfer, but the server may already have accepted it: refresh before retrying.
- **AI workspace:** independent questions using `/api/v1/query`, or retrieval-only
  requests using `/api/v1/retrieve`. Dense, BM25, and hybrid selection, optional
  cross-encoder reranking, and integer top-k 1-100 are supported.
- **Answers:** `answered`, `insufficient_evidence`, and `conflicting_evidence`
  remain distinct, visible states. Quotations, filenames, page ranges, and citation
  strings are taken from the API. Citations expand to show the returned quote.
- **Evidence:** select a response to inspect its returned passages, document and
  chunk IDs, section/subheading, chunk type, rank, retriever, and raw score.
  Scores are not confidence percentages and are not comparable across strategies.

History is retained in React memory while navigating the screens. It clears on
reload or tab closure; nothing is written to localStorage/sessionStorage. Each
question is sent independently without earlier conversation context. The UI has
no streaming or automatic retries. Requests time out after 180 seconds; health
checks after 5 seconds. “Stop waiting” aborts the browser request but does not
claim that the server cancels its computation. Retry restores the failed request
to the composer for editing and explicit resubmission.

Controls have labels, visible keyboard focus, and native expandable sections.
Navigation adapts to a compact top bar on mobile; evidence stacks below the chat.
Reduced-motion preferences disable the loading spinner animation.

## Document management and remaining gaps

The screen uses raw PDF uploads to `POST /api/v1/documents?filename=...`, inventory
polling at `GET /api/v1/documents`, document-specific DELETE after confirmation, and
`POST /api/v1/documents/recover` for pending cleanup. All requests use the configured
base URL/proxy. The server is authoritative for size, type, and filename validation.
The default upload limit is 20 MiB. Empty/non-PDF drops are rejected locally.

A 202 response means accepted, not indexed. States and errors come from the backend.
Deletion failures remain visible and retryable. Parsed content is removed along
with the owned PDF and vector points; metadata tombstones remain. Older answers in
tab history retain their evidence snapshots. See [API.md](API.md) for the integrity
contract and recovery behavior.

Authentication, including session restoration and expiry handling, is implemented.
Remaining gaps are PDF download/viewer links, service readiness beyond liveness,
server reranking-limit discovery, and a distributed job queue.
There is no automatic adoption/deletion of legacy CLI documents. Run one API worker
and do not run external collection writers concurrently. API-managed mutations
refresh retrieval caches automatically; offline CLI indexing requires a restart.

## Validation

From `frontend/`:

```powershell
npm run typecheck
npm run test
npm run build
npm run format:check
npm run test:browser
```

Vitest/React Testing Library tests mock HTTP responses and cover the client
contract, input validation, error mapping, grounding states, citations, settings,
history across navigation, upload/deletion workflows, and safe text rendering. Playwright
uses an already installed Microsoft Edge by default, with desktop 1440x1000 and
mobile 390x844 viewports. It exercises all screens, API request payloads, source
expansion, errors, and horizontal-overflow checks. It captures screenshots to its
ignored output directory. These are simulated mobile viewports, not physical-device
certification or a cross-browser compatibility claim.

If Edge is unavailable, use another already installed Playwright-supported channel:

```powershell
$env:PLAYWRIGHT_CHANNEL = "chrome"
npm run test:browser
```

No browser installation is performed by these commands. Synthetic browser results
verify functionality only; they are not real retrieval-quality or LLM benchmarks.
The package lock records the dependency versions used for reproducible installs.

## File layout

`src/App.tsx` owns navigation, transient history, settings, and request lifecycle.
`src/components.tsx` contains dashboard, citation, and evidence views.
`src/Documents.tsx` owns document management and `src/documents-api.ts` its typed
HTTP client with byte-transfer progress.
`src/api.ts` centralizes endpoints, runtime response validation, and sanitized
failure messages. `src/types.ts` mirrors the used API contracts. `src/styles.css`
defines the responsive visual system with Tailwind integration. Tests and fixtures
are isolated from the production entry point.
