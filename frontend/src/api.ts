import type {
  Citation,
  Passage,
  QueryResponse,
  RetrievalRequest,
  RetrievalResponse,
} from "./types";

const base = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object";
const string = (value: unknown): value is string => typeof value === "string";
const number = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
const nullableString = (value: unknown) => value === null || string(value);
function citation(value: unknown): value is Citation {
  return (
    object(value) &&
    ["evidence_id", "chunk_id", "source_file", "quote", "citation"].every((k) =>
      string(value[k]),
    ) &&
    number(value.page_start) &&
    number(value.page_end)
  );
}
function passage(value: unknown): value is Passage {
  return (
    object(value) &&
    [
      "chunk_id",
      "doc_id",
      "source_file",
      "chunk_type",
      "text",
      "retriever",
      "citation",
    ].every((k) => string(value[k])) &&
    ["page_start", "page_end", "rank", "score"].every((k) =>
      number(value[k]),
    ) &&
    nullableString(value.section_heading) &&
    nullableString(value.subheading)
  );
}
export function isRetrieval(value: unknown): value is RetrievalResponse {
  return (
    object(value) &&
    string(value.question) &&
    ["dense", "bm25", "hybrid"].includes(String(value.strategy)) &&
    typeof value.reranked === "boolean" &&
    Array.isArray(value.evidence) &&
    value.evidence.every(passage)
  );
}
export function isQuery(value: unknown): value is QueryResponse {
  return (
    isRetrieval(value) &&
    object(value) &&
    ["answered", "insufficient_evidence", "conflicting_evidence"].includes(
      String(value.status),
    ) &&
    string(value.answer) &&
    nullableString(value.reason) &&
    Array.isArray(value.citations) &&
    value.citations.every(citation)
  );
}
export class APIError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
async function request(
  path: string,
  signal: AbortSignal,
  body?: RetrievalRequest,
): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(base + path, {
      signal,
      ...(body
        ? {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }
        : {}),
    });
  } catch (error) {
    if (signal.aborted)
      throw new APIError(
        0,
        "Request stopped or timed out. The server may still finish processing.",
      );
    throw new APIError(
      0,
      "Cannot reach the backend. Check the API connection and try again.",
    );
  }
  if (!response.ok) {
    const messages: Record<number, string> = {
      422: "The request was rejected. Check top-k and retrieval settings; reranking cannot exceed the server candidate limit.",
      502: "The generated response failed evidence validation. No answer was accepted.",
      503: "A required service or model is unavailable. Check Qdrant, the configured models, and the generation provider.",
    };
    // Never display arbitrary upstream HTML, exception text, or credentials.
    throw new APIError(
      response.status,
      messages[response.status] ||
        `The request failed (HTTP ${response.status}). Please try again.`,
    );
  }
  try {
    return await response.json();
  } catch {
    throw new APIError(502, "The backend returned an invalid response.");
  }
}
export async function health(signal: AbortSignal): Promise<void> {
  const data = await request("/health", signal);
  if (!object(data) || data.status !== "ok")
    throw new APIError(502, "Unexpected health response.");
}
export async function retrieve(
  body: RetrievalRequest,
  signal: AbortSignal,
): Promise<RetrievalResponse> {
  const data = await request("/api/v1/retrieve", signal, body);
  if (!isRetrieval(data))
    throw new APIError(
      502,
      "The backend returned an invalid evidence response.",
    );
  return data;
}
export async function query(
  body: RetrievalRequest,
  signal: AbortSignal,
): Promise<QueryResponse> {
  const data = await request("/api/v1/query", signal, body);
  if (!isQuery(data))
    throw new APIError(502, "The backend returned an invalid answer response.");
  return data;
}

export { base as apiBaseURL };
