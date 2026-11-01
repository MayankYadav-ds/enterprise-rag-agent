import { apiBaseURL, APIError } from "./api";

export interface DocumentRecord {
  doc_id: string;
  filename: string;
  status:
    | "queued"
    | "processing"
    | "indexing"
    | "ready"
    | "failed"
    | "cleanup_required"
    | "deleting"
    | "deleted"
    | "external";
  managed: boolean;
  owns_vectors: boolean;
  size_bytes: number | null;
  page_count: number | null;
  chunk_count: number;
  updated_at: string | null;
  error: string | null;
}
function valid(value: unknown): value is DocumentRecord {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.doc_id === "string" &&
    typeof v.filename === "string" &&
    typeof v.managed === "boolean" &&
    typeof v.owns_vectors === "boolean" &&
    typeof v.status === "string" &&
    [
      "queued",
      "processing",
      "indexing",
      "ready",
      "failed",
      "cleanup_required",
      "deleting",
      "deleted",
      "external",
    ].includes(v.status) &&
    typeof v.chunk_count === "number" &&
    Number.isFinite(v.chunk_count) &&
    (v.size_bytes === null || typeof v.size_bytes === "number") &&
    (v.page_count === null || typeof v.page_count === "number") &&
    (v.updated_at === null || typeof v.updated_at === "string") &&
    (v.error === null || typeof v.error === "string")
  );
}
function failure(status: number): APIError {
  const messages: Record<number, string> = {
    409: "This PDF is already registered or indexed. Refresh the inventory to inspect its status.",
    413: "The PDF exceeds the server upload size limit.",
    415: "The file is not a supported PDF.",
    422: "The filename or request is invalid. Use a plain .pdf filename without path characters.",
    404: "This managed document was not found. Refresh the inventory.",
    503: "Document services are unavailable or cleanup is incomplete. Refresh status and retry recovery if needed.",
  };
  return new APIError(
    status,
    messages[status] ||
      "The document operation failed. Refresh inventory before retrying.",
  );
}
async function call(
  path: string,
  method: string,
  signal?: AbortSignal,
): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(apiBaseURL + "/api/v1/documents" + path, {
      method,
      signal,
    });
  } catch {
    throw new APIError(
      0,
      "Cannot reach document services. Check the connection and refresh.",
    );
  }
  if (!response.ok) throw failure(response.status);
  if (response.status === 204) return null;
  try {
    return await response.json();
  } catch {
    throw new APIError(502, "Invalid document response.");
  }
}
export async function listDocuments(
  signal?: AbortSignal,
): Promise<DocumentRecord[]> {
  const data = await call("", "GET", signal);
  if (!Array.isArray(data) || !data.every(valid))
    throw new APIError(502, "Invalid document inventory response.");
  return data;
}
export async function deleteDocument(id: string): Promise<DocumentRecord> {
  const data = await call("/" + encodeURIComponent(id), "DELETE");
  if (!valid(data) || data.status !== "deleted")
    throw new APIError(
      502,
      "Deletion was not confirmed. Refresh status before retrying.",
    );
  return data;
}
export async function recoverDocuments(): Promise<void> {
  await call("/recover", "POST");
}
export function uploadPDF(
  file: File,
  onProgress: (percent: number) => void,
): { promise: Promise<DocumentRecord>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<DocumentRecord>((resolve, reject) => {
    xhr.open(
      "POST",
      apiBaseURL +
        "/api/v1/documents?filename=" +
        encodeURIComponent(file.name),
    );
    xhr.setRequestHeader("Content-Type", "application/pdf");
    xhr.timeout = 180000;
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable)
        onProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.onerror = () =>
      reject(
        new APIError(
          0,
          "Upload connection failed. Refresh inventory before retrying.",
        ),
      );
    xhr.ontimeout = xhr.onabort = () =>
      reject(
        new APIError(
          0,
          "Upload interrupted. The server may have accepted it; refresh inventory before retrying.",
        ),
      );
    xhr.onload = () => {
      if (xhr.status !== 202) {
        reject(failure(xhr.status));
        return;
      }
      try {
        const data: unknown = JSON.parse(xhr.responseText);
        if (!valid(data)) throw new Error();
        resolve(data);
      } catch {
        reject(
          new APIError(
            502,
            "Invalid upload response. Refresh inventory before retrying.",
          ),
        );
      }
    };
    xhr.send(file);
  });
  return { promise, abort: () => xhr.abort() };
}
