import { useCallback, useEffect, useRef, useState } from "react";
import { FileText, RefreshCw, Trash2, UploadCloud } from "lucide-react";
import {
  deleteDocument,
  listDocuments,
  recoverDocuments,
  uploadPDF,
  type DocumentRecord,
} from "./documents-api";

export function Documents() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [inventoryError, setInventoryError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [confirm, setConfirm] = useState<DocumentRecord | null>(null);
  const [dragging, setDragging] = useState(false);
  const abortUpload = useRef<(() => void) | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const result = await listDocuments(signal);
      if (!signal?.aborted) {
        setDocuments(result);
        setInventoryError("");
      }
    } catch (e) {
      if (!signal?.aborted)
        setInventoryError(e instanceof Error ? e.message : "Inventory failed.");
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    const interval = setInterval(() => void refresh(controller.signal), 5000);
    return () => {
      clearInterval(interval);
      controller.abort();
      abortUpload.current?.();
    };
  }, [refresh]);
  const upload = async (files: FileList | null) => {
    if (busy || !files?.length) return;
    setError("");
    setNotice("");
    if (files.length !== 1) {
      setError("Please upload one PDF at a time.");
      return;
    }
    const file = files[0];
    if (!file.name.toLowerCase().endsWith(".pdf") || file.size === 0) {
      setError("Choose a non-empty PDF file.");
      return;
    }
    setBusy(true);
    setProgress(0);
    const transfer = uploadPDF(file, setProgress);
    abortUpload.current = transfer.abort;
    try {
      await transfer.promise;
      setNotice("Upload accepted. Processing status is shown below.");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setBusy(false);
      setProgress(null);
      abortUpload.current = null;
      if (input.current) input.current.value = "";
    }
  };
  const remove = async () => {
    if (!confirm || busy) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await deleteDocument(confirm.doc_id);
      setConfirm(null);
      setNotice("Document and its vector points were deleted.");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Deletion failed.");
    } finally {
      setBusy(false);
    }
  };
  const recover = async () => {
    setBusy(true);
    setError("");
    try {
      await recoverDocuments();
      setNotice("Recovery completed. Failed uploads may now be retried.");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Recovery failed.");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="page-stack">
      <div className="page-intro">
        <div className="eyebrow">KNOWLEDGE LIBRARY</div>
        <h1>Documents</h1>
        <p>
          Upload a PDF, follow its processing status, and manage the evidence
          available to your workspace.
        </p>
      </div>
      <section
        className={`upload-panel ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void upload(e.dataTransfer.files);
        }}
        aria-label="PDF upload"
      >
        <div className="icon-tile">
          <UploadCloud size={30} />
        </div>
        <h2>Bring your documents into focus</h2>
        <p>
          Drag a PDF here or choose a file. The server validates its contents
          and size before processing. Default size limit: 20 MB.
        </p>
        <input
          ref={input}
          type="file"
          accept=".pdf,application/pdf"
          aria-label="Choose PDF"
          disabled={busy}
          onChange={(e) => void upload(e.target.files)}
        />
        {progress !== null && (
          <div className="upload-progress" role="status">
            <label htmlFor="upload-progress">
              Transferring PDF · {progress}%
            </label>
            <progress id="upload-progress" max="100" value={progress} />
            <p>Transfer progress is not indexing progress.</p>
          </div>
        )}
      </section>
      {inventoryError && (
        <div role="alert" className="error-message">
          {inventoryError}
        </div>
      )}
      {error && (
        <div role="alert" className="error-message">
          {error}
        </div>
      )}
      {notice && (
        <div role="status" className="success-notice">
          {notice}
        </div>
      )}
      <section className="panel">
        <div className="panel-heading">
          <h2>Document inventory</h2>
          <button
            className="secondary"
            onClick={() => void refresh()}
            disabled={busy}
          >
            <RefreshCw size={15} />
            Refresh inventory
          </button>
        </div>
        <p className="evidence-help">
          Statuses refresh every five seconds. Legacy CLI documents are
          read-only here.
        </p>
        {loading && <p role="status">Loading inventory…</p>}
        {!loading && !documents.length && !inventoryError && (
          <p>No indexed or managed documents found.</p>
        )}
        {documents.map((doc) => (
          <article className="document-row" key={doc.doc_id}>
            <div className="document-main">
              <div className="source-row">
                <FileText size={18} />
                <strong>{doc.filename}</strong>
                <span
                  className={`answer-status ${doc.status === "ready" ? "answered" : "insufficient_evidence"}`}
                >
                  {doc.status.replaceAll("_", " ")}
                </span>
              </div>
              <p>
                {doc.page_count === null
                  ? "Page count unavailable"
                  : `${doc.page_count} pages`}{" "}
                · {doc.chunk_count} chunks
                {doc.size_bytes !== null
                  ? ` · ${(doc.size_bytes / 1024).toFixed(1)} KB`
                  : ""}
              </p>
              <details>
                <summary>Document metadata</summary>
                <dl>
                  <dt>Document ID</dt>
                  <dd>{doc.doc_id}</dd>
                  <dt>Updated</dt>
                  <dd>{doc.updated_at ?? "Not recorded"}</dd>
                  <dt>Ownership</dt>
                  <dd>
                    {doc.managed ? "API-managed" : "External / read-only"}
                  </dd>
                </dl>
              </details>
              {doc.error && <p className="document-error">{doc.error}</p>}
            </div>
            {doc.managed && (
              <button
                className="secondary danger"
                aria-label={`Delete ${doc.filename}`}
                disabled={
                  busy ||
                  ["queued", "processing", "indexing"].includes(doc.status)
                }
                onClick={() => setConfirm(doc)}
              >
                <Trash2 size={15} />
                {doc.status === "deleting" ? "Retry deletion" : "Delete"}
              </button>
            )}
          </article>
        ))}
        {documents.some((doc) =>
          ["cleanup_required", "deleting"].includes(doc.status),
        ) && (
          <div className="recovery-note">
            <p>Retrieval is paused until pending cleanup is confirmed.</p>
            <button
              className="secondary"
              disabled={busy}
              onClick={() => void recover()}
            >
              Retry recovery
            </button>
          </div>
        )}
      </section>
      {confirm && (
        <section
          className="panel delete-confirmation"
          role="alertdialog"
          aria-labelledby="delete-title"
          aria-describedby="delete-description"
        >
          <h2 id="delete-title">Delete {confirm.filename}?</h2>
          <p id="delete-description">
            This removes this document’s vector points and uploaded PDF. Other
            documents are preserved. Parsed content is also removed; only a
            status tombstone remains.
          </p>
          <div>
            <button
              className="secondary"
              disabled={busy}
              onClick={() => setConfirm(null)}
            >
              Cancel deletion
            </button>
            <button
              className="primary danger"
              disabled={busy}
              onClick={() => void remove()}
            >
              Confirm deletion
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
