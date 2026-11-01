import {
  ArrowUpRight,
  BookOpen,
  FileText,
  FolderOpen,
  Layers3,
  MessageSquare,
  ShieldCheck,
} from "lucide-react";
import type { Passage, QueryResponse, Turn } from "./types";

export type Screen = "Overview" | "Documents" | "AI workspace";
export type Health = "checking" | "online" | "offline";
export function HealthBadge({ health }: { health: Health }) {
  return (
    <span className={`health ${health}`} role="status">
      <span className="dot" />
      {health === "online"
        ? "Backend online"
        : health === "offline"
          ? "Backend unavailable"
          : "Checking backend"}
    </span>
  );
}
export function Dashboard({
  health,
  turns,
  navigate,
  refresh,
}: {
  health: Health;
  turns: Turn[];
  navigate: (screen: Screen) => void;
  refresh: () => void;
}) {
  const completed = turns.filter((t) => t.result);
  const sources = new Set(
    completed.flatMap((t) => t.result!.evidence.map((e) => e.doc_id)),
  );
  return (
    <div className="page-stack">
      <section className="welcome">
        <div>
          <div className="eyebrow">YOUR DOCUMENT INTELLIGENCE WORKSPACE</div>
          <h1>
            Find the answer.
            <br />
            <span>Keep the evidence.</span>
          </h1>
          <p>
            Explore your indexed documents with answers you can trace back to
            the source.
          </p>
          <button className="primary" onClick={() => navigate("AI workspace")}>
            Open AI workspace <ArrowUpRight size={17} />
          </button>
        </div>
        <div className="welcome-mark" aria-hidden="true">
          <Layers3 size={70} strokeWidth={1} />
          <span>EVIDENCE FIRST</span>
          <div className="mark-line" />
        </div>
      </section>
      <div className="stats-grid">
        <section className="stat-card">
          <div className="stat-label">
            API connection <ShieldCheck size={18} />
          </div>
          <HealthBadge health={health} />
          <p>Liveness only · model readiness is not reported</p>
          <button className="text-button" onClick={refresh}>
            Check connection
          </button>
        </section>
        <section className="stat-card">
          <div className="stat-label">
            Completed requests <MessageSquare size={18} />
          </div>
          <strong className="stat-number">{completed.length}</strong>
          <p>In this browser tab session</p>
        </section>
        <section className="stat-card">
          <div className="stat-label">
            Sources encountered <FileText size={18} />
          </div>
          <strong className="stat-number">{sources.size}</strong>
          <p>Unique document IDs in session results</p>
        </section>
      </div>
      <div className="overview-grid">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">A TRACEABLE WORKFLOW</div>
              <h2>From question to source</h2>
            </div>
            <BookOpen size={21} />
          </div>
          <ol className="workflow">
            <li>
              <span>01</span>
              <div>
                <h3>Choose your search</h3>
                <p>
                  Combine semantic and keyword retrieval, or explore either
                  approach on its own.
                </p>
              </div>
            </li>
            <li>
              <span>02</span>
              <div>
                <h3>Ask with confidence</h3>
                <p>
                  Answers use source excerpts. Missing or conflicting evidence
                  is called out explicitly.
                </p>
              </div>
            </li>
            <li>
              <span>03</span>
              <div>
                <h3>Inspect the evidence</h3>
                <p>
                  Review filenames, page ranges, supporting quotes, and the
                  retrieved passages.
                </p>
              </div>
            </li>
          </ol>
        </section>
        <section className="panel library-summary">
          <div className="icon-tile">
            <FolderOpen size={24} />
          </div>
          <h2>Your document library</h2>
          <p>
            Upload PDFs, follow processing status, and manage indexed documents
            in your library.
          </p>
          <span className="neutral-tag">
            Open Documents for the live inventory
          </span>
          <button className="secondary" onClick={() => navigate("Documents")}>
            Manage documents <ArrowUpRight size={16} />
          </button>
        </section>
      </div>
    </div>
  );
}
export { Documents } from "./Documents";
export function EvidencePanel({
  evidence,
  hasResult,
}: {
  evidence: Passage[];
  hasResult: boolean;
}) {
  return (
    <aside className="evidence-panel" aria-label="Retrieved evidence">
      <div className="panel-heading">
        <div>
          <div className="eyebrow">SOURCE EXPLORER</div>
          <h2>Retrieved evidence</h2>
        </div>
        <span className="count">{evidence.length}</span>
      </div>
      <p className="evidence-help">
        {hasResult
          ? "Passages returned for the selected response."
          : "Select a response to inspect its supporting passages."}
      </p>
      {!evidence.length && (
        <div className="evidence-empty">
          <BookOpen size={30} />
          <h3>{hasResult ? "No passages returned" : "Evidence lives here"}</h3>
          <p>
            {hasResult
              ? "Try a more specific question or adjust retrieval settings."
              : "Ask a question or run retrieval to explore actual source material."}
          </p>
        </div>
      )}
      {evidence.map((item, index) => (
        <details className="evidence-card" key={`${item.chunk_id}-${index}`}>
          <summary>
            <div className="source-row">
              <FileText size={17} />
              <strong>{item.source_file}</strong>
            </div>
            <span className="page-label">
              {pageLabel(item.page_start, item.page_end)} · Rank {item.rank}
            </span>
            <span className="score-label">
              {item.retriever} · raw score {item.score.toPrecision(4)}
            </span>
          </summary>
          <div className="passage-body">
            <p>{item.text}</p>
            <dl>
              <dt>Document ID</dt>
              <dd>{item.doc_id}</dd>
              <dt>Chunk ID</dt>
              <dd>{item.chunk_id}</dd>
              <dt>Type</dt>
              <dd>{item.chunk_type}</dd>
              {item.section_heading && (
                <>
                  <dt>Section</dt>
                  <dd>{item.section_heading}</dd>
                </>
              )}
              {item.subheading && (
                <>
                  <dt>Subheading</dt>
                  <dd>{item.subheading}</dd>
                </>
              )}
            </dl>
            <small>{item.citation}</small>
          </div>
        </details>
      ))}
      {!!evidence.length && (
        <p className="score-note">
          Raw scores are not confidence percentages and are not comparable
          across retrieval methods.
        </p>
      )}
    </aside>
  );
}
export function pageLabel(start: number, end: number) {
  return start === end ? `Page ${start}` : `Pages ${start}–${end}`;
}
export function AnswerCard({ result }: { result: QueryResponse }) {
  const label = {
    answered: "Evidence-grounded answer",
    insufficient_evidence: "Insufficient evidence",
    conflicting_evidence: "Conflicting evidence",
  }[result.status];
  return (
    <>
      <span className={`answer-status ${result.status}`}>{label}</span>
      <p className="answer-text">{result.answer}</p>
      {result.reason && <p className="reason">{result.reason}</p>}
      {!!result.citations.length && (
        <div className="citations">
          <div className="eyebrow">CITED SOURCES</div>
          {result.citations.map((citation, i) => (
            <details key={`${citation.evidence_id}-${i}`}>
              <summary>
                <FileText size={15} />
                <span>{citation.source_file}</span>
                <span>{pageLabel(citation.page_start, citation.page_end)}</span>
              </summary>
              <blockquote>{citation.quote}</blockquote>
              <small>{citation.citation}</small>
            </details>
          ))}
        </div>
      )}
    </>
  );
}
