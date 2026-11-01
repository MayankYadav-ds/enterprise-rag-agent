import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  BookOpen,
  Check,
  ChevronRight,
  CircleHelp,
  FileText,
  LayoutDashboard,
  Layers3,
  LoaderCircle,
  MessageSquare,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Square,
} from "lucide-react";
import * as api from "./api";
import {
  AnswerCard,
  Dashboard,
  Documents,
  EvidencePanel,
  HealthBadge,
  type Health,
  type Screen,
} from "./components";
import type { RetrievalRequest, Turn } from "./types";

export default function App() {
  const [screen, setScreen] = useState<Screen>("Overview");
  const [health, setHealth] = useState<Health>("checking");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [selected, setSelected] = useState<string>();
  const [question, setQuestion] = useState("");
  const [settings, setSettings] = useState<Omit<RetrievalRequest, "question">>({
    strategy: "hybrid",
    top_k: 5,
    rerank: false,
  });
  const [mode, setMode] = useState<"query" | "retrieve">("query");
  const [pending, setPending] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const healthController = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const checkHealth = useCallback(async () => {
    healthController.current?.abort();
    const active = new AbortController();
    healthController.current = active;
    setHealth("checking");
    const timeout = setTimeout(() => active.abort(), 5000);
    try {
      await api.health(active.signal);
      if (healthController.current === active) setHealth("online");
    } catch {
      if (healthController.current === active) setHealth("offline");
    } finally {
      clearTimeout(timeout);
    }
  }, []);
  useEffect(() => {
    void checkHealth();
    const interval = setInterval(() => void checkHealth(), 60000);
    return () => {
      clearInterval(interval);
      healthController.current?.abort();
      controller.current?.abort();
    };
  }, [checkHealth]);
  useEffect(() => {
    if (screen === "AI workspace")
      endRef.current?.scrollIntoView({ block: "nearest" });
  }, [turns.length, pending, screen]);
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (controller.current || !question.trim() || question.trim().length > 4000)
      return;
    const request = { ...settings, question: question.trim() };
    const id = crypto.randomUUID();
    const active = new AbortController();
    controller.current = active;
    const timeout = setTimeout(() => active.abort(), 180000);
    setTurns((previous) => [...previous, { id, request, mode }]);
    setSelected(id);
    setPending(true);
    setQuestion("");
    try {
      const result =
        mode === "query"
          ? await api.query(request, active.signal)
          : await api.retrieve(request, active.signal);
      setTurns((previous) =>
        previous.map((turn) => (turn.id === id ? { ...turn, result } : turn)),
      );
    } catch (error) {
      const message =
        error instanceof api.APIError
          ? error.message
          : "An unexpected error occurred. Please try again.";
      setTurns((previous) =>
        previous.map((turn) =>
          turn.id === id ? { ...turn, error: message } : turn,
        ),
      );
    } finally {
      clearTimeout(timeout);
      controller.current = null;
      setPending(false);
    }
  };
  const activeTurn = turns.find((turn) => turn.id === selected);
  const navigation = [
    { name: "Overview", icon: LayoutDashboard },
    { name: "Documents", icon: FileText },
    { name: "AI workspace", icon: MessageSquare },
  ] as const;
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(event) => {
            event.preventDefault();
            setScreen("Overview");
          }}
        >
          <span className="brand-icon">
            <Layers3 size={24} />
          </span>
          <span>
            Enterprise RAG<small>DOCUMENT INTELLIGENCE</small>
          </span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.map(({ name, icon: Icon }) => (
            <button
              key={name}
              aria-current={screen === name ? "page" : undefined}
              className={screen === name ? "nav-item active" : "nav-item"}
              onClick={() => setScreen(name)}
            >
              <Icon size={18} />
              <span>{name}</span>
              {screen === name && (
                <ChevronRight size={15} className="nav-arrow" />
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <ShieldCheck size={20} />
          <strong>Grounded by design</strong>
          <p>
            Every answer starts with evidence. Every citation leads back to a
            source.
          </p>
        </div>
        <div className="sidebar-bottom">
          <span className="avatar">ER</span>
          <div>
            Local workspace<small>Session-only history</small>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={14} />
            <strong>{screen}</strong>
          </div>
          <HealthBadge health={health} />
        </header>
        <main
          id="main"
          className={
            screen === "AI workspace"
              ? "main-content chat-page"
              : "main-content"
          }
        >
          {screen === "Overview" && (
            <Dashboard
              health={health}
              turns={turns}
              navigate={setScreen}
              refresh={() => void checkHealth()}
            />
          )}
          {screen === "Documents" && <Documents />}
          {screen === "AI workspace" && (
            <>
              <div className="chat-heading">
                <div>
                  <div className="eyebrow">ASK. VERIFY. UNDERSTAND.</div>
                  <h1>AI workspace</h1>
                  <p>Answers anchored to your indexed documents.</p>
                </div>
                <button
                  className="secondary"
                  aria-expanded={settingsOpen}
                  aria-controls="retrieval-settings"
                  onClick={() => setSettingsOpen(!settingsOpen)}
                >
                  <SlidersHorizontal size={16} /> Retrieval settings
                </button>
              </div>
              {settingsOpen && (
                <section
                  className="settings-panel"
                  id="retrieval-settings"
                  aria-label="Retrieval settings"
                >
                  <label>
                    Retrieval strategy
                    <select
                      disabled={pending}
                      value={settings.strategy}
                      onChange={(event) =>
                        setSettings({
                          ...settings,
                          strategy: event.target
                            .value as RetrievalRequest["strategy"],
                        })
                      }
                    >
                      <option value="hybrid">
                        Hybrid · semantic + keyword
                      </option>
                      <option value="dense">Dense · semantic search</option>
                      <option value="bm25">BM25 · keyword search</option>
                    </select>
                  </label>
                  <label>
                    Top-k passages
                    <input
                      type="number"
                      min="1"
                      max="100"
                      step="1"
                      required
                      disabled={pending}
                      value={
                        Number.isFinite(settings.top_k) ? settings.top_k : ""
                      }
                      onChange={(event) =>
                        setSettings({
                          ...settings,
                          top_k: event.target.valueAsNumber,
                        })
                      }
                    />
                  </label>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      disabled={pending}
                      checked={settings.rerank}
                      onChange={(event) =>
                        setSettings({
                          ...settings,
                          rerank: event.target.checked,
                        })
                      }
                    />{" "}
                    Cross-encoder reranking
                  </label>
                  <p>
                    <CircleHelp size={14} /> Reranking top-k must fit the server
                    candidate limit (default 20; the API does not report its
                    configured value).
                  </p>
                </section>
              )}
              <div className="chat-grid">
                <section className="conversation" aria-label="Conversation">
                  <div className="conversation-toolbar">
                    <span>
                      <span className="dot" /> {settings.strategy.toUpperCase()}
                      {settings.rerank ? " + RERANKING" : ""}
                    </span>
                    <span>
                      Top-k{" "}
                      {Number.isFinite(settings.top_k) ? settings.top_k : "—"}
                    </span>
                  </div>
                  <div
                    className="messages"
                    aria-live="polite"
                    aria-busy={pending}
                  >
                    {!turns.length && (
                      <div className="chat-empty">
                        <div className="icon-tile">
                          <BookOpen size={30} />
                        </div>
                        <h2>What would you like to understand?</h2>
                        <p>
                          Ask a specific question about your documents.
                          <br />
                          We’ll bring the evidence into focus.
                        </p>
                        <div className="suggestions">
                          {[
                            "What risks are described in the reports?",
                            "What do the documents say about revenue?",
                          ].map((prompt) => (
                            <button
                              key={prompt}
                              onClick={() => setQuestion(prompt)}
                            >
                              {prompt}
                              <ArrowUp size={15} />
                            </button>
                          ))}
                        </div>
                        <small>
                          Each question is independent. Conversation history is
                          not sent to the model.
                        </small>
                      </div>
                    )}
                    {turns.map((turn) => (
                      <article
                        key={turn.id}
                        className={`turn ${selected === turn.id ? "selected" : ""}`}
                      >
                        <div className="question">
                          <span className="message-label">YOU</span>
                          <p>{turn.request.question}</p>
                        </div>
                        <div className="answer">
                          <div className="answer-heading">
                            <span className="assistant-icon">
                              <Layers3 size={17} />
                            </span>
                            <strong>Enterprise RAG</strong>
                            <small>
                              {turn.request.strategy} · top-k{" "}
                              {turn.request.top_k}
                              {turn.request.rerank ? " · reranked" : ""}
                            </small>
                          </div>
                          {turn.error ? (
                            <div role="alert" className="error-message">
                              <strong>Request not completed</strong>
                              <p>{turn.error}</p>
                              <button
                                className="text-button"
                                disabled={pending}
                                onClick={() => {
                                  setQuestion(turn.request.question);
                                  setSettings({
                                    strategy: turn.request.strategy,
                                    top_k: turn.request.top_k,
                                    rerank: turn.request.rerank,
                                  });
                                  setMode(turn.mode);
                                }}
                              >
                                Edit and try again
                              </button>
                            </div>
                          ) : turn.result ? (
                            <>
                              {"status" in turn.result ? (
                                <AnswerCard result={turn.result} />
                              ) : (
                                <>
                                  <span className="answer-status retrieved">
                                    Retrieval complete
                                  </span>
                                  <p>
                                    {turn.result.evidence.length} passages
                                    returned. Answer generation was not
                                    requested.
                                  </p>
                                </>
                              )}
                              <button
                                className="evidence-button"
                                aria-pressed={selected === turn.id}
                                onClick={() => setSelected(turn.id)}
                              >
                                <Search size={14} /> Inspect{" "}
                                {turn.result.evidence.length} passages{" "}
                                {selected === turn.id && <Check size={14} />}
                              </button>
                            </>
                          ) : (
                            <div className="loading">
                              <LoaderCircle className="spin" size={17} />
                              {turn.mode === "query"
                                ? "Retrieving evidence and preparing an answer…"
                                : "Searching indexed passages…"}
                            </div>
                          )}
                        </div>
                      </article>
                    ))}
                    <div ref={endRef} />
                  </div>
                  <form className="composer" onSubmit={submit}>
                    <label className="sr-only" htmlFor="question">
                      Ask a question
                    </label>
                    <textarea
                      id="question"
                      placeholder="Ask a question about your documents…"
                      maxLength={4000}
                      value={question}
                      onChange={(event) => setQuestion(event.target.value)}
                      rows={3}
                      disabled={pending}
                    />
                    <div className="composer-bottom">
                      <label>
                        <span className="sr-only">Response mode</span>
                        <select
                          aria-label="Response mode"
                          value={mode}
                          disabled={pending}
                          onChange={(event) =>
                            setMode(event.target.value as "query" | "retrieve")
                          }
                        >
                          <option value="query">Grounded answer</option>
                          <option value="retrieve">
                            Retrieve evidence only
                          </option>
                        </select>
                      </label>
                      {pending ? (
                        <button
                          className="secondary"
                          type="button"
                          onClick={() => controller.current?.abort()}
                        >
                          <Square size={13} /> Stop waiting
                        </button>
                      ) : (
                        <button
                          className="primary send"
                          type="submit"
                          disabled={
                            !question.trim() ||
                            !Number.isInteger(settings.top_k) ||
                            settings.top_k < 1 ||
                            settings.top_k > 100
                          }
                        >
                          Send <ArrowUp size={17} />
                        </button>
                      )}
                    </div>
                    <p className="composer-note">
                      Verify important answers against the cited sources.
                      History clears on reload.
                    </p>
                  </form>
                </section>
                <EvidencePanel
                  evidence={activeTurn?.result?.evidence || []}
                  hasResult={!!activeTurn?.result}
                />
              </div>
            </>
          )}
        </main>
        <footer className="footer">
          Enterprise RAG <span>Evidence-grounded document intelligence</span>
        </footer>
      </div>
    </div>
  );
}
