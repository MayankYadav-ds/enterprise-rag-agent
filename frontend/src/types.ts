export type Strategy = "dense" | "bm25" | "hybrid";
export interface RetrievalRequest {
  question: string;
  strategy: Strategy;
  top_k: number;
  rerank: boolean;
}
export interface Passage {
  chunk_id: string;
  doc_id: string;
  source_file: string;
  page_start: number;
  page_end: number;
  section_heading: string | null;
  subheading: string | null;
  chunk_type: string;
  text: string;
  score: number;
  rank: number;
  retriever: string;
  citation: string;
}
export interface Citation {
  evidence_id: string;
  chunk_id: string;
  source_file: string;
  page_start: number;
  page_end: number;
  quote: string;
  citation: string;
}
export interface RetrievalResponse {
  question: string;
  strategy: Strategy;
  reranked: boolean;
  evidence: Passage[];
}
export type AnswerStatus =
  "answered" | "insufficient_evidence" | "conflicting_evidence";
export interface QueryResponse extends RetrievalResponse {
  status: AnswerStatus;
  answer: string;
  citations: Citation[];
  reason: string | null;
}
export interface Turn {
  id: string;
  request: RetrievalRequest;
  mode: "query" | "retrieve";
  result?: QueryResponse | RetrievalResponse;
  error?: string;
}
