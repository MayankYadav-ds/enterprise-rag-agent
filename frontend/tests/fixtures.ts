import type { QueryResponse } from "../src/types";
// Synthetic test data only; never loaded by the application.
export const answer: QueryResponse = {
  question: "What was revenue?",
  strategy: "hybrid",
  reranked: false,
  status: "answered",
  answer: "Revenue was $10 million. [report.pdf, pp. 7-8, Revenue]",
  reason: null,
  citations: [
    {
      evidence_id: "E1",
      chunk_id: "c1",
      source_file: "report.pdf",
      page_start: 7,
      page_end: 8,
      quote: "Revenue was $10 million.",
      citation: "[report.pdf, pp. 7-8, Revenue]",
    },
  ],
  evidence: [
    {
      chunk_id: "c1",
      doc_id: "d1",
      source_file: "report.pdf",
      page_start: 7,
      page_end: 8,
      section_heading: "Revenue",
      subheading: null,
      chunk_type: "text",
      text: "Revenue was $10 million.",
      score: 0.82,
      rank: 1,
      retriever: "hybrid",
      citation: "[report.pdf, pp. 7-8, Revenue]",
    },
  ],
};
