"""Explicit public contracts for retrieval, answers, and sanitized failures."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Strategy = Literal["dense", "bm25", "hybrid"]


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)
    strategy: Strategy = "hybrid"
    top_k: int = Field(default=5, ge=1, le=100)
    rerank: bool = False
    doc_id: str | None = Field(default=None, min_length=1, max_length=256)
    chunk_type: Literal["text", "table"] | None = None
    min_score: float | None = Field(default=None, allow_inf_nan=False)


class Passage(BaseModel):
    chunk_id: str
    doc_id: str
    source_file: str
    page_start: int
    page_end: int
    section_heading: str | None
    subheading: str | None
    chunk_type: str
    text: str
    score: float = Field(allow_inf_nan=False)
    rank: int
    retriever: str
    citation: str


class Citation(BaseModel):
    evidence_id: str
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    quote: str
    citation: str


class RetrievalResponse(BaseModel):
    question: str
    strategy: Strategy
    reranked: bool
    evidence: list[Passage]


class QueryResponse(RetrievalResponse):
    status: Literal["answered", "insufficient_evidence", "conflicting_evidence"]
    answer: str
    citations: list[Citation]
    reason: str | None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str
