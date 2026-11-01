"""Typed evidence selections and validated, extractive answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["answered", "insufficient_evidence", "conflicting_evidence"]


class Selection(BaseModel):
    """A model may select text; it may not supply citation metadata."""

    model_config = ConfigDict(extra="forbid", strict=True)
    evidence_id: str = Field(min_length=1, max_length=20)
    quote: str = Field(min_length=1, max_length=8000)


class EvidenceResponse(BaseModel):
    """Wire contract shared by the prompt, provider, and validator."""

    model_config = ConfigDict(extra="forbid", strict=True)
    status: Status
    selections: list[Selection] = Field(max_length=12)


@dataclass(frozen=True, slots=True)
class CitedExcerpt:
    evidence_id: str
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    quote: str
    citation: str


@dataclass(frozen=True, slots=True)
class Answer:
    """Only validated excerpts and application-authored framing are rendered."""

    question: str
    status: Status
    text: str
    excerpts: tuple[CitedExcerpt, ...] = ()
    reason: str | None = None
    evidence_ids: tuple[str, ...] = ()


class GenerationError(RuntimeError):
    """Base for operational generation failures, distinct from abstention."""


class ProviderError(GenerationError):
    """The configured model service failed or returned an invalid envelope."""


class InvalidResponseError(GenerationError):
    """The model response failed the evidence contract; no answer is emitted."""
