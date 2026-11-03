"""Thin synchronous endpoints; blocking work runs in FastAPI's worker pool.

All endpoints that expose or modify data are now protected using the
authentication dependencies defined in ``rag_agent.auth.dependencies``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from rag_agent.api.dependencies import (
    GeneratorFactory,
    RetrievalService,
    get_generator_factory,
    get_retriever,
)
from rag_agent.api.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from rag_agent.auth.dependencies import csrf_required, get_current_user
from rag_agent.retrieval.citation import to_dict

router = APIRouter()
errors = {code: {"model": ErrorResponse} for code in (422, 500, 502, 503)}


def validate_options(body: RetrievalRequest, request: Request) -> None:
    if body.rerank and body.top_k > request.app.state.runtime.settings.rerank_candidates:
        raise HTTPException(422, "top_k exceeds the configured reranking candidate count")


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Process liveness only; does not contact services or load models."""
    return HealthResponse()


@router.post(
    "/api/v1/retrieve",
    response_model=RetrievalResponse,
    responses=errors,
    dependencies=[Depends(get_current_user), Depends(csrf_required)],
)
def retrieve(
    body: RetrievalRequest,
    request: Request,
    retriever: Annotated[RetrievalService, Depends(get_retriever)],
) -> RetrievalResponse:
    validate_options(body, request)
    hits = retriever(body)
    return RetrievalResponse(
        question=body.question,
        strategy=body.strategy,
        reranked=body.rerank,
        evidence=[to_dict(hit) for hit in hits],
    )


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    responses=errors,
    dependencies=[Depends(get_current_user), Depends(csrf_required)],
)
def query(
    body: RetrievalRequest,
    request: Request,
    retriever: Annotated[RetrievalService, Depends(get_retriever)],
    generator_factory: Annotated[GeneratorFactory, Depends(get_generator_factory)],
) -> QueryResponse:
    validate_options(body, request)
    hits = retriever(body)
    answer = generator_factory().generate(body.question, hits)
    return QueryResponse(
        question=answer.question,
        strategy=body.strategy,
        reranked=body.rerank,
        evidence=[to_dict(hit) for hit in hits],
        status=answer.status,
        answer=answer.text,
        citations=[asdict(excerpt) for excerpt in answer.excerpts],
        reason=answer.reason,
    )
