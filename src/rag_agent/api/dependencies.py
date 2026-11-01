"""Dependency seams for offline tests and alternative implementations."""

from collections.abc import Callable

from fastapi import Request

from rag_agent.api.schemas import RetrievalRequest
from rag_agent.generation import AnswerGenerator
from rag_agent.retrieval.models import RetrievedChunk

RetrievalService = Callable[[RetrievalRequest], list[RetrievedChunk]]
GeneratorFactory = Callable[[], AnswerGenerator]


def get_retriever(request: Request) -> RetrievalService:
    return request.app.state.runtime.retrieve


def get_generator_factory(request: Request) -> GeneratorFactory:
    return request.app.state.runtime.generator
