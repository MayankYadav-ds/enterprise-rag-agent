"""Evidence-constrained answer generation. Imports never load or call a model."""

from rag_agent.generation.config import GenerationConfig
from rag_agent.generation.engine import AnswerGenerator, AnswerPipeline
from rag_agent.generation.models import (
    Answer,
    CitedExcerpt,
    GenerationError,
    InvalidResponseError,
    ProviderError,
)
from rag_agent.generation.provider import LLMProvider, OllamaProvider, build_provider

__all__ = [
    "Answer",
    "AnswerGenerator",
    "AnswerPipeline",
    "CitedExcerpt",
    "GenerationConfig",
    "GenerationError",
    "InvalidResponseError",
    "LLMProvider",
    "OllamaProvider",
    "ProviderError",
    "build_provider",
]
