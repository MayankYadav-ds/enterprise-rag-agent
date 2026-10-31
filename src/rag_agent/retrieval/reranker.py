"""Cross-encoder reranker for retrieval results.

The reranker wraps a cross-encoder model (e.g. BAAI/bge-reranker-base or
cross-encoder/ms-marco-MiniLM-L-6-v2) and re-scores retrieved candidate chunks.
It is an optional post-processing step that sits on top of any retriever
implementing the ``Retriever`` protocol.

Key design points:
- The reranker is **lazy-loaded** on first use so ordinary searches never
  trigger a model download unless ``--rerank`` is explicitly requested.
- Unit tests use a ``FakeCrossEncoder`` that produces deterministic scores
  without any model download or torch dependency.
- The reranker preserves all citation metadata (chunk_id, doc_id, physical page
  numbers, section information, subheading, chunk_type, text) from the input hits.
- Score semantics: cross-encoder outputs are raw logits (unbounded) that are
  higher for more relevant pairs. They are **not** probabilities and must not
  be compared to dense cosine similarity or BM25 scores.
- ``min_score`` on the reranker threshold applies to the cross-encoder logit,
  not to the upstream retriever's score.
- NaN and infinite model scores are rejected with ``ValueError`` before
  filtering or ranking; invalid model output must not become a retrieval hit.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

from rag_agent.retrieval.models import RetrievedChunk
from rag_agent.retrieval.protocol import Retriever

logger = logging.getLogger(__name__)

# Default cross-encoder models to consider. These are candidates for
# benchmarking, not mandatory dependencies. The default is the lighter model.
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_ALTERNATIVE_MODEL = "BAAI/bge-reranker-base"

# bge-reranker-base accepts up to 512 tokens; ms-marco-MiniLM-L-6-v2 accepts
# up to 512 tokens. The reranker truncates longer pairs.
_MAX_MODEL_TOKENS = 512


class CrossEncoderUnavailable(RuntimeError):
    """Raised when a cross-encoder model cannot be loaded."""

    pass


class FakeCrossEncoder:
    """Deterministic fake cross-encoder for testing.

    Returns scores that are inversely proportional to the length of the query
    plus chunk text, so shorter combined texts score higher (simulating better
    relevance). This is purely for deterministic ordering in tests and does not
    reflect real semantic relevance.
    """

    def __init__(self, *, seed: int = 42) -> None:
        self._seed = seed
        self._counter = 0

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        """Return deterministic scores for query-chunk pairs.

        Score = 1000 / (len(query) + len(chunk) + 1) + small seed-based noise.
        """
        scores: list[float] = []
        for query, chunk in pairs:
            self._counter += 1
            base = 1000.0 / (len(query) + len(chunk) + 1)
            # No noise added to keep scores purely based on length
            scores.append(base)
        return scores


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """Configuration for the cross‑encoder reranker.

    Attributes:
        model_name: HF model id (e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"
            or "BAAI/bge-reranker-base"). Overridable via the
            ``RERANKER_MODEL`` environment variable. ``None`` or a blank name
            resolves to the environment value, then the default, when the
            configuration is created. Resolving a name does not load a model.
        top_k: Final number of hits to return after reranking. Must be >= 1.
        candidate_count: Number of candidates to fetch from the upstream
            retriever before reranking. Must be >= top_k.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        batch_size: Number of query‑chunk pairs per model call.
        min_score: Optional threshold on the cross‑encoder logit.
    """

    model_name: str | None = None
    top_k: int = 5
    candidate_count: int = 20
    device: str = "cpu"
    batch_size: int = 32
    min_score: float | None = None

    def __post_init__(self) -> None:
        # Every config carries the actual model name, while model loading stays lazy.
        for name in (self.model_name, os.environ.get("RERANKER_MODEL"), _DEFAULT_MODEL):
            if name is None:
                continue
            if not isinstance(name, str):
                raise ValueError(f"model_name must be a string or None, got {name!r}")
            if name.strip():
                object.__setattr__(self, "model_name", name.strip())
                break

        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if not isinstance(self.candidate_count, int) or self.candidate_count < self.top_k:
            raise ValueError(
                f"candidate_count ({self.candidate_count}) must be >= top_k ({self.top_k})"
            )
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    @classmethod
    def from_env(cls, **overrides: Any) -> RerankerConfig:
        """Create config from environment variables with optional overrides.

        Environment variables:
        - ``RERANKER_MODEL``: model name (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
        - ``RERANKER_TOP_K``: final result count (default: 5)
        - ``RERANKER_CANDIDATE_COUNT``: candidate pool size (default: 20)
        - ``RERANKER_DEVICE``: torch device (default: cpu)
        - ``RERANKER_BATCH_SIZE``: batch size (default: 32)
        - ``RERANKER_MIN_SCORE``: optional score threshold (default: none)
        """
        values: dict[str, Any] = {
            "model_name": os.environ.get("RERANKER_MODEL"),
            "top_k": os.environ.get("RERANKER_TOP_K", "5"),
            "candidate_count": os.environ.get("RERANKER_CANDIDATE_COUNT", "20"),
            "device": os.environ.get("RERANKER_DEVICE", "cpu"),
            "batch_size": os.environ.get("RERANKER_BATCH_SIZE", "32"),
            "min_score": os.environ.get("RERANKER_MIN_SCORE"),
        }
        values.update(overrides)
        # Parse only environment-derived values. Invalid overridden environment
        # values must not fail before explicit values reach normal validation.
        for key in ("top_k", "candidate_count", "batch_size"):
            if key not in overrides:
                values[key] = int(values[key])
        if "min_score" not in overrides:
            values["min_score"] = float(values["min_score"]) if values["min_score"] else None
        return cls(**values)


class CrossEncoderReranker:
    """Rerank retrieved candidates using a cross-encoder model.

    Usage:
        retriever = HybridRetriever(dense, bm25)
        reranker = CrossEncoderReranker(retriever, config=RerankerConfig(...))
        hits = reranker.retrieve("query", top_k=5)

    The reranker is a **pass-through retriever**: it implements the same
    ``Retriever`` protocol, delegates candidate retrieval to the wrapped
    retriever, then re-scores and re-ranks the candidates.

    The ``min_score`` parameter in ``retrieve()`` is forwarded to the upstream
    retriever as a pre-filter. The reranker's own ``config.min_score`` (if set)
    is applied as a post-filter on the cross-encoder logits. Both can be used
    together.
    """

    def __init__(
        self,
        retriever: Retriever,
        config: RerankerConfig | None = None,
        *,
        cross_encoder: Any | None = None,  # for test injection
    ) -> None:
        self._retriever = retriever
        self._config = config or RerankerConfig.from_env()
        self._cross_encoder = cross_encoder  # injected for tests; None = lazy load
        self._model_loaded = False

    @property
    def config(self) -> RerankerConfig:
        """Return the resolved configuration without loading the model."""
        return self._config

    @property
    def base_retriever(self) -> Retriever:
        """Return the upstream retriever for evaluation metadata."""
        return self._retriever

    # ---------------------------------------------------------------------
    # Model loading
    # ---------------------------------------------------------------------
    def _load_model(self) -> Any:
        """Lazy-load the cross-encoder model.

        Returns the model instance (sentence_transformers.CrossEncoder).
        Raises ``CrossEncoderUnavailable`` if the model cannot be loaded.
        """
        if self._cross_encoder is not None:
            # Test-injected fake or pre-loaded model
            return self._cross_encoder

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise CrossEncoderUnavailable(
                "sentence-transformers is required for CrossEncoderReranker. "
                'Install it with: python -m pip install -e ".[embeddings]"'
            ) from exc

        try:
            model = CrossEncoder(
                self._config.model_name,
                device=self._config.device,
                max_length=_MAX_MODEL_TOKENS,
            )
            self._cross_encoder = model
            self._model_loaded = True
            logger.info(
                "Loaded cross-encoder model %r on %s", self._config.model_name, self._config.device
            )
            return model
        except Exception as exc:  # noqa: BLE001
            raise CrossEncoderUnavailable(
                f"Failed to load cross-encoder model {self._config.model_name!r}: {exc}"
            ) from exc

    # ---------------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        /,
        *,
        top_k: int | None = None,
        doc_id: str | None = None,
        chunk_type: str | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and rerank candidates.

        Returns:
            Top ``top_k`` chunks reranked by cross-encoder score, descending.
            Each hit's ``score`` is the cross-encoder logit; ``retriever`` is set
            to ``"reranked"``; ``rank`` is 1-based in the reranked list.

        Raises:
            ValueError: If any model score is NaN or infinite, or a model batch
                returns a different number of scores than input pairs. No
                partial ranking is returned, even if an invalid score would
                otherwise be filtered out or fall outside the final top-k.
        """
        effective_top_k = top_k if top_k is not None else self._config.top_k
        effective_candidate_count = self._config.candidate_count

        # Fetch candidates from the upstream retriever (honouring its filters and min_score)
        candidates = self._retriever.retrieve(
            query,
            top_k=effective_candidate_count,
            doc_id=doc_id,
            chunk_type=chunk_type,
            min_score=min_score,
        )

        if not candidates:
            return []

        # Load the cross-encoder model (lazy)
        model = self._load_model()

        # Build query‑chunk pairs for scoring
        pairs = [(query, hit.text or "") for hit in candidates]

        # Score in batches
        scores: list[float] = []
        for start in range(0, len(pairs), self._config.batch_size):
            batch = pairs[start : start + self._config.batch_size]
            batch_scores = [float(score) for score in model.predict(batch)]
            if len(batch_scores) != len(batch):
                raise ValueError(
                    f"Cross-encoder returned {len(batch_scores)} scores for {len(batch)} pairs"
                )
            for hit, score in zip(
                candidates[start : start + len(batch)], batch_scores, strict=True
            ):
                if not math.isfinite(score):
                    raise ValueError(
                        f"Cross-encoder returned non-finite score {score!r} "
                        f"for chunk {hit.chunk_id!r}"
                    )
            scores.extend(batch_scores)

        # Attach scores and apply reranker's min_score threshold
        scored_hits: list[tuple[RetrievedChunk, float]] = []
        for hit, score in zip(candidates, scores, strict=True):
            if self._config.min_score is not None and score < self._config.min_score:
                continue
            scored_hits.append((hit, score))

        if not scored_hits:
            return []

        # Sort by cross‑encoder score descending. Tie‑break by original rank
        # (the upstream retriever's ranking) so the output is deterministic.
        scored_hits.sort(key=lambda pair: (-pair[1], pair[0].rank))

        # Build final results with reranked scores and new ranks
        results: list[RetrievedChunk] = []
        for rank, (hit, score) in enumerate(scored_hits[:effective_top_k], start=1):
            results.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    source_file=hit.source_file,
                    page_start=hit.page_start,
                    page_end=hit.page_end,
                    section_heading=hit.section_heading,
                    subheading=hit.subheading,
                    chunk_type=hit.chunk_type,
                    text=hit.text,
                    score=score,
                    rank=rank,
                    retriever="reranked",
                )
            )
        return results


def build_reranked_retriever(
    base_retriever: Retriever,
    *,
    enable: bool = False,
    config: RerankerConfig | None = None,
    cross_encoder: Any | None = None,
) -> Retriever:
    """Wrap a base retriever with optional cross‑encoder reranking.

    Args:
        base_retriever: The upstream retriever (dense, BM25, or hybrid).
        enable: If False, returns the base retriever unchanged. If True,
            wraps it with CrossEncoderReranker.
        config: Reranker configuration. If None, uses ``RerankerConfig.from_env()``.
        cross_encoder: Optional pre‑loaded or fake cross‑encoder for testing.

    Returns:
        A ``Retriever`` that either passes through to ``base_retriever`` or
        reranks its candidates with a cross‑encoder.
    """
    if not enable:
        return base_retriever
    return CrossEncoderReranker(base_retriever, config=config, cross_encoder=cross_encoder)
