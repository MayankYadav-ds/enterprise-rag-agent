"""Real embedder backed by sentence-transformers (BAAI/bge-small-en-v1.5)."""

from __future__ import annotations

import os

from rag_agent.embeddings.protocol import Embedder

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# bge-small-en-v1.5 accepts up to 512 WordPiece tokens including special tokens.
_MAX_MODEL_TOKENS = 512


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a Hugging Face sentence-transformers model.

    The ``sentence-transformers`` package is imported lazily so that the base
    package installs and imports without it. Constructing this embedder without
    the ``[embeddings]`` extra installed raises a clear error.

    Args:
        model_name: HF model id, overridable via the ``EMBEDDING_MODEL``
            environment variable.
        instruction: Prefix prepended to query texts. bge models are
            instruction-aware: documents are embedded raw while queries get
            this prefix so the model's query-task is consistent.
        batch_size: Number of texts embedded per model call.
        device: Torch device string; defaults to CPU.
        normalize_embeddings: L2-normalise every vector so cosine similarity
            equals the dot product in Qdrant.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        instruction: str = _DEFAULT_INSTRUCTION,
        batch_size: int = 32,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised in tests
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                'Install it with: python -m pip install -e ".[embeddings]"'
            ) from exc

        self._model_name = model_name or os.environ.get("EMBEDDING_MODEL", _DEFAULT_MODEL)
        self.instruction = instruction
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(self.model_name, device=device)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings = self._model.encode(
                batch,
                batch_size=self.batch_size,
                device=self.device,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=False,
                show_progress_bar=False,
            )
            vectors.extend([list(map(float, row)) for row in embeddings])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        prefixed = self.instruction + text if self.instruction else text
        embeddings = self._model.encode(
            [prefixed],
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return list(map(float, embeddings[0]))


def model_max_tokens() -> int:
    """Maximum WordPiece token length accepted by the bge embedding model."""
    return _MAX_MODEL_TOKENS
