"""Retrieval evaluation runner.

Runs a ``Retriever`` over a ``GoldSet`` and computes Recall@K, Precision@K and
MRR for each query, aggregating to per-K and mean scores. The runner is
dependency-light: it drives any object implementing the ``Retriever`` protocol
against a gold set of real chunk IDs, so it works with a live Qdrant collection
or with a mocked retriever.

The output is both a human-readable text summary and a reproducible JSON
report that records the dataset version, query count, retrieval settings and
per-query metrics, so a baseline and any later improvement are directly
comparable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rag_agent.evaluation.gold_set import GOLD_SET_PATH, GoldQuery, GoldSet, load_gold_set
from rag_agent.evaluation.metrics import PrecisionAtK, RecallAtK, mean_reciprocal_rank

if TYPE_CHECKING:
    from rag_agent.retrieval.protocol import Retriever


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Per-query evaluation result.

    Attributes:
        query: The query text.
        relevant_chunk_ids: Gold-relevant chunk IDs for this query.
        retrieved_chunk_ids: Chunk IDs returned by the retriever, ranked.
        recall_at_k: Recall@K for each evaluated K, keyed by K.
        precision_at_k: Precision@K for each evaluated K, keyed by K.
        mrr: Mean Reciprocal Rank for this query (0.0 if no relevant hit).
    """

    query: str
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregated evaluation result across all gold queries.

    Attributes:
        dataset_version: SHA-256 prefix of the gold set file used.
        num_queries: Number of queries in the gold set.
        ks: K values evaluated.
        retrieval_settings: Snapshot of retrieval settings (top_k, etc.).
        mean_recall: Mean Recall@K across queries for each K.
        mean_precision: Mean Precision@K across queries for each K.
        mean_mrr: Mean reciprocal rank averaged across all queries.
        per_query: One ``QueryResult`` per query, in gold-set order.
        reranked: Whether cross-encoder reranking was applied, or ``None``
            when historical results did not record this setting.
    """

    dataset_version: str
    num_queries: int
    ks: tuple[int, ...]
    retrieval_settings: dict[str, object]
    mean_recall: dict[int, float]
    mean_precision: dict[int, float]
    mean_mrr: float
    per_query: tuple[QueryResult, ...] = field(default_factory=tuple)
    reranked: bool | None = None


def _compute_retrieved_ids(hits: list, relevant: frozenset[str], ks: list[int]) -> list[str]:
    """Extract ranked chunk IDs from retrieval hits."""
    return [hit.chunk_id for hit in hits]


def _coerce_gold_set(gold_set: GoldSet | list[dict]) -> GoldSet:
    """Normalise a gold set passed as a ``GoldSet`` or a list of query dicts."""
    if isinstance(gold_set, GoldSet):
        return gold_set
    if isinstance(gold_set, list):
        queries = tuple(
            GoldQuery(
                query=item["query"],
                relevant_chunk_ids=frozenset(item["relevant_chunk_ids"]),
                note=item.get("note", ""),
            )
            for item in gold_set
        )
        return GoldSet(version="inline", queries=queries)
    raise TypeError(
        f"gold_set must be a GoldSet or a list of query dicts, got {type(gold_set).__name__}"
    )


def _retrieval_strategy(retriever: Retriever, strategy: str | None) -> str:
    """Identify built-in retrievers without guessing missing historical settings.

    Custom retrievers may supply their strategy explicitly. For built-in
    retrievers the actual implementation takes precedence over a conflicting
    label, which is rejected rather than written into an inaccurate report.
    """
    from rag_agent.retrieval.bm25 import BM25Retriever
    from rag_agent.retrieval.dense import DenseRetriever
    from rag_agent.retrieval.hybrid import HybridRetriever
    from rag_agent.retrieval.reranker import CrossEncoderReranker

    while isinstance(retriever, CrossEncoderReranker):
        retriever = retriever.base_retriever
    for retriever_type, actual in (
        (DenseRetriever, "dense"),
        (BM25Retriever, "bm25"),
        (HybridRetriever, "hybrid"),
    ):
        if isinstance(retriever, retriever_type):
            if strategy is not None and strategy != actual:
                raise ValueError(
                    f"strategy {strategy!r} does not match actual retriever {actual!r}"
                )
            return actual
    return strategy or "unknown"


def run_evaluation(
    retriever: Retriever,
    gold_set: GoldSet | list[dict],
    *,
    ks: list[int] | None = None,
    top_k: int = 10,
    strategy: str | None = None,
    rerank: bool = False,
    rerank_candidates: int = 20,
    rerank_model: str | None = None,
    **retriever_kwargs: object,
) -> EvaluationResult:
    """Run a retriever over a gold set and compute retrieval metrics.

    Args:
        retriever: Any object implementing the ``Retriever`` protocol.
        gold_set: The gold set of queries and relevant chunk IDs. May be a
            ``GoldSet`` instance or a list of query dicts (``{"query": ...,
            "relevant_chunk_ids": [...]}``).
        ks: K values at which to compute Recall@K and Precision@K. Defaults
            to ``[1, 3, 5, 10]``. Each K must be a positive int.
        top_k: How many hits to request from the retriever per query. Must be
            at least as large as the largest K in ``ks`` so every K is scored.
        strategy: Retrieval strategy label for custom retrievers. Built-in
            dense, BM25 and hybrid strategies are inferred and checked against
            any explicit label. Unidentified custom retrievers record unknown.
        rerank: If True, wrap ``retriever`` with ``CrossEncoderReranker`` so
            candidates are re-scored by a cross-encoder before metrics are
            computed. The reranker uses ``RerankerConfig.from_env()`` with
            ``top_k=top_k`` and ``candidate_count=rerank_candidates``.
        rerank_candidates: Number of candidates to fetch from the upstream
            retriever before reranking. Only used when ``rerank=True``.
        rerank_model: Explicit cross-encoder model name, taking precedence over
            ``RERANKER_MODEL`` and the default. Only used when ``rerank=True``.
        **retriever_kwargs: Extra filters forwarded to ``retriever.retrieve``
            (e.g. ``doc_id``, ``chunk_type``, ``min_score``).

    Returns:
        An ``EvaluationResult`` with per-query and mean metrics.

    Raises:
        ValueError: If ``ks`` is empty or contains non-positive values, or if
            ``top_k`` is smaller than the largest K in ``ks``.
    """
    gold_set = _coerce_gold_set(gold_set)
    if ks is None:
        ks = [1, 3, 5, 10]
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"ks must be a non-empty list of positive ints, got {ks}")
    max_k = max(ks)
    if top_k < max_k:
        raise ValueError(f"top_k ({top_k}) must be >= max(ks) ({max_k}) so every K is scored")

    from rag_agent.retrieval.reranker import (
        CrossEncoderReranker,
        RerankerConfig,
        build_reranked_retriever,
    )

    actual_strategy = _retrieval_strategy(retriever, strategy)
    if rerank:
        if isinstance(retriever, CrossEncoderReranker):
            raise ValueError("retriever is already reranked; pass rerank=False to evaluate it")
        rerank_config = RerankerConfig.from_env(
            top_k=top_k,
            candidate_count=rerank_candidates,
            model_name=rerank_model,
        )
        retriever = build_reranked_retriever(retriever, enable=True, config=rerank_config)
    rerank_config = retriever.config if isinstance(retriever, CrossEncoderReranker) else None
    actual_reranked = rerank_config is not None

    # Sort and de-duplicate K values for stable, readable reports.
    ks_sorted = sorted(set(ks))

    per_query: list[QueryResult] = []
    recall_sums: dict[int, float] = {k: 0.0 for k in ks_sorted}
    precision_sums: dict[int, float] = {k: float(k) for k in ks_sorted}  # zero-init
    precision_sums = {k: 0.0 for k in ks_sorted}
    mrr_sum = 0.0

    for gold_query in gold_set.queries:
        relevant = gold_query.relevant_chunk_ids
        hits = retriever.retrieve(
            gold_query.query,
            top_k=top_k,
            **retriever_kwargs,  # type: ignore[arg-type]
        )
        retrieved_ids = _compute_retrieved_ids(hits, relevant, ks_sorted)

        q_recall: dict[int, float] = {}
        q_precision: dict[int, float] = {}
        for k in ks_sorted:
            rk = RecallAtK(k=k)
            pk = PrecisionAtK(k=k)
            r = rk.compute(retrieved_ids, relevant)
            p = pk.compute(retrieved_ids, relevant)
            q_recall[k] = r
            q_precision[k] = p
            recall_sums[k] += r
            precision_sums[k] += p

        mrr = mean_reciprocal_rank(retrieved_ids, relevant)
        mrr_sum += mrr

        per_query.append(
            QueryResult(
                query=gold_query.query,
                relevant_chunk_ids=tuple(sorted(relevant)),
                retrieved_chunk_ids=tuple(retrieved_ids),
                recall_at_k=q_recall,
                precision_at_k=q_precision,
                mrr=mrr,
            )
        )

    n = len(gold_set.queries)
    if n == 0:
        mean_recall: dict[int, float] = {k: 0.0 for k in ks_sorted}
        mean_precision: dict[int, float] = {k: 0.0 for k in ks_sorted}
        mean_mrr = 0.0
    else:
        mean_recall = {k: recall_sums[k] / n for k in ks_sorted}
        mean_precision = {k: precision_sums[k] / n for k in ks_sorted}
        mean_mrr = mrr_sum / n

    retrieval_settings: dict[str, object] = {
        **retriever_kwargs,
        "strategy": actual_strategy,
        "reranked": actual_reranked,
        "rerank_model": rerank_config.model_name if rerank_config else None,
        "rerank_candidates": rerank_config.candidate_count if rerank_config else None,
        "top_k": top_k,
    }
    if rerank_config is not None:
        retrieval_settings.update(
            rerank_device=rerank_config.device,
            rerank_batch_size=rerank_config.batch_size,
            rerank_min_score=rerank_config.min_score,
        )
    # Keep only JSON-serialisable settings (filter out live retriever objects).
    safe_settings = {
        key: value
        for key, value in retrieval_settings.items()
        if isinstance(value, str | int | float | bool | type(None))
    }

    return EvaluationResult(
        dataset_version=gold_set.version,
        num_queries=n,
        ks=tuple(ks_sorted),
        retrieval_settings=safe_settings,
        mean_recall=mean_recall,
        mean_precision=mean_precision,
        mean_mrr=mean_mrr,
        per_query=tuple(per_query),
        reranked=actual_reranked,
    )


def result_to_dict(result: EvaluationResult) -> dict:
    """Serialise an ``EvaluationResult`` to a JSON-plain dict."""
    return {
        "dataset_version": result.dataset_version,
        "num_queries": result.num_queries,
        "ks": list(result.ks),
        "retrieval_settings": result.retrieval_settings,
        "mean_recall": dict(sorted(result.mean_recall.items())),
        "mean_precision": dict(sorted(result.mean_precision.items())),
        "mean_mrr": result.mean_mrr,
        "reranked": result.reranked,
        "per_query": [
            {
                "query": q.query,
                "relevant_chunk_ids": list(q.relevant_chunk_ids),
                "retrieved_chunk_ids": list(q.retrieved_chunk_ids),
                "recall_at_k": {str(k): v for k, v in sorted(q.recall_at_k.items())},
                "precision_at_k": {str(k): v for k, v in sorted(q.precision_at_k.items())},
                "mrr": q.mrr,
            }
            for q in result.per_query
        ],
    }


def render_report(result: EvaluationResult) -> str:
    """Render a human-readable evaluation report as text."""
    lines: list[str] = []
    lines.append(f"Evaluation — dataset version: {result.dataset_version}")
    lines.append(f"Queries: {result.num_queries}  |  K values: {list(result.ks)}")
    settings = result.retrieval_settings
    strategy = settings.get("strategy") or "unknown (not recorded)"
    reranked = result.reranked if result.reranked is not None else "unknown (not recorded)"
    lines.append(f"Strategy: {strategy}  |  reranked: {reranked}")
    lines.append(f"Final top-k: {settings.get('top_k', 'unknown (not recorded)')}")
    if result.reranked:
        lines.append(
            f"Cross-encoder: {settings.get('rerank_model') or 'unknown (not recorded)'}"
            f"  |  candidates: {settings.get('rerank_candidates', 'unknown (not recorded)')}"
        )
    lines.append("")
    lines.append("Mean metrics:")
    header = "  K"
    for k in result.ks:
        header += f" |   R@{k:<2} |   P@{k:<2}"
    lines.append(header + " |  MRR   ")
    row = "  --"
    for k in result.ks:
        r = result.mean_recall.get(k, 0.0)
        p = result.mean_precision.get(k, 0.0)
        row += f" | {r:>6.3f} | {p:>6.3f}"
    row += f" | {result.mean_mrr:>6.3f}"
    lines.append(row)
    lines.append("")
    lines.append("Per-query results:")
    for q in result.per_query:
        recall_vals = ", ".join(f"R@{k}={q.recall_at_k.get(k, 0.0):.2f}" for k in result.ks)
        lines.append(
            f"  Q: {q.query!r}"
            f"\n    retrieved={len(q.retrieved_chunk_ids)}  mrr={q.mrr:.3f}"
            f"\n    {recall_vals}"
        )
    return "\n".join(lines)


def save_json_report(result: EvaluationResult, path: Path) -> None:
    """Write the machine-readable JSON report to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result_to_dict(result), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def save_text_report(result: EvaluationResult, path: Path) -> None:
    """Write the human-readable text report to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(result) + "\n", encoding="utf-8")


# --- CLI entry point ----------------------------------------------
# ``python -m rag_agent.evaluation.runner`` runs this function. The CLI
# imports are deferred to function scope so the module-level import of the
# runner (used by tests and the package __init__) does not require the
# sentence-transformers extra to be installed.


def _parse_ks(value: str) -> list[int]:
    """Parse a comma-separated list of K values into sorted positive ints."""
    try:
        ks = sorted({int(k.strip()) for k in value.split(",") if k.strip()})
    except ValueError as exc:
        raise ValueError(f"--ks must be comma-separated integers, got {value!r}") from exc
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"--ks must contain only positive integers, got {value!r}")
    return ks


DEFAULT_STRATEGY = "dense"
STRATEGIES = ("dense", "bm25", "hybrid")


def _build_retriever(
    embedder_dim: int,
    collection: str,
    qdrant_url: str | None,
    use_fake: bool,
    strategy: str = DEFAULT_STRATEGY,
) -> Retriever:
    """Build the retriever for the requested evaluation strategy.

    Imports are deferred so the module remains importable without the
    ``[embeddings]`` extra installed (used by unit tests with a fake embedder).

    ``dense`` and ``hybrid`` need the embedder for scoring. ``bm25`` scores text
    only and never calls the embedder, but the store still needs an embedder to
    open the collection (the collection's vector dimension is checked on open).
    """
    from rag_agent.embeddings.fake import FakeEmbedder
    from rag_agent.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from rag_agent.embeddings.store import QdrantStore
    from rag_agent.retrieval.bm25 import BM25Retriever
    from rag_agent.retrieval.dense import DenseRetriever
    from rag_agent.retrieval.hybrid import HybridRetriever

    if use_fake or strategy == "bm25":
        embedder = FakeEmbedder(dimension=embedder_dim)
    else:
        try:
            embedder = SentenceTransformerEmbedder()
        except RuntimeError as exc:
            raise SystemExit(f"Embedder not available: {exc}") from exc
    store = QdrantStore(
        embedder=embedder,
        collection_name=collection,
        url=qdrant_url or os.environ.get("QDRANT_URL") or "http://localhost:6333",
        validate_dimension=strategy != "bm25",
    )
    if strategy == "dense":
        return DenseRetriever(embedder, store)
    if strategy == "bm25":
        return BM25Retriever(index=store.scroll_payloads())
    if strategy == "hybrid":
        # hybrid: fuse dense and BM25 rankings with RRF.
        return HybridRetriever(
            DenseRetriever(embedder, store),
            BM25Retriever(index=store.scroll_payloads()),
        )
    raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES!r}")


def main() -> int:
    """CLI: run evaluation and write JSON + optional text reports."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        prog="python -m rag_agent.evaluation.runner",
        description="Run retrieval evaluation against a gold set of verified chunk IDs.",
    )
    parser.add_argument(
        "--gold-set",
        type=Path,
        default=GOLD_SET_PATH,
        help=f"Path to gold_set.json (default: {GOLD_SET_PATH}).",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="1,3,5,10",
        help="Comma-separated K values for Recall@K and Precision@K (default: 1,3,5,10).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of hits to request per query from the retriever (default: 10).",
    )
    parser.add_argument(
        "--collection",
        default="rag_chunks",
        help="Qdrant collection name (default: rag_chunks).",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant URL. Use ':memory:' for an in-memory store (default: env "
        "QDRANT_URL, fallback http://localhost:6333).",
    )
    parser.add_argument(
        "--fake-embedder",
        action="store_true",
        help="Use a deterministic fake embedder instead of downloading the real model.",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=STRATEGIES,
        help="Retrieval strategy: dense (default), bm25, or hybrid (dense + BM25 RRF).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking on top of the selected strategy.",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=20,
        help="Number of candidates to fetch for reranking (default: 20).",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="Cross-encoder model name (default: RERANKER_MODEL environment variable, "
        "or cross-encoder/ms-marco-MiniLM-L-6-v2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/results.json"),
        help="Path to write the JSON evaluation report (default: data/eval/results.json).",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional path to write a human-readable text report.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    ks = _parse_ks(args.ks)
    if args.top_k < max(ks):
        logging.error(
            "--top-k (%d) must be >= max(ks) (%d) so every K is scored",
            args.top_k,
            max(ks),
        )
        return 1

    gold_set = load_gold_set(args.gold_set)
    if gold_set.size == 0:
        logging.error("Gold set at %s contains no queries.", args.gold_set)
        return 1
    logging.info(
        "Loaded %d queries from gold set version %s.",
        gold_set.size,
        gold_set.version,
    )

    retriever = _build_retriever(
        embedder_dim=32,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        use_fake=args.fake_embedder,
        strategy=args.strategy,
    )
    logging.info(
        "Querying collection %r with strategy=%s, top_k=%d, ks=%s, rerank=%s",
        args.collection,
        args.strategy,
        args.top_k,
        ks,
        args.rerank,
    )

    result = run_evaluation(
        retriever,
        gold_set,
        ks=ks,
        top_k=args.top_k,
        strategy=args.strategy,
        rerank=args.rerank,
        rerank_candidates=args.rerank_candidates,
        rerank_model=args.rerank_model,
    )
    save_json_report(result, args.output)
    logging.info("JSON report written to %s", args.output)
    if args.text_output is not None:
        save_text_report(result, args.text_output)
        logging.info("Text report written to %s", args.text_output)

    print(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
