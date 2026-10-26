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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rag_agent.evaluation.gold_set import GOLD_SET_PATH, GoldQuery, GoldSet, load_gold_set
from rag_agent.evaluation.metrics import PrecisionAtK, RecallAtK, mean_reciprocal_rank

if TYPE_CHECKING:
    from rag_agent.retrieval.dense import DenseRetriever
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
    """

    dataset_version: str
    num_queries: int
    ks: tuple[int, ...]
    retrieval_settings: dict[str, object]
    mean_recall: dict[int, float]
    mean_precision: dict[int, float]
    mean_mrr: float
    per_query: tuple[QueryResult, ...] = field(default_factory=tuple)


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


def run_evaluation(
    retriever: Retriever,
    gold_set: GoldSet | list[dict],
    *,
    ks: list[int] | None = None,
    top_k: int = 10,
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

    retrieval_settings: dict[str, object] = {"top_k": top_k, **dict(retriever_kwargs)}
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


def _build_retriever(
    embedder_dim: int,
    collection: str,
    qdrant_url: str | None,
    use_fake: bool,
) -> DenseRetriever:
    """Build a DenseRetriever with a real or fake embedder.

    Imports are deferred so the module remains importable without the
    ``[embeddings]`` extra installed (used by unit tests with a fake embedder).
    """
    from rag_agent.embeddings.fake import FakeEmbedder
    from rag_agent.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from rag_agent.embeddings.store import QdrantStore
    from rag_agent.retrieval.dense import DenseRetriever

    if use_fake:
        embedder = FakeEmbedder(dimension=embedder_dim)
    else:
        try:
            embedder = SentenceTransformerEmbedder()
        except RuntimeError as exc:
            raise SystemExit(f"Embedder not available: {exc}") from exc
    store = QdrantStore(
        embedder=embedder,
        collection_name=collection,
        url=qdrant_url or "http://localhost:6333",
    )
    return DenseRetriever(embedder, store)


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
    )
    logging.info(
        "Querying collection %r with top_k=%d, ks=%s",
        args.collection,
        args.top_k,
        ks,
    )

    result = run_evaluation(retriever, gold_set, ks=ks, top_k=args.top_k)
    save_json_report(result, args.output)
    logging.info("JSON report written to %s", args.output)
    if args.text_output is not None:
        save_text_report(result, args.text_output)
        logging.info("Text report written to %s", args.text_output)

    print(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
