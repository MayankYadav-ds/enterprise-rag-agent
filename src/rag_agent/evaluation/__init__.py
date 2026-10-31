"""Offline RAG and retrieval evaluation.

This package provides retrieval-quality metrics (Recall@K, Precision@K, MRR)
against a reproducible gold set of question-to-relevant-chunk annotations.
It is deliberately dependency-light: the metrics operate on plain ranked hit
lists, so they can be run against any ``Retriever`` implementation.

The gold set lives in ``data/eval/gold_set.json`` and is generated from the
indexed documents; every annotation names a real chunk ID, so the evaluation
is grounded in the actual collection rather than synthetic expectations.
"""

from __future__ import annotations

from rag_agent.evaluation.gold_set import GoldSet, load_gold_set
from rag_agent.evaluation.metrics import (
    PrecisionAtK,
    RecallAtK,
    mean_reciprocal_rank,
)
from rag_agent.evaluation.runner import (
    EvaluationResult,
    QueryResult,
    render_report,
    run_evaluation,
    save_json_report,
    save_text_report,
)

__all__ = [
    "EvaluationResult",
    "GoldSet",
    "PrecisionAtK",
    "RecallAtK",
    "QueryResult",
    "load_gold_set",
    "mean_reciprocal_rank",
    "render_report",
    "run_evaluation",
    "save_json_report",
    "save_text_report",
]
