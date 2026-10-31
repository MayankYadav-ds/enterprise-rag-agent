"""Reproducible gold set for retrieval evaluation.

Each entry is a question paired with the chunk IDs that a human annotator
judged relevant, based on the actual indexed documents. The annotations name
real chunk IDs from the live ``rag_chunks`` collection, so the evaluation is
grounded in the actual index rather than synthetic expectations.

The gold set is versioned by the SHA-256 of this file's content, so a change
to an annotation is a new version and the runner records which version was
used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

GOLD_SET_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "eval" / "gold_set.json"
)


@dataclass(frozen=True, slots=True)
class GoldQuery:
    """One evaluation query and its relevance annotations."""

    query: str
    relevant_chunk_ids: frozenset[str] = field(default_factory=frozenset)
    note: str = ""


@dataclass(frozen=True, slots=True)
class GoldSet:
    """A versioned collection of gold queries."""

    version: str
    queries: tuple[GoldQuery, ...]

    @property
    def size(self) -> int:
        return len(self.queries)

    def relevant_ids_for(self, query: str) -> frozenset[str]:
        for gold in self.queries:
            if gold.query == query:
                return gold.relevant_chunk_ids
        return frozenset()


def _file_version(path: Path = GOLD_SET_PATH) -> str:
    """SHA-256 of the gold set file, so the runner records the exact version."""
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_gold_set(path: Path = GOLD_SET_PATH) -> GoldSet:
    """Load the gold set from disk, deriving its version from the file hash."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Gold set not found at {path}. Generate it with "
            f"scripts/build_gold_set.py from the live Qdrant collection."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries = tuple(
        GoldQuery(
            query=item["query"],
            relevant_chunk_ids=frozenset(item["relevant_chunk_ids"]),
            note=item.get("note", ""),
        )
        for item in raw["queries"]
    )
    return GoldSet(version=_file_version(path), queries=queries)


def gold_set_to_dict(gold_set: GoldSet) -> dict:
    """Serialise a gold set for the evaluation report."""
    return {
        "version": gold_set.version,
        "queries": [
            {
                "query": q.query,
                "relevant_chunk_ids": sorted(q.relevant_chunk_ids),
                "note": q.note,
            }
            for q in gold_set.queries
        ],
    }
