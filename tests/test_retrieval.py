"""Tests for the retrieval package's data model and retriever protocol.

``RetrievedChunk`` is the retrieval layer's view of a hit: the fields a
citation needs plus the fields the answer generator needs. It is deliberately
frozen so a hit cannot be silently mutated between retrieval and citation
formatting. ``Retriever`` is the interface every retriever implements, so the
answer generator and citation layer are retriever-agnostic.
"""

from __future__ import annotations

import dataclasses

import pytest

from rag_agent.retrieval.models import RetrievedChunk
from rag_agent.retrieval.protocol import Retriever


def _hit(**overrides) -> RetrievedChunk:
    defaults = dict(
        chunk_id="c1",
        doc_id="d",
        source_file="nps-2013-10k.pdf",
        page_start=1,
        page_end=1,
        section_heading=None,
        subheading=None,
        chunk_type="text",
        text="t",
        score=1.0,
        rank=1,
        retriever="dense",
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


class TestRetrievedChunk:
    def test_is_frozen(self):
        hit = _hit()
        with pytest.raises(dataclasses.FrozenInstanceError):
            hit.rank = 2  # type: ignore[misc]

    def test_has_no_instance_dict(self):
        assert not hasattr(_hit(), "__dict__")

    def test_equality_is_value_based(self):
        # Equality compares every field, so two hits that differ only in their
        # chunk id are not equal — the id is part of the hit's identity.
        assert _hit() == _hit()
        assert _hit() != _hit(chunk_id="other")

    def test_inequality_on_text(self):
        assert _hit(text="a") != _hit(text="b")


class TestRetrieverProtocol:
    def test_protocol_exposes_retrieve(self):
        assert hasattr(Retriever, "retrieve")

    def test_retrieved_chunk_is_not_a_retriever(self):
        # The protocol is for retrievers, not for the result model.
        assert not isinstance(_hit(), Retriever)
