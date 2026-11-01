"""Tests for citation formatting.

A citation is the shortest traceable reference a reader needs to check a claim:
the source file, the page range, and the section heading. It is formatted from
a ``RetrievedChunk`` so the same function serves the CLI, the future API and
the answer generator — there is one citation shape, not three.
"""

from __future__ import annotations

from rag_agent.retrieval.citation import format_citation, to_dict
from rag_agent.retrieval.models import RetrievedChunk


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


class TestFormatCitation:
    def test_single_page(self):
        assert (
            format_citation(_hit(source_file="nps-2013-10k.pdf", page_start=22, page_end=22))
            == "[nps-2013-10k.pdf, p. 22]"
        )

    def test_page_range(self):
        assert format_citation(_hit(page_start=22, page_end=23)) == "[nps-2013-10k.pdf, pp. 22-23]"

    def test_long_range(self):
        assert format_citation(_hit(page_start=5, page_end=100)) == "[nps-2013-10k.pdf, pp. 5-100]"

    def test_heading_omitted_when_none(self):
        assert format_citation(_hit(section_heading=None)) == "[nps-2013-10k.pdf, p. 1]"

    def test_heading_included_when_set(self):
        assert (
            format_citation(_hit(section_heading="ITEM 1. RISK FACTORS"))
            == "[nps-2013-10k.pdf, p. 1, ITEM 1. RISK FACTORS]"
        )

    def test_unicode_apostrophe_preserved(self):
        # The citation is a literal rendering of the chunk's metadata, not a
        # sanitised copy, so a possessive heading keeps its apostrophe.
        assert (
            format_citation(_hit(section_heading="MANAGEMENT'S DISCUSSION"))
            == "[nps-2013-10k.pdf, p. 1, MANAGEMENT'S DISCUSSION]"
        )

    def test_long_heading_preserved_in_full(self):
        heading = "ITEM 1A. ADDITIONAL FACTORS THAT MAY AFFECT OUR BUSINESS"
        assert (
            format_citation(_hit(section_heading=heading)) == f"[nps-2013-10k.pdf, p. 1, {heading}]"
        )

    def test_malformed_range_collapses_to_start(self):
        # A chunk whose page_end precedes page_start is malformed; the range
        # collapses to the start page rather than producing a backwards range.
        assert format_citation(_hit(page_start=10, page_end=5)) == "[nps-2013-10k.pdf, p. 10]"

    def test_bare_file_name_only(self):
        # Citations use the bare file name, never the indexed path, so a
        # renamed source file is reflected without re-hydrating the chunk.
        citation = format_citation(_hit(source_file="nps-pharmaceuticals-2013-10k.pdf"))
        assert citation == "[nps-pharmaceuticals-2013-10k.pdf, p. 1]"
        assert "/" not in citation and "\\" not in citation

    def test_deterministic(self):
        hit = _hit(page_start=7, page_end=9, section_heading="RISK FACTORS")
        assert format_citation(hit) == format_citation(hit)


class TestToDict:
    def test_keys(self):
        d = to_dict(_hit())
        assert set(d) == {
            "chunk_id",
            "doc_id",
            "source_file",
            "page_start",
            "page_end",
            "section_heading",
            "subheading",
            "chunk_type",
            "text",
            "score",
            "rank",
            "retriever",
            "citation",
        }

    def test_citation_derived_from_fields(self):
        hit = _hit(page_start=3, page_end=3, section_heading="ITEM 1. RISK FACTORS")
        assert to_dict(hit)["citation"] == "[nps-2013-10k.pdf, p. 3, ITEM 1. RISK FACTORS]"

    def test_score_and_rank_preserved(self):
        d = to_dict(_hit(score=0.75, rank=4))
        assert d["score"] == 0.75
        assert d["rank"] == 4

    def test_retriever_name_preserved(self):
        assert to_dict(_hit(retriever="dense"))["retriever"] == "dense"

    def test_subheading_preserved(self):
        assert (
            to_dict(_hit(subheading="Management Discussion"))["subheading"]
            == "Management Discussion"
        )
