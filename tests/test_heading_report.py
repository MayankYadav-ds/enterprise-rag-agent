"""Tests for the offline heading-coverage report.

``heading_report.py`` has two modes: ``--collection`` (live Qdrant) and
``--from-pdf DIR`` (in-process ingest + chunk, no embedding). The collection
mode needs a running store, so only the offline mode and its helpers are
tested here.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

# scripts/ is not an importable package and the repo root is not on sys.path
# in CI, so add it explicitly before importing the report helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.heading_report import _chunk_payloads_from_pdf, report  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_report.pdf"


@pytest.fixture(autouse=True)
def isolate_report_output(tmp_path, monkeypatch):
    """The report ingests into data/processed relative to the working directory."""
    monkeypatch.chdir(tmp_path)


def test_chunk_payloads_from_pdf_shapes_qdrant_payloads() -> None:
    """The offline mode returns point-shaped payloads without embedding."""
    payloads = _chunk_payloads_from_pdf(FIXTURE_PATH)

    assert len(payloads) >= 1
    for payload in payloads:
        assert payload["source_file"] == "sample_report.pdf"
        assert "\\" not in payload["source_file"]
        assert payload["chunk_id"]
        assert payload["doc_id"]
        assert isinstance(payload["page_start"], int)
        assert isinstance(payload["page_end"], int)
        assert "text" in payload and "section_heading" in payload


def test_chunk_payloads_from_pdf_attaches_section_headings() -> None:
    """The fixture's two headings reach every chunk on their pages."""
    payloads = _chunk_payloads_from_pdf(FIXTURE_PATH)

    assert all(payload["section_heading"] for payload in payloads)
    assert {payload["section_heading"] for payload in payloads} == {
        "Financial Overview",
        "Operational Notes",
    }


def test_report_counts_chunks_with_and_without_headings() -> None:
    """report() separates headed chunks from unheaded ones and counts docs."""
    payloads = [
        {"section_heading": "A", "page_start": 1, "doc_id": "d1", "source_file": "a.pdf"},
        {"section_heading": None, "page_start": 2, "doc_id": "d1", "source_file": "a.pdf"},
        {"section_heading": "B", "page_start": 3, "doc_id": "d2", "source_file": "b.pdf"},
    ]

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        report(payloads, top_k=10, source_label="in-process")

    output = buffer.getvalue()
    assert output.count("\n===") == 2  # two documents printed
    assert "chunks=2 with_heading=1 without_heading=1" in output
    assert "chunks=1 with_heading=1 without_heading=0" in output
    assert "first=p1" in output
    assert "last=p1" in output
    assert "first=p3" in output
    assert "last=p3" in output


def test_main_from_pdf_prints_per_document_coverage(tmp_path: Path) -> None:
    """The --from-pdf CLI path prints chunk counts for each document."""
    from scripts.heading_report import main

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    import shutil

    shutil.copy(FIXTURE_PATH, raw_dir / "sample_report.pdf")

    argv = ["--from-pdf", str(raw_dir), "--top-k", "5"]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), pytest.MonkeyPatch().context() as monkey:
        monkey.setattr("sys.argv", ["heading_report.py", *argv])
        exit_code = main()

    assert exit_code == 0
    output = buffer.getvalue()
    assert "sample_report.pdf" in output
    assert "chunks=" in output
    assert "with_heading=" in output


def test_main_from_pdf_errors_on_empty_directory(tmp_path: Path) -> None:
    """An empty --from-pdf directory is a usage error, not a silent report."""
    from scripts.heading_report import main

    with pytest.MonkeyPatch().context() as monkey:
        monkey.setattr("sys.argv", ["heading_report.py", "--from-pdf", str(tmp_path)])
        exit_code = main()

    assert exit_code != 0
