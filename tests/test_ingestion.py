"""Integration tests for JSON persistence and idempotent directory ingestion."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rag_agent.ingestion.exceptions import PdfCorruptError
from rag_agent.ingestion.pipeline import ingest_directory, ingest_pdf

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_report.pdf"


def test_ingest_pdf_persists_a_page_aware_json_record(tmp_path: Path) -> None:
    """The pipeline saves one deterministic JSON artifact for a parsed PDF."""
    output_dir = tmp_path / "processed"

    document, pages = ingest_pdf(FIXTURE_PATH, output_dir)

    stored_path = output_dir / f"{document.doc_id}.json"
    stored_payload = json.loads(stored_path.read_text(encoding="utf-8"))
    assert stored_payload["document"]["doc_id"] == document.doc_id
    assert stored_payload["pages"][0]["page_number"] == 1
    assert pages[0].tables[0].startswith("| Metric |")


def test_ingest_directory_skips_an_already_processed_file(tmp_path: Path) -> None:
    """Hash-named output makes repeated directory runs idempotent."""
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()
    shutil.copy(FIXTURE_PATH, input_dir / "sample_report.pdf")

    first_run = ingest_directory(input_dir, output_dir)
    second_run = ingest_directory(input_dir, output_dir)

    assert first_run.processed_count == 1
    assert first_run.skipped_count == 0
    assert second_run.processed_count == 0
    assert second_run.skipped_count == 1
    assert first_run.records[0].document.doc_id == second_run.records[0].document.doc_id


def test_ingest_pdf_surfaces_corrupt_source_error(tmp_path: Path) -> None:
    """Single-file callers receive the parser's explicit corrupt-PDF error."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"not a PDF")

    with pytest.raises(PdfCorruptError, match="corrupt or unsupported"):
        ingest_pdf(corrupt_file, tmp_path / "processed")
