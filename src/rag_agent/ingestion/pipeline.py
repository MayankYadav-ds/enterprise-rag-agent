"""Idempotent file and directory ingestion with JSON persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from rag_agent.ingestion.exceptions import PdfIngestionError
from rag_agent.ingestion.models import Document, PageContent, document_id_from_hash, sha256_for_file
from rag_agent.ingestion.pdf_parser import ParseWarning, parse_pdf

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionRecord:
    """One file's outcome from a directory-ingestion run."""

    source_path: Path
    status: str
    document: Document | None
    pages: list[PageContent]
    warnings: list[ParseWarning]
    duration_seconds: float
    error: str | None = None

    @property
    def table_count(self) -> int:
        """Return the number of extracted tables in this record."""
        return sum(len(page.tables) for page in self.pages)


@dataclass(frozen=True, slots=True)
class DirectoryIngestionResult:
    """Aggregated, inspectable result of an idempotent directory ingestion."""

    records: list[IngestionRecord]

    @property
    def processed_count(self) -> int:
        """Return the number of PDFs parsed during this run."""
        return sum(record.status == "processed" for record in self.records)

    @property
    def skipped_count(self) -> int:
        """Return the number of PDFs already represented by persisted output."""
        return sum(record.status == "skipped" for record in self.records)

    @property
    def failed_count(self) -> int:
        """Return the number of PDFs that could not be parsed."""
        return sum(record.status == "failed" for record in self.records)

    @property
    def page_count(self) -> int:
        """Return total PDF pages represented by successful records."""
        return sum(record.document.num_pages for record in self.records if record.document)

    @property
    def table_count(self) -> int:
        """Return the number of tables extracted from successful records."""
        return sum(record.table_count for record in self.records)

    @property
    def warning_count(self) -> int:
        """Return non-fatal parsing warnings observed in this run."""
        return sum(len(record.warnings) for record in self.records)

    @property
    def low_quality_page_count(self) -> int:
        """Return the number of parsed pages below the configured text-quality threshold."""
        return sum(
            warning.kind == "low_text_quality"
            for record in self.records
            for warning in record.warnings
        )


def ingest_pdf(
    path: Path, output_dir: Path = Path("data/processed")
) -> tuple[Document, list[PageContent]]:
    """Parse one PDF and atomically write its deterministic JSON representation."""
    record = _ingest_new_pdf(Path(path), Path(output_dir))
    if record.document is None:
        msg = record.error or f"Could not ingest PDF: {path}"
        raise PdfIngestionError(msg)
    return record.document, record.pages


def ingest_directory(
    path: Path, output_dir: Path = Path("data/processed")
) -> DirectoryIngestionResult:
    """Ingest every PDF below a directory, skipping already-persisted content hashes."""
    input_dir = Path(path)
    processed_dir = Path(output_dir)
    if not input_dir.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise ValueError(msg)

    records: list[IngestionRecord] = []
    pdf_paths = sorted(
        candidate for candidate in input_dir.rglob("*") if candidate.suffix.lower() == ".pdf"
    )
    for pdf_path in pdf_paths:
        file_hash = sha256_for_file(str(pdf_path))
        output_path = _output_path(processed_dir, file_hash)
        if output_path.is_file():
            document, pages, warnings = _load_persisted_output(output_path)
            logger.info("Skipping unchanged PDF %s (%s)", pdf_path, document.doc_id)
            records.append(
                IngestionRecord(
                    source_path=pdf_path,
                    status="skipped",
                    document=document,
                    pages=pages,
                    warnings=warnings,
                    duration_seconds=0.0,
                )
            )
            continue
        try:
            records.append(_ingest_new_pdf(pdf_path, processed_dir, file_hash=file_hash))
        except PdfIngestionError as error:
            logger.error("Failed to ingest %s: %s", pdf_path, error)
            records.append(
                IngestionRecord(
                    source_path=pdf_path,
                    status="failed",
                    document=None,
                    pages=[],
                    warnings=[],
                    duration_seconds=0.0,
                    error=str(error),
                )
            )
    return DirectoryIngestionResult(records=records)


def _ingest_new_pdf(path: Path, output_dir: Path, file_hash: str | None = None) -> IngestionRecord:
    """Parse and persist a previously unseen PDF."""
    started_at = perf_counter()
    content_hash = file_hash or sha256_for_file(str(path))
    document_id = document_id_from_hash(content_hash)
    parsed = parse_pdf(path, document_id)
    document = Document(
        doc_id=document_id,
        source_path=str(path.resolve()),
        title=parsed.title,
        num_pages=parsed.num_pages,
        file_hash=content_hash,
    )
    _write_parsed_output(output_dir, document, parsed.pages, parsed.warnings)
    elapsed = perf_counter() - started_at
    logger.info(
        "Ingested %s: %d/%d content pages, %d tables, %d warnings in %.2fs",
        path,
        len(parsed.pages),
        document.num_pages,
        sum(len(page.tables) for page in parsed.pages),
        len(parsed.warnings),
        elapsed,
    )
    return IngestionRecord(
        source_path=path,
        status="processed",
        document=document,
        pages=parsed.pages,
        warnings=parsed.warnings,
        duration_seconds=elapsed,
    )


def _output_path(output_dir: Path, file_hash: str) -> Path:
    """Return the one stable JSON path for a source file hash."""
    return output_dir / f"{document_id_from_hash(file_hash)}.json"


def _write_parsed_output(
    output_dir: Path,
    document: Document,
    pages: list[PageContent],
    warnings: list[ParseWarning],
) -> None:
    """Atomically save source metadata and page records as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = _output_path(output_dir, document.file_hash)
    temporary_destination = destination.with_suffix(".json.tmp")
    payload = {
        "document": document.model_dump(mode="json"),
        "pages": [page.model_dump(mode="json") for page in pages],
        "warnings": [asdict(warning) for warning in warnings],
    }
    temporary_destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    temporary_destination.replace(destination)


def _load_persisted_output(path: Path) -> tuple[Document, list[PageContent], list[ParseWarning]]:
    """Load an existing idempotency record and validate its stored schema."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        document = Document.model_validate(payload["document"])
        pages = [PageContent.model_validate(page) for page in payload["pages"]]
        warnings = [ParseWarning(**warning) for warning in payload.get("warnings", [])]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        msg = f"Persisted ingestion output is invalid: {path}"
        raise PdfIngestionError(msg) from error
    return document, pages, warnings
