"""Domain-specific errors raised while ingesting PDF files."""

from __future__ import annotations


class PdfIngestionError(Exception):  # pragma: no cover
    """Base error for a PDF that cannot be ingested."""


class PdfNotFoundError(PdfIngestionError):  # pragma: no cover
    """Raised when an input PDF path does not exist."""


class PdfCorruptError(PdfIngestionError):  # pragma: no cover
    """Raised when a PDF cannot be opened or decoded."""


class PdfEncryptedError(PdfIngestionError):  # pragma: no cover
    """Raised when a password-protected PDF is encountered."""
