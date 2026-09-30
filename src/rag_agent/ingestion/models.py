"""Validated records produced by the PDF ingestion stage."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

from pydantic import BaseModel, Field, field_validator, model_validator


def document_id_from_hash(file_hash: str) -> str:
    """Return the stable document identifier for a SHA-256 file digest."""
    normalized_hash = file_hash.lower()
    is_sha256 = len(normalized_hash) == 64 and all(
        char in "0123456789abcdef" for char in normalized_hash
    )
    if not is_sha256:
        msg = "file_hash must be a 64-character SHA-256 hexadecimal digest"
        raise ValueError(msg)
    return f"doc_{normalized_hash}"


class Document(BaseModel):
    """Source-level metadata for an ingested PDF."""

    doc_id: str = Field(min_length=68, pattern=r"^doc_[0-9a-f]{64}$")
    source_path: str = Field(min_length=1)
    title: str = Field(min_length=1)
    num_pages: int = Field(ge=1)
    file_hash: str = Field(min_length=64, max_length=64)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("file_hash")
    @classmethod
    def normalise_file_hash(cls, value: str) -> str:
        """Validate and lowercase the content hash."""
        return document_id_from_hash(value)[4:]

    @model_validator(mode="after")
    def validate_document_id(self) -> Document:
        """Ensure re-ingestion of identical bytes always uses the same ID."""
        expected_id = document_id_from_hash(self.file_hash)
        if self.doc_id != expected_id:
            msg = "doc_id must be derived from file_hash"
            raise ValueError(msg)
        return self


class PageContent(BaseModel):
    """Clean, page-aware content that is ready for later chunking."""

    doc_id: str = Field(min_length=68, pattern=r"^doc_[0-9a-f]{64}$")
    page_number: int = Field(ge=1)
    text: str = ""
    tables: list[str] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    char_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def set_char_count(self) -> PageContent:
        """Make the recorded character count agree with the cleaned text."""
        expected_count = len(self.text)
        if self.char_count not in (0, expected_count):
            msg = "char_count must equal the length of text"
            raise ValueError(msg)
        self.char_count = expected_count
        return self


def sha256_for_file(path: str) -> str:
    """Compute the SHA-256 digest for a file without loading it all into memory."""
    digest = sha256()
    with open(path, "rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
