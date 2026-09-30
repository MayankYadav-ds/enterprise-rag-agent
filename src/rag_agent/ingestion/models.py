"""Validated records produced by the PDF ingestion stage."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from hashlib import sha256

from pydantic import BaseModel, Field, field_validator, model_validator

_CID_ARTIFACT_PATTERN = re.compile(r"\(cid:\d+\)")


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
    text_quality: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def set_derived_fields(self) -> PageContent:
        """Derive text length and quality consistently from the stored page text."""
        expected_count = len(self.text)
        if self.char_count not in (0, expected_count):
            msg = "char_count must equal the length of text"
            raise ValueError(msg)
        self.char_count = expected_count
        self.text_quality = text_quality_score(self.text)
        return self


def text_quality_score(text: str) -> float:
    """Score extracted text from 0 to 1, penalising known PDF decoding artifacts.

    A score of 1.0 has no known artifacts. The score counts characters in
    ``(cid:N)`` sequences, literal replacement characters, and Unicode control
    characters. Newlines and tabs are excluded because they are valid layout
    whitespace before the parser normalises a page.
    """
    if not text:
        return 1.0
    cid_artifact_characters = sum(
        match.end() - match.start() for match in _CID_ARTIFACT_PATTERN.finditer(text)
    )
    replacement_characters = text.count("\ufffd")
    control_characters = sum(
        unicodedata.category(character).startswith("C") and character not in "\n\r\t"
        for character in text
    )
    artifact_characters = cid_artifact_characters + replacement_characters + control_characters
    return round(max(0.0, 1.0 - artifact_characters / len(text)), 6)


def sha256_for_file(path: str) -> str:
    """Compute the SHA-256 digest for a file without loading it all into memory."""
    digest = sha256()
    with open(path, "rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
