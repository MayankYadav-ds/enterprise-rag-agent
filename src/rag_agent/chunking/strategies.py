"""Chunking strategies for the RAG pipeline."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import tiktoken

from rag_agent.chunking.config import LOW_TEXT_QUALITY_THRESHOLD
from rag_agent.chunking.models import Chunk
from rag_agent.ingestion.models import PageContent


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_size: int = 10):
        """Initialize the chunking strategy.

        Args:
            chunk_size: Target size for chunks in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be valid
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

    @abstractmethod
    def chunk(self, pages: list[PageContent], doc_id: str) -> list[Chunk]:
        """Chunk a list of pages into Chunk objects.

        Args:
            pages: List of PageContent objects
            doc_id: Document identifier

        Returns:
            List of Chunk objects
        """
        pass

    def _is_low_quality_page(self, page: PageContent) -> bool:
        """Return True when ingestion scored this page below the shared threshold."""
        return page.text_quality < LOW_TEXT_QUALITY_THRESHOLD

    def _split_text_recursively(self, text: str) -> list[str]:
        """Split text recursively by paragraph, sentence, then word.

        Args:
            text: Text to split

        Returns:
            List of text segments
        """
        # Try splitting by paragraph first (double newline)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs

        # Try splitting by sentence
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 1:
            return sentences

        # Fallback to word-based splitting
        words = text.split()
        return [
            " ".join(words[i : i + self.chunk_size // 4])
            for i in range(0, len(words), self.chunk_size // 4)
        ]

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Merge splits into chunks with overlap.

        Args:
            splits: List of text segments to merge

        Returns:
            List of merged chunks
        """
        if not splits:
            return []

        chunks = []
        current_chunk = splits[0]

        for i in range(1, len(splits)):
            split = splits[i]
            split_tokens = len(self.tokenizer.encode(split))

            # Handle case where split itself is larger than chunk size
            if split_tokens > self.chunk_size:
                # Save current chunk if it has content
                if (
                    current_chunk
                    and len(self.tokenizer.encode(current_chunk)) >= self.min_chunk_size
                ):
                    chunks.append(current_chunk)

                # Split the oversized split into chunks by character count
                # Estimate characters needed for target token count (4 chars per token)
                max_chars = self.chunk_size * 4
                for j in range(0, len(split), max_chars):
                    chunk_text = split[j : j + max_chars]
                    if len(self.tokenizer.encode(chunk_text)) >= self.min_chunk_size:
                        chunks.append(chunk_text)

                # Start fresh current chunk
                current_chunk = ""
                continue  # Skip the normal processing below

            # Try to add the next split to current chunk
            combined = current_chunk + " " + split
            if len(self.tokenizer.encode(combined)) <= self.chunk_size:
                current_chunk = combined
            else:
                # Current chunk is full, save it and start new one
                if len(self.tokenizer.encode(current_chunk)) >= self.min_chunk_size:
                    chunks.append(current_chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + split if overlap_text else split

        # Don't forget the last chunk
        if current_chunk and len(self.tokenizer.encode(current_chunk)) >= self.min_chunk_size:
            chunks.append(current_chunk)

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk.

        Args:
            text: Source text

        Returns:
            Overlap text for the next chunk
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.chunk_overlap:
            return text
        overlap_tokens = tokens[-self.chunk_overlap :]
        return self.tokenizer.decode(overlap_tokens)

    def _process_table(
        self,
        table_markdown: str,
        doc_id: str,
        page_number: int,
        chunk_index: int,
        section_heading: str | None = None,
    ) -> list[Chunk]:
        """Process a table markdown string into chunks, handling large tables by splitting rows.

        Args:
            table_markdown: The table in Markdown format
            doc_id: Document identifier
            page_number: Page number where the table appears
            chunk_index: Starting chunk index
            section_heading: Optional section heading for context

        Returns:
            List of Chunk objects representing the table (or table parts)
        """
        # Parse the markdown table to extract header and rows
        lines = [line.strip() for line in table_markdown.split("\n") if line.strip()]

        if len(lines) < 3:  # Need at least header, separator, and one row
            # If it's too small to be a proper table, treat as text
            if len(self.tokenizer.encode(table_markdown)) >= self.min_chunk_size:
                chunk = Chunk.create(
                    doc_id=doc_id,
                    text=table_markdown.strip(),
                    chunk_type="table",
                    page_start=page_number,
                    page_end=page_number,
                    section_heading=section_heading,
                    chunk_index=chunk_index,
                    tokenizer=self.tokenizer,
                )
                return [chunk]
            return []

        # Extract header (first line) and separator (second line)
        header_line = lines[0]
        separator_line = lines[1] if len(lines) > 1 else ""
        data_lines = lines[2:] if len(lines) > 2 else []

        # Reconstruct the full header with separator for each chunk
        table_header = f"{header_line}\n{separator_line}"

        chunks = []
        current_chunk_index = chunk_index

        # If there are no data rows, just return the header
        if not data_lines:
            table_text = table_header
            if len(self.tokenizer.encode(table_text)) >= self.min_chunk_size:
                chunk = Chunk.create(
                    doc_id=doc_id,
                    text=table_text,
                    chunk_type="table",
                    page_start=page_number,
                    page_end=page_number,
                    section_heading=section_heading,
                    chunk_index=current_chunk_index,
                    tokenizer=self.tokenizer,
                )
                chunks.append(chunk)
            return chunks

        # Try to fit as many rows as possible in each chunk
        current_rows = []
        current_tokens = len(self.tokenizer.encode(table_header))  # Start with header tokens

        for row_line in data_lines:
            row_tokens = len(self.tokenizer.encode(row_line))
            # Account for newline that will be added before this row
            # First row doesn't need preceding newline in join
            newline_tokens = 1 if current_rows else 0

            # If adding this row would exceed chunk size, save current chunk and start new one
            if current_tokens + row_tokens + newline_tokens > self.chunk_size and current_rows:
                # Create chunk with current rows
                table_content = table_header + "\n" + "\n".join(current_rows)
                chunk = Chunk.create(
                    doc_id=doc_id,
                    text=table_content,
                    chunk_type="table",
                    page_start=page_number,
                    page_end=page_number,
                    section_heading=section_heading,
                    chunk_index=current_chunk_index,
                    tokenizer=self.tokenizer,
                )
                chunks.append(chunk)
                current_chunk_index += 1

                # Start new chunk with header
                current_rows = [row_line]
                current_tokens = len(self.tokenizer.encode(table_header)) + row_tokens
            else:
                # Add row to current chunk
                current_rows.append(row_line)
                current_tokens += row_tokens + newline_tokens

        # Don't forget the last chunk
        if current_rows:
            table_content = table_header + "\n" + "\n".join(current_rows)
            if len(self.tokenizer.encode(table_content)) >= self.min_chunk_size:
                chunk = Chunk.create(
                    doc_id=doc_id,
                    text=table_content,
                    chunk_type="table",
                    page_start=page_number,
                    page_end=page_number,
                    section_heading=section_heading,
                    chunk_index=current_chunk_index,
                    tokenizer=self.tokenizer,
                )
                chunks.append(chunk)

        return chunks


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with overlap."""

    def chunk(self, pages: list[PageContent], doc_id: str) -> list[Chunk]:
        """Chunk using fixed-size strategy with overlap.

        Args:
            pages: List of PageContent objects
            doc_id: Document identifier

        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_index = 0

        for page in pages:
            # Skip low-quality pages if they have no tables
            if self._is_low_quality_page(page):
                continue

            # Skip pages with no text and no tables
            if not page.text.strip() and not page.tables:
                continue

            # Process text content
            if page.text.strip():
                # Split page text into sentences for better boundaries
                sentences = re.split(r"(?<=[.!?])\s+", page.text)
                sentences = [s.strip() for s in sentences if s.strip()]

                if sentences:
                    # Group sentences into chunks
                    current_chunk = ""
                    current_tokens = 0

                    for sentence in sentences:
                        sentence_tokens = len(self.tokenizer.encode(sentence))

                        # If adding this sentence would exceed chunk size, save current chunk
                        if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                            if len(self.tokenizer.encode(current_chunk)) >= self.min_chunk_size:
                                chunk = Chunk.create(
                                    doc_id=doc_id,
                                    text=current_chunk.strip(),
                                    chunk_type="text",
                                    page_start=page.page_number,
                                    page_end=page.page_number,
                                    chunk_index=chunk_index,
                                    tokenizer=self.tokenizer,
                                )
                                chunks.append(chunk)
                                chunk_index += 1

                            # Start new chunk with overlap
                            overlap_text = self._get_overlap_text(current_chunk)
                            current_chunk = (
                                overlap_text + " " + sentence if overlap_text else sentence
                            )
                            current_tokens = len(self.tokenizer.encode(current_chunk))
                        else:
                            # Add sentence to current chunk
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                            current_tokens += sentence_tokens

                    # Handle case where a single sentence exceeds chunk size
                    if (
                        current_chunk
                        and len(self.tokenizer.encode(current_chunk)) > self.chunk_size
                    ):
                        # Save current chunk if it meets minimum size
                        if len(self.tokenizer.encode(current_chunk)) >= self.min_chunk_size:
                            chunk = Chunk.create(
                                doc_id=doc_id,
                                text=current_chunk.strip(),
                                chunk_type="text",
                                page_start=page.page_number,
                                page_end=page.page_number,
                                chunk_index=chunk_index,
                                tokenizer=self.tokenizer,
                            )
                            chunks.append(chunk)
                            chunk_index += 1

                        # Split the oversized sentence by characters (estimate 4 chars per token)
                        max_chars = self.chunk_size * 4
                        sentence_text = current_chunk
                        current_chunk = ""
                        for j in range(0, len(sentence_text), max_chars):
                            chunk_text = sentence_text[j : j + max_chars]
                            if len(self.tokenizer.encode(chunk_text)) >= self.min_chunk_size:
                                chunk = Chunk.create(
                                    doc_id=doc_id,
                                    text=chunk_text.strip(),
                                    chunk_type="text",
                                    page_start=page.page_number,
                                    page_end=page.page_number,
                                    chunk_index=chunk_index,
                                    tokenizer=self.tokenizer,
                                )
                                chunks.append(chunk)
                                chunk_index += 1

                        # Start fresh current chunk
                        current_chunk = ""
                        current_tokens = 0

                    # Don't forget the last chunk on the page
                    if (
                        current_chunk
                        and len(self.tokenizer.encode(current_chunk)) >= self.min_chunk_size
                    ):
                        chunk = Chunk.create(
                            doc_id=doc_id,
                            text=current_chunk.strip(),
                            chunk_type="text",
                            page_start=page.page_number,
                            page_end=page.page_number,
                            chunk_index=chunk_index,
                            tokenizer=self.tokenizer,
                        )
                        chunks.append(chunk)
                        chunk_index += 1

            # Process tables
            for _table_idx, table_markdown in enumerate(page.tables):
                # Process each table, potentially splitting it into multiple chunks if large
                table_chunks = self._process_table(
                    table_markdown, doc_id, page.page_number, chunk_index
                )

                for table_chunk in table_chunks:
                    chunks.append(table_chunk)
                    chunk_index += 1

        return chunks


class RecursiveChunking(ChunkingStrategy):
    """Recursive chunking strategy."""

    def chunk(self, pages: list[PageContent], doc_id: str) -> list[Chunk]:
        """Chunk using recursive strategy.

        Args:
            pages: List of PageContent objects
            doc_id: Document identifier

        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_index = 0

        for page in pages:
            # Skip low-quality pages if they have no tables
            if self._is_low_quality_page(page):
                continue

            # Skip pages with no text and no tables
            if not page.text.strip() and not page.tables:
                continue

            # Process text content
            if page.text.strip():
                # Split text recursively
                splits = self._split_text_recursively(page.text)

                # Merge splits into chunks with overlap
                merged_chunks = self._merge_splits(splits)

                # Create Chunk objects
                for chunk_text in merged_chunks:
                    if len(self.tokenizer.encode(chunk_text)) >= self.min_chunk_size:
                        chunk = Chunk.create(
                            doc_id=doc_id,
                            text=chunk_text.strip(),
                            chunk_type="text",
                            page_start=page.page_number,
                            page_end=page.page_number,
                            chunk_index=chunk_index,
                            tokenizer=self.tokenizer,
                        )
                        chunks.append(chunk)
                        chunk_index += 1

            # Process tables
            for _table_idx, table_markdown in enumerate(page.tables):
                # Process each table, potentially splitting it into multiple chunks if large
                table_chunks = self._process_table(
                    table_markdown, doc_id, page.page_number, chunk_index
                )

                for table_chunk in table_chunks:
                    chunks.append(table_chunk)
                    chunk_index += 1

        return chunks


class StructureAwareChunking(ChunkingStrategy):
    """Structure-aware chunking: split on headings, then recursive merge within sections."""

    _HEADING_PATTERN_UPPER = re.compile(r"^[A-Z][A-Z\s\d.\-,:]+$")
    _HEADING_PATTERN_NUMBERED = re.compile(r"^[\d.]+\s+[A-Z]")
    _HEADING_PATTERN_SHORT_CAPS = re.compile(r"^[A-Z][A-Z\s\d.\-]{2,}$")

    def _fallback_heading_positions(self, lines: list[str]) -> list[tuple[int, str]]:
        """Infer heading lines from uppercase / numbered patterns when metadata is missing."""
        headings: list[tuple[int, str]] = []
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line or len(line) >= 150:
                continue
            if (
                self._HEADING_PATTERN_UPPER.match(line)
                or self._HEADING_PATTERN_NUMBERED.match(line)
                or self._HEADING_PATTERN_SHORT_CAPS.match(line)
            ):
                headings.append((index, line))
        return headings

    def _heading_positions(self, page: PageContent, lines: list[str]) -> list[tuple[int, str]]:
        """Locate ingestion headings in page text, falling back to line patterns."""
        headings: list[tuple[int, str]] = []
        if page.headings:
            text = page.text
            for heading_text in page.headings:
                start_pos = 0
                while True:
                    pos = text.find(heading_text, start_pos)
                    if pos == -1:
                        break
                    headings.append((text[:pos].count("\n"), heading_text))
                    start_pos = pos + 1
            headings = list(dict.fromkeys(headings))
            headings.sort(key=lambda item: item[0])
        if not headings:
            headings = self._fallback_heading_positions(lines)
        return headings

    def _sections_for_page(self, page: PageContent) -> list[tuple[str | None, str]]:
        """Build heading/body sections; never emit a heading-only section."""
        lines = page.text.split("\n")
        headings = self._heading_positions(page, lines)
        if not headings:
            return [(None, page.text)]

        sections: list[tuple[str | None, str]] = []
        orphan_headings: list[str] = []
        first_heading_line = headings[0][0]
        if first_heading_line > 0:
            preamble = "\n".join(lines[:first_heading_line]).strip()
            if preamble:
                sections.append((None, preamble))

        for index, (heading_line, heading_text) in enumerate(headings):
            end_line = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
            body = "\n".join(lines[heading_line + 1 : end_line]).strip()
            if not body:
                orphan_headings.append(heading_text)
                continue
            prefix = "\n\n".join([*orphan_headings, heading_text])
            orphan_headings.clear()
            sections.append((heading_text, f"{prefix}\n\n{body}"))

        if orphan_headings:
            extra = "\n\n".join(orphan_headings)
            if sections:
                previous_heading, previous_text = sections[-1]
                sections[-1] = (previous_heading, f"{previous_text}\n\n{extra}")
            else:
                sections.append((orphan_headings[-1], extra))
        return sections

    def _coalesce_undersized(self, parts: list[str]) -> list[str]:
        """Merge leftover fragments below min_chunk_size into a neighbour when it still fits."""
        if not parts:
            return []
        coalesced: list[str] = []
        for part in parts:
            token_count = len(self.tokenizer.encode(part))
            if coalesced and token_count < self.min_chunk_size:
                combined = f"{coalesced[-1]}\n\n{part}"
                if len(self.tokenizer.encode(combined)) <= self.chunk_size:
                    coalesced[-1] = combined
                    continue
            coalesced.append(part)
        if len(coalesced) > 1 and len(self.tokenizer.encode(coalesced[0])) < self.min_chunk_size:
            combined = f"{coalesced[0]}\n\n{coalesced[1]}"
            if len(self.tokenizer.encode(combined)) <= self.chunk_size:
                coalesced = [combined, *coalesced[2:]]
        return [
            part for part in coalesced if len(self.tokenizer.encode(part)) >= self.min_chunk_size
        ]

    def chunk(self, pages: list[PageContent], doc_id: str) -> list[Chunk]:
        """Chunk using structure-aware strategy."""
        chunks = []
        chunk_index = 0

        for page in pages:
            if self._is_low_quality_page(page):
                continue
            if not page.text.strip() and not page.tables:
                continue

            for heading, section_text in self._sections_for_page(page):
                if not section_text.strip():
                    continue
                merged_chunks = self._coalesce_undersized(
                    self._merge_splits(self._split_text_recursively(section_text))
                )
                for chunk_text in merged_chunks:
                    chunk = Chunk.create(
                        doc_id=doc_id,
                        text=chunk_text.strip(),
                        chunk_type="text",
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section_heading=heading,
                        chunk_index=chunk_index,
                        tokenizer=self.tokenizer,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

            for table_markdown in page.tables:
                table_chunks = self._process_table(
                    table_markdown, doc_id, page.page_number, chunk_index
                )
                for table_chunk in table_chunks:
                    chunks.append(table_chunk)
                    chunk_index += 1

        return chunks
