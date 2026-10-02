"""Chunking strategies for the RAG pipeline."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import tiktoken

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

    def _is_low_quality(self, text: str, threshold: float = 0.9) -> bool:
        """Check if text quality is below threshold.

        Args:
            text: Text to check
            threshold: Quality threshold (0-1)

        Returns:
            True if text quality is below threshold
        """
        if not text:
            return True

        # Simple quality check based on length and special characters
        # In a real implementation, we'd reuse the text_quality_score from ingestion
        special_chars = text.count("�") + len(re.findall(r"\(cid:\d+\)", text))
        total_chars = len(text)
        if total_chars == 0:
            return True

        quality = 1.0 - (special_chars / total_chars)
        return quality < threshold

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
            if self._is_low_quality(page.text) and not page.tables:
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
            if self._is_low_quality(page.text) and not page.tables:
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
    """Structure-aware chunking strategy."""

    def chunk(self, pages: list[PageContent], doc_id: str) -> list[Chunk]:
        """Chunk using structure-aware strategy.

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
            if self._is_low_quality(page.text) and not page.tables:
                continue

            # Skip pages with no text and no tables
            if not page.text.strip() and not page.tables:
                continue

            # Extract lines from text once
            lines = page.text.split("\n")

            # Use headings from ingestion if available, otherwise extract from text
            headings = []
            if page.headings:
                # Use pre-extracted headings from ingestion
                # Find where each heading appears in the text to create section boundaries
                text = page.text

                for heading_text in page.headings:
                    # Find all occurrences of this heading in the text
                    start_pos = 0
                    while True:
                        pos = text.find(heading_text, start_pos)
                        if pos == -1:
                            break
                        # Convert character position to approximate line number
                        # (count newlines before this position)
                        line_num = text[:pos].count("\n")
                        headings.append((line_num, heading_text))
                        start_pos = pos + 1  # Continue searching after this occurrence

                # Remove duplicates (same heading at same position) and sort by line number
                headings = list(
                    dict.fromkeys(headings)
                )  # Removes duplicates while preserving order
                headings.sort(key=lambda x: x[0])

                # If we didn't find any headings via substring search, fall back to pattern matching
                if not headings:
                    # Fallback: extract headings from text using improved regex patterns

                    # Improved heading patterns that are more inclusive
                    # Pattern 1: Lines that are mostly uppercase (allowing some
                    # lowercase/numbers/special chars)
                    heading_pattern1 = re.compile(r"^[A-Z][A-Z\s\d\.\-,:]+$")
                    # Pattern 2: Numbered headings like "1. ", "2. ", etc.
                    heading_pattern2 = re.compile(r"^[\d\.]+\s+[A-Z]")
                    # Pattern 3: ALL CAPS reasonable length lines
                    heading_pattern3 = re.compile(r"^[A-Z][A-Z\s\d\.\-]{2,}$")

                    # Find potential headings
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line and len(line) < 150:  # Increased limit for heading length
                            # Check if line matches any of our heading patterns
                            if (
                                heading_pattern1.match(line)
                                or heading_pattern2.match(line)
                                or heading_pattern3.match(line)
                            ):
                                headings.append((i, line))
            else:
                # Fallback: extract headings from text using improved regex patterns

                # Improved heading patterns that are more inclusive
                # Pattern 1: Lines that are mostly uppercase (allowing some
                # lowercase/numbers/special chars)
                heading_pattern1 = re.compile(r"^[A-Z][A-Z\s\d\.\-,:]+$")
                # Pattern 2: Numbered headings like "1. ", "2. ", etc.
                heading_pattern2 = re.compile(r"^[\d\.]+\s+[A-Z]")
                # Pattern 3: ALL CAPS reasonable length lines
                heading_pattern3 = re.compile(r"^[A-Z][A-Z\s\d\.\-]{2,}$")

                # Find potential headings
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line and len(line) < 150:  # Increased limit for heading length
                        # Check if line matches any of our heading patterns
                        if (
                            heading_pattern1.match(line)
                            or heading_pattern2.match(line)
                            or heading_pattern3.match(line)
                        ):
                            headings.append((i, line))

            # If we found headings, split by them
            if headings:
                # Sort headings by line number
                headings.sort(key=lambda x: x[0])

                # Create sections based on headings
                sections = []

                # Text before first heading (if any)
                first_heading_line = headings[0][0]
                if first_heading_line > 0:
                    section_text = "\n".join(lines[0:first_heading_line]).strip()
                    if section_text:
                        sections.append((None, section_text))

                # Process each heading with the content that follows it
                for i, (heading_line, heading_text) in enumerate(headings):
                    # Determine the end of this section (start of next heading or end of document)
                    if i + 1 < len(headings):
                        end_line = headings[i + 1][0]
                    else:
                        end_line = len(lines)

                    # Section content is from after this heading to before next heading
                    section_start_line = heading_line + 1
                    section_end_line = end_line

                    if section_start_line < section_end_line:
                        section_text = "\n".join(lines[section_start_line:section_end_line]).strip()
                        if section_text:
                            # Include heading with its content
                            full_section_text = heading_text + "\n\n" + section_text
                            sections.append((heading_text, full_section_text))
                    else:
                        # No content after heading, just the heading itself
                        sections.append((heading_text, heading_text))
            else:
                # No headings found, treat whole page as one section
                sections = [(None, page.text)]

            # Chunk each section
            for heading, section_text in sections:
                if not section_text.strip():
                    continue

                # Use recursive chunking within each section
                splits = self._split_text_recursively(section_text)
                merged_chunks = self._merge_splits(splits)

                for chunk_text in merged_chunks:
                    if len(self.tokenizer.encode(chunk_text)) >= self.min_chunk_size:
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
