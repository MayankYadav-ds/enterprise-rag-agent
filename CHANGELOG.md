# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Foundation repository scaffold, Python packaging, linting, testing, and CI.
- Page-aware PDF ingestion with deterministic IDs, local JSON persistence, a CLI, and validation fixtures.
- Per-page extraction-quality scores and warnings for broken font mappings or other text artifacts.
- Three chunking strategies (fixed-size, recursive, structure-aware) with deterministic IDs, table-aware splitting, overlap, and quality-based page filtering.
- Chunk model with metadata including chunk_id, doc_id, text, chunk_type, page range, section heading, token count, text quality, and chunk index.
- CLI option to run ingestion then chunking, saving JSONL output.
- Comparison script for chunking strategies with metrics (chunk count, token statistics, mid-sentence share, runtime).
- Comprehensive test suite for chunking strategies and models.
