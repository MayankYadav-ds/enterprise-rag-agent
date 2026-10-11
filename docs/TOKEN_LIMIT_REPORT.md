# Model token-limit report: recursive chunks

**Measured on**: 2026-10-11
**Embedding model**: BAAI/bge-small-en-v1.5 (max 512 WordPiece tokens incl. CLS/SEP)
**Tokenizer used**: tokenizers >=0.20, the model's own `tokenizer.json` from Hugging Face
**Strategy measured**: `RecursiveChunking` only (per request)

### Sec Form 10 K Pdf Fixture

- **Config**: recursive chunks, size=512, overlap=50
- **Model limit (bge-small-en-v1.5, WordPiece incl. special tokens)**: 512
- **Chunk set measured**: all prose (chunk_type=text) chunks produced by `RecursiveChunking(chunk_size=512, chunk_overlap=50)` on the processed pages of this document. Table chunks are excluded.

| Metric | Value |
|--------|------|
| Chunk count | 234 |
| Chunks over model limit | 33 (14.1%) |
| Median model tokens | 458.0 |
| P95 model tokens | 528 |
| P99 model tokens | 567 |
| Max model tokens | 603 |
| Median tiktoken tokens | 474.0 |
| Median ratio model_tokens / tiktoken_tokens | 1.00 |

### Sec Form 10 K Pdf Fixture

- **Config**: recursive chunks, size=400, overlap=39
- **Model limit (bge-small-en-v1.5, WordPiece incl. special tokens)**: 512
- **Chunk set measured**: all prose (chunk_type=text) chunks produced by `RecursiveChunking(chunk_size=400, chunk_overlap=39)` on the processed pages of this document. Table chunks are excluded.

| Metric | Value |
|--------|------|
| Chunk count | 294 |
| Chunks over model limit | 1 (0.3%) |
| Median model tokens | 365.5 |
| P95 model tokens | 415 |
| P99 model tokens | 456 |
| Max model tokens | 569 |
| Median tiktoken tokens | 369.5 |
| Median ratio model_tokens / tiktoken_tokens | 1.01 |

### N Able 2024 Annual Financial Report

- **Config**: recursive chunks, size=512, overlap=50
- **Model limit (bge-small-en-v1.5, WordPiece incl. special tokens)**: 512
- **Chunk set measured**: all prose (chunk_type=text) chunks produced by `RecursiveChunking(chunk_size=512, chunk_overlap=50)` on the processed pages of this document. Table chunks are excluded.

| Metric | Value |
|--------|------|
| Chunk count | 194 |
| Chunks over model limit | 41 (21.1%) |
| Median model tokens | 484.0 |
| P95 model tokens | 531 |
| P99 model tokens | 543 |
| Max model tokens | 551 |
| Median tiktoken tokens | 469.0 |
| Median ratio model_tokens / tiktoken_tokens | 1.03 |

### N Able 2024 Annual Financial Report

- **Config**: recursive chunks, size=400, overlap=39
- **Model limit (bge-small-en-v1.5, WordPiece incl. special tokens)**: 512
- **Chunk set measured**: all prose (chunk_type=text) chunks produced by `RecursiveChunking(chunk_size=400, chunk_overlap=39)` on the processed pages of this document. Table chunks are excluded.

| Metric | Value |
|--------|------|
| Chunk count | 236 |
| Chunks over model limit | 0 (0.0%) |
| Median model tokens | 373.0 |
| P95 model tokens | 418 |
| P99 model tokens | 429 |
| Max model tokens | 452 |
| Median tiktoken tokens | 368.5 |
| Median ratio model_tokens / tiktoken_tokens | 1.03 |
