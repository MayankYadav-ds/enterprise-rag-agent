# Decisions and trade-offs

## Chunking configuration

**Provisional defaults (still to be validated in Phase 8):** chunk size
400 tokens, overlap 39 tokens, minimum chunk size 10 tokens. These were
lowered from 512/50 after measuring the real embedding model's WordPiece
tokenizer on the two filings — see [`docs/TOKEN_LIMIT_REPORT.md`](TOKEN_LIMIT_REPORT.md).

**Why 400/39 beats 512/50.** The chunking package sizes chunks in tiktoken
`cl100k_base` tokens, but the embedding model counts with its own WordPiece
tokenizer and silently truncates text beyond 512 tokens. Measuring the
recursive strategy on the real chunks:

| Document | Config | Text chunks | Over limit | Median | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NPS 10-K | 512/50 | 234 | 14.1% | 458 | 528 | 567 | 603 |
| NPS 10-K | 400/39 | 294 | 0.3% | 366 | 415 | 456 | 569 |
| N-able 2024 | 512/50 | 194 | 21.1% | 484 | 531 | 543 | 551 |
| N-able 2024 | 400/39 | 236 | 0.0% | 373 | 418 | 429 | 452 |

At 512/50, 14–21% of chunks exceeded the model limit and were cut off mid-way;
at 400/39 the p99 is 429–456 and only 0–0.3% exceed the limit, leaving a safety
margin. The ratio model_tokens / tiktoken_tokens is 1.00–1.03 at the median,
so the two tokenisers agree closely on financial prose.

**Tokeniser:** tiktoken `cl100k_base` (GPT-4). This is NOT the embedding
model's tokenizer — embedding models use a different tokenisation that would
make chunk boundaries inconsistent with the retrieval scoring. The chunking
package uses tiktoken for all token-counting, and falls back to a character-based
estimate only if tiktoken is unavailable.

## Three strategies

| Strategy | Boundary precision | Speed | Heading awareness |
|----------|-------------------|-------|-------------------|
| fixed_size | lowest | fastest | none |
| recursive | medium | medium | none |
| structure_aware | highest (sentence-split oversized paragraphs) | slower | yes |

**Recursive is the default.** Structure-aware is NOT claimed to be
better: on the two real filings it produces more chunks and a higher
mid-sentence rate than recursive (NPS 15.5% vs 11.5%; N-able 24.8% vs
24.2%), and its heading-aware splitting is a hypothesis, not a
verified win. The choice between recursive and structure-aware is
deferred to Phase 8, when retrieval and answer-quality scores are
available to judge boundary placement against downstream quality.

## Heading detection

SEC "Item NN." headings are bold but rendered at body font size,
so font-size rules alone miss them. Both the ingestion detector
and the chunker's fallback use a regex pattern (`^Item\s+\d+[A-Z]?\.`)
in addition to font-size and uppercase rules.

## Table handling

Tables are treated as atomic units and split by rows with header
repetition only when a table exceeds chunk size. Table chunks carry
`chunk_type=table` and should be excluded from sentence-completeness
metrics.

## Embedding model

**BAAI/bge-small-en-v1.5** (33M parameters, 384-dimensional, multilingual
English). It is instruction-aware: documents are embedded raw while queries get
a search prefix (`"Represent this sentence for searching relevant passages: "`)
so the model's query-task is consistent. Vectors are L2-normalised on output, so
cosine similarity equals the dot product computed by Qdrant.

The model accepts up to 512 WordPiece tokens including the CLS/SEP special
tokens, and silently truncates beyond that. This is the reason chunk size was
lowered from 512 to 400 (above). The model can be overridden with
`EMBEDDING_MODEL` for ablation, but a collection built with one model rejects
another at open time — see `QdrantStore._ensure_collection`.

## Vector storage

**Qdrant with cosine distance.** Qdrant stores vectors and computes cosine
similarity; because every vector is normalised, cosine equals the dot product
and the two are interchangeable.

**UUID5 point IDs.** Each chunk maps to `uuid.uuid5(PROJECT_NAMESPACE,
chunk_id)`, so the same chunk always produces the same point and re-runs never
duplicate points. The ID is derived from the chunk ID, not from a random or
timestamped value, so a diff-based sync can compare stored points against the
target set exactly.

**Diff-based sync.** `QdrantStore.sync(doc_id, chunks, ...)` computes
`existing = scroll_ids(doc_id)` and `target = {point_id(c.chunk_id) for c in
chunks}`, then deletes `existing - target` and upserts only chunks whose ID is
not already stored. It returns `{"upserted", "deleted", "skipped"}` and is
scoped to one document, so syncing one document never deletes another's points.
Re-running the indexer on unchanged input embeds 0 chunks.

## Quality threshold

Pages scoring below 0.900 (e.g. N-able's 44 pages with broken font
mappings producing non-whitespace control characters) are excluded
from chunking by default; their tables are still retained.

## Known limitations

- PDF font encodings can produce unreadable glyph mappings (N-able:
  44 pages affected). See [#14](https://github.com/MayankYadav-ds/enterprise-rag-agent/issues/14).
- Unruled multi-year financial tables fragment (NPS page 80). See [#13](https://github.com/MayankYadav-ds/enterprise-rag-agent/issues/13).
- Superscript footnotes fuse into adjacent values (e.g. `2015¹` becomes `20151`). See [#12](https://github.com/MayankYadav-ds/enterprise-rag-agent/issues/12).
- Mid-sentence chunk rate is ~12-25% depending on strategy; sentence-boundary splitting reduces it but does not eliminate it for oversized paragraphs.
- Structure-aware is not better than recursive on the two real filings (NPS 15.5% vs 11.5%, N-able 24.8% vs 24.2%); it also produces more chunks. See [#18](https://github.com/MayankYadav-ds/enterprise-rag-agent/issues/18).
- Chunks whose `section_heading` is null are embedded as raw text with no heading context; fixed-size and recursive produce such chunks. See [#23](https://github.com/MayankYadav-ds/enterprise-rag-agent/issues/23).
- Table chunks are embedded as raw Markdown (pipe characters, alignment dashes) rather than prose; the bge model was trained on natural text. See [#24](https://github.com/MayankYadav-ds/enterprise-rag-agent/issues/24).

## Future work

- OCR for image-only pages.
- Table-region merging for fragmented unruled tables.
- Reranker-guided chunk size adaptation.
- Page-render-based quality checks.
