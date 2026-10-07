# Decisions and trade-offs

## Chunking configuration

**Provisional defaults:** chunk size 512 tokens, overlap 50 tokens,
minimum chunk size 10 tokens. These were chosen because they produce
a balanced trade-off between context window pressure (fewer, larger
chunks) and boundary precision (smaller chunks end more often at
sentence boundaries). The values are not optimised for any specific
document type; update the config file once real retrieval quality
scores are available.

**Tokeniser:** tiktoken `cl100k_base` (GPT-4). This is NOT the
embedding model's tokenizer — embedding models use a different
tokenisation that would make chunk boundaries inconsistent with the
retrieval scoring. The chunking package uses tiktoken for all
token-counting, and falls back to a character-based estimate only
if tiktoken is unavailable.

## Three strategies

| Strategy | Boundary precision | Speed | Heading awareness |
|----------|-------------------|-------|-------------------|
| fixed_size | lowest | fastest | none |
| recursive | medium | medium | none |
| structure_aware | highest (sentence-split oversized paragraphs) | slower | yes |

Structure-aware is the default for SEC filings because "Item NN."
headings are real section boundaries; fixed-size/recursive ignore
them.

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

## Future work

- OCR for image-only pages.
- Table-region merging for fragmented unruled tables.
- Reranker-guided chunk size adaptation.
- Page-render-based quality checks.
