# Decisions and trade-offs

## Heading assignment strategy

**Decision:** Use character-offset-based position matching within page text, not page-level last-heading carry-forward.

**Why:** Page-level last-heading assignment fails when a chunk starts mid-page before a heading that is *inside* its text span. For example, N-able p75 chunk 191 starts at offset 2003 but contains `ITEM 8` at offset 2876 — carry-forward from p74's `ITEM 7A` would be correct only if no ITEM heading exists within the chunk's span. The fix checks for contained ITEM headings first, then falls back to backwards carry-forward.

**Alternatives considered:**
- *Sentence-level heading detection*: too fragile with PDF text extraction noise (hyphenation, wrapping, private-use characters).
- *Font-size-based section detection*: works at ingestion time but not reliably at chunk time where heading text may be split across chunks.
- *Full-document heading tree with nesting*: overkill for a flat ITEM structure; 10-K filings are flat (ITEM 1, 1A, 2, ...).

**Trade-offs:**
- Contained-ITEM check adds O(n) per chunk but only within the chunk's own page — negligible cost.
- Non-ITEM headings (table labels, statement titles) are no longer section_heading in 10-K docs; they are optionally captured as `subheading`.

## Subheading field

**Decision:** Added `subheading: str | None` to the Chunk model and Qdrant payload.

**Why:** Some documents have meaningful non-ITEM subheadings (e.g. "Risk Factors" under `ITEM 1A`). These provide better context for retrieval than the ITEM heading alone. However, in 10-K-structured docs, most non-ITEM "headings" detected by the PDF parser are table labels and statement titles — so only strict subheading candidates (bold, larger than body font, short, not a table label) are stored.

**Trade-off:** Adds one field to the payload schema. The change is backward-compatible since existing points simply gain a `subheading: null` field when patched.

**Subheading hygiene.** `_is_subheading` checks text patterns only — it has no access to the bold or font-size signals the ingestion detector used, so a candidate that passes is not guaranteed to be bold or larger than the body font. It also rejects headings containing non-printable characters (the N-able report's broken-font-mapping pages produce control-character soup such as `"\x06,-70\x05\x01\x1b9.\x07"`), headings that are pure punctuation/symbols (`:=`, the only such token on the real filings), and headings that are mostly symbols — a run of at least two consecutive letters is required, which rejects `$6%24)3).'"` and `$0A09@0` from the same broken-font-mapping pages. A non-ITEM heading that appears *before* an ITEM heading on the same page (e.g. `PART I`, the auditor's report title) is never captured as that ITEM's subheading — it names the enclosing part or the report, not a subheading of the ITEM.

## Payload diff convergence

**Decision:** The patch pass normalises the stored-vs-fresh payload comparison so that `None`, a missing key and `""` are all "no value", and `_payload` writes optional fields as `None` when absent rather than omitting them.

**Why:** Qdrant's `upsert` keeps `null`-valued payload keys, but `set_payload` and `batch_update_points` drop them entirely (verified on the live server at `http://localhost:6333`). The 543 points indexed before the `subheading` field existed carried no such key, while the fresh payload carries `subheading: None`; under a plain `dict ==` comparison those are not equal, so the patch pass re-applied the same 258 points on every run and never converged. Normalising the comparison makes the stored form consistent, so a second sync reports `0 payload updated`.

The optional fields are written as `None` rather than omitted for a second reason, discovered on the live server: **omitting a key from a `set_payload` / `batch_update_points` patch does not remove it — only an explicit `null` does** (verified by inserting a probe point and patching it both ways). The 41 N-able points carrying `subheading: "PART I"` and the 47 carrying the auditor's report title were written by a pre-whitelist run, when section markers leaked into the `subheading` field; an omission-based patch left those values in place forever and the diff never converged. Writing `None` is what clears them. The normaliser still treats "stored missing" and "fresh None" as equal, so a point that never had the key converges after one patch.

**Trade-off:** The normalised comparison treats `""` and `None` as equal, which is the intended semantic for an absent value. A real empty-string field would be indistinguishable from a missing one; no field in the payload is ever legitimately empty. The patch now always carries every optional field, which is a larger payload per `SetPayloadOperation` but is bounded by the number of optional fields (one today).

**Why not `delete_payload`.** To clear a field whose fresh value is absent, the sync writes `None` in the same `SetPayloadOperation` batch that carries the other payload changes. Qdrant's explicit delete (`DeletePayloadOperation`, which `batch_update_points` accepts, checked against the installed `qdrant-client`) is a viable alternative. It is not used because writing `None` already works and needs no extra operation type.

*Verified:* (1) `DeletePayloadOperation` exists in `qdrant_client.http.models` and `batch_update_points` accepts it, so the earlier claim that a delete cannot be batched was wrong. (2) On the live server, omitting a key from a set-payload patch does not remove it, while writing `null` clears it (probe recorded in DEVLOG). (3) From reading `_payloads_equal`: it treats `None`, a missing key and `""` as equal, so the stored shape (null versus missing) does not decide whether a second sync reports 0 payload updates; a delete-based patch would converge as well. This third point comes from reading the code and has no dedicated test.

*Not verified:* how the live server handles a delete of a key that is not present (no-op or error), and the request cost compared with writing `None`.

The one case where `delete_payload` would be strictly better is a field that is being *removed from the schema entirely* — there, `None` leaves a tombstone key on every point while `delete_payload` removes it. That is a one-time migration, not a per-sync operation, and is tracked separately.

## 10-K structure detection threshold

**Decision:** `_MIN_ITEM_HEADINGS = 3`.

**Why:** Documents with fewer than 3 distinct ITEM headings are treated as non-10-K, where any valid heading (not just ITEM headings) can set `section_heading`. This preserves the original behavior for non-SEC documents while specialising for 10-K filings. The threshold of 3 was chosen because all real 10-Ks have at least 5–6 ITEM headings, and 3 provides a safe margin above any false positives from cover pages or table-of-contents.

## Contained-ITEM rule

**Decision:** A chunk whose text span contains an `ITEM` heading on its own page takes that heading; a chunk that straddles two `ITEM` boundaries is labelled with the later `ITEM`.

**Why:** In a 10-K-structured document only `ITEM` headings set `section_heading`, so the last `ITEM` on a page carries forward over everything that follows it — including the financial-statement pages that come after the last `ITEM`. On the real filings this produces the wrong labels: N-able pages 77–125 (the financial statements and the auditor's report, 47 chunks) are labelled `ITEM 14. PRINCIPAL ACCOUNTANT FEES AND SERVICES`, and NPS page 96 loses its `ITEM 14` label (one of its two chunks becomes `ITEM 13`, the other `ITEM 9B`). The contained-ITEM check is the first thing tried for each chunk, so a chunk that literally contains an `ITEM` heading takes that heading rather than a carry-forward from an earlier page.

**Trade-off:** This is a heuristic, not a parse of the document's actual section boundaries. 16 chunks in NPS and 21 in N-able contain an `ITEM` heading inside their span without starting at it — those chunks are labelled with the later `ITEM`, which is correct when the heading names the section the chunk belongs to and wrong when the chunk is tail text of the previous section. Treating `ITEM` headings as hard chunk boundaries would fix the boundary but requires re-embedding, because chunk text and chunk IDs change. Tracked as a known limitation.

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
not already stored. It returns `{"upserted", "deleted", "skipped",
"payload_updated"}` and is scoped to one document, so syncing one document
never deletes another's points. Re-running the indexer on unchanged input
embeds 0 chunks.

**What the sync can and cannot prove.** The sync is diff-based, so re-running
it on unchanged input converges: the second run reports `0 upserted, 0
payload_updated, 0 stale deleted`. That is the only property asserted here —
nothing about `batch_update_points` is claimed. Measured on the live server,
`batch_update_points` is **not** atomic: a batch whose second operation targets
a non-existent point raises a 404, but the first operation is still applied
(verified by inserting two probe points and sending a batch whose second
operation references a point that was never inserted). The patch pass is
therefore not a transaction; it relies on the diff converging, not on the
batch being all-or-nothing.

**Payload patch convergence.** The patch pass compares each stored payload
against the fresh one and patches only the differences. Qdrant's `upsert`
keeps `null`-valued payload keys, but `set_payload` and `batch_update_points`
drop them (verified on the live server). `_payload` therefore **writes** every
optional field, carrying `None` when it has no value, and the comparison
normalises `None`, a missing key and `""` as equal, so a point that never had a
`subheading` key converges after one patch instead of being re-patched on
every run. Writing `None` (rather than omitting the key) is also what clears a
stale value on the live server — only an explicit `null` removes a key from a
patch, which cleared 41 points carrying `subheading: "PART I"` and 47 carrying
the auditor's report title once the 10-K marker whitelist stopped leaking
those values into the subheading field.

## Dense retrieval and citations

**Dense-only baseline.** `DenseRetriever` embeds the query with the same
embedder that indexed the collection and asks Qdrant for the nearest
neighbours, returning `RetrievedChunk` hits ranked 1..N best first. It is the
baseline retriever: BM25, reciprocal-rank fusion and a cross-encoder reranker
arrive in later phases and implement the same `Retriever` protocol, so the
search CLI and the citation layer are retriever-agnostic. No default score
threshold is chosen — a value passed as `min_score` is an explicit caller
decision, not a silent filter.

**Why no default threshold.** Cosine similarity on normalised bge vectors has
no natural cut-off between "relevant" and "not relevant": two chunks sharing
a heading score ~0.83 while an unrelated chunk scores ~0.72, and the gap
shrinks as the collection grows. A fixed threshold would silently drop
legitimate hits on some queries and keep noise on others, so the retriever
returns everything the store ranks and lets the caller decide.

**Citations use `section_heading`.** A citation is `[<file>, p. 22,
<section_heading>]`, rendered from the chunk's stored metadata in one place
(`rag_agent.retrieval.citation.format_citation`). It uses the bare file name
(`Path(source_file).name`), so a renamed source file is reflected without
re-hydrating the chunk, and the section heading is the third element, dropped
entirely when the chunk has none — the bracket never contains an empty slot.
The heading is taken from the payload rather than re-derived, so a hit is
always traceable to exactly what was indexed.

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
