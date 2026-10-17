# Design Decisions

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

## 10-K structure detection threshold

**Decision:** `_MIN_ITEM_HEADINGS = 3`.

**Why:** Documents with fewer than 3 distinct ITEM headings are treated as non-10-K, where any valid heading (not just ITEM headings) can set `section_heading`. This preserves the original behavior for non-SEC documents while specialising for 10-K filings. The threshold of 3 was chosen because all real 10-Ks have at least 5–6 ITEM headings, and 3 provides a safe margin above any false positives from cover pages or table-of-contents.
