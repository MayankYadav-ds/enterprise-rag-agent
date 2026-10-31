# Evaluation artifact provenance

These are preserved historical development records, not a validated benchmark
of the final code. The canonical gold set and all existing reports are unchanged.
Known label and corpus-coverage issues prevent strong retrieval-quality claims.
The current BM25 implementation corrects IDF to log(1 + odds); historical BM25,
hybrid, and reranked results used the earlier raw-odds implementation. Do not
compare a new run to these reports as though only reranking changed.

- `gold_set.json`: original 20-query labels, version `b4b9bd873762`.
- `all_chunks.json`: 542-chunk text/metadata manifest from two public SEC filings.
- `baseline_dense.*`: previously tracked dense baseline, retained unchanged.
- `dense_latest.json`, `bm25_latest.json`, `hybrid_latest.json`: historical strategy runs.
  These older JSON files omit strategy metadata; filenames provide their historical
  labels. Seven accompanying local text reports are excluded because their
  generated headers contain trailing whitespace; some also incorrectly say dense.
  All remain untouched locally. The previously tracked baseline text is retained.
- `hybrid_phase54_baseline.json`, `reranked_hybrid_latest.json`: comparison with
  component depth 10 versus 20; does not isolate reranking.
- `hybrid_controlled_baseline.json`, `reranked_hybrid_controlled.json`: same captured
  depth-20 candidate pools, top-k 10, no retrieval filters. Reports record corpus
  hashes, model revisions, CPU device, and configuration. The reranked report
  identifies the real CrossEncoder class. The ad hoc capture script is not part
  of the project CLI, so this is an archived experiment, not a one-command recipe.
- Local `results.json` is deliberately excluded: generic all-zero smoke output
  without sufficient provenance for a quality comparison.

Finalization recomputed every JSON report's per-query and aggregate Recall,
Precision and MRR from its stored rankings and gold labels; they agree. Every
recorded retrieved/relevant chunk ID resolves in the saved manifest. This is
internal consistency verification, not a fresh live evaluation or verification
of model execution. No fake scores establish retrieval quality.

The controlled historical baseline has MRR 0.314167 versus 0.291250 reranked;
Recall@10 is 0.625 for both. The data does not support an improvement claim.
The corpus omits some relevant PDF content and the gold labels need human
adjudication before another benchmark. No candidate from private Temp audits
has been substituted for the canonical gold set.

## SHA256 manifest

Hashes below identify Git-stored bytes (LF line endings after Git normalization).
Working-tree CRLF files can have different raw-byte hashes. Filenames ending in
latest are historical snapshots, not a promise of current-code results.

| File | SHA256 |
| --- | --- |
| `all_chunks.json` | `e941f26152934aefe8068e1024f5dfacd9c2c0b1954f930df788312f5bed46b1` |
| `baseline_dense.json` | `ace8ffa4407746cfc9feb21a9f896e620db3b8de1ea5734f3f831fc759367cc1` |
| `baseline_dense.txt` | `80202d2430e75985fdb08f1b438edfb8f84a7af7dfcd0d672fa9c408fadbedd3` |
| `bm25_latest.json` | `c4a9d20a5bc006f7c5cae56f531289a96bab5c9da0029393cad58c93349c4a51` |
| `dense_latest.json` | `70246b9a0066d8781b30488fe29f59897332140a519a33ccc1ddb65131a619bb` |
| `gold_set.json` | `b4b9bd873762ac59ecc86fdc5d0693f0616a50814ecafbd16206a65430daee80` |
| `hybrid_controlled_baseline.json` | `d3b706d209f2a3bd5f42c13242b3af471f76149a5d290b67cd7bce8a025a22d3` |
| `hybrid_latest.json` | `3f07dc11e885456f3292d024768bf6deba236bd7c52b9007d082ff54abe90c38` |
| `hybrid_phase54_baseline.json` | `3f6769ec0e838ad58f484e6a2d3cfdcfd53faf62315f800ee921a626e155c39a` |
| `reranked_hybrid_controlled.json` | `54e1641caf79cf3806dd99e603c2eb7457558adf79a641fa0169b7b455245bd0` |
| `reranked_hybrid_latest.json` | `5c99d42f9c4844b9621c28d6e298a8457d177346091c27f80bbea9d3705a4d2d` |
