# Evaluation

## Status

The evaluation framework is implemented and the first baseline has been
measured against a live Qdrant collection. RAGAS generation metrics are
pending (Phase 6 answer generation must land first).

## Framework

The evaluation lives in `src/rag_agent/evaluation/`:

| Module | Responsibility |
|---|---|
| `gold_set.py` | `GoldSet` / `GoldQuery` dataclasses, SHA-256 versioning, disk loading |
| `metrics.py` | Pure functions: `PrecisionAtK`, `RecallAtK`, `mean_reciprocal_rank` |
| `runner.py` | `run_evaluation()` driver, CLI, JSON/text report output |

### Design

- **Metrics are pure functions.** Each metric takes a ranked list of chunk IDs
  and a set of gold-relevant chunk IDs, returning a float in `[0, 1]`. They
  never raise on empty input and never depend on any retriever implementation.
- **`run_evaluation`** accepts any object implementing the `Retriever`
  protocol (the `DenseRetriever` in production). It also accepts the gold set
  as a `GoldSet` instance or a plain list of dicts, so unit tests can pass
  inline fixtures without file I/O.
- **Deferred imports.** The CLI's `_build_retriever()` imports
  `sentence-transformers` only at call time so the module remains importable
  without torch installed.
- **Determinism.** The gold set version is the SHA-256 of
  `data/eval/gold_set.json` (truncated to 12 hex chars). The version string is
  recorded in every report, so baselines are directly comparable.

### Metric definitions

- **Recall@K** = (relevant chunks found in top-K) ÷ (total relevant chunks).
  Returns 0.0 when the gold set is empty.
- **Precision@K** = (relevant chunks in top-K) ÷ (number of chunks returned).
  Returns 0.0 when K ≤ 0 or no hits are returned.
- **MRR** = 1 ÷ (rank of first relevant hit). Averaged across all queries;
  queries with no relevant hit contribute 0.

## Gold set

`data/eval/gold_set.json` contains **20 questions** grounded in two SEC filings:

1. **N-able 2024 Annual Report** — 242 chunks
2. **NPS Pharmaceuticals 2013 10-K** — 300 chunks

Each query pairs a natural-language question with the **chunk IDs** (not just
page numbers) that a human annotator verified relevant, plus a `note` citing
the page. All 19 unique chunk IDs were verified against both
`data/eval/all_chunks.json` (the chunk manifest) and the live Qdrant
`rag_chunks` collection (542 points). Every ID resolves to text that actually
answers the question.

The gold IDs are real index IDs, so evaluation is grounded in the actual
index rather than synthetic expectations.

Version: `b4b9bd873762` (SHA-256 of `gold_set.json`).

## How to run evaluation

Start Qdrant (Docker), then run the evaluation CLI:

```bash
# Start the vector database
docker compose up -d

# Run evaluation against the live collection
.venv/Scripts/python -m rag_agent.evaluation.runner \
    --qdrant-url "http://localhost:6333" \
    --ks "1,3,5,10" \
    --top-k 10 \
    --output data/eval/baseline_dense.json \
    --text-output data/eval/baseline_dense.txt
```

The command loads the gold set from `data/eval/gold_set.json`, queries each
question against the `rag_chunks` collection with `top_k=10`, and writes a
machine-readable JSON report plus an optional human-readable text report.

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--gold-set` | `data/eval/gold_set.json` | Path to gold set file |
| `--ks` | `1,3,5,10` | K values for Recall@K and Precision@K |
| `--top-k` | `10` | Hits requested per query (must be ≥ max K) |
| `--collection` | `rag_chunks` | Qdrant collection name |
| `--qdrant-url` | env `QDRANT_URL` or `http://localhost:6333` | Qdrant endpoint |
| `--fake-embedder` | off | Use deterministic fake embedder (testing only) |
| `--output` | `data/eval/results.json` | JSON report path |
| `--text-output` | (none) | Optional text report path |

For offline tests without torch, use `--fake-embedder` with an in-memory
store:

```bash
.venv/Scripts/python -m rag_agent.evaluation.runner \
    --qdrant-url ":memory:" \
    --fake-embedder \
    --gold-set data/eval/gold_set.json
```

## Measured results

### Baseline: dense retrieval (bge-small-en-v1.5)

Run against the live Qdrant collection on 2025-10-26, Qdrant v1.11,
collection `rag_chunks` (542 points).

| Metric | @1 | @3 | @5 | @10 | Mean |
|---|---|---|---|---|---|
| **Recall** | 0.100 | 0.250 | 0.300 | 0.525 | — |
| **Precision** | 0.010 | 0.029 | 0.029 | 0.016 | — |
| **MRR** | — | — | — | — | 0.220 |

**Per-query highlights** (from `data/eval/baseline_dense.json`):

- **7 of 20 queries achieved Recall@10 = 1.0** (all relevant chunks retrieved in top-10).
- **8 of 20 queries had Recall@10 = 0.0** — the relevant chunk was not in the
  top-10 results at all. These are typically questions where the relevant
  chunk's embedding is far from the query embedding.
- Questions about financial figures (revenue, ARR, market risk) performed
  moderately; questions about risk factor summaries and IP licensing performed
  poorly.

### Reproducibility

The baseline report (`data/eval/baseline_dense.json`) records:

- `dataset_version`: `b4b9bd873762`
- `num_queries`: 20
- `retrieval_settings`: `{top_k: 10}`
- Per-query recalls, precisions, MRR, and retrieved vs. relevant IDs.

Re-running the same command against the same collection and gold set produces
identical results (dense retrieval is deterministic given the same query
embedding and Qdrant index).

## Limitations

- **Small gold set.** 20 queries is sufficient for relative comparison between
  retrievers but too small for strong statistical claims. A larger set (50–100
  queries) is planned.
- **Single dense baseline.** Only bge-small-en-v1.5 dense retrieval is
  measured. Hybrid retrieval (BM25 + dense fusion) and cross-encoder reranking
  are not yet implemented.
- **No RAGAS metrics.** Answer generation (Phase 6) and its API (Phase 7) must
  land before faithfulness, answer relevancy, and context precision/recall can
  be computed.
- **No generation-side evaluation.** The gold set measures retrieval quality
  only; whether the answer generator uses the right evidence is not yet tested.
- **Chunk-level relevance.** The gold set is annotated at chunk granularity.
  A query whose answer spans two adjacent chunks will only get credit for the
  one(s) were explicitly annotated.

## Future work

- Expand the gold set to 50–100 queries across more documents.
- Implement BM25 retrieval (Phase 5.2) and reciprocal-rank fusion (Phase 5.3).
- Add cross-encoder reranking (Phase 5.4) and measure the improvement.
- Collect RAGAS metrics once answer generation is implemented.
