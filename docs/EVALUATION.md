# Evaluation

> **Historical-results notice:** The retained gold set has known relevance-label
> and corpus-coverage limitations; IDs resolving successfully does not establish
> answer relevance. Q13/Q17/Q19/Q20 require adjudication. No labels were changed
> during finalization. Historical BM25/hybrid/reranker runs predate the correction
> from raw IDF odds to `log(1 + odds)` and are not current-code benchmarks.
> See [artifact provenance](../data/eval/README.md). No new live evaluation was run.

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
  recorded in every report, which identifies the labels; corpus, models, filters, and candidate depth
  must also match for a controlled comparison.

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
`rag_chunks` collection (542 points). ID existence was checked, but subsequent source review found semantic relevance
issues; these labels must not be described as fully validated.

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
    --output data/eval/dense_new_run.json \
    --text-output data/eval/dense_new_run.txt
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
| **Precision** | 0.100 | 0.083 | 0.060 | 0.055 | — |
| **MRR** | — | — | — | — | 0.220 |

**Per-query highlights** (from `data/eval/baseline_dense.json`):

- **7 of 20 queries achieved Recall@10 = 1.0** (all relevant chunks retrieved in top-10).
- **8 of 20 queries had Recall@10 = 0.0** — the relevant chunk was not in the
  top-10 results at all. These are typically questions where the relevant
  chunk's embedding is far from the query embedding.
- Questions about financial figures (revenue, ARR, market risk) performed
  moderately; questions about risk factor summaries and IP licensing performed
  poorly.

### Multi-strategy comparison (2026-10-28)

With the evaluation runner extended to support `--strategy dense|bm25|hybrid`,
all three strategies were run against the same live `rag_chunks` collection
(542 points) and the same 20-query gold set (`b4b9bd873762`), top-k 10,
K values 1, 3, 5, 10. Results are saved separately from the historical
baseline:

| Report | Strategy | File |
|---|---|---|
| Dense (baseline) | bge-small-en-v1.5 | `data/eval/baseline_dense.json` |
| Dense (latest) | bge-small-en-v1.5 | `data/eval/dense_latest.json` |
| BM25 | pure-Python BM25 | `data/eval/bm25_latest.json` |
| Hybrid RRF | dense + BM25 fusion | `data/eval/hybrid_latest.json` |

#### Measured results

| Strategy | R@1 | R@3 | R@5 | R@10 | P@1 | P@3 | P@5 | P@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| Dense (baseline) | 0.100 | 0.250 | 0.300 | 0.525 | 0.100 | 0.083 | 0.060 | 0.055 | 0.220 |
| Dense (latest) | 0.100 | 0.250 | 0.300 | 0.525 | 0.100 | 0.083 | 0.060 | 0.055 | 0.220 |
| BM25 | 0.200 | 0.250 | 0.300 | 0.450 | 0.200 | 0.083 | 0.060 | 0.045 | 0.248 |
| Hybrid RRF | 0.150 | 0.300 | 0.400 | 0.550 | 0.150 | 0.100 | 0.080 | 0.055 | 0.272 |

**Key observations:**

- **Dense is unchanged.** The latest dense run reproduces the historical
  baseline exactly (Recall@10 0.525, MRR 0.220), confirming the collection
  and gold set are stable.
- **BM25 improves Recall@1 (0.10 → 0.20) and MRR (0.220 → 0.248)**, but
  loses Recall@10 (0.525 → 0.450). BM25 is stronger at ranking the *first*
  relevant hit but weaker at covering *all* relevant chunks in the top-10.
- **Hybrid RRF is the best overall.** It lifts Recall@3, @5 and @10 above
  both single strategies and achieves the highest MRR (0.272). The fusion
  combines BM25's strong top-1 ranking with dense's broad coverage.
- **Precision follows the same pattern** as recall: hybrid leads at @3 and
  @5, BM25 leads at @1, dense leads at @10.

#### Score semantics

- **Dense** scores are cosine similarity in `[-1, 1]` (bge-small-en-v1.5).
- **BM25** scores are unbounded BM25 contributions; they are **not**
  comparable to cosine similarity and must not be thresholded with a dense
  `min_score` value.
- **Hybrid** scores are sums of RRF contributions: `k/(k+rank_dense) +
  k/(k+rank_bm25)`, living in `(0, 2)`. They are rank-based fusion scores, not
  similarities. When `min_score` is passed to a hybrid retriever it is
  forwarded to each component *before* fusion, so it filters dense hits by
  cosine similarity and BM25 hits by BM25 score independently.

#### How to run

```bash
# Start Qdrant (Docker)
docker compose up -d

# Dense (default)
python -m rag_agent.evaluation.runner \
    --qdrant-url "http://localhost:6333" \
    --strategy dense \
    --top-k 10 --ks "1,3,5,10" \
    --output data/eval/dense_latest.json

# BM25
python -m rag_agent.evaluation.runner \
    --qdrant-url "http://localhost:6333" \
    --strategy bm25 \
    --top-k 10 --ks "1,3,5,10" \
    --output data/eval/bm25_latest.json

# Hybrid RRF
python -m rag_agent.evaluation.runner \
    --qdrant-url "http://localhost:6333" \
    --strategy hybrid \
    --top-k 10 --ks "1,3,5,10" \
    --output data/eval/hybrid_latest.json
```

### Reproducibility

The baseline report (`data/eval/baseline_dense.json`) records:

- `dataset_version`: `b4b9bd873762`
- `num_queries`: 20
- `retrieval_settings`: `{top_k: 10}`
- Per-query recalls, precisions, MRR, and retrieved vs. relevant IDs.

Re-running the same command against the same collection and gold set produces
comparable results only when model revisions, corpus, filters, and candidate
depth are also fixed; approximate search and software versions can affect rankings.

### Phase 5.4 reranking metadata

New reports retain the existing metric fields and `dataset_version`, and record
the actual `strategy`, `reranked` state, `rerank_model`, `rerank_candidates`, and
final `top_k` in `retrieval_settings`. Enabled reranking also records its device,
batch size, and score threshold. The gold-set version hashes the supplied file,
including a custom `--gold-set` path.

Both search and evaluation accept `--rerank-model`. Model names resolve when
configuration is created: explicit argument, then `RERANKER_MODEL`, then
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Model loading remains deferred until
reranking has candidates. Environment settings supply defaults; explicit
configuration overrides take precedence.

Historical reports are preserved. Missing strategy, model, candidate count, or
reranking state is reported as unknown rather than inferred from current
defaults. Disabled reranking records `null` model and candidate-count values.

NaN, positive infinity, and negative infinity from the cross-encoder raise a
`ValueError` identifying the affected chunk before filtering or ranking. Invalid
scores are never converted into hits or returned as a partial ranking.

For comparisons, retain the same gold set and final top-k and record candidate
depth. Hybrid retrieval fetches the requested number of hits from each component
before fusion: reranking 20 candidates uses component depth 20, while a top-10
hybrid baseline uses depth 10. A difference between those runs includes the
effect of the larger candidate pool.

### Real reranking evaluation (2026-10-30)

The cached `cross-encoder/ms-marco-MiniLM-L-6-v2` model was evaluated offline
against gold set `b4b9bd873762` (20 queries) and the existing 542-point
`rag_chunks` collection, using 20 hybrid candidates and final top-k 10. The
fresh hybrid baseline reproduces the saved hybrid metrics. New reports record
model revisions, corpus and gold-set hashes, library versions, and settings:

- [Fresh hybrid baseline](../data/eval/hybrid_phase54_baseline.json)
- [Reranked hybrid JSON](../data/eval/reranked_hybrid_latest.json)
- Historical text copies are retained locally; JSON reports are the archived deliverables.

| Metric | Hybrid baseline | Reranked hybrid | Difference |
|---|---:|---:|---:|
| Recall@1 | 0.150 | 0.125 | -0.025 |
| Recall@3 | 0.300 | 0.375 | +0.075 |
| Recall@5 | 0.400 | 0.425 | +0.025 |
| Recall@10 | 0.550 | 0.625 | +0.075 |
| Precision@1 | 0.150 | 0.150 | 0.000 |
| Precision@3 | 0.100 | 0.133333 | +0.033333 |
| Precision@5 | 0.080 | 0.090 | +0.010 |
| Precision@10 | 0.055 | 0.065 | +0.010 |
| MRR | 0.272222 | 0.291250 | +0.019028 |

These differences include the change from component depth 10 to 20 and do not
isolate reranking alone. Recall@1 decreased even though later-cutoff recall and
MRR increased. The payload corpus fingerprint was unchanged before and after
the run, and existing evaluation files and gold annotations were preserved.

## Limitations

- **Small gold set.** 20 queries provide only an exploratory comparison, especially
  with unresolved label quality; they do not support strong statistical claims. A larger set (50–100
  queries) is planned.
- **Single dense baseline.** Only bge-small-en-v1.5 dense retrieval is
  measured. Hybrid retrieval (BM25 + dense fusion) and cross-encoder reranking
  are implemented, but historical metrics predate the BM25 correction.
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
- Re-evaluate implemented retrieval strategies only after labels and corpus coverage
  are adjudicated; do not assume reranking improves quality.
- Collect RAGAS metrics once answer generation is implemented.

## Controlled historical comparison

The preserved controlled reports use identical captured hybrid pools, depth 20
for both components, final top-k 10, and the same gold set. Baseline MRR is
0.314167 versus 0.291250 reranked; Recall@10 is 0.625 for both. This isolates
reranking better than the earlier depth-10/depth-20 comparison, but retains the
old BM25 formula and unresolved gold labels. It is not a validated quality claim.

Always choose unused output paths: the runner currently overwrites a supplied
report path. The simple CLI does not expose independent baseline component depth;
controlled comparisons require an explicit candidate-pool setup rather than
assuming `--top-k` and `--rerank-candidates` are equivalent. Saved controlled
artifacts contain those pools; the custom capture script is not a shipped CLI.
