# Chunking Strategy Comparison

Comparison of three chunking strategies on real PDF documents.

| Strat | Chunks | Mean | Med | P95 | %OldMid | %NewComp | %NewInd | Tbl | Time |
|--------|----------|------|------|-----|------------|------------|------------|------|--------|
| fixed_size | 444 | 386.7 | 469.0 | 510.0 | 17.2% | 82.8% | 82.8% | 13 | 0.17 |
| recursive | 441 | 384.5 | 469.0 | 510.0 | 17.3% | 82.7% | 82.7% | 13 | 0.40 |
| structure_aware | 509 | 372.6 | 466.0 | 512.0 | 19.4% | 80.6% | 80.6% | 13 | 0.50 |

## Metric notes

- **%OldMid** is `chunk_ends_mid_sentence` (text does not end in `. ! ?`, optionally followed by a closing quote/paren). Table chunks are excluded.
- **%NewComp** is `chunk_ends_with_complete_sentence`, the complement of OLD.
- **%NewInd** is the same metric computed independently by `report_metrics.py` on the same chunks.
- **OLD and NEW are identical in every row.** `%NewComp = 1 - %OldMid` holds exactly because the two helpers are complements; `%NewInd` matches `%NewComp` to the printed precision. The NEW column is kept as the headline boundary-quality metric; OLD is retained only for continuity with earlier reports.

## Per-document breakdown

### NPS Pharmaceuticals 2013 Form 10-K (104 pages, 0 low-quality pages, 6 tables)

| Strat | Chunks | Mean | Med | P95 | %OldMid | %NewComp | %NewInd | Tbl | Time |
|--------|----------|------|------|-----|------------|------------|------------|------|--------|
| fixed_size | 242 | 390.8 | 474.0 | 514.0 | 11.4% | 88.6% | 88.6% | 6 | 0.10 |
| recursive | 240 | 385.5 | 470.5 | 511.0 | 11.5% | 88.5% | 88.5% | 6 | 0.23 |
| structure_aware | 296 | 365.6 | 459.5 | 512.0 | 15.5% | 84.5% | 84.5% | 6 | 0.29 |

### N-able 2024 Annual Financial Report (125 pages, 44 low-quality pages, 11 tables)

| Strat | Chunks | Mean | Med | P95 | %OldMid | %NewComp | %NewInd | Tbl | Time |
|--------|----------|------|------|-----|------------|------------|------------|------|--------|
| fixed_size | 202 | 381.8 | 466.0 | 508.0 | 24.1% | 75.9% | 75.9% | 7 | 0.07 |
| recursive | 201 | 383.4 | 466.0 | 509.0 | 24.2% | 75.8% | 75.8% | 7 | 0.19 |
| structure_aware | 213 | 382.4 | 468.0 | 510.0 | 24.8% | 75.2% | 75.2% | 7 | 0.22 |

## Example Chunks

### Text Chunk Example

**Chunk ID**: `chunk_doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669_71688c38_65dc36646be487a5`

**Document ID**: `doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669`

**Chunk Type**: `text`

**Page Range**: 1-1

**Section Heading**: N/A

**Token Count**: 451

**Text Quality**: 1.0

**Chunk Index**: 0

**Text Preview**:

```
UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-K ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 For the fiscal year ended December 31, 2013 Commission File Number 0-23272 NPS PHARMACEUTICALS, INC. (Exact Name of Registrant as Specified in Its Charter) Delaware 87-0439579 (State or Other jurisdiction of (I.R.S. Employer Incorporation or Organization) Identification No.) 550 Hills Drive, 3rd Floor, Bedminster, New Jersey 07921 ...
```

### Table Chunk Example

**Chunk ID**: `chunk_doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669_4f56898e_c43ed4bab4275f72`

**Document ID**: `doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669`

**Chunk Type**: `table`

**Page Range**: 21-21

**Section Heading**: N/A

**Token Count**: 111

**Text Quality**: 1.0

**Chunk Index**: 52

**Table Preview**:

```
| Territory | General Subject Matter | Expiration |
| --- | --- | --- |
| U.S. | Glucagon-like peptide-2 analogs | 20151 |
| U.S. | GLP-2 formulations | 20221 |
| Europe | Glucagon-like peptide-2 analogs | 20172 |
| Europe | GLP-2 formulations | 2020 |
| Japan | Glucagon-like peptide-2 analogs | 20173 |
| Japan | GLP-2 formulations | 20203 |
```
