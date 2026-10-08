# Chunking Strategy Comparison

Comparison of three chunking strategies on real PDF documents.

| Strat | Chunks | Mean | Med | P95 | %OldMid | %NewComp | %NewInd | Tbl | Time |
|--------|----------|------|------|-----|------------|------------|------------|------|--------|
| fixed_size | 444 | 386.7 | 469.0 | 510.0 | 17.2% | 82.8% | 82.8% | 13 | 0.16 |
| recursive | 441 | 384.5 | 469.0 | 510.0 | 17.3% | 82.7% | 82.7% | 13 | 0.42 |
| structure_aware | 509 | 372.6 | 466.0 | 512.0 | 19.4% | 80.6% | 80.6% | 13 | 0.48 |

## Metric notes

- **%OldMid** is `chunk_ends_mid_sentence` (text does not end in `. ! ?`, optionally followed by a closing quote/paren). Table chunks are excluded.
- **%NewComp** is `chunk_ends_with_complete_sentence`, the complement of OLD.
- **%NewInd** is the headline metric computed by this script's own metric loop. It was cross-checked against the now-removed `report_metrics.py` on the same chunks and matches `%NewComp` to the printed precision.
- **OLD and NEW are identical in every row.** `%NewComp = 1 - %OldMid` holds exactly because the two helpers are complements; `%NewInd` matches `%NewComp` to the printed precision. The NEW column is kept as the headline boundary-quality metric; OLD is retained only for continuity with earlier reports.

## Per-document breakdown

### Sec Form 10 K Pdf Fixture (104 pages, 0 low-quality pages)

| Strat | Chunks | Mean | Med | P95 | %OldMid | %NewComp | %NewInd | Tbl | Time |
|--------|----------|------|------|-----|------------|------------|------------|------|--------|
| fixed_size | 242 | 390.8 | 474.0 | 514.0 | 11.4% | 88.6% | 88.6% | 6 | 0.08 |
| recursive | 240 | 385.5 | 470.5 | 511.0 | 11.5% | 88.5% | 88.5% | 6 | 0.23 |
| structure_aware | 296 | 365.6 | 459.5 | 512.0 | 15.5% | 84.5% | 84.5% | 6 | 0.27 |

### N Able 2024 Annual Financial Report (125 pages, 44 low-quality pages)

| Strat | Chunks | Mean | Med | P95 | %OldMid | %NewComp | %NewInd | Tbl | Time |
|--------|----------|------|------|-----|------------|------------|------------|------|--------|
| fixed_size | 202 | 381.8 | 466.0 | 508.0 | 24.1% | 75.9% | 75.9% | 7 | 0.07 |
| recursive | 201 | 383.4 | 466.0 | 509.0 | 24.2% | 75.8% | 75.8% | 7 | 0.19 |
| structure_aware | 213 | 382.4 | 468.0 | 510.0 | 24.8% | 75.2% | 75.2% | 7 | 0.21 |


## Example Chunks

### Text Chunk Example

**Chunk ID**: `chunk_doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669_767411dd_18f188de3c985fd9`

**Document ID**: `doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669`

**Chunk Type**: `text`

**Page Range**: 1-1

**Section Heading**: UNITED STATES SECURITIES AND EXCHANGE COMMISSION

**Token Count**: 512

**Text Quality**: 1.0

**Chunk Index**: 1

**Text Preview**:

```
Washington, D.C. 20549 FORM 10-K ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 For the fiscal year ended December 31, 2013 Commission File Number 0-23272 NPS PHARMACEUTICALS, INC. (Exact Name of Registrant as Specified in Its Charter) Delaware 87-0439579 (State or Other jurisdiction of (I.R.S. Employer Incorporation or Organization) Identification No.) 550 Hills Drive, 3rd Floor, Bedminster, New Jersey 07921 (Address of Principal Executive Offices) (Zip Cod...
```

### Table Chunk Example

**Chunk ID**: `chunk_doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669_04265bbe_c43ed4bab4275f72`

**Document ID**: `doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669`

**Chunk Type**: `table`

**Page Range**: 21-21

**Section Heading**: N/A

**Token Count**: 111

**Text Quality**: 1.0

**Chunk Index**: 60

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

## Phase 4 new chunk size 400/39

Comparison of three chunking strategies on real PDF documents.

| Strat | Chunks | Mean | Med | P95 | %OldMid | %NewComp | %NewInd | Tbl | Time |
|--------|----------|------|------|-----|------------|------------|------------|------|--------|
| fixed_size | 548 | 316.4 | 365.5 | 398.0 | 14.2% | 85.8% | 85.8% | 13 | 0.17 |
| recursive | 543 | 314.6 | 367.0 | 398.0 | 14.3% | 85.7% | 85.7% | 13 | 0.38 |
| structure_aware | 638 | 298.6 | 363.0 | 398.0 | 17.1% | 82.9% | 82.9% | 13 | 0.47 |

## Metric notes

- **%OldMid** is `chunk_ends_mid_sentence` (text does not end in `. ! ?`, optionally followed by a closing quote/paren). Table chunks are excluded.
- **%NewComp** is `chunk_ends_with_complete_sentence`, the complement of OLD.
- **%NewInd** is the headline metric computed by this script. `report_metrics.py` independently reproduces the same values on the same chunks; it matches `%NewComp` to the printed precision.
- **OLD and NEW are identical in every row.** `%NewComp = 1 - %OldMid` holds exactly because the two helpers are complements; `%NewInd` matches `%NewComp` to the printed precision. The NEW column is kept as the headline boundary-quality metric; OLD is retained only for continuity with earlier reports.

## Per-document breakdown

### Sec Form 10 K Pdf Fixture (104 pages, 0 low-quality pages)

| Strat | Chunks | Mean | Med | P95 | %OldMid | %NewComp | %NewInd | Tbl | Time |
|--------|----------|------|------|-----|------------|------------|------------|------|--------|
| fixed_size | 303 | 316.3 | 367.0 | 398.0 | 9.8% | 90.2% | 90.2% | 6 | 0.10 |
| recursive | 300 | 311.2 | 367.5 | 398.0 | 9.9% | 90.1% | 90.1% | 6 | 0.21 |
| structure_aware | 374 | 290.7 | 363.0 | 400.0 | 14.7% | 85.3% | 85.3% | 6 | 0.27 |

### N Able 2024 Annual Financial Report (125 pages, 44 low-quality pages)

| Strat | Chunks | Mean | Med | P95 | %OldMid | %NewComp | %NewInd | Tbl | Time |
|--------|----------|------|------|-----|------------|------------|------------|------|--------|
| fixed_size | 245 | 316.5 | 365.0 | 398.0 | 19.7% | 80.3% | 80.3% | 7 | 0.07 |
| recursive | 243 | 318.8 | 366.0 | 398.0 | 19.9% | 80.1% | 80.1% | 7 | 0.17 |
| structure_aware | 264 | 309.9 | 365.5 | 398.0 | 20.6% | 79.4% | 79.4% | 7 | 0.21 |


## Example Chunks

### Text Chunk Example

**Chunk ID**: `chunk_doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669_767411dd_d7a353619d84611d`

**Document ID**: `doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669`

**Chunk Type**: `text`

**Page Range**: 1-1

**Section Heading**: UNITED STATES SECURITIES AND EXCHANGE COMMISSION

**Token Count**: 362

**Text Quality**: 1.0

**Chunk Index**: 1

**Text Preview**:

```
Washington, D.C. 20549 FORM 10-K ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 For the fiscal year ended December 31, 2013 Commission File Number 0-23272 NPS PHARMACEUTICALS, INC. (Exact Name of Registrant as Specified in Its Charter) Delaware 87-0439579 (State or Other jurisdiction of (I.R.S. Employer Incorporation or Organization) Identification No.) 550 Hills Drive, 3rd Floor, Bedminster, New Jersey 07921 (Address of Principal Executive Offices) (Zip Cod...
```

### Table Chunk Example

**Chunk ID**: `chunk_doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669_fe9ea50a_c43ed4bab4275f72`

**Document ID**: `doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669`

**Chunk Type**: `table`

**Page Range**: 21-21

**Section Heading**: N/A

**Token Count**: 111

**Text Quality**: 1.0

**Chunk Index**: 70

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
