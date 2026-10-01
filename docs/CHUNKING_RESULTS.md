# Chunking Strategy Comparison

Comparison of three chunking strategies on real PDF documents.

| Strategy | Chunk Count | Mean Tokens | Median Tokens | P95 Tokens | % Mid-Sentence | Table Chunks | Runtime (s) |
|----------|-------------|-------------|---------------|------------|----------------|--------------|-------------|
| fixed_size | 584 | 584.5 | 474.5 | 2269.0 | 26.0% | 35 | 0.31 |
| recursive | 588 | 580.9 | 475.0 | 2200.0 | 26.7% | 35 | 0.62 |
| structure_aware | 588 | 580.9 | 475.0 | 2200.0 | 26.7% | 35 | 0.61 |

## Example Chunks

### Text Chunk Example

**Chunk ID**: `chunk_doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669_71688c38_df0880102655c8db`

**Document ID**: `doc_7ad99a6a1e620aa1db7988b662f4d23ae0dd1e0c178c9d5d4bfedb5c16f58669`

**Chunk Type**: `text`

**Page Range**: 1-1

**Section Heading**: N/A

**Token Count**: 469

**Text Quality**: 1.0

**Chunk Index**: 0

**Text Preview**:

```
UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-K  ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 For the fiscal year ended December 31, 2013 Commission File Number 0-23272 NPS PHARMACEUTICALS, INC. (Exact Name of Registrant as Specified in Its Charter) Delaware 87-0439579 (State or Other jurisdiction of (I.R.S. Employer Incorporation or Organization) Identification No.) 550 Hills Drive, 3rd Floor, Bedminster, New Jersey 0792...
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
