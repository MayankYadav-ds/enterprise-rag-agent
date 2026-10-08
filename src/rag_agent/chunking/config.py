"""Chunking-specific configuration defaults."""

# Minimum text_quality threshold; pages below this are excluded.
LOW_TEXT_QUALITY_THRESHOLD = 0.900

# Default chunk size and overlap, measured in tiktoken (cl100k_base) tokens.
# These were lowered from 512/50 to 400/39 after measuring the real BAAI
# bge-small-en-v1.5 WordPiece tokenizer on the two filings: at 512/50, 17.3%
# of chunks exceeded the model's 512-token limit and were silently truncated.
# At 400/39 the p99 model-token count is 429 and only 0.2% of chunks exceed
# the limit, leaving a safety margin. See docs/CHUNKING_RESULTS.md.
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 39
DEFAULT_MIN_CHUNK_SIZE = 10
