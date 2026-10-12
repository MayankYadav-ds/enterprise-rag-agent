#!/bin/bash
set -eu

STRATEGY="${1:-recursive}"
QDRANT_URL="${2:-http://qdrant:6333}"
SKIP_SECOND_RUN="${3:-false}"

echo "===== RUN 1 (recreate) ====="
python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install --quiet -e ".[embeddings]"
python scripts/index_documents.py --input data/raw --strategy "${STRATEGY}" --recreate --qdrant-url "${QDRANT_URL}"
python scripts/smoke_queries.py --collection rag_chunks --qdrant-url "${QDRANT_URL}"

if [[ "${SKIP_SECOND_RUN}" != "true" ]]; then
    echo "===== RUN 2 (idempotent) ====="
    python -m pip install --quiet -e ".[embeddings]"
    python scripts/index_documents.py --input data/raw --strategy "${STRATEGY}" --qdrant-url "${QDRANT_URL}"
    python scripts/smoke_queries.py --collection rag_chunks --qdrant-url "${QDRANT_URL}"
else
    echo "SKIP: second run (--skip-second-run)"
fi
