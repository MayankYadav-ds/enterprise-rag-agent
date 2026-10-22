#!/bin/bash
# Container-side entry point for scripts/run_in_docker.ps1.
#
# Runs a search command inside the indexing container: install the embeddings
# extra (the bge model needs torch, which Windows Application Control blocks on
# this machine), then exec the command passed on the argv. Output is streamed
# to the host so the caller sees progress instead of a single blob at the end.
set -eu

# "$@" is the command to run, e.g. "python scripts/search.py 'What risks?'"
if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command...>" >&2
    exit 2
fi

python -m pip install --quiet -e ".[embeddings]"
exec "$@"
