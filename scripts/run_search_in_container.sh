#!/bin/bash
# Container-side entry point for scripts/run_in_docker.ps1.
#
# Runs a search command inside the indexing container: install the embeddings
# extra (the bge model needs torch, which Windows Application Control blocks on
# this machine), then exec the command passed on the argv. Output is streamed
# to the host so the caller sees progress instead of a single blob at the end.
set -eu

# The search command is passed through the SEARCH_COMMAND environment variable
# by run_in_docker.ps1, not as argv: a command like
#   python scripts/search.py '...' --top-k 5 --queries-file ...
# is a single shell string, and passing it as one argv element to bash would
# make it $0, leaving "$@" empty. Passing it through SEARCH_COMMAND and
# word-splitting with 'eval set --' lets 'exec "$@"' run it with correct argv.
if [ -n "${SEARCH_COMMAND:-}" ]; then
    eval set -- "$SEARCH_COMMAND"
fi

# "$@" is the command to run, e.g. python scripts/search.py '...' --top-k 5
if [ "$#" -eq 0 ] && [ -z "${SEARCH_COMMAND:-}" ]; then
    echo "usage: $0 <command...> or set SEARCH_COMMAND=<command>" >&2
    exit 2
fi

# Install the embeddings extra (the bge model needs torch, which Windows
# Application Control blocks on this machine). The install is skipped when
# SKIP_EMBEDDINGS_INSTALL is set, so argument-handling tests can exercise the
# script without downloading PyTorch or Sentence Transformers. A real container
# run leaves the flag unset, so the install still happens.
if [ -z "${SKIP_EMBEDDINGS_INSTALL:-}" ]; then
    python -m pip install --quiet -e ".[embeddings]"
fi

exec "$@"
