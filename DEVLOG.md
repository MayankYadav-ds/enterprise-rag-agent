# Development log

## Phase 1 — Foundation

I set up the package layout and a deliberately small configuration object so later components have one clear place to read non-secret settings. I chose LangChain for the eventual orchestration layer because its retriever and streaming interfaces work well with a FastAPI service, while keeping the project’s retrieval logic testable in plain Python.

The main snags at this stage are external GitHub automation and a Windows application-control restriction: the GitHub CLI is not installed, and its policy blocks pre-commit's bundled YAML checker. I kept the local gate dependable by retaining end-of-file, whitespace, merge-conflict, Ruff lint, and Ruff format hooks; GitHub Actions remains the server-side quality gate. The codebase itself is provider-agnostic and does not require an API key to run the current checks.
