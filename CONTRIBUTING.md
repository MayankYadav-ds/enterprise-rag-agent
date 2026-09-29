# Contributing

## Local setup

Use Python 3.11, then install the project with development dependencies:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Run `ruff check .` and `pytest` before opening a pull request. Keep changes focused, typed, documented where public, and covered by meaningful tests.

## Workflow

- Create a branch using `feat/`, `fix/`, `docs/`, `test/`, or `chore/`.
- Follow Conventional Commits, for example `feat(retrieval): add reciprocal rank fusion`.
- Explain the behaviour, validation, and any trade-offs in the pull request description.
- Never add credentials, private documents, or `.env` files.

See the issue tracker for planned work and [docs/DECISIONS.md](docs/DECISIONS.md) for project choices.
