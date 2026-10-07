"""Quality helpers for comparing chunking strategies."""

from __future__ import annotations

_TRAILING_CLOSERS = "\"'”’)]}"


def chunk_ends_mid_sentence(text: str) -> bool:
    """Return True when a text chunk does not finish a sentence.

    A chunk is complete if it ends with ``.``, ``!``, or ``?``, optionally
    followed by a closing quote or parenthesis. Table chunks should be excluded
    by the caller.
    """
    stripped = text.strip()
    if not stripped:
        return False
    while stripped and stripped[-1] in _TRAILING_CLOSERS:
        stripped = stripped[:-1].rstrip()
    return not stripped or stripped[-1] not in ".!?"


def chunk_ends_with_complete_sentence(text: str) -> bool:
    """Return True when a text chunk finishes with a sentence terminator.

    This is the complement of :func:`chunk_ends_mid_sentence` and is the
    headline metric for chunk-boundary quality. Table chunks should be
    excluded by the caller, as they end with a table row, not prose.
    """
    return not chunk_ends_mid_sentence(text)
