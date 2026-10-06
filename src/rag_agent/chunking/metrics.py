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
