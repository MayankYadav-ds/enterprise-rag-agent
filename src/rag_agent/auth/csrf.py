"""CSRF token utilities.

Implemented functions:

* ``generate_csrf_token`` – Returns a new random string that can be sent to the front‑end.
* ``hash_csrf`` – Returns a SHA‑256 hex digest of the supplied CSRF token.

The raw CSRF token is never stored in the database; only its hash is kept.
"""

import hashlib
import secrets

__all__ = ["generate_csrf_token", "hash_csrf"]


def generate_csrf_token() -> str:
    """Return a new CSRF token using ``secrets.token_urlsafe``.

    The returned token is a URL‑safe base64 string typically 27‑34 characters.
    """
    return secrets.token_urlsafe(32)


def hash_csrf(token: str) -> str:
    """Return the SHA‑256 hex digest of the CSRF token.

    This hash is stored in the ``sessions`` table.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
