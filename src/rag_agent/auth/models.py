"""SQLite authentication models and helpers.

The RAG application already uses a single SQLite file (``inventory.sqlite3``) for
document metadata.  Authentication re‑uses this file and adds two tables:

* ``users`` – stores a UUID primary key, a unique username, an Argon2id password
  hash, an ``is_admin`` flag and a creation timestamp.
* ``sessions`` – stores a SHA‑256 hash of the session token (the raw token is
  never persisted), a foreign key to ``users.id``, creation/expiration timestamps
  and a SHA‑256 hash of a CSRF token.

All helper functions operate on a ``sqlite3.Connection`` obtained via the
``Runtime.connection()`` context manager.  Foreign‑key enforcement is enabled
once per connection (SQLite disables it by default).
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta

from rag_agent.auth.csrf import hash_csrf

# ---------------------------------------------------------------------------
# Table creation – idempotent
# ---------------------------------------------------------------------------


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create ``users`` and ``sessions`` tables if they do not exist.

    The ``users`` table stores authentication credentials.  ``sessions`` stores a
    hash of the session token and a hash of a CSRF token.  Both tables use a
    UNICODE ``TEXT`` primary key compatible with UUID hex strings.
    """
    # Enable foreign‑key enforcement for this connection.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            csrf_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS csrf_tokens (
            session_hash TEXT NOT NULL REFERENCES sessions(token_hash) ON DELETE CASCADE,
            csrf_hash TEXT NOT NULL,
            PRIMARY KEY (session_hash, csrf_hash)
        );
        """
    )


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------


def create_user(
    conn: sqlite3.Connection,
    username: str,
    password_hash: str,
    *,
    is_admin: bool = False,
) -> str:
    """Insert a new user and return its ``id`` (a UUID4 hex string)."""
    if not isinstance(password_hash, str) or not password_hash.startswith("$argon2id$"):
        raise ValueError("password_hash must be an Argon2id encoded hash")
    user_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO users (id, username, password_hash, is_admin, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, username, password_hash, int(is_admin), datetime.now(UTC).isoformat()),
    )
    return user_id


def get_user_by_username(conn: sqlite3.Connection, username: str) -> tuple[str, str, bool] | None:
    """Return ``(id, password_hash, is_admin)`` for *username* or ``None``."""
    row = conn.execute(
        "SELECT id, password_hash, is_admin FROM users WHERE username = ?", (username,)
    ).fetchone()
    if row is None:
        return None
    user_id, pw_hash, is_admin_int = row
    return user_id, pw_hash, bool(is_admin_int)


def get_user_by_id(conn: sqlite3.Connection, user_id: str) -> tuple[str, bool] | None:
    """Return ``(username, is_admin)`` for *user_id* or ``None``."""
    row = conn.execute("SELECT username, is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    username, is_admin_int = row
    return username, bool(is_admin_int)


# ---------------------------------------------------------------------------
# Session helpers – raw token never persisted
# ---------------------------------------------------------------------------


def _hash_token(token: str) -> str:
    """Hash an opaque raw session token once before database lookup/storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(
    conn: sqlite3.Connection,
    user_id: str,
    *,
    lifetime: timedelta = timedelta(hours=8),
) -> tuple[str, str]:
    """Create a new session for *user_id*.

    Returns a tuple ``(raw_token, csrf_token)``.  The raw token is sent to the
    client in a cookie; the ``csrf_token`` is returned to the client (e.g. in the
    login response).  Both are stored as SHA‑256 hashes in the ``sessions`` table.
    """
    raw_token = secrets.token_urlsafe(48)
    token_hash = _hash_token(raw_token)
    csrf_token = secrets.token_urlsafe(32)
    csrf_hash = hash_csrf(csrf_token)
    now = datetime.now(UTC)
    expires = now + lifetime
    conn.execute(
        "INSERT INTO sessions (token_hash, user_id, csrf_hash, created_at, expires_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (token_hash, user_id, csrf_hash, now.isoformat(), expires.isoformat()),
    )
    conn.execute(
        "INSERT INTO csrf_tokens (session_hash, csrf_hash) VALUES (?, ?)",
        (token_hash, csrf_hash),
    )
    return raw_token, csrf_token


def get_session(
    conn: sqlite3.Connection, token_hash: str
) -> tuple[str, str, datetime, datetime] | None:
    """Lookup a session by its already-hashed token.

    Returns ``(session_token_hash, user_id, created_at, expires_at)`` or ``None``.
    """
    row = conn.execute(
        "SELECT token_hash, user_id, created_at, expires_at FROM sessions WHERE token_hash = ?",
        (token_hash,),
    ).fetchone()
    if row is None:
        return None
    token_hash, user_id, created_iso, expires_iso = row
    created = datetime.fromisoformat(created_iso).astimezone(UTC)
    expires = datetime.fromisoformat(expires_iso).astimezone(UTC)
    return token_hash, user_id, created, expires


def get_csrf_hash(conn: sqlite3.Connection, token_hash: str) -> str | None:
    row = conn.execute(
        "SELECT csrf_hash FROM csrf_tokens WHERE session_hash = ? ORDER BY rowid DESC LIMIT 1",
        (token_hash,),
    ).fetchone()
    if row:
        return row[0]
    row = conn.execute(
        "SELECT csrf_hash FROM sessions WHERE token_hash = ?", (token_hash,)
    ).fetchone()
    return row[0] if row else None


def get_csrf_hashes(conn: sqlite3.Connection, token_hash: str) -> list[str]:
    """Return recent CSRF hashes for this session, supporting multiple tabs."""
    rows = conn.execute(
        "SELECT csrf_hash FROM csrf_tokens WHERE session_hash = ? ORDER BY rowid DESC LIMIT 16",
        (token_hash,),
    ).fetchall()
    if rows:
        return [row[0] for row in rows]
    legacy = get_csrf_hash(conn, token_hash)
    return [legacy] if legacy else []


def set_csrf_hash(conn: sqlite3.Connection, token_hash: str, csrf_hash: str) -> bool:
    """Add a CSRF verifier for this session; retain the 16 latest tab tokens."""
    exists = conn.execute("SELECT 1 FROM sessions WHERE token_hash = ?", (token_hash,)).fetchone()
    if exists is None:
        return False
    conn.execute(
        "INSERT OR IGNORE INTO csrf_tokens (session_hash, csrf_hash) VALUES (?, ?)",
        (token_hash, csrf_hash),
    )
    conn.execute(
        "DELETE FROM csrf_tokens WHERE session_hash = ? AND csrf_hash NOT IN "
        "(SELECT csrf_hash FROM csrf_tokens WHERE session_hash = ? "
        "ORDER BY rowid DESC LIMIT 16)",
        (token_hash, token_hash),
    )
    return True


def delete_session(conn: sqlite3.Connection, token_hash: str) -> None:
    conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


def rotate_session(
    conn: sqlite3.Connection, old_token_hash: str, new_lifetime: timedelta = timedelta(hours=8)
) -> tuple[str, str]:
    """Replace an existing session with a fresh token (used after login).

    The old session record is removed and a new one created with the same user_id.
    Returns ``(new_raw_token, new_csrf_token)``.
    """
    # Fetch the user_id first.
    row = conn.execute(
        "SELECT user_id FROM sessions WHERE token_hash = ?", (old_token_hash,)
    ).fetchone()
    if row is None:
        raise RuntimeError("Attempted to rotate a non‑existent session")
    user_id = row[0]
    # Delete the old session.
    delete_session(conn, old_token_hash)
    # Create a fresh one.
    return create_session(conn, user_id, lifetime=new_lifetime)


# ---------------------------------------------------------------------------
# Initialization helper – called from ``Runtime`` at startup
# ---------------------------------------------------------------------------


def ensure_auth_schema(runtime) -> None:
    """Create auth tables using the same SQLite file as ``DocumentManager``.

    ``runtime`` is the ``Runtime`` instance passed to ``create_app``.  The
    function re‑uses the ``Runtime.connection()`` context manager so the same
    transaction rules apply.
    """
    with runtime.connection() as conn:
        _ensure_tables(conn)
        conn.commit()
