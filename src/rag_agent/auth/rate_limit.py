"""Persistent, bounded-window login throttling keyed by the direct peer address."""

import hashlib
import sqlite3
import time


def login_bucket(request_host: str | None) -> str:
    """Hash the transport peer identifier so raw addresses are not stored in SQLite."""
    host = request_host or "unknown"
    return hashlib.sha256(host.encode("utf-8", errors="replace")).hexdigest()


def retry_after(
    conn: sqlite3.Connection,
    bucket: str,
    *,
    now: float | None = None,
    window_seconds: int,
    limit: int,
) -> int | None:
    """Return remaining block time when the bucket reached its failure limit."""
    current = time.time() if now is None else now
    row = conn.execute(
        "SELECT window_started, failures FROM login_attempts WHERE bucket = ?", (bucket,)
    ).fetchone()
    if row is None:
        return None
    window_started, failures = float(row[0]), int(row[1])
    if current - window_started >= window_seconds:
        conn.execute("DELETE FROM login_attempts WHERE bucket = ?", (bucket,))
        return None
    if failures < limit:
        return None
    return min(window_seconds, max(1, int(window_seconds - (current - window_started))))


def record_failure(
    conn: sqlite3.Connection,
    bucket: str,
    *,
    now: float | None = None,
    window_seconds: int,
) -> None:
    """Increment this peer's failure window and prune expired/overflow buckets."""
    current = time.time() if now is None else now
    conn.execute(
        "DELETE FROM login_attempts WHERE window_started <= ?", (current - window_seconds,)
    )
    row = conn.execute(
        "SELECT window_started, failures FROM login_attempts WHERE bucket = ?", (bucket,)
    ).fetchone()
    if row is None or current - float(row[0]) >= window_seconds:
        conn.execute(
            "INSERT INTO login_attempts (bucket, window_started, failures) VALUES (?, ?, 1) "
            "ON CONFLICT(bucket) DO UPDATE SET window_started=excluded.window_started, failures=1",
            (bucket, current),
        )
    else:
        conn.execute(
            "UPDATE login_attempts SET failures = failures + 1 WHERE bucket = ?", (bucket,)
        )
    # Bound storage if a large number of distinct peers fail within one window.
    conn.execute(
        "DELETE FROM login_attempts WHERE bucket IN ("
        "SELECT bucket FROM login_attempts ORDER BY window_started DESC LIMIT -1 OFFSET 10000)"
    )


def clear_failures(conn: sqlite3.Connection, bucket: str) -> None:
    conn.execute("DELETE FROM login_attempts WHERE bucket = ?", (bucket,))
