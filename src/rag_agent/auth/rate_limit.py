"""Persistent SQLite login throttling keyed by the direct peer address.

An immediate transaction reserves a request slot before password verification.
The transaction ends before Argon2 work starts; completion then records a failure
or clears prior failures in a second short transaction. SQLite's database-level
writer lock coordinates independent connections and worker processes.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class LoginReservation:
    bucket: str
    window_id: str
    window_seconds: int


@dataclass(frozen=True)
class LoginAdmission:
    reservation: LoginReservation | None
    retry_after: int | None


def login_bucket(request_host: str | None) -> str:
    """Hash the transport peer identifier so raw addresses are not stored in SQLite."""
    host = request_host or "unknown"
    return hashlib.sha256(host.encode("utf-8", errors="replace")).hexdigest()


@contextmanager
def _immediate_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Serialize a short limiter update across SQLite connections and processes."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _prune_locked(
    conn: sqlite3.Connection, *, now: float, window_seconds: int, preserve_bucket: str
) -> None:
    """Prune old windows and excess inactive peers inside the caller's transaction."""
    conn.execute(
        "DELETE FROM login_attempts WHERE window_started <= ?",
        (now - window_seconds,),
    )
    conn.execute(
        "DELETE FROM login_attempts WHERE bucket IN ("
        "SELECT bucket FROM login_attempts WHERE in_flight = 0 "
        "AND bucket != ? ORDER BY window_started DESC LIMIT -1 OFFSET 10000)",
        (preserve_bucket,),
    )


def reserve_login_attempt(
    conn: sqlite3.Connection,
    bucket: str,
    *,
    now: float | None = None,
    window_seconds: int,
    limit: int,
) -> LoginAdmission:
    """Atomically admit a login slot or return its retry delay.

    In-flight verifications count against the limit, preventing simultaneous
    requests from all passing the threshold. Expiring a window invalidates its
    old reservations; a late completion from one cannot alter the new window.
    """
    current = time.time() if now is None else now
    with _immediate_transaction(conn):
        _prune_locked(conn, now=current, window_seconds=window_seconds, preserve_bucket=bucket)
        row = conn.execute(
            "SELECT window_started, failures, in_flight, window_id "
            "FROM login_attempts WHERE bucket = ?",
            (bucket,),
        ).fetchone()
        if row is None:
            window_started, failures, in_flight, window_id = current, 0, 0, uuid.uuid4().hex
            conn.execute(
                "INSERT INTO login_attempts "
                "(bucket, window_started, failures, in_flight, window_id) "
                "VALUES (?, ?, 0, 0, ?)",
                (bucket, current, window_id),
            )
        else:
            window_started, failures, in_flight, window_id = (
                float(row[0]),
                int(row[1]),
                int(row[2]),
                str(row[3]),
            )
            if current - window_started >= window_seconds:
                window_started, failures, in_flight, window_id = (
                    current,
                    0,
                    0,
                    uuid.uuid4().hex,
                )
                conn.execute(
                    "UPDATE login_attempts SET window_started = ?, failures = 0, in_flight = 0, "
                    "window_id = ? WHERE bucket = ?",
                    (current, window_id, bucket),
                )

        if failures + in_flight >= limit:
            wait = min(window_seconds, max(1, int(window_seconds - (current - window_started))))
            return LoginAdmission(None, wait)

        conn.execute(
            "UPDATE login_attempts SET in_flight = in_flight + 1 WHERE bucket = ?", (bucket,)
        )
        return LoginAdmission(LoginReservation(bucket, window_id, window_seconds), None)


def finish_login_attempt(
    conn: sqlite3.Connection,
    reservation: LoginReservation,
    *,
    succeeded: bool,
    now: float | None = None,
) -> None:
    """Release a reservation and atomically account for failure or success."""
    with _immediate_transaction(conn):
        row = conn.execute(
            "SELECT window_started, failures, in_flight, window_id "
            "FROM login_attempts WHERE bucket = ?",
            (reservation.bucket,),
        ).fetchone()
        if row is None:
            return
        if str(row[3]) != reservation.window_id:
            # A verified login still clears the peer's current failures, while a
            # stale failed attempt must not be charged to a newer window.
            if succeeded:
                conn.execute(
                    "UPDATE login_attempts SET window_started = ?, failures = 0 WHERE bucket = ?",
                    (time.time() if now is None else now, reservation.bucket),
                )
            return
        window_started, failures, in_flight = float(row[0]), int(row[1]), int(row[2])
        current = time.time() if now is None else now
        if current - window_started >= reservation.window_seconds:
            # An expired ticket cannot add a failure to a later window.
            conn.execute(
                "DELETE FROM login_attempts WHERE bucket = ? AND window_id = ?",
                (reservation.bucket, reservation.window_id),
            )
            return
        if in_flight <= 0:
            raise RuntimeError("Login reservation accounting is inconsistent")
        if succeeded:
            if in_flight == 1:
                conn.execute(
                    "DELETE FROM login_attempts WHERE bucket = ? AND window_id = ?",
                    (reservation.bucket, reservation.window_id),
                )
            else:
                conn.execute(
                    "UPDATE login_attempts SET window_started = ?, failures = 0, "
                    "in_flight = in_flight - 1 WHERE bucket = ? AND window_id = ?",
                    (current, reservation.bucket, reservation.window_id),
                )
        else:
            next_started = current if failures == 0 else window_started
            conn.execute(
                "UPDATE login_attempts SET window_started = ?, failures = failures + 1, "
                "in_flight = in_flight - 1 WHERE bucket = ? AND window_id = ?",
                (next_started, reservation.bucket, reservation.window_id),
            )
