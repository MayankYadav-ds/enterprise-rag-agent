"""Authentication and authorization dependencies.

This module contains FastAPI dependency callables that are used by the
application routes to enforce authentication, CSRF protection and, where
appropriate, administrative entitlement.

The runtime instance is obtained through ``get_runtime`` which pulls the
``Runtime`` object stored on ``app.state.runtime`` by ``create_app``.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from rag_agent.api.runtime import Runtime
from rag_agent.auth.csrf import hash_csrf
from rag_agent.auth.models import (
    _hash_token,
    delete_session,
    get_csrf_hashes,
    get_session,
    get_user_by_id,
)

# ----- Runtime helper ----------------------------------------------------


def get_runtime(request: Request) -> Runtime:
    """Return the ``Runtime`` instance from the application state.

    This function is intended to be used as a FastAPI dependency.
    """
    return request.app.state.runtime


# ----- User model --------------------------------------------------------
@dataclass
class User:
    id: str
    username: str
    is_admin: bool
    session_hash: str


# ----- Core authentication dependency ------------------------------------


def get_current_user(request: Request, runtime: Annotated[Runtime, Depends(get_runtime)]) -> User:
    """Return the currently authenticated user.

    Looks for the ``session`` cookie, verifies its associated row in the
    SQLite database, expires if past the ``expires_at`` timestamp, and then
    returns a :class:`User` instance.
    """
    raw_token: str | None = request.cookies.get("session")
    if not raw_token:
        raise HTTPException(status_code=401, detail="Authentication required")

    token_hash = _hash_token(raw_token)
    failure: str | None = None
    authenticated: User | None = None
    with runtime.connection() as conn:
        session = get_session(conn, token_hash)
        if session is None:
            failure = "Invalid session"
        else:
            stored_hash, user_id, _, expires = session
            if datetime.now(UTC) >= expires:
                delete_session(conn, stored_hash)
                failure = "Session expired"
            else:
                user_row = get_user_by_id(conn, user_id)
                if user_row is None:
                    failure = "User not found"
                else:
                    username, is_admin = user_row
                    authenticated = User(
                        id=user_id,
                        username=username,
                        is_admin=is_admin,
                        session_hash=stored_hash,
                    )
    if authenticated is None:
        raise HTTPException(status_code=401, detail=failure or "Authentication required")
    return authenticated


# ----- Admin enforcement -------------------------------------------------


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    """Allow the request only if the authenticated user is an administrator."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Administrator privilege required")
    return user


# ----- CSRF validation ---------------------------------------------------


def csrf_required(
    request: Request,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    user: Annotated[User, Depends(get_current_user)],
) -> None:
    """Ensure the ``X‑CSRF‑Token`` header matches the stored CSRF hash.

    The ``X‑CSRF‑Token`` header must be present on all state‑changing POST,
    DELETE or PATCH requests.  The header value is hashed (just once) and
    compared using :func:`hmac.compare_digest` to the value stored in
    ``sessions``.  On mismatch a ``403`` error is returned.
    """
    header = request.headers.get("X-CSRF-Token")
    if not header:
        raise HTTPException(status_code=403, detail="Missing CSRF token")

    presented_hash = hash_csrf(header)
    with runtime.connection() as conn:
        stored_hashes = get_csrf_hashes(conn, user.session_hash)
    if not stored_hashes:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    if not any(hmac.compare_digest(presented_hash, stored) for stored in stored_hashes):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")


# End of file
