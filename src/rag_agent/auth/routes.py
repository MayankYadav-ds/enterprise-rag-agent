"""Authentication endpoints backed by opaque server-side sessions."""

import sqlite3
from datetime import timedelta
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_agent.api.runtime import Runtime
from rag_agent.auth.csrf import generate_csrf_token, hash_csrf
from rag_agent.auth.dependencies import User, csrf_required, get_current_user, get_runtime
from rag_agent.auth.models import (
    create_session,
    delete_session,
    get_user_by_username,
    set_csrf_hash,
)
from rag_agent.auth.password import dummy_password_hash, is_argon2id_hash, verify_password
from rag_agent.auth.rate_limit import finish_login_attempt, login_bucket, reserve_login_attempt

router = APIRouter(prefix="/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=1024)


def _request_origin(request: Request) -> str | None:
    origin = request.headers.get("origin")
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme in {"http", "https"} and parsed.netloc and parsed.path in {"", "/"}:
            return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return None
    referer = request.headers.get("referer")
    if referer:
        parsed = urlsplit(referer)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    return None


@router.post("/login")
def login(
    request: Request,
    payload: LoginRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
):
    """Validate a trusted browser origin, then issue a session and CSRF token."""
    origin = _request_origin(request)
    if origin is None or origin not in runtime.settings.trusted_origins:
        raise HTTPException(status_code=400, detail="Untrusted Origin")

    bucket = login_bucket(request.client.host if request.client else None)
    try:
        with runtime.connection() as conn:
            admission = reserve_login_attempt(
                conn,
                bucket,
                window_seconds=runtime.settings.login_rate_limit_window_seconds,
                limit=runtime.settings.login_rate_limit_attempts,
            )
            if admission.reservation is not None:
                user = get_user_by_username(conn, payload.username)
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503,
            detail="Authentication temporarily unavailable",
            headers={"Retry-After": "1"},
        ) from exc

    if admission.reservation is None:
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts",
            headers={"Retry-After": str(admission.retry_after or 1)},
        )

    reservation = admission.reservation
    password_hash = user[1] if user and is_argon2id_hash(user[1]) else dummy_password_hash()
    password_valid = verify_password(password_hash, payload.password)
    authenticated = user is not None and password_valid

    try:
        with runtime.connection() as conn:
            finish_login_attempt(conn, reservation, succeeded=authenticated)
            response_data = (
                create_session(conn, user[0], lifetime=timedelta(hours=8))
                if authenticated and user is not None
                else None
            )
    except sqlite3.Error as exc:
        # Do not issue a session if accounting could not be durably completed.
        raise HTTPException(
            status_code=503,
            detail="Authentication temporarily unavailable",
            headers={"Retry-After": "1"},
        ) from exc

    if not authenticated or response_data is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    raw_token, csrf_token = response_data

    response = JSONResponse(status_code=200, content={"csrf_token": csrf_token})
    response.set_cookie(
        key="session",
        value=raw_token,
        httponly=True,
        secure=runtime.settings.environment == "production",
        samesite="lax",
        path="/",
        max_age=8 * 60 * 60,
    )
    return response


@router.get("/me")
def me(user: Annotated[User, Depends(get_current_user)]) -> dict[str, str | bool]:
    """Return minimal information for the authenticated session."""
    return {"id": user.id, "username": user.username, "is_admin": user.is_admin}


@router.post("/csrf")
def refresh_csrf(
    request: Request,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    user: Annotated[User, Depends(get_current_user)],
) -> dict[str, str]:
    """Issue a fresh in-memory CSRF token after a page reload."""
    origin = _request_origin(request)
    if origin is None or origin not in runtime.settings.trusted_origins:
        raise HTTPException(status_code=400, detail="Untrusted Origin")
    token = generate_csrf_token()
    with runtime.connection() as conn:
        if not set_csrf_hash(conn, user.session_hash, hash_csrf(token)):
            raise HTTPException(status_code=401, detail="Invalid session")
    return {"csrf_token": token}


@router.post("/logout")
def logout(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    user: Annotated[User, Depends(get_current_user)],
    _: Annotated[None, Depends(csrf_required)],
) -> JSONResponse:
    """Revoke the server-side session and expire its browser cookie."""
    with runtime.connection() as conn:
        delete_session(conn, user.session_hash)
    response = JSONResponse(status_code=200, content={"message": "Logged out"})
    response.delete_cookie(
        key="session",
        path="/",
        httponly=True,
        secure=runtime.settings.environment == "production",
        samesite="lax",
    )
    return response
