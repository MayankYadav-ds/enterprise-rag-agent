"""Authentication regression tests with per-test SQLite databases."""

import hashlib
import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app
from rag_agent.api.settings import APISettings
from rag_agent.auth.models import (
    _hash_token as model_hash_token,
)
from rag_agent.auth.models import (
    create_session,
    create_user,
    delete_session,
    ensure_auth_schema,
    get_csrf_hash,
    get_csrf_hashes,
    get_session,
)
from rag_agent.auth.password import hash_password, verify_password

_CREATE_USER_PATH = Path(__file__).resolve().parent.parent / "scripts" / "create_user.py"
_CREATE_USER_SPEC = importlib.util.spec_from_file_location(
    "rag_agent_scripts_create_user", _CREATE_USER_PATH
)
assert _CREATE_USER_SPEC is not None and _CREATE_USER_SPEC.loader is not None
_CREATE_USER_MODULE = importlib.util.module_from_spec(_CREATE_USER_SPEC)
sys.modules["rag_agent_scripts_create_user"] = _CREATE_USER_MODULE
_CREATE_USER_SPEC.loader.exec_module(_CREATE_USER_MODULE)
create_user_command = _CREATE_USER_MODULE.main

ORIGIN = "http://localhost:5173"
PASSWORD = "correct horse battery staple 42"


@pytest.fixture
def auth_app(tmp_path):
    app = create_app(APISettings(document_dir=tmp_path / "database", environment="test"))
    runtime = app.state.runtime
    ensure_auth_schema(runtime)
    with runtime.connection() as conn:
        user_id = create_user(conn, "alice", hash_password(PASSWORD), is_admin=True)
    return app, runtime, user_id


@pytest.fixture
def authenticated(auth_app):
    app, runtime, _ = auth_app
    with TestClient(app) as client:
        response = client.post(
            "/auth/login",
            headers={"Origin": ORIGIN},
            json={"username": "alice", "password": PASSWORD},
        )
        assert response.status_code == 200
        cookie_header = response.headers["set-cookie"].lower()
        assert "httponly" in cookie_header and "samesite=lax" in cookie_header
        csrf = response.json()["csrf_token"]
        client.headers["X-CSRF-Token"] = csrf
        yield client, runtime, csrf


def test_argon2id_password_hash_salted_and_verified():
    first = hash_password(PASSWORD)
    second = hash_password(PASSWORD)
    assert first.startswith("$argon2id$")
    assert first != PASSWORD and first != second
    assert verify_password(first, PASSWORD)
    assert not verify_password(first, "incorrect password")
    assert not verify_password(hashlib.sha256(PASSWORD.encode()).hexdigest(), PASSWORD)
    assert not verify_password("not-an-argon-hash", PASSWORD)


def test_user_provisioning_command_stores_only_argon2id(tmp_path, monkeypatch):
    import sys

    from rag_agent.api.runtime import Runtime

    monkeypatch.setenv("RAG_DOCUMENT_DIR", str(tmp_path / "managed"))
    monkeypatch.setenv("RAG_ENV", "test")
    monkeypatch.setattr(sys, "argv", ["create_user.py", "new-user", "--admin"])
    answers = iter([PASSWORD, PASSWORD])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(answers))
    assert create_user_command() == 0
    runtime = Runtime(APISettings(document_dir=tmp_path / "managed"))
    with runtime.connection() as conn:
        row = conn.execute(
            "SELECT password_hash, is_admin FROM users WHERE username = ?", ("new-user",)
        ).fetchone()
    runtime.close()
    assert row and row[0].startswith("$argon2id$") and row[1] == 1
    assert verify_password(row[0], PASSWORD)


def test_session_hash_representation_and_revocation(auth_app):
    _, runtime, user_id = auth_app
    with runtime.connection() as conn:
        raw_token, csrf = create_session(conn, user_id)
        token_hash = model_hash_token(raw_token)
        assert token_hash == hashlib.sha256(raw_token.encode()).hexdigest()
        assert raw_token != token_hash
        stored = conn.execute("SELECT token_hash, csrf_hash FROM sessions").fetchone()
        assert stored == (token_hash, hashlib.sha256(csrf.encode()).hexdigest())
        session = get_session(conn, token_hash)
        assert session and session[0] == token_hash and session[1] == user_id
        assert get_csrf_hash(conn, token_hash) == stored[1]
        delete_session(conn, token_hash)
        assert get_session(conn, token_hash) is None


def test_login_origin_and_generic_credentials(auth_app):
    app, _, _ = auth_app
    with TestClient(app) as client:
        assert (
            client.post("/auth/login", json={"username": "alice", "password": PASSWORD}).status_code
            == 400
        )
        assert (
            client.post(
                "/auth/login",
                headers={"Origin": "https://attacker.invalid"},
                json={"username": "alice", "password": PASSWORD},
            ).status_code
            == 400
        )
        wrong = client.post(
            "/auth/login",
            headers={"Origin": ORIGIN},
            json={"username": "alice", "password": "short"},
        )
        missing = client.post(
            "/auth/login",
            headers={"Origin": ORIGIN},
            json={"username": "missing", "password": "short"},
        )
        assert wrong.status_code == missing.status_code == 401
        assert {k: v for k, v in wrong.json().items() if k != "request_id"} == {
            k: v for k, v in missing.json().items() if k != "request_id"
        }


def test_cookie_flags_me_and_csrf_token_hash(authenticated):
    client, runtime, csrf = authenticated
    cookie = client.cookies.get("session")
    assert cookie
    assert client.cookies.get("session") != model_hash_token(cookie)
    me = client.get("/auth/me")
    assert me.status_code == 200 and me.json()["username"] == "alice"
    with runtime.connection() as conn:
        session = get_session(conn, model_hash_token(cookie))
        assert session
        assert get_csrf_hash(conn, session[0]) == hashlib.sha256(csrf.encode()).hexdigest()


def test_production_cookie_is_secure_and_http_only(tmp_path):
    app = create_app(
        APISettings(
            document_dir=tmp_path / "database",
            environment="production",
            trusted_origins=("https://app.example.com",),
        )
    )
    runtime = app.state.runtime
    ensure_auth_schema(runtime)
    with runtime.connection() as conn:
        create_user(conn, "alice", hash_password(PASSWORD), is_admin=True)
    with TestClient(app, base_url="https://testserver") as client:
        response = client.post(
            "/auth/login",
            headers={"Origin": "https://app.example.com"},
            json={"username": "alice", "password": PASSWORD},
        )
        cookie = response.headers["set-cookie"].lower()
        assert response.status_code == 200
        assert all(flag in cookie for flag in ("httponly", "secure", "samesite=lax", "path=/"))
        assert "max-age=28800" in cookie
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        logout = client.post("/auth/logout")
        cleared = logout.headers["set-cookie"].lower()
        assert logout.status_code == 200
        assert all(
            flag in cleared
            for flag in ("httponly", "secure", "samesite=lax", "path=/", "max-age=0")
        )


def test_development_cookie_is_http_usable_but_restricted(tmp_path):
    app = create_app(
        APISettings(
            document_dir=tmp_path / "database",
            environment="development",
            trusted_origins=(ORIGIN,),
        )
    )
    runtime = app.state.runtime
    ensure_auth_schema(runtime)
    with runtime.connection() as conn:
        create_user(conn, "alice", hash_password(PASSWORD))
    with TestClient(app) as client:
        response = client.post(
            "/auth/login",
            headers={"Origin": ORIGIN},
            json={"username": "alice", "password": PASSWORD},
        )
        cookie = response.headers["set-cookie"].lower()
        assert response.status_code == 200
        assert all(flag in cookie for flag in ("httponly", "samesite=lax", "path=/"))
        assert "secure" not in cookie


def test_expired_session_is_rejected_and_deleted(auth_app):
    app, runtime, user_id = auth_app
    with runtime.connection() as conn:
        token, _ = create_session(conn, user_id, lifetime=timedelta(seconds=-1))
    with TestClient(app) as client:
        client.cookies.set("session", token)
        assert client.get("/auth/me").status_code == 401
    with runtime.connection() as conn:
        assert get_session(conn, model_hash_token(token)) is None


def test_logout_requires_csrf_revokes_session_and_clears_cookie(authenticated):
    client, runtime, csrf = authenticated
    client.headers.pop("X-CSRF-Token")
    assert client.post("/auth/logout").status_code == 403
    client.headers["X-CSRF-Token"] = "bad-token"
    assert client.post("/auth/logout").status_code == 403
    client.headers["X-CSRF-Token"] = csrf
    response = client.post("/auth/logout")
    assert response.status_code == 200
    cleared = response.headers["set-cookie"].lower()
    assert all(flag in cleared for flag in ("httponly", "samesite=lax", "path=/", "max-age=0"))
    assert "secure" not in cleared
    assert client.get("/auth/me").status_code == 401


def test_csrf_refresh_and_api_route_protection(authenticated):
    client, runtime, old_token = authenticated
    assert (
        client.post("/auth/csrf", headers={"Origin": "https://attacker.invalid"}).status_code == 400
    )
    refreshed = client.post("/auth/csrf", headers={"Origin": ORIGIN})
    assert refreshed.status_code == 200
    new_token = refreshed.json()["csrf_token"]
    with runtime.connection() as conn:
        session = get_session(conn, model_hash_token(client.cookies.get("session")))
        assert session
        assert get_csrf_hash(conn, session[0]) == hashlib.sha256(new_token.encode()).hexdigest()
        assert hashlib.sha256(old_token.encode()).hexdigest() in get_csrf_hashes(conn, session[0])
    assert old_token != new_token
    client.headers["X-CSRF-Token"] = new_token
    assert client.post("/auth/logout").status_code == 200


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/v1/retrieve"),
        ("post", "/api/v1/query"),
        ("post", "/api/v1/documents?filename=x.pdf"),
        ("get", "/api/v1/documents"),
        ("get", "/api/v1/documents/doc_" + "a" * 64),
        ("delete", "/api/v1/documents/doc_" + "a" * 64),
        ("post", "/api/v1/documents/recover"),
    ],
)
def test_every_private_api_route_requires_authentication(auth_app, method, path):
    app, _, _ = auth_app
    with TestClient(app) as client:
        needs_question = "/retrieve" in path or "/query" in path
        response = client.request(
            method.upper(), path, json={"question": "x"} if needs_question else None
        )
        assert response.status_code == 401


def test_state_changing_routes_require_csrf(authenticated):
    client, _, _ = authenticated
    client.headers.pop("X-CSRF-Token")
    assert client.post("/api/v1/retrieve", json={"question": "x"}).status_code == 403
    assert client.post("/api/v1/query", json={"question": "x"}).status_code == 403
    assert client.post("/api/v1/documents/recover").status_code == 403


def test_recovery_requires_admin(auth_app):
    app, runtime, _ = auth_app
    with runtime.connection() as conn:
        create_user(conn, "ordinary-user", hash_password(PASSWORD), is_admin=False)
    with TestClient(app) as client:
        login = client.post(
            "/auth/login",
            headers={"Origin": ORIGIN},
            json={"username": "ordinary-user", "password": PASSWORD},
        )
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        assert client.post("/api/v1/documents/recover").status_code == 403


def test_health_is_public(auth_app):
    app, _, _ = auth_app
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
