"""Authentication regression tests with per-test SQLite databases."""

import hashlib
import importlib.util
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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
    reset_owner_password,
)
from rag_agent.auth.password import hash_password, verify_password
from rag_agent.auth.rate_limit import (
    finish_login_attempt,
    login_bucket,
    reserve_login_attempt,
)

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


def test_unknown_username_uses_argon2_dummy_verification(auth_app, monkeypatch):
    app, _, _ = auth_app
    calls = []

    def verify(hash_, password):
        calls.append((hash_, password))
        return False

    monkeypatch.setattr("rag_agent.auth.routes.verify_password", verify)
    with TestClient(app) as client:
        response = client.post(
            "/auth/login",
            headers={"Origin": ORIGIN},
            json={"username": "unknown", "password": PASSWORD},
        )
    assert response.status_code == 401
    assert calls and calls[0][0].startswith("$argon2id$")
    assert calls[0][1] == PASSWORD


def test_malformed_stored_hash_uses_dummy_verification(auth_app, monkeypatch):
    app, runtime, _ = auth_app
    with runtime.connection() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?", ("legacy-hash", "alice")
        )
    calls = []
    monkeypatch.setattr(
        "rag_agent.auth.routes.verify_password",
        lambda hash_, password: calls.append(hash_) or False,
    )
    with TestClient(app) as client:
        response = client.post(
            "/auth/login",
            headers={"Origin": ORIGIN},
            json={"username": "alice", "password": PASSWORD},
        )
    assert response.status_code == 401
    assert calls and calls[0].startswith("$argon2id$") and calls[0] != "legacy-hash"


def test_login_attempts_are_persistently_throttled(tmp_path):
    settings = APISettings(
        document_dir=tmp_path / "database",
        environment="test",
        login_rate_limit_attempts=2,
        login_rate_limit_window_seconds=60,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        headers = {"Origin": ORIGIN}
        for _ in range(2):
            assert (
                client.post(
                    "/auth/login",
                    headers=headers,
                    json={"username": "unknown", "password": PASSWORD},
                ).status_code
                == 401
            )
        blocked = client.post(
            "/auth/login",
            headers=headers,
            json={"username": "another-unknown", "password": PASSWORD},
        )
        assert blocked.status_code == 429
        assert 1 <= int(blocked.headers["retry-after"]) <= 60
    # The limiter is stored in the same persistent application SQLite registry.
    runtime = app.state.runtime
    with runtime.connection() as conn:
        assert conn.execute("SELECT failures FROM login_attempts").fetchone() == (2,)
    runtime.close()


def test_concurrent_failures_reserve_only_the_configured_number_of_slots(tmp_path):
    database = tmp_path / "rate-limit.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE login_attempts (bucket TEXT PRIMARY KEY, window_started REAL NOT NULL, "
            "failures INTEGER NOT NULL, in_flight INTEGER NOT NULL DEFAULT 0, "
            "window_id TEXT NOT NULL DEFAULT '')"
        )
    workers, limit = 12, 3
    barrier = threading.Barrier(workers)
    bucket = login_bucket("192.0.2.10")

    def failed_attempt() -> bool:
        barrier.wait(timeout=10)
        with sqlite3.connect(database, timeout=10) as connection:
            admission = reserve_login_attempt(
                connection, bucket, now=100, window_seconds=60, limit=limit
            )
            if admission.reservation is None:
                return False
            finish_login_attempt(connection, admission.reservation, succeeded=False, now=101)
            return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        admitted = list(executor.map(lambda _: failed_attempt(), range(workers)))

    assert sum(admitted) == limit
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT failures, in_flight FROM login_attempts WHERE bucket = ?", (bucket,)
        ).fetchone()
        assert row == (limit, 0)
        blocked = reserve_login_attempt(connection, bucket, now=102, window_seconds=60, limit=limit)
    assert blocked.reservation is None
    assert 1 <= blocked.retry_after <= 60


def test_concurrent_different_peers_have_independent_limits(tmp_path):
    database = tmp_path / "rate-limit.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE login_attempts (bucket TEXT PRIMARY KEY, window_started REAL NOT NULL, "
            "failures INTEGER NOT NULL, in_flight INTEGER NOT NULL DEFAULT 0, "
            "window_id TEXT NOT NULL DEFAULT '')"
        )
    peers = ("192.0.2.20", "192.0.2.21")
    barrier = threading.Barrier(len(peers))

    def attempt(peer: str) -> bool:
        barrier.wait(timeout=10)
        with sqlite3.connect(database, timeout=10) as connection:
            admission = reserve_login_attempt(
                connection, login_bucket(peer), now=200, window_seconds=60, limit=1
            )
            if admission.reservation is None:
                return False
            finish_login_attempt(connection, admission.reservation, succeeded=False, now=201)
            return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        admitted = list(executor.map(attempt, peers))
    assert admitted == [True, True]


def test_success_clears_failures_for_the_same_peer(auth_app):
    app, runtime, _ = auth_app
    peer = ("198.51.100.30", 43000)
    bucket = login_bucket(peer[0])
    with TestClient(app, client=peer) as client:
        headers = {"Origin": ORIGIN}
        failed = client.post(
            "/auth/login", headers=headers, json={"username": "alice", "password": "wrong"}
        )
        assert failed.status_code == 401
        success = client.post(
            "/auth/login", headers=headers, json={"username": "alice", "password": PASSWORD}
        )
        assert success.status_code == 200
    with runtime.connection() as connection:
        row = connection.execute(
            "SELECT failures, in_flight FROM login_attempts WHERE bucket = ?", (bucket,)
        ).fetchone()
    assert row is None


def test_expired_window_resets_and_late_reservation_cannot_change_new_window(tmp_path):
    database = tmp_path / "rate-limit.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE login_attempts (bucket TEXT PRIMARY KEY, window_started REAL NOT NULL, "
            "failures INTEGER NOT NULL, in_flight INTEGER NOT NULL DEFAULT 0, "
            "window_id TEXT NOT NULL DEFAULT '')"
        )
        bucket = login_bucket("192.0.2.30")
        old = reserve_login_attempt(
            connection, bucket, now=100, window_seconds=10, limit=1
        ).reservation
        assert old is not None
        fresh = reserve_login_attempt(
            connection, bucket, now=111, window_seconds=10, limit=1
        ).reservation
        assert fresh is not None
        finish_login_attempt(connection, old, succeeded=False, now=112)
        row = connection.execute(
            "SELECT window_started, failures, in_flight FROM login_attempts WHERE bucket = ?",
            (bucket,),
        ).fetchone()
        assert row == (111.0, 0, 1)
        finish_login_attempt(connection, fresh, succeeded=False, now=113)
        assert connection.execute(
            "SELECT failures, in_flight FROM login_attempts WHERE bucket = ?", (bucket,)
        ).fetchone() == (1, 0)


def test_database_write_contention_does_not_admit_an_attempt(tmp_path):
    database = tmp_path / "rate-limit.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE login_attempts (bucket TEXT PRIMARY KEY, window_started REAL NOT NULL, "
            "failures INTEGER NOT NULL, in_flight INTEGER NOT NULL DEFAULT 0, "
            "window_id TEXT NOT NULL DEFAULT '')"
        )
    bucket = login_bucket("192.0.2.40")
    lock = sqlite3.connect(database, timeout=0)
    contender = sqlite3.connect(database, timeout=0)
    try:
        lock.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            reserve_login_attempt(contender, bucket, now=300, window_seconds=60, limit=2)
    finally:
        lock.rollback()
        lock.close()
    admission = reserve_login_attempt(contender, bucket, now=301, window_seconds=60, limit=2)
    assert admission.reservation is not None
    contender.close()


def test_login_database_contention_fails_closed(auth_app, monkeypatch):
    app, _, _ = auth_app

    def contend(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("rag_agent.auth.routes.reserve_login_attempt", contend)
    with TestClient(app) as client:
        response = client.post(
            "/auth/login",
            headers={"Origin": ORIGIN},
            json={"username": "alice", "password": PASSWORD},
        )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"


def test_existing_rate_limit_table_is_migrated_for_reservations(tmp_path):
    from rag_agent.api.runtime import Runtime

    managed = tmp_path / "managed"
    managed.mkdir()
    with sqlite3.connect(managed / "inventory.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE login_attempts (bucket TEXT PRIMARY KEY, window_started REAL NOT NULL, "
            "failures INTEGER NOT NULL)"
        )
    runtime = Runtime(APISettings(document_dir=managed))
    ensure_auth_schema(runtime)
    with runtime.connection() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(login_attempts)")}
    runtime.close()
    assert {"in_flight", "window_id"} <= columns


def test_owner_password_reset_revokes_all_sessions(auth_app):
    _, runtime, user_id = auth_app
    with runtime.connection() as conn:
        create_user(conn, "ordinary", hash_password(PASSWORD), is_admin=False)
        first, _ = create_session(conn, user_id)
        second, _ = create_session(conn, user_id)
        assert reset_owner_password(conn, "alice", hash_password("new owner password 42"))
        assert get_session(conn, model_hash_token(first)) is None
        assert get_session(conn, model_hash_token(second)) is None
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", ("alice",)
        ).fetchone()
        assert row and verify_password(row[0], "new owner password 42")
        assert not reset_owner_password(conn, "missing", hash_password("new owner password 42"))
        assert not reset_owner_password(conn, "ordinary", hash_password("new owner password 42"))


def test_owner_bootstrap_refuses_existing_accounts(tmp_path, monkeypatch):
    from rag_agent.api.runtime import Runtime

    managed = tmp_path / "managed"
    monkeypatch.setenv("RAG_DOCUMENT_DIR", str(managed))
    monkeypatch.setenv("RAG_ENV", "test")
    runtime = Runtime(APISettings(document_dir=managed))
    ensure_auth_schema(runtime)
    with runtime.connection() as conn:
        create_user(conn, "existing", hash_password(PASSWORD), is_admin=True)
    runtime.close()

    import sys

    monkeypatch.setattr(sys, "argv", ["create_user.py", "second", "--admin"])
    answers = iter([PASSWORD, PASSWORD])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(answers))
    with pytest.raises(SystemExit):
        create_user_command()


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
