"""Tests for the auth service.

Covers signup, login, and /auth/me per the API contract at
docs/api/v1.md. Uses the in-memory state adapter
(`DATABASE_URL=memory://test`) so no Postgres is required.
"""
from __future__ import annotations

import os
import time
import importlib

import pytest


# Ensure tests use the in-memory state adapter BEFORE importing the app.
os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


def _make_client():
    """Build a fresh FastAPI TestClient with reset in-memory tenant state.

    Reloads the state adapter so module-level dicts start empty for every test.
    """
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    # Reset in-memory tenant dicts in case reload did not.
    if hasattr(state_mod, "_MEM_TENANTS"):
        state_mod._MEM_TENANTS.clear()

    # Reload services that depend on the freshly reloaded state module.
    import services.auth.auth as auth_mod
    importlib.reload(auth_mod)
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)

    from fastapi.testclient import TestClient
    return TestClient(main_mod.app)


# --- signup ---------------------------------------------------------------


def test_signup_happy_path():
    c = _make_client()
    r = c.post(
        "/auth/signup",
        json={"email": "alice@example.com", "password": "password123"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert "token" in body and isinstance(body["token"], str) and body["token"]
    tenant = body["tenant"]
    assert tenant["email"] == "alice@example.com"
    assert tenant["id"].startswith("ten_")
    assert tenant["plan"] == "free"
    assert "created_at" in tenant

    # State adapter sees the new tenant.
    from adapters.state import state as state_mod
    found = state_mod.get_tenant_by_email("alice@example.com")
    assert found is not None
    assert found["id"] == tenant["id"]


def test_signup_duplicate_email():
    c = _make_client()
    payload = {"email": "dup@example.com", "password": "password123"}
    r1 = c.post("/auth/signup", json=payload)
    assert r1.status_code == 201
    r2 = c.post("/auth/signup", json=payload)
    assert r2.status_code == 409, r2.text
    assert r2.json()["error"]["code"] == "email_taken"


def test_signup_invalid_email():
    c = _make_client()
    r = c.post(
        "/auth/signup",
        json={"email": "not-an-email", "password": "password123"},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_email"


def test_signup_weak_password():
    c = _make_client()
    r = c.post(
        "/auth/signup",
        json={"email": "weak@example.com", "password": "short"},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "weak_password"


# --- login ----------------------------------------------------------------


def test_login_happy_path():
    c = _make_client()
    c.post(
        "/auth/signup",
        json={"email": "bob@example.com", "password": "password123"},
    )
    r = c.post(
        "/auth/login",
        json={"email": "bob@example.com", "password": "password123"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["token"], str) and body["token"]
    assert body["tenant"]["email"] == "bob@example.com"

    # Verify password was actually hashed with bcrypt by checking the stored row.
    import bcrypt
    from adapters.state import state as state_mod
    row = state_mod.get_tenant_by_email("bob@example.com")
    assert row is not None
    assert bcrypt.checkpw(b"password123", row["password_hash"].encode())


def test_login_wrong_password():
    c = _make_client()
    c.post(
        "/auth/signup",
        json={"email": "carol@example.com", "password": "password123"},
    )
    r = c.post(
        "/auth/login",
        json={"email": "carol@example.com", "password": "wrongpassword"},
    )
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "invalid_credentials"


def test_login_unknown_email():
    c = _make_client()
    r = c.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "password123"},
    )
    assert r.status_code == 401, r.text
    # Same code as wrong password — must not leak which side was wrong.
    assert r.json()["error"]["code"] == "invalid_credentials"


# --- /auth/me -------------------------------------------------------------


def test_me_with_valid_jwt():
    c = _make_client()
    s = c.post(
        "/auth/signup",
        json={"email": "dan@example.com", "password": "password123"},
    )
    assert s.status_code == 201
    token = s.json()["token"]
    r = c.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    assert r.json()["tenant"]["email"] == "dan@example.com"


def test_me_missing_token():
    c = _make_client()
    r = c.get("/auth/me")
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "unauthorized"


def test_me_malformed_jwt():
    c = _make_client()
    r = c.get("/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "unauthorized"


def test_me_expired_jwt():
    c = _make_client()
    s = c.post(
        "/auth/signup",
        json={"email": "eve@example.com", "password": "password123"},
    )
    tenant_id = s.json()["tenant"]["id"]

    # Manually mint a JWT with `exp` in the past using the same secret.
    import jwt as pyjwt
    secret = os.environ["JWT_SECRET"]
    past = int(time.time()) - 3600
    expired = pyjwt.encode(
        {"sub": tenant_id, "iat": past - 1, "exp": past},
        secret,
        algorithm="HS256",
    )
    r = c.get("/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "unauthorized"
