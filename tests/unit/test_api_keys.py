"""Tests for the API-key surface.

Covers `POST/GET/DELETE /auth/keys`, the `first_api_key` field on
signup, and dual JWT-or-API-key authentication on `/auth/me`. Per the
v1 contract (`docs/api/v1.md`):

  - Signup auto-issues a "Default" key and returns the raw key once.
  - `POST /auth/keys` is JWT-only (cannot bootstrap keys from a key).
  - `GET`/`DELETE /auth/keys` accept either JWT or API key auth.
  - Cross-tenant access returns 404, never 403 (no existence leak).
  - Revoked keys no longer authenticate.
  - `last_used_at` updates on successful auth.

All tests use the in-memory state adapter so no Postgres is required.
"""
from __future__ import annotations

import importlib
import os
import re
import time


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


# `rb_live_` + 32 URL-safe base64 characters (digits, letters, `_`, `-`).
KEY_RE = re.compile(r"^rb_live_[A-Za-z0-9_-]{32}$")


def _make_client():
    """Build a fresh FastAPI TestClient with reset in-memory state.

    Mirrors the helper in `test_auth.py`. Reloading the state module
    drops the module-level dicts so each test starts with a clean slate.
    """
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_API_KEYS_BY_HASH"):
        if hasattr(state_mod, attr):
            obj = getattr(state_mod, attr)
            if isinstance(obj, dict):
                obj.clear()
            elif isinstance(obj, list):
                obj.clear()

    import services.auth.jwt_utils as jwt_utils
    importlib.reload(jwt_utils)
    import services.auth.auth as auth_mod
    importlib.reload(auth_mod)
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)

    from fastapi.testclient import TestClient
    return TestClient(main_mod.app)


def _signup(client, email="alice@example.com", password="password123"):
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


# --- signup auto-issues a Default key ------------------------------------


def test_signup_returns_first_api_key():
    c = _make_client()
    body = _signup(c)
    fk = body.get("first_api_key")
    assert isinstance(fk, dict), f"missing first_api_key: {body}"
    assert fk["name"] == "Default"
    assert fk["id"].startswith("key_")
    assert "created_at" in fk and fk["created_at"]
    assert KEY_RE.match(fk["key"]), f"unexpected key format: {fk['key']!r}"


# --- POST /auth/keys ------------------------------------------------------


def test_create_api_key_returns_raw_once():
    c = _make_client()
    body = _signup(c)
    jwt = body["token"]
    r = c.post(
        "/auth/keys",
        headers={"Authorization": f"Bearer {jwt}"},
        json={"name": "test"},
    )
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["name"] == "test"
    assert out["id"].startswith("key_")
    assert KEY_RE.match(out["key"]), f"unexpected key format: {out['key']!r}"
    assert "created_at" in out and out["created_at"]


def test_create_api_key_with_api_key_rejected():
    """POST /auth/keys is JWT-only — cannot mint a key with an API key.

    Per the v1 contract this avoids chicken/egg + scope creep: customer
    code authenticating with a key cannot escalate to issuing more keys.
    """
    c = _make_client()
    body = _signup(c)
    api_key = body["first_api_key"]["key"]
    r = c.post(
        "/auth/keys",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"name": "from-api-key"},
    )
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "unauthorized"


def test_create_api_key_invalid_name_empty():
    c = _make_client()
    body = _signup(c)
    jwt = body["token"]
    r = c.post(
        "/auth/keys",
        headers={"Authorization": f"Bearer {jwt}"},
        json={"name": ""},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_name"


def test_create_api_key_invalid_name_too_long():
    c = _make_client()
    body = _signup(c)
    jwt = body["token"]
    r = c.post(
        "/auth/keys",
        headers={"Authorization": f"Bearer {jwt}"},
        json={"name": "x" * 65},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_name"


# --- GET /auth/keys -------------------------------------------------------


def test_list_keys_no_raw_key():
    c = _make_client()
    body = _signup(c)
    jwt = body["token"]
    r = c.get("/auth/keys", headers={"Authorization": f"Bearer {jwt}"})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert "keys" in payload
    keys = payload["keys"]
    assert len(keys) == 1  # The auto-issued Default key
    entry = keys[0]
    assert set(entry.keys()) == {"id", "name", "created_at", "last_used_at", "revoked_at"}
    assert "key" not in entry  # Raw key MUST never appear


def test_list_keys_tenant_isolation():
    c = _make_client()
    a = _signup(c, email="a@example.com")
    b = _signup(c, email="b@example.com")
    # Tenant A creates an extra key
    r = c.post(
        "/auth/keys",
        headers={"Authorization": f"Bearer {a['token']}"},
        json={"name": "a-only"},
    )
    assert r.status_code == 201, r.text
    a_key_id = r.json()["id"]

    # Tenant B's list does not include it
    r = c.get("/auth/keys", headers={"Authorization": f"Bearer {b['token']}"})
    assert r.status_code == 200
    b_ids = [k["id"] for k in r.json()["keys"]]
    assert a_key_id not in b_ids


# --- DELETE /auth/keys/{id} ----------------------------------------------


def test_revoke_key_204():
    c = _make_client()
    body = _signup(c)
    jwt = body["token"]
    key_id = body["first_api_key"]["id"]
    raw_key = body["first_api_key"]["key"]

    r = c.delete(f"/auth/keys/{key_id}", headers={"Authorization": f"Bearer {jwt}"})
    assert r.status_code == 204, r.text
    assert r.content in (b"", b"null")

    # GET shows revoked_at set
    r = c.get("/auth/keys", headers={"Authorization": f"Bearer {jwt}"})
    rows = [k for k in r.json()["keys"] if k["id"] == key_id]
    assert len(rows) == 1
    assert rows[0]["revoked_at"] is not None

    # The revoked key can no longer auth
    r = c.get("/auth/me", headers={"Authorization": f"Bearer {raw_key}"})
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "unauthorized"


def test_revoke_other_tenants_key_404():
    """Cross-tenant revoke returns 404 `not_found` — we never leak existence."""
    c = _make_client()
    a = _signup(c, email="a@example.com")
    b = _signup(c, email="b@example.com")
    a_key_id = a["first_api_key"]["id"]
    r = c.delete(
        f"/auth/keys/{a_key_id}",
        headers={"Authorization": f"Bearer {b['token']}"},
    )
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "not_found"


def test_revoke_nonexistent_key_404():
    c = _make_client()
    body = _signup(c)
    jwt = body["token"]
    r = c.delete(
        "/auth/keys/key_doesnotexist",
        headers={"Authorization": f"Bearer {jwt}"},
    )
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "not_found"


# --- Auth via API key ----------------------------------------------------


def test_auth_via_api_key_to_me():
    c = _make_client()
    body = _signup(c)
    api_key = body["first_api_key"]["key"]
    r = c.get("/auth/me", headers={"Authorization": f"Bearer {api_key}"})
    assert r.status_code == 200, r.text
    assert r.json()["tenant"]["email"] == "alice@example.com"


def test_auth_via_revoked_key_rejected():
    c = _make_client()
    body = _signup(c)
    jwt = body["token"]
    api_key = body["first_api_key"]["key"]
    key_id = body["first_api_key"]["id"]
    c.delete(f"/auth/keys/{key_id}", headers={"Authorization": f"Bearer {jwt}"})
    r = c.get("/auth/me", headers={"Authorization": f"Bearer {api_key}"})
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "unauthorized"


def test_auth_via_malformed_api_key_rejected():
    c = _make_client()
    _signup(c)
    r = c.get(
        "/auth/me",
        headers={"Authorization": "Bearer rb_live_garbage"},
    )
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "unauthorized"


def test_last_used_at_updates_on_successful_auth():
    c = _make_client()
    body = _signup(c)
    jwt = body["token"]
    api_key = body["first_api_key"]["key"]

    # Brand-new key: last_used_at is null in the list.
    r = c.get("/auth/keys", headers={"Authorization": f"Bearer {jwt}"})
    keys = r.json()["keys"]
    assert keys[0]["last_used_at"] is None

    before = time.time()
    r = c.get("/auth/me", headers={"Authorization": f"Bearer {api_key}"})
    assert r.status_code == 200, r.text

    # After one authed request the field is a recent timestamp.
    r = c.get("/auth/keys", headers={"Authorization": f"Bearer {jwt}"})
    keys = r.json()["keys"]
    used = keys[0]["last_used_at"]
    assert isinstance(used, str) and used, f"expected timestamp, got {used!r}"
    # Parse ISO 8601 `YYYY-MM-DDTHH:MM:SSZ` and assert it's within the last 60 seconds.
    import datetime as _dt
    parsed = _dt.datetime.strptime(used, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
    delta = abs(parsed.timestamp() - before)
    assert delta < 60, f"last_used_at {used} is not recent (delta {delta}s)"


# --- O(1) resolution (SHA-256 indexed lookup) ----------------------------


def test_api_key_resolution_is_constant_time_and_no_bcrypt():
    """Auth resolves correctly and identically regardless of key count.

    `rb_live_` keys are stored as a deterministic SHA-256 digest and
    resolved via a single indexed lookup, NOT a linear bcrypt scan.
    This test proves the resolution does not depend on the number of keys
    in the system (works the same with 1 key and with ~30 keys across
    multiple tenants) and that `bcrypt.checkpw` is never invoked on the
    API-key auth path.
    """
    import bcrypt

    c = _make_client()

    # Tenant A: signup gives one key. Resolve it with only 1 key present.
    a = _signup(c, email="a@example.com")
    a_first_key = a["first_api_key"]["key"]
    r = c.get("/auth/me", headers={"Authorization": f"Bearer {a_first_key}"})
    assert r.status_code == 200, r.text
    assert r.json()["tenant"]["email"] == "a@example.com"

    # Tenant B: signup, then create ~30 more keys across A and B so the
    # system holds many keys. A linear bcrypt scan would now be slow; an
    # indexed lookup is unaffected.
    b = _signup(c, email="b@example.com")
    target_key = None
    for i in range(15):
        ra = c.post(
            "/auth/keys",
            headers={"Authorization": f"Bearer {a['token']}"},
            json={"name": f"a-key-{i}"},
        )
        assert ra.status_code == 201, ra.text
        rb = c.post(
            "/auth/keys",
            headers={"Authorization": f"Bearer {b['token']}"},
            json={"name": f"b-key-{i}"},
        )
        assert rb.status_code == 201, rb.text
        if i == 7:
            # Pick one specific B key in the middle of the pack as the
            # one we will authenticate with.
            target_key = rb.json()["key"]

    assert target_key is not None

    # Spy on bcrypt.checkpw — it must not be touched by API-key auth.
    calls = {"n": 0}
    real_checkpw = bcrypt.checkpw

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return real_checkpw(*args, **kwargs)

    bcrypt.checkpw = _spy
    try:
        # The same first key still resolves with 32 keys present.
        r = c.get("/auth/me", headers={"Authorization": f"Bearer {a_first_key}"})
        assert r.status_code == 200, r.text
        assert r.json()["tenant"]["email"] == "a@example.com"

        # A specific B key resolves to tenant B — not A — even buried
        # among many keys.
        r = c.get("/auth/me", headers={"Authorization": f"Bearer {target_key}"})
        assert r.status_code == 200, r.text
        assert r.json()["tenant"]["email"] == "b@example.com"

        # A garbage rb_live_ token still 401s (SHA-256 not in the table).
        r = c.get(
            "/auth/me",
            headers={"Authorization": "Bearer rb_live_" + "z" * 32},
        )
        assert r.status_code == 401, r.text
        assert r.json()["error"]["code"] == "unauthorized"
    finally:
        bcrypt.checkpw = real_checkpw

    assert calls["n"] == 0, (
        f"bcrypt.checkpw was called {calls['n']}x on the API-key path; "
        "API keys must resolve via an indexed SHA-256 lookup, not bcrypt"
    )
