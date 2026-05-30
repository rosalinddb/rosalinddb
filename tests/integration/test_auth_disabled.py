"""OSS opt-in: `RB_REQUIRE_AUTH` unset/false → the entire auth + tenancy
stack is bypassed.

The self-host headline path (`docker compose up`) defaults to this mode — a
self-hoster running a single tenant on their own box should not have to
signup, ferry JWTs around, or mint API keys for what is functionally a
single-user install. These tests prove that with the env var off:

  - every request resolves to the bootstrap "default" tenant without
    needing an `Authorization` header;
  - the v1 dataset / ingest / query surface is fully reachable
    unauthenticated;
  - the `/auth/{signup,login,me,keys}` surface returns 404 (the routes
    look like they don't exist on this deployment);
  - `GET /auth/usage` returns `{"enabled": false}` with no tenant context
    (same behaviour as when quotas are also off).

The fixture explicitly clears `RB_REQUIRE_AUTH` (overriding the suite-wide
default in `tests/conftest.py`) so this whole module exercises the OFF path.
A separate test confirms the existing `RB_ENABLE_QUOTAS=off` flow still
works alongside auth-off, proving the two gates are independent.

Tests run in memory mode (`DATABASE_URL=memory://test`) — no Postgres needed.
"""
from __future__ import annotations

import importlib
import json
import logging
import os

import pytest


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def client(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Fresh TestClient with `RB_REQUIRE_AUTH` explicitly UNSET.

    Mirrors the `test_quotas_disabled.py` fixture but inverts a different
    master switch — we test that the auth + tenancy code path short-circuits
    end-to-end. Quotas are left at the OSS default (off) too so a single
    fixture exercises the headline self-host configuration; an extra test
    flips quotas on independently to prove the two gates do not depend on
    each other.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")
    # The point of this module: BOTH OSS gates default OFF. tests/conftest.py
    # pins `RB_REQUIRE_AUTH=true` for the suite as a whole; this delenv
    # explicitly cancels that pin for the duration of these tests.
    monkeypatch.delenv("RB_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("RB_ENABLE_QUOTAS", raising=False)

    from adapters.queue.queue import consume as _consume
    for _topic in ("VALIDATE_DATASET", "DATASET_READY", "RUN_EPHEMERAL_QUERY", "RESULT_READY"):
        while _consume(_topic, block=False):
            pass

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_DATASETS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, list):
            obj.clear()
    state_mod._MEM_SHARDS.clear()

    import services.auth.jwt_utils as jwt_utils
    importlib.reload(jwt_utils)
    import services.auth.quota as quota_mod
    importlib.reload(quota_mod)
    quota_mod.reset_rate_limiter()
    import services.auth.auth as auth_mod
    importlib.reload(auth_mod)
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)
    import services.validator_worker.run as validator
    importlib.reload(validator)
    import services.index_builder.run as builder
    importlib.reload(builder)
    import services.ephemeral_runner.run as ephemeral
    importlib.reload(ephemeral)
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    v1_query.cache_clear()
    v1_query._RESULTS.clear()

    main_mod.app.include_router(v1_query.router)

    from fastapi.testclient import TestClient
    # Using TestClient as a context manager fires the FastAPI startup event,
    # which is where the OSS bootstrap of the "default" tenant happens.
    with TestClient(main_mod.app) as c:
        yield c


def _run_pipeline_once():
    """Drain VALIDATE_DATASET and run the builder synchronously."""
    from adapters.queue.queue import consume
    from services.validator_worker.run import process_uri
    from services.index_builder.run import run_once

    pending = []
    while True:
        msg = consume("VALIDATE_DATASET", block=False)
        if not msg:
            break
        try:
            process_uri(msg["dataset"], msg["tenant"], msg["uri"], msg.get("file_type"))
            pending.append(msg)
        except Exception:
            pass
    for msg in pending:
        run_once(msg["dataset"], msg["tenant"])


# --- helper --------------------------------------------------------------


def test_auth_helper_reports_disabled(monkeypatch):
    """The `auth_required()` helper is False with the env var unset.

    `tests/conftest.py` pins `RB_REQUIRE_AUTH=true` for the suite-wide default
    so existing tests keep passing; this test explicitly unsets it (mirroring
    what the `client` fixture does) before reading the helper.
    """
    monkeypatch.delenv("RB_REQUIRE_AUTH", raising=False)
    import services.auth.jwt_utils as jwt_utils
    assert jwt_utils.auth_required() is False
    assert jwt_utils.DEFAULT_TENANT_ID == "default"


def test_default_tenant_bootstrapped(client):
    """The bootstrap default-tenant row exists in the state store on startup."""
    import adapters.state.state as state_mod
    row = state_mod.get_tenant_by_id("default")
    assert row is not None, "expected the OSS bootstrap to seed tenant_id=default"
    assert row["email"] == "self-host@localhost"
    assert row["password_hash"] == "!disabled!"
    assert row["plan"] == "oss"


# --- v1 surface is open --------------------------------------------------


def test_datasets_list_without_authorization_header(client):
    """`GET /v1/datasets` with no Authorization header → 200, empty list."""
    r = client.get("/v1/datasets")
    assert r.status_code == 200, r.text
    assert r.json() == {"datasets": []}


def test_create_ingest_query_without_auth(client):
    """End-to-end: create + ingest + query with no auth header at any step.

    Proves the "default" tenant short-circuit threads through every v1 surface
    a self-hoster would use (catalog, ingest, query) — not just `/v1/datasets`.
    """
    # Create a dataset.
    r = client.post("/v1/datasets", json={"name": "smoke", "dimension": 4})
    assert r.status_code == 201, r.text
    assert r.json()["name"] == "smoke"

    # Ingest a few vectors.
    records = [{"id": f"v{i}", "values": [float(i), 0.0, 0.0, 0.0]} for i in range(3)]
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        "/v1/datasets/smoke/vectors",
        headers={"Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    assert r.json()["accepted"] == 3

    # Run the pipeline so the dataset becomes queryable.
    _run_pipeline_once()

    # Query — no Authorization header.
    r = client.post(
        "/v1/query",
        json={"dataset": "smoke", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 200, r.text


def test_dataset_isolation_collapses_to_default(client):
    """All datasets created in OSS mode belong to tenant_id="default".

    Two `POST /v1/datasets` calls — neither with auth — must see each other,
    proving they end up on the same tenant. (In SaaS mode each caller has
    their own JWT and the lists are isolated.)
    """
    assert client.post("/v1/datasets", json={"name": "a", "dimension": 4}).status_code == 201
    assert client.post("/v1/datasets", json={"name": "b", "dimension": 4}).status_code == 201
    r = client.get("/v1/datasets")
    assert r.status_code == 200, r.text
    names = sorted(d["name"] for d in r.json()["datasets"])
    assert names == ["a", "b"]


# --- /auth/* surface is hidden -------------------------------------------


def test_signup_returns_404_when_auth_disabled(client):
    """POST /auth/signup → 404 with the `auth_disabled` code."""
    r = client.post("/auth/signup", json={"email": "a@b.com", "password": "password123"})
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "auth_disabled"


def test_login_returns_404_when_auth_disabled(client):
    """POST /auth/login → 404."""
    r = client.post("/auth/login", json={"email": "a@b.com", "password": "password123"})
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "auth_disabled"


def test_me_returns_404_when_auth_disabled(client):
    """GET /auth/me → 404 (no per-caller principal to describe)."""
    r = client.get("/auth/me")
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "auth_disabled"


def test_keys_endpoints_return_404_when_auth_disabled(client):
    """The /auth/keys* surface is hidden in OSS mode."""
    r = client.get("/auth/keys")
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "auth_disabled"

    r = client.post("/auth/keys", json={"name": "default"})
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "auth_disabled"

    r = client.delete("/auth/keys/anything")
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "auth_disabled"


# --- /auth/usage ---------------------------------------------------------


def test_usage_returns_disabled_envelope_no_auth(client):
    """`GET /auth/usage` with no auth header → `{"enabled": false}`.

    No 401 — there is no auth to demand. No tenant lookup either — there is no
    per-caller tenant to project. The dashboard binds to the same envelope it
    gets when quotas are off, so the disabled flow is uniform.
    """
    r = client.get("/auth/usage")
    assert r.status_code == 200, r.text
    assert r.json() == {"enabled": False}


def test_usage_disabled_envelope_ignores_bogus_auth_header(client):
    """A bogus Authorization header is silently ignored in OSS mode."""
    r = client.get("/auth/usage", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 200, r.text
    assert r.json() == {"enabled": False}


# --- startup warning -----------------------------------------------------


def test_startup_logs_auth_disabled_warning(client, caplog):
    """The startup hook emits a WARNING when auth is disabled.

    `client` is already built (so the startup event already fired); we re-run
    the startup hook with caplog capturing to assert the banner shape.
    """
    import services.source_registry.main as main_mod

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="services.source_registry.main"):
        main_mod._oss_startup()
    # One WARNING line carrying the canonical banner.
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "RB_REQUIRE_AUTH=false" in r.getMessage() for r in warning_records
    ), [r.getMessage() for r in warning_records]


# --- gates are independent -----------------------------------------------


def test_auth_off_quotas_on_still_works(client, monkeypatch):
    """`RB_REQUIRE_AUTH=off` + `RB_ENABLE_QUOTAS=on` is a coherent state.

    Quotas-on means the rate limiter and the per-tenant counters are active;
    they target whatever `current_tenant_id` resolves to — which in OSS mode
    is always "default". The bootstrap row exists, so the quota path finds a
    tenant and the request goes through.
    """
    monkeypatch.setenv("RB_ENABLE_QUOTAS", "true")
    import services.auth.quota as quota_mod
    importlib.reload(quota_mod)
    quota_mod.reset_rate_limiter()
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)
    # `_bootstrap_default_tenant_memory` is idempotent — re-fire it after the
    # reload to ensure the row is present.
    import adapters.state.state as state_mod
    state_mod._bootstrap_default_tenant_memory()

    from fastapi.testclient import TestClient
    with TestClient(main_mod.app) as c:
        r = c.get("/v1/datasets")
        assert r.status_code == 200, r.text
