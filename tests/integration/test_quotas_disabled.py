"""OSS opt-in: `RB_ENABLE_QUOTAS` unset/false → the entire quota + rate-limit
subsystem is a no-op.

The self-host headline path (`docker compose up`) defaults to this mode — a
self-hoster running their own database should never be throttled by the
default 10k-queries/day cap. These tests prove that with the env var off:

  - the rate limiter does not throttle even under burst conditions;
  - ingest does not 429 when the per-tenant vector quota would be exceeded;
  - queries do not 429 when the per-tenant daily query quota would be exceeded;
  - `GET /auth/usage` returns the honest `{"enabled": false}` envelope instead
    of the full v1 usage shape.

Tests run in memory mode (`DATABASE_URL=memory://test`) — no Postgres needed.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def client(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Fresh TestClient with `RB_ENABLE_QUOTAS` explicitly UNSET.

    Mirrors the `test_quotas.py` fixture but inverts the master switch — we
    test that all the quota + rate-limit code paths short-circuit.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")
    # The point of this module: the env var is OFF.
    monkeypatch.delenv("RB_ENABLE_QUOTAS", raising=False)
    # Even with the per-tenant quotas set to zero, requests must pass — proves
    # the runtime check is gated, not the schema/defaults.
    monkeypatch.setenv("RB_TEST_VECTOR_QUOTA", "0")
    monkeypatch.setenv("RB_TEST_QUERY_QUOTA", "0")

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
    return TestClient(main_mod.app)


def _signup(client, email="alice@example.com", password="password123"):
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


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


# --- helper ---------------------------------------------------------------


def test_quotas_helper_reports_disabled():
    """The `quotas_enabled()` helper is False with the env var unset."""
    import services.auth.quota as quota_mod
    assert quota_mod.quotas_enabled() is False


# --- ingest ---------------------------------------------------------------


def test_ingest_passes_when_vector_quota_would_block(client):
    """`RB_TEST_VECTOR_QUOTA=0` would normally reject every upload; with quotas
    disabled the upload is accepted unchanged."""
    s = _signup(client)
    r = client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "ds", "dimension": 4})
    assert r.status_code == 201, r.text
    records = [{"id": f"v{i}", "values": [float(i), 0.0, 0.0, 0.0]} for i in range(5)]
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        "/v1/datasets/ds/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    # With `RB_ENABLE_QUOTAS` off the per-tenant vector cap is ignored even
    # though the test override pinned it to 0 — the upload is accepted.
    assert r.status_code == 202, r.text
    assert r.json()["accepted"] == 5


def test_import_admission_passes_when_vector_quota_would_block(client):
    """The bulk-import admission check is a no-op when quotas are disabled."""
    s = _signup(client)
    r = client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "ds", "dimension": 4}
    )
    assert r.status_code == 201, r.text
    # `RB_TEST_VECTOR_QUOTA=0` would make admission reject the import; with
    # quotas off the import-create endpoint returns the staged-upload target.
    r = client.post(
        "/v1/datasets/ds/imports",
        headers=_auth(s["token"]),
        json={"format": "ndjson"},
    )
    assert r.status_code == 201, r.text
    assert "upload" in r.json()


# --- query ----------------------------------------------------------------


def test_query_passes_when_daily_quota_would_block(client):
    """`RB_TEST_QUERY_QUOTA=0` would 429 every query; with quotas disabled it
    succeeds and `queries_today` is not bumped."""
    s = _signup(client)
    # Build a tiny indexed dataset so the query has something to search.
    r = client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "ds", "dimension": 4})
    assert r.status_code == 201, r.text
    records = [{"id": f"v{i}", "values": [float(i), 0.0, 0.0, 0.0]} for i in range(3)]
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        "/v1/datasets/ds/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    _run_pipeline_once()

    # Query → 200, never 429.
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "ds", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    # With the test override pinning the daily query quota to 0 we'd normally
    # 429 immediately; with quotas off this is allowed through. The result
    # may be a hot-path 200 or an enqueued ephemeral job (also 200) — either
    # way the quota path did NOT block the request.
    assert r.status_code == 200, r.text


# --- /auth/usage ----------------------------------------------------------


def test_usage_endpoint_returns_disabled_envelope(client):
    """With quotas disabled, `GET /auth/usage` returns the honest payload."""
    s = _signup(client)
    r = client.get("/auth/usage", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert r.json() == {"enabled": False}


def test_usage_still_requires_auth(client):
    """The disabled path still rejects unauthenticated callers (401)."""
    r = client.get("/auth/usage")
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "unauthorized"


# --- rate limiter ---------------------------------------------------------


def test_rate_limiter_disabled_even_with_tiny_bucket(client, monkeypatch):
    """`RB_RATE_LIMIT_RPS=0` + tiny burst would 429 in three calls; with
    quotas disabled the limiter is a no-op and N requests all return 200."""
    monkeypatch.setenv("RB_RATE_LIMIT_RPS", "0")
    monkeypatch.setenv("RB_RATE_LIMIT_BURST", "1")
    # Reload quota module so the new RATE_LIMIT_* are picked up; the
    # `quotas_enabled()` short-circuit still gates the dependency.
    import services.auth.quota as quota_mod
    importlib.reload(quota_mod)
    quota_mod.reset_rate_limiter()
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)
    from fastapi.testclient import TestClient
    c = TestClient(main_mod.app)

    s = _signup(c)
    h = _auth(s["token"])
    # 10 consecutive calls — would 429 after 1 if the limiter were on.
    statuses = [c.get("/v1/datasets", headers=h).status_code for _ in range(10)]
    assert statuses == [200] * 10, statuses
