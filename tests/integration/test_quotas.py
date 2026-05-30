"""Per-tenant quota enforcement, `GET /auth/usage`, and the
per-API-key rate limiter.

Covers `state.get_usage` / `try_consume_query` / `try_consume_vectors`, the
`GET /auth/usage` endpoint, the 429 `query_quota_exceeded` /
`vector_quota_exceeded` / `rate_limited` responses, the lazy daily reset, and
tenant isolation. All tests run with `DATABASE_URL=memory://test` so no
Postgres is required; the ingest pipeline is driven inline where needed.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import os

import pytest


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def client(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Fresh TestClient over the source_registry app + mounted v1_query router.

    Mirrors the query-api test fixture: per-test MinIO landing/index prefixes
    + a local FAISS shard cache, resets in-memory state, reloads the pipeline
    modules so module-level env reads pick up the patched values, and clears
    the rate-limiter token buckets so each test starts with full buckets.
    The `RB_TEST_*` quota overrides are cleared by default; individual tests
    set them via `monkeypatch` before calling `_signup`.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")
    monkeypatch.delenv("RB_TEST_VECTOR_QUOTA", raising=False)
    monkeypatch.delenv("RB_TEST_QUERY_QUOTA", raising=False)
    # The quota subsystem is opt-in (`RB_ENABLE_QUOTAS`). This whole module
    # exists to exercise it, so the fixture turns it on for every test here.
    # The "quotas-disabled" path is covered explicitly in test_quotas_disabled.
    monkeypatch.setenv("RB_ENABLE_QUOTAS", "true")

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


def _make_indexed_dataset(client, token, name="test", dimension=4, n=5):
    """Create + populate + index a small dataset."""
    r = client.post("/v1/datasets", headers=_auth(token), json={"name": name, "dimension": dimension})
    assert r.status_code == 201, r.text
    records = [
        {"id": f"doc-{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"t": i}}
        for i in range(n)
    ]
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        f"/v1/datasets/{name}/vectors",
        headers={**_auth(token), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    _run_pipeline_once()
    ds = client.get(f"/v1/datasets/{name}", headers=_auth(token)).json()
    assert ds["status"] == "indexed", ds


# --- state.get_usage ------------------------------------------------------


def test_get_usage_default_shape(client):
    """A fresh tenant reports zero usage and the default quota values."""
    s = _signup(client)
    import adapters.state.state as state_mod

    usage = state_mod.get_usage(s["tenant"]["id"])
    assert usage == {
        "vectors_used": 0,
        "vector_quota": 100000,
        "queries_today": 0,
        "daily_query_quota": 10000,
        "queries_reset_at": dt.date.today().isoformat(),
    }


# --- query quota ----------------------------------------------------------


def test_query_under_quota_increments(client):
    """A successful query increments queries_today by exactly one."""
    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 3},
    )
    assert r.status_code == 200, r.text
    usage = client.get("/auth/usage", headers=_auth(s["token"])).json()
    assert usage["queries_today"] == 1


def test_query_at_quota_returns_429(client, monkeypatch):
    """At the daily query cap, the next query → 429 query_quota_exceeded."""
    monkeypatch.setenv("RB_TEST_QUERY_QUOTA", "1")
    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    q = {"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]}
    r1 = client.post("/v1/query", headers=_auth(s["token"]), json=q)
    assert r1.status_code == 200, r1.text
    r2 = client.post("/v1/query", headers=_auth(s["token"]), json=q)
    assert r2.status_code == 429, r2.text
    err = r2.json()["error"]
    assert err["code"] == "query_quota_exceeded"
    assert err["details"]["limit"] == 1
    assert err["details"]["reset_at"] == dt.date.today().isoformat()


def test_failed_validation_query_does_not_consume_quota(client, monkeypatch):
    """A dimension-mismatch query is rejected before quota is consumed."""
    monkeypatch.setenv("RB_TEST_QUERY_QUOTA", "1")
    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    # Wrong vector length → 400 dimension_mismatch, must NOT burn the 1 unit.
    bad = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0]},
    )
    assert bad.status_code == 400, bad.text
    assert bad.json()["error"]["code"] == "dimension_mismatch"
    usage = client.get("/auth/usage", headers=_auth(s["token"])).json()
    assert usage["queries_today"] == 0
    # The single quota unit is still available for a valid query.
    ok = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert ok.status_code == 200, ok.text


def test_lazy_daily_reset(client):
    """A queries_reset_at in the past zeroes queries_today on next access."""
    s = _signup(client)
    import adapters.state.state as state_mod

    tid = s["tenant"]["id"]
    row = state_mod._MEM_TENANTS[tid]
    row["queries_today"] = 42
    row["queries_reset_at"] = (dt.date.today() - dt.timedelta(days=1)).isoformat()

    usage = state_mod.get_usage(tid)
    assert usage["queries_today"] == 0
    assert usage["queries_reset_at"] == dt.date.today().isoformat()


# --- vector quota ---------------------------------------------------------


def test_ingest_under_vector_quota_increments(client):
    """A successful upload raises vectors_used by the accepted count."""
    s = _signup(client)
    r = client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "ds", "dimension": 4})
    assert r.status_code == 201, r.text
    records = [{"id": f"v{i}", "values": [float(i), 0.0, 0.0, 0.0]} for i in range(7)]
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        "/v1/datasets/ds/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    assert r.json()["accepted"] == 7
    usage = client.get("/auth/usage", headers=_auth(s["token"])).json()
    assert usage["vectors_used"] == 7


def test_ingest_over_vector_quota_returns_429_and_persists_nothing(client, monkeypatch):
    """An upload that would cross the cap → 429, and nothing is published."""
    monkeypatch.setenv("RB_TEST_VECTOR_QUOTA", "3")
    s = _signup(client)
    r = client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "ds", "dimension": 4})
    assert r.status_code == 201, r.text
    # 5 records, quota is 3 → all-or-nothing rejection.
    records = [{"id": f"v{i}", "values": [float(i), 0.0, 0.0, 0.0]} for i in range(5)]
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        "/v1/datasets/ds/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 429, r.text
    err = r.json()["error"]
    assert err["code"] == "vector_quota_exceeded"
    assert err["details"]["limit"] == 3
    assert err["details"]["used"] == 0
    # vectors_used untouched and no VALIDATE_DATASET message published.
    usage = client.get("/auth/usage", headers=_auth(s["token"])).json()
    assert usage["vectors_used"] == 0
    from adapters.queue.queue import consume
    assert consume("VALIDATE_DATASET", block=False) is None


# --- rate limiter ---------------------------------------------------------


def test_rate_limiter_burst_then_429(client, monkeypatch):
    """Requests within the burst allowance pass; the next one → rate_limited."""
    # Tiny bucket: 0 refill, burst 3 → exactly 3 requests then 429.
    monkeypatch.setenv("RB_RATE_LIMIT_RPS", "0")
    monkeypatch.setenv("RB_RATE_LIMIT_BURST", "3")
    import services.auth.quota as quota_mod
    importlib.reload(quota_mod)
    quota_mod.reset_rate_limiter()
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    main_mod.app.include_router(v1_query.router)
    from fastapi.testclient import TestClient
    c = TestClient(main_mod.app)

    s = _signup(c)
    h = _auth(s["token"])
    statuses = [c.get("/v1/datasets", headers=h).status_code for _ in range(3)]
    assert statuses == [200, 200, 200]
    r = c.get("/v1/datasets", headers=h)
    assert r.status_code == 429, r.text
    assert r.json()["error"]["code"] == "rate_limited"


def test_rate_limiter_is_per_key(client, monkeypatch):
    """Key A exhausting its bucket does not 429 key B."""
    monkeypatch.setenv("RB_RATE_LIMIT_RPS", "0")
    monkeypatch.setenv("RB_RATE_LIMIT_BURST", "2")
    import services.auth.quota as quota_mod
    importlib.reload(quota_mod)
    quota_mod.reset_rate_limiter()
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    main_mod.app.include_router(v1_query.router)
    from fastapi.testclient import TestClient
    c = TestClient(main_mod.app)

    a = _signup(c, email="a@example.com")
    b = _signup(c, email="b@example.com")
    key_a = a["first_api_key"]["key"]
    key_b = b["first_api_key"]["key"]

    # Exhaust key A's bucket (burst 2).
    assert c.get("/v1/datasets", headers=_auth(key_a)).status_code == 200
    assert c.get("/v1/datasets", headers=_auth(key_a)).status_code == 200
    assert c.get("/v1/datasets", headers=_auth(key_a)).status_code == 429
    # Key B's bucket is independent and still full.
    assert c.get("/v1/datasets", headers=_auth(key_b)).status_code == 200


# --- GET /auth/usage ------------------------------------------------------


def test_usage_requires_auth(client):
    """GET /auth/usage without a token → 401 unauthorized."""
    r = client.get("/auth/usage")
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "unauthorized"


def test_usage_endpoint_shape(client):
    """GET /auth/usage returns exactly the v1 contract fields."""
    s = _signup(client)
    r = client.get("/auth/usage", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {
        "vectors_used",
        "vector_quota",
        "queries_today",
        "daily_query_quota",
        "queries_reset_at",
    }
    assert body["vectors_used"] == 0
    assert body["queries_today"] == 0


# --- tenant isolation -----------------------------------------------------


def test_tenant_isolation_counters(client):
    """Tenant B's uploads/queries do not move tenant A's counters."""
    a = _signup(client, email="a@example.com")
    b = _signup(client, email="b@example.com")

    _make_indexed_dataset(client, b["token"], name="bds", n=4)
    client.post(
        "/v1/query",
        headers=_auth(b["token"]),
        json={"dataset": "bds", "vector": [0.0, 0.0, 0.0, 0.0]},
    )

    usage_a = client.get("/auth/usage", headers=_auth(a["token"])).json()
    assert usage_a["vectors_used"] == 0
    assert usage_a["queries_today"] == 0

    usage_b = client.get("/auth/usage", headers=_auth(b["token"])).json()
    assert usage_b["vectors_used"] == 4
    assert usage_b["queries_today"] == 1
