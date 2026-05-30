"""Tests for the DP-trust `/v1/query` router.

The customer-facing query path runs behind a Control Plane (CP) that
authenticates, and a private Data Plane (DP) that trusts the CP. This module
covers the DP-trust router (`services/query_api/dp_query.py`) and
the extracted auth/quota-free query core in `v1_query.py`.

Asserted behaviour:
  - a query via the trusted `X-RB-Tenant-Id` header succeeds and returns the
    exact same result shape as the authenticated route;
  - a missing/empty tenant header → 400 with the v1 error envelope;
  - the `RB_PROXY_SECRET` shared-secret check: set + correct secret → ok,
    set + wrong/missing secret → 403 `proxy_unauthorized`, unset → skipped;
  - the DP path consumes NO query quota — a tenant at quota 0 can still query;
  - the status endpoint works;
  - the extracted core (`validate_query_body` / `run_query` /
    `execute_v1_query`) is genuinely auth-free and quota-free.

The dataset/index setup reuses the authenticated `source_registry` app + the
ingest pipeline, exactly like `test_query_api.py`; the DP router is mounted on
its own bare FastAPI app so it is exercised with no auth dependency at all.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def env(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Reset state + reload the pipeline, returning the wired-up modules.

    Mirrors the `test_query_api.py` fixture: per-test MinIO prefixes, a local
    FAISS shard cache, fresh in-memory state, reloaded pipeline modules. By
    default `RB_PROXY_SECRET` and the quota override are cleared so each test
    opts in explicitly.

    Returns a small object exposing:
      - `auth_client`     — TestClient over the authenticated source_registry
                            app (with the v1_query router) for dataset setup
                            and authenticated-path parity checks;
      - `dp_client`       — TestClient over a bare app mounting ONLY the DP
                            router (no auth dependency exists on it);
      - `v1_query`        — the reloaded `v1_query` module;
      - `dp_query`        — the reloaded `dp_query` module.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")
    monkeypatch.delenv("RB_PROXY_SECRET", raising=False)
    monkeypatch.delenv("RB_TEST_QUERY_QUOTA", raising=False)
    # A handful of tests in this module assert the authenticated-route 429
    # (`test_dp_query_consumes_no_quota`) — that requires the quota subsystem
    # to be on. Turn it on for the whole fixture; the DP path is unaffected
    # because the DP never runs the quota check regardless of the env var.
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
    import services.query_api.dp_query as dp_query
    importlib.reload(dp_query)
    v1_query.cache_clear()
    v1_query._RESULTS.clear()

    main_mod.app.include_router(v1_query.router)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    dp_app = FastAPI(title="dp-test")
    dp_app.include_router(dp_query.router)

    class _Env:
        pass

    e = _Env()
    e.auth_client = TestClient(main_mod.app)
    e.dp_client = TestClient(dp_app)
    e.v1_query = v1_query
    e.dp_query = dp_query
    return e


# --- helpers --------------------------------------------------------------


def _signup(client, email="alice@example.com", password="password123"):
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _tenant_id(client, signup_body):
    r = client.get("/auth/me", headers=_auth(signup_body["token"]))
    return r.json()["tenant"]["id"]


def _run_pipeline_once():
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


def _make_indexed_dataset(client, token, name="test", dimension=4, records=None):
    r = client.post("/v1/datasets", headers=_auth(token), json={"name": name, "dimension": dimension})
    assert r.status_code == 201, r.text
    if records is None:
        records = [
            {"id": f"doc-{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"title": f"t{i}"}}
            for i in range(10)
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
    return records


# --- DP query happy path --------------------------------------------------


def test_dp_query_via_tenant_header_succeeds(env):
    """A query carrying `X-RB-Tenant-Id` (and no Authorization) succeeds."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    r = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Tenant-Id": tid},
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] in ("hot", "cold")
    assert isinstance(body["latency_ms"], int)
    assert len(body["matches"]) > 0
    for m in body["matches"]:
        assert isinstance(m["id"], str)
        assert isinstance(m["score"], (int, float))
        assert isinstance(m["metadata"], dict)


def test_dp_query_result_matches_authenticated_path(env):
    """The DP route returns the same matches as the authenticated route."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    q = {"dataset": "test", "vector": [3.0, 0.0, 0.0, 0.0], "top_k": 10}

    auth_resp = env.auth_client.post("/v1/query", headers=_auth(s["token"]), json=q)
    assert auth_resp.status_code == 200, auth_resp.text
    dp_resp = env.dp_client.post("/v1/query", headers={"X-RB-Tenant-Id": tid}, json=q)
    assert dp_resp.status_code == 200, dp_resp.text

    # `latency_ms` varies; the matches + their ordering must be identical.
    assert dp_resp.json()["matches"] == auth_resp.json()["matches"]


def test_dp_query_no_auth_header_needed(env):
    """The DP route reads the tenant from `X-RB-Tenant-Id`, not Authorization.

    Even an Authorization header that would normally fail is irrelevant — the
    DP never parses it.
    """
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    r = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Tenant-Id": tid, "Authorization": "Bearer garbage-token"},
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 3},
    )
    assert r.status_code == 200, r.text


# --- missing tenant header ------------------------------------------------


def test_dp_query_missing_tenant_header_400(env):
    """No `X-RB-Tenant-Id` → 400 with the v1 error envelope."""
    r = env.dp_client.post(
        "/v1/query",
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_request"


def test_dp_query_empty_tenant_header_400(env):
    """An empty `X-RB-Tenant-Id` is treated as missing → 400."""
    r = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Tenant-Id": ""},
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_request"


# --- proxy shared-secret --------------------------------------------------


def test_dp_query_proxy_secret_set_and_correct_ok(env, monkeypatch):
    """`RB_PROXY_SECRET` set + matching `X-RB-Proxy-Secret` → query runs."""
    monkeypatch.setenv("RB_PROXY_SECRET", "s3cr3t")
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    r = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Tenant-Id": tid, "X-RB-Proxy-Secret": "s3cr3t"},
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 3},
    )
    assert r.status_code == 200, r.text


def test_dp_query_proxy_secret_set_and_wrong_403(env, monkeypatch):
    """`RB_PROXY_SECRET` set + wrong `X-RB-Proxy-Secret` → 403."""
    monkeypatch.setenv("RB_PROXY_SECRET", "s3cr3t")
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    r = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Tenant-Id": tid, "X-RB-Proxy-Secret": "wrong"},
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "proxy_unauthorized"


def test_dp_query_proxy_secret_set_and_missing_403(env, monkeypatch):
    """`RB_PROXY_SECRET` set but no `X-RB-Proxy-Secret` header → 403."""
    monkeypatch.setenv("RB_PROXY_SECRET", "s3cr3t")
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    r = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Tenant-Id": tid},
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "proxy_unauthorized"


def test_dp_query_proxy_secret_unset_no_secret_needed(env):
    """`RB_PROXY_SECRET` unset → the secret check is skipped entirely."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    # No `X-RB-Proxy-Secret` header at all — still allowed.
    r = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Tenant-Id": tid},
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 3},
    )
    assert r.status_code == 200, r.text


def test_dp_proxy_secret_checked_before_tenant_header(env, monkeypatch):
    """The secret check runs first: a bad secret 403s even with no tenant."""
    monkeypatch.setenv("RB_PROXY_SECRET", "s3cr3t")
    r = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Proxy-Secret": "wrong"},
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "proxy_unauthorized"


# --- DP path consumes no quota --------------------------------------------


def test_dp_query_consumes_no_quota(env, monkeypatch):
    """A tenant at quota 0 can still query through the DP route.

    `RB_TEST_QUERY_QUOTA=0` means the daily query quota is fully exhausted —
    the authenticated route would 429. The DP route must NOT consume or check
    query quota (it moved to the CP), so the query still succeeds.
    """
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    # Drive the tenant to exhausted query quota.
    monkeypatch.setenv("RB_TEST_QUERY_QUOTA", "0")
    import adapters.state.state as state_mod
    state_mod._MEM_TENANTS[tid]["daily_query_quota"] = 0

    # Authenticated route is now over-quota → 429 (sanity check).
    auth_resp = env.auth_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert auth_resp.status_code == 429, auth_resp.text
    assert auth_resp.json()["error"]["code"] == "query_quota_exceeded"

    # DP route ignores query quota entirely → still 200.
    dp_resp = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Tenant-Id": tid},
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert dp_resp.status_code == 200, dp_resp.text

    # And it left `queries_today` untouched — no quota was consumed.
    usage = state_mod.get_usage(tid)
    assert usage["queries_today"] == 0


# --- DP body re-validation ------------------------------------------------


def test_dp_query_revalidates_body(env):
    """The DP re-validates the body — a dimension mismatch is a 400 there."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    r = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Tenant-Id": tid},
        json={"dataset": "test", "vector": [0.0, 0.0]},  # 2 != 4
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "dimension_mismatch"


def test_dp_query_cross_tenant_404(env):
    """A tenant header for tenant B querying tenant A's dataset → 404."""
    a = _signup(env.auth_client, email="a@example.com")
    b = _signup(env.auth_client, email="b@example.com")
    _make_indexed_dataset(env.auth_client, a["token"], name="a-only")
    b_tid = _tenant_id(env.auth_client, b)

    r = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Tenant-Id": b_tid},
        json={"dataset": "a-only", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "dataset_not_found"


# --- DP status endpoint ---------------------------------------------------


def test_dp_query_status_unknown_job_not_ready(env):
    """The DP status endpoint answers a poll for an unknown job."""
    r = env.dp_client.get("/v1/query/status/job_does_not_exist")
    assert r.status_code == 200, r.text
    assert r.json() == {"ready": False}


def test_dp_query_status_returns_ready_result(env):
    """An ephemeral result stashed in the store is visible via DP status."""
    job_id = "job_dp_status_test"
    env.v1_query._RESULTS[job_id] = {
        "matches": [{"id": "doc-1", "score": 0.0, "metadata": {}}],
        "latency_ms": 7,
    }
    r = env.dp_client.get(f"/v1/query/status/{job_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ready"] is True
    assert body["mode"] == "ephemeral"
    assert body["matches"][0]["id"] == "doc-1"


def test_dp_query_status_proxy_secret_enforced(env, monkeypatch):
    """The status endpoint also enforces the proxy secret when set."""
    monkeypatch.setenv("RB_PROXY_SECRET", "s3cr3t")
    bad = env.dp_client.get("/v1/query/status/job_x", headers={"X-RB-Proxy-Secret": "wrong"})
    assert bad.status_code == 403, bad.text
    ok = env.dp_client.get("/v1/query/status/job_x", headers={"X-RB-Proxy-Secret": "s3cr3t"})
    assert ok.status_code == 200, ok.text


def test_dp_query_no_shard_ephemeral_fallback(env):
    """A dataset with no shard falls back to the ephemeral path on the DP."""
    s = _signup(env.auth_client)
    r = env.auth_client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "empty", "dimension": 4}
    )
    assert r.status_code == 201, r.text
    tid = _tenant_id(env.auth_client, s)

    r = env.dp_client.post(
        "/v1/query",
        headers={"X-RB-Tenant-Id": tid},
        json={"dataset": "empty", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "ephemeral"
    assert body["matches"] == []
    assert body["job_id"].startswith("job_")


# --- extracted core: auth-free / quota-free -------------------------------


def test_validate_query_body_is_pure(env):
    """`validate_query_body` validates with no auth/quota and no quota burn."""
    from fastapi.responses import JSONResponse

    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    parsed = env.v1_query.validate_query_body(
        {"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 4}, tid
    )
    assert not isinstance(parsed, JSONResponse)
    assert parsed.dataset_name == "test"
    assert parsed.top_k == 4
    assert parsed.vector == [0.0, 0.0, 0.0, 0.0]

    # A bad body returns the v1 error response, still no auth involved.
    bad = env.v1_query.validate_query_body(
        {"dataset": "test", "vector": [0.0, 0.0]}, tid
    )
    assert isinstance(bad, JSONResponse)
    assert bad.status_code == 400

    # No quota was consumed by validation.
    import adapters.state.state as state_mod
    assert state_mod.get_usage(tid)["queries_today"] == 0


def test_execute_v1_query_skips_quota_when_callback_none(env, monkeypatch):
    """`execute_v1_query(..., consume_quota=None)` consumes no quota."""
    from fastapi.responses import JSONResponse

    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    import adapters.state.state as state_mod
    state_mod._MEM_TENANTS[tid]["daily_query_quota"] = 0

    result = env.v1_query.execute_v1_query(
        tid,
        {"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 3},
        consume_quota=None,
    )
    assert not isinstance(result, JSONResponse)
    assert result["mode"] in ("hot", "cold")
    assert state_mod.get_usage(tid)["queries_today"] == 0


def test_execute_v1_query_invokes_quota_callback_after_validation(env):
    """The quota callback fires only AFTER body validation succeeds."""
    from fastapi.responses import JSONResponse

    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    calls = {"n": 0}

    def _cb():
        calls["n"] += 1
        return None

    # A body that fails validation → callback never runs.
    bad = env.v1_query.execute_v1_query(
        tid, {"dataset": "test", "vector": [0.0]}, consume_quota=_cb
    )
    assert isinstance(bad, JSONResponse)
    assert bad.status_code == 400
    assert calls["n"] == 0

    # A valid body → callback runs exactly once, before the search.
    ok = env.v1_query.execute_v1_query(
        tid,
        {"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 2},
        consume_quota=_cb,
    )
    assert not isinstance(ok, JSONResponse)
    assert calls["n"] == 1


def test_execute_v1_query_quota_callback_can_reject(env):
    """A quota callback returning a JSONResponse short-circuits the search."""
    from fastapi.responses import JSONResponse

    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    rejection = JSONResponse(status_code=429, content={"error": {"code": "x", "message": "y"}})
    result = env.v1_query.execute_v1_query(
        tid,
        {"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
        consume_quota=lambda: rejection,
    )
    assert result is rejection
