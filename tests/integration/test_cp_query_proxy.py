"""Tests for the CP reverse-proxy query router.

The customer-facing query path runs behind a Control Plane (CP) that
authenticates + rate-limits + quota-checks, then reverse-proxies the request
to a private Query Data Plane (DP). This module covers the CP side —
`services/query_api/query_proxy.py` and the assembled CP app
`services/control_plane/cp_app.py`.

The proxy is exercised **in-process, no real network**: the proxy's
`httpx.AsyncClient` is wired to the real `dp_app.py` ASGI app via
`httpx.ASGITransport`, registered under the base URL `resolve_dp_base_url`
produces. So a `POST /v1/query` to the CP genuinely flows
CP-handler -> httpx -> DP-app -> DP-handler and back.

Asserted behaviour:
  - a query proxied CP->DP returns the same result as calling the DP directly;
  - the CP consumes query quota — a tenant at quota 0 gets 429 from the CP and
    the request never reaches the DP;
  - the CP-verified `X-RB-Tenant-Id` header is set on the DP call (and the
    customer `Authorization` is NOT forwarded);
  - auth is enforced at the CP edge (no/invalid key -> 401);
  - `X-RB-Proxy-Secret` is forwarded when `RB_PROXY_SECRET` is set;
  - failure mapping: DP 4xx forwarded verbatim, connect failure -> 503
    `query_unavailable`, CP->DP timeout -> 504 `query_timeout`;
  - the status endpoint proxies (and consumes no quota / no rate limit).
"""
from __future__ import annotations

import importlib
import json
import os

import httpx
import pytest


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def env(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Reset state + reload the pipeline, wiring the CP proxy at the real DP app.

    Mirrors the `test_dp_query.py` fixture, then additionally:
      - builds the Query-DP ASGI app (`dp_app.app`);
      - registers an `httpx.AsyncClient` on the proxy's per-pool client
        registry, wired to that DP app via `httpx.ASGITransport`, under the
        base URL `resolve_dp_base_url('shared')` produces — so the CP proxy's
        CP->DP hop runs entirely in-process.

    Returns an object exposing:
      - `cp_client`   — TestClient over the assembled CP app (auth + datasets
                        + the proxied query surface);
      - `dp_client`   — TestClient over the bare DP app, for parity checks;
      - `state`       — the reloaded state module;
      - `query_proxy` — the reloaded proxy module.
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
    # `test_cp_proxy_consumes_query_quota` asserts the proxy's query-quota
    # consume and 429. That path is gated behind `RB_ENABLE_QUOTAS`; turn it
    # on for the whole fixture so the proxy's quota check is exercised.
    monkeypatch.setenv("RB_ENABLE_QUOTAS", "true")
    monkeypatch.setenv("QUERY_DP_URL", "http://query-dp.test")

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
    import services.query_api.dp_app as dp_app_mod
    importlib.reload(dp_app_mod)
    import services.query_api.query_proxy as query_proxy
    importlib.reload(query_proxy)
    v1_query.cache_clear()
    v1_query._RESULTS.clear()

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # The CP app: source_registry app + the proxy router. Mounting v1_query's
    # router too so dataset setup + an authenticated-path parity check are
    # available — but the proxy router is what the query tests hit. They use
    # the SAME `/v1/query` path; FastAPI matches the FIRST registered, so the
    # v1_query router is mounted AFTER the proxy router would shadow it. To
    # keep dataset setup on `main_mod.app` and the proxy isolated, the proxy
    # router goes on its own CP app and v1_query stays on `main_mod.app`.
    main_mod.app.include_router(v1_query.router)
    cp_app = FastAPI(title="cp-test")
    # The CP app reuses the auth + datasets surface, then mounts the proxy.
    cp_app.include_router(auth_mod.router, prefix="/auth")
    quota_mod.install_rate_limit_handler(cp_app)
    auth_mod.install_exception_handlers(cp_app)
    cp_app.include_router(query_proxy.router)

    # The bare Query-DP app — the proxy target.
    dp_app = dp_app_mod.app

    # Wire the proxy's CP->DP client to the DP ASGI app, in-process. The base
    # URL must match what `resolve_dp_base_url('shared')` returns.
    dp_base = query_proxy.resolve_dp_base_url("shared")
    asgi_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=dp_app),
        base_url=dp_base,
    )
    query_proxy.register_dp_client(dp_base, asgi_client)

    class _Env:
        pass

    e = _Env()
    e.auth_client = TestClient(main_mod.app)   # dataset setup + authed parity
    e.cp_client = TestClient(cp_app)           # the proxy under test
    e.dp_client = TestClient(dp_app)           # direct-DP parity
    e.state = state_mod
    e.query_proxy = query_proxy
    e.dp_base = dp_base
    yield e

    import anyio
    anyio.run(query_proxy.reset_dp_clients)


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


# --- happy path: CP proxies to DP -----------------------------------------


def test_cp_proxy_query_succeeds(env):
    """A query through the CP proxy returns a normal v1 search result."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])

    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
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


def test_cp_proxy_result_matches_direct_dp_call(env):
    """The CP-proxied result is identical to calling the DP directly."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    q = {"dataset": "test", "vector": [3.0, 0.0, 0.0, 0.0], "top_k": 10}

    cp_resp = env.cp_client.post("/v1/query", headers=_auth(s["token"]), json=q)
    assert cp_resp.status_code == 200, cp_resp.text
    dp_resp = env.dp_client.post("/v1/query", headers={"X-RB-Tenant-Id": tid}, json=q)
    assert dp_resp.status_code == 200, dp_resp.text

    # `latency_ms` varies; matches + ordering must be identical.
    assert cp_resp.json()["matches"] == dp_resp.json()["matches"]


# --- auth enforced at the CP edge -----------------------------------------


def test_cp_proxy_requires_auth(env):
    """No `Authorization` header → 401, the request never reaches the DP."""
    r = env.cp_client.post(
        "/v1/query",
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 401, r.text


def test_cp_proxy_rejects_invalid_key(env):
    """A garbage bearer token → 401 at the CP edge."""
    r = env.cp_client.post(
        "/v1/query",
        headers={"Authorization": "Bearer rb_live_totally-bogus-key"},
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 401, r.text


# --- the CP consumes query quota ------------------------------------------


def test_cp_proxy_consumes_query_quota(env, monkeypatch):
    """A tenant at quota 0 gets 429 from the CP — the request never reaches DP."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    # Exhaust the daily query quota.
    env.state._MEM_TENANTS[tid]["daily_query_quota"] = 0

    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 429, r.text
    assert r.json()["error"]["code"] == "query_quota_exceeded"


def test_cp_proxy_increments_queries_today(env):
    """A successful proxied query consumes exactly one unit of query quota."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    assert env.state.get_usage(tid)["queries_today"] == 0
    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 200, r.text
    assert env.state.get_usage(tid)["queries_today"] == 1


# --- trusted headers on the CP->DP hop ------------------------------------


def test_cp_proxy_sets_tenant_header_on_dp_call(env, monkeypatch):
    """The CP->DP call carries `X-RB-Tenant-Id` (and not `Authorization`).

    A stub transport captures the outbound CP->DP request and asserts the
    trusted header is the CP-verified tenant and the customer key is dropped.
    """
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])
    tid = _tenant_id(env.auth_client, s)

    captured = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = request.content
        return httpx.Response(200, json={"matches": [], "latency_ms": 1, "mode": "hot"})

    stub = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url=env.dp_base
    )
    env.query_proxy.register_dp_client(env.dp_base, stub)

    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 200, r.text
    assert captured["headers"].get("x-rb-tenant-id") == tid
    # The customer's Authorization header is NOT forwarded to the DP.
    assert "authorization" not in captured["headers"]
    # The body is forwarded byte-for-byte.
    assert json.loads(captured["body"])["dataset"] == "test"


def test_cp_proxy_forwards_proxy_secret_when_set(env, monkeypatch):
    """`X-RB-Proxy-Secret` is sent on the CP->DP call when `RB_PROXY_SECRET` set."""
    monkeypatch.setenv("RB_PROXY_SECRET", "s3cr3t")
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])

    captured = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"matches": [], "latency_ms": 1, "mode": "hot"})

    stub = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url=env.dp_base
    )
    env.query_proxy.register_dp_client(env.dp_base, stub)

    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 200, r.text
    assert captured["headers"].get("x-rb-proxy-secret") == "s3cr3t"


def test_cp_proxy_no_proxy_secret_when_unset(env):
    """No `X-RB-Proxy-Secret` is sent when `RB_PROXY_SECRET` is unset."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])

    captured = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"matches": [], "latency_ms": 1, "mode": "hot"})

    stub = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url=env.dp_base
    )
    env.query_proxy.register_dp_client(env.dp_base, stub)

    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 200, r.text
    assert "x-rb-proxy-secret" not in captured["headers"]


# --- failure mapping ------------------------------------------------------


def test_cp_proxy_forwards_dp_4xx_verbatim(env):
    """A DP 4xx is forwarded with the same status code and body."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])

    # A dimension mismatch makes the real DP return 400 `dimension_mismatch`.
    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0]},  # 2 != 4
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "dimension_mismatch"


def test_cp_proxy_forwards_dp_5xx_verbatim(env):
    """A DP 5xx is forwarded verbatim — NOT retried, NOT remapped."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])

    calls = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            500, json={"error": {"code": "internal", "message": "boom"}}
        )

    stub = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url=env.dp_base
    )
    env.query_proxy.register_dp_client(env.dp_base, stub)

    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 500, r.text
    assert r.json()["error"]["code"] == "internal"
    # A DP that answered with a 5xx is NOT retried.
    assert calls["n"] == 1


def test_cp_proxy_connect_failure_maps_to_503(env, monkeypatch):
    """A CP->DP connect failure (after retries) → 503 `query_unavailable`."""
    monkeypatch.setenv("RB_QUERY_DP_CONNECT_RETRIES", "2")
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])

    calls = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("connection refused")

    stub = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url=env.dp_base
    )
    env.query_proxy.register_dp_client(env.dp_base, stub)

    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "query_unavailable"
    # A connect failure IS retried: 1 initial + 2 retries = 3 attempts.
    assert calls["n"] == 3


def test_cp_proxy_timeout_maps_to_504(env):
    """A CP->DP read timeout → 504 `query_timeout`, and is NOT retried."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])

    calls = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("too slow")

    stub = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url=env.dp_base
    )
    env.query_proxy.register_dp_client(env.dp_base, stub)

    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 504, r.text
    assert r.json()["error"]["code"] == "query_timeout"
    # A timeout means the DP may have done work — NOT retried.
    assert calls["n"] == 1


def test_cp_proxy_remote_protocol_error_maps_to_502(env):
    """A `RemoteProtocolError` from the DP transport → v1-envelope 502.

    The DP closed the connection / sent an HTTP/2 GOAWAY mid-response. The
    request reached the DP and may have done work, so it is NOT retried. The
    CP must surface a v1 `{"error": {...}}` 502 — never a bare ASGI 500.
    """
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])

    calls = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.RemoteProtocolError("server disconnected mid-response")

    stub = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url=env.dp_base
    )
    env.query_proxy.register_dp_client(env.dp_base, stub)

    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "bad_gateway"
    # NOT retried — the request reached the DP.
    assert calls["n"] == 1


def test_cp_proxy_read_error_maps_to_502(env):
    """An `httpx.ReadError` (broken response stream) → v1-envelope 502."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])

    calls = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadError("connection reset reading response")

    stub = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url=env.dp_base
    )
    env.query_proxy.register_dp_client(env.dp_base, stub)

    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "bad_gateway"
    assert calls["n"] == 1


def test_cp_proxy_pool_timeout_maps_to_502(env):
    """An `httpx.PoolTimeout` (CP-side client pool) → v1-envelope 502."""
    s = _signup(env.auth_client)
    _make_indexed_dataset(env.auth_client, s["token"])

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.PoolTimeout("no connection slot available")

    stub = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url=env.dp_base
    )
    env.query_proxy.register_dp_client(env.dp_base, stub)

    r = env.cp_client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "bad_gateway"


# --- status endpoint proxies ----------------------------------------------


def test_cp_proxy_status_unknown_job(env):
    """The CP proxies a status poll for an unknown job to the DP."""
    s = _signup(env.auth_client)
    r = env.cp_client.get(
        "/v1/query/status/job_does_not_exist", headers=_auth(s["token"])
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"ready": False}


def test_cp_proxy_status_returns_ready_result(env):
    """A stashed ephemeral result is visible through the CP status proxy."""
    s = _signup(env.auth_client)
    job_id = "job_cp_status_test"
    import services.query_api.v1_query as v1_query
    v1_query._RESULTS[job_id] = {
        "matches": [{"id": "doc-1", "score": 0.0, "metadata": {}}],
        "latency_ms": 7,
    }
    r = env.cp_client.get(f"/v1/query/status/{job_id}", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ready"] is True
    assert body["mode"] == "ephemeral"
    assert body["matches"][0]["id"] == "doc-1"


def test_cp_proxy_status_requires_auth(env):
    """The status poll is authenticated at the CP edge — no key → 401."""
    r = env.cp_client.get("/v1/query/status/job_x")
    assert r.status_code == 401, r.text


def test_cp_proxy_status_consumes_no_quota(env):
    """A status poll consumes no query quota (parity with the legacy route)."""
    s = _signup(env.auth_client)
    tid = _tenant_id(env.auth_client, s)
    # Drive quota to 0 — a status poll must still work (no quota call).
    env.state._MEM_TENANTS[tid]["daily_query_quota"] = 0
    r = env.cp_client.get("/v1/query/status/job_x", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert env.state.get_usage(tid)["queries_today"] == 0


# --- the assembled CP app -------------------------------------------------


def test_cp_app_mounts_proxy_not_in_process_router(env):
    """`cp_app.app` exposes the proxied query surface and the catalog.

    The CP app must carry `/v1/query` + `/v1/datasets` (auth + catalog +
    proxied query) but NOT the in-process search internals.
    """
    import services.control_plane.cp_app as cp_app_mod
    importlib.reload(cp_app_mod)
    paths = {getattr(r, "path", None) for r in cp_app_mod.app.routes}
    assert "/v1/query" in paths
    assert "/v1/query/status/{job_id}" in paths
    assert "/v1/datasets" in paths
    assert "/auth/signup" in paths
