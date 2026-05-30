"""Hermetic unit tests for the CP query-proxy plumbing.

These cover the parts of the Control Plane query proxy that need no real
network, no MinIO and no indexed dataset:

  - `resolve_dp_base_url` — the DP-pool -> base-URL routing convention;
  - `get_tenant_dp_pool` — the `tenants.dp_pool` read, with the `'shared'`
    default for an unknown tenant / NULL column (memory mode);
  - migration `006_tenants_dp_pool.sql` is registered and well-formed
    (additive, idempotent `ADD COLUMN IF NOT EXISTS`).

The end-to-end CP->DP proxy behaviour (auth, quota, failure mapping, the
verbatim stream-back) is exercised in-process against the real `dp_app` in
`tests/integration/test_query_proxy.py`.
"""
from __future__ import annotations

import importlib

import pytest


# --- resolve_dp_base_url --------------------------------------------------


def test_resolve_shared_pool_uses_query_dp_url(monkeypatch):
    """`'shared'` resolves to `QUERY_DP_URL`."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("QUERY_DP_URL", "http://dp.internal:9000")
    assert qp.resolve_dp_base_url("shared") == "http://dp.internal:9000"


def test_resolve_shared_pool_dev_default(monkeypatch):
    """With `QUERY_DP_URL` unset, `'shared'` falls back to the dev default."""
    import services.query_api.query_proxy as qp

    monkeypatch.delenv("QUERY_DP_URL", raising=False)
    assert qp.resolve_dp_base_url("shared") == "http://localhost:8090"


def test_resolve_strips_trailing_slash(monkeypatch):
    """A trailing slash on `QUERY_DP_URL` is stripped (callers append paths)."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("QUERY_DP_URL", "http://dp.internal:9000/")
    assert qp.resolve_dp_base_url("shared") == "http://dp.internal:9000"


def test_resolve_empty_pool_treated_as_shared(monkeypatch):
    """An empty pool name resolves to the shared pool (fail-safe default)."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("QUERY_DP_URL", "http://shared:8090")
    assert qp.resolve_dp_base_url("") == "http://shared:8090"


def test_resolve_dedicated_pool_uses_per_tenant_env(monkeypatch):
    """`'dedicated-<tenant>'` resolves via the per-tenant `QUERY_DP_URL_<T>`."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("QUERY_DP_URL_TEN_ABC", "http://dedicated-abc:9100")
    assert (
        qp.resolve_dp_base_url("dedicated-ten_abc") == "http://dedicated-abc:9100"
    )


def test_resolve_dedicated_pool_unprovisioned_falls_back_to_shared(monkeypatch):
    """A dedicated pool with no env var falls back to the shared pool.

    No dedicated Query-DP pool is deployed yet — a tenant flagged for
    one but not actually provisioned must keep using `shared`, never route to
    an unreachable host.
    """
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("QUERY_DP_URL", "http://shared:8090")
    monkeypatch.delenv("QUERY_DP_URL_TEN_XYZ", raising=False)
    assert qp.resolve_dp_base_url("dedicated-ten_xyz") == "http://shared:8090"


def test_resolve_unrecognised_pool_falls_back_to_shared(monkeypatch):
    """An unrecognised pool name resolves to the shared pool."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("QUERY_DP_URL", "http://shared:8090")
    assert qp.resolve_dp_base_url("garbage-pool") == "http://shared:8090"


# --- get_tenant_dp_pool (memory mode) -------------------------------------


@pytest.fixture
def state(monkeypatch):
    """A fresh in-memory state adapter with cleared tenant tables."""
    monkeypatch.setenv("DATABASE_URL", "memory://test")
    import adapters.state.state as state_mod

    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL"):
        getattr(state_mod, attr).clear()
    yield state_mod
    importlib.reload(state_mod)


def test_get_tenant_dp_pool_defaults_to_shared_for_unknown_tenant(state):
    """An unknown tenant id resolves to `'shared'` — routing never fails open."""
    assert state.get_tenant_dp_pool("no_such_tenant") == "shared"


def test_get_tenant_dp_pool_defaults_to_shared_for_new_tenant(state):
    """A freshly-created tenant (no `dp_pool` key) reads back as `'shared'`."""
    state.create_tenant("ten_new", "new@example.com", "pw")
    assert state.get_tenant_dp_pool("ten_new") == "shared"


def test_get_tenant_dp_pool_returns_stored_value(state):
    """A tenant row carrying an explicit `dp_pool` returns that value."""
    state.create_tenant("ten_ded", "ded@example.com", "pw")
    state._MEM_TENANTS["ten_ded"]["dp_pool"] = "dedicated-ten_ded"
    assert state.get_tenant_dp_pool("ten_ded") == "dedicated-ten_ded"


def test_get_tenant_dp_pool_null_value_defaults_to_shared(state):
    """A NULL/empty `dp_pool` value resolves to `'shared'`."""
    state.create_tenant("ten_null", "null@example.com", "pw")
    state._MEM_TENANTS["ten_null"]["dp_pool"] = None
    assert state.get_tenant_dp_pool("ten_null") == "shared"


def test_resolver_maps_default_pool(state, monkeypatch):
    """End-to-end: the `'shared'` default a new tenant gets maps to a base URL."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("QUERY_DP_URL", "http://shared:8090")
    state.create_tenant("ten_e2e", "e2e@example.com", "pw")
    pool = state.get_tenant_dp_pool("ten_e2e")
    assert pool == "shared"
    assert qp.resolve_dp_base_url(pool) == "http://shared:8090"


# --- migration 006 file integrity -----------------------------------------


def test_migration_006_registered():
    """`006_tenants_dp_pool` is registered in `_MIGRATION_VERSIONS`.

    The position pin (was "last") was relaxed when migration 007 was
    appended for the DP residency registry — a strict last-position
    assertion would force every future migration to also edit this test.
    Membership is the load-bearing invariant for 006; head-position is
    pinned independently (`001_tenants_and_keys` must remain the
    bootstrap migration, since the migrator runs in declared order and
    every later migration assumes 001's tables exist).
    """
    import adapters.state.state as state_mod

    assert "006_tenants_dp_pool" in state_mod._MIGRATION_VERSIONS
    # Head-position pin: 001 is the bootstrap and must stay first.
    # Without this, a future loop that prepends a migration would
    # break every later migration's implicit "tenants table exists"
    # assumption — far worse than the position drift this test relaxed.
    assert state_mod._MIGRATION_VERSIONS[0] == "001_tenants_and_keys", (
        f"head migration must remain 001_tenants_and_keys; got "
        f"{state_mod._MIGRATION_VERSIONS[0]!r}"
    )


def test_migration_006_file_is_additive_and_idempotent():
    """The 006 SQL is a non-destructive, idempotent `ADD COLUMN IF NOT EXISTS`."""
    from pathlib import Path

    import adapters.state.state as state_mod

    sql = (
        Path(state_mod.__file__).parent
        / "migrations"
        / "006_tenants_dp_pool.sql"
    ).read_text(encoding="utf-8")
    lowered = sql.lower()
    assert "alter table tenants" in lowered
    assert "add column if not exists dp_pool" in lowered
    assert "default 'shared'" in lowered
    # Non-destructive: no DROP / DELETE / TRUNCATE.
    for destructive in ("drop ", "delete ", "truncate "):
        assert destructive not in lowered, f"006 must not contain {destructive!r}"


# --- Async offload of the sync state calls --------------------------------
#
# `query_proxy.py`'s `async def` handlers call the SYNC state functions
# `get_tenant_dp_pool` / `try_consume_query` — blocking DB I/O (possibly
# with a block-with-timeout sleep). These are offloaded with
# `asyncio.to_thread` so the event loop is never blocked. These tests prove
# (a) the call runs OFF the event loop's thread and (b) a request-scoped
# connection contextvar bound before the offload still resolves inside it.


def test_proxy_runs_get_tenant_dp_pool_off_the_event_loop(monkeypatch):
    """`_proxy` calls `get_tenant_dp_pool` on a worker thread, not the loop.

    `asyncio.to_thread` runs the sync state call on a thread distinct from the
    event loop's. The stub records the thread it ran on; it must differ from
    the loop thread — proof the blocking DB I/O is offloaded.
    """
    import asyncio
    import threading

    import services.query_api.query_proxy as qp

    loop_thread = threading.get_ident()
    ran_on = {}

    def _fake_get_tenant_dp_pool(tenant_id):
        ran_on["thread"] = threading.get_ident()
        return "shared"

    monkeypatch.setattr(qp, "get_tenant_dp_pool", _fake_get_tenant_dp_pool)

    # Stub the HTTP hop so the test stays offline.
    class _FakeResp:
        content = b"{}"
        status_code = 200
        headers = {"content-type": "application/json"}

    class _FakeClient:
        async def request(self, *_a, **_kw):
            return _FakeResp()

    monkeypatch.setattr(qp, "_get_client", lambda _base: _FakeClient())

    async def _run():
        return await qp._proxy("GET", "/v1/query/status/x", "ten_off")

    resp = asyncio.run(_run())
    assert resp.status_code == 200
    assert ran_on.get("thread") is not None
    assert ran_on["thread"] != loop_thread, (
        "get_tenant_dp_pool ran on the event loop thread — it must be "
        "offloaded with asyncio.to_thread"
    )


# --- httpx transport errors never escape `_proxy` -------------------------


def _stub_pool(monkeypatch):
    """Stub `get_tenant_dp_pool` so `_proxy` stays offline."""
    import services.query_api.query_proxy as qp

    monkeypatch.setattr(qp, "get_tenant_dp_pool", lambda _t: "shared")


@pytest.mark.parametrize(
    "exc",
    [
        "RemoteProtocolError",
        "ReadError",
        "NetworkError",
        "WriteError",
        "PoolTimeout",
    ],
)
def test_proxy_other_transport_errors_map_to_502(monkeypatch, exc):
    """Any non-connect/non-timeout `httpx` transport error → v1-envelope 502.

    `RemoteProtocolError`, `ReadError`, `NetworkError`, `WriteError`,
    `PoolTimeout` must NOT escape `_proxy` as a bare ASGI 500 — they map to a
    v1 `{"error": {"code": "bad_gateway", ...}}` 502.
    """
    import asyncio
    import json

    import httpx
    import services.query_api.query_proxy as qp

    _stub_pool(monkeypatch)
    exc_cls = getattr(httpx, exc)

    class _FakeClient:
        async def request(self, *_a, **_kw):
            raise exc_cls("boom")

    monkeypatch.setattr(qp, "_get_client", lambda _base: _FakeClient())

    async def _run():
        return await qp._proxy("POST", "/v1/query", "ten_x", body=b"{}")

    resp = asyncio.run(_run())
    assert resp.status_code == 502
    body = json.loads(resp.body)
    assert body["error"]["code"] == "bad_gateway"
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


def test_proxy_other_transport_error_is_not_retried(monkeypatch):
    """A non-connect transport error is NOT retried (the DP may have worked)."""
    import asyncio

    import httpx
    import services.query_api.query_proxy as qp

    _stub_pool(monkeypatch)
    monkeypatch.setenv("RB_QUERY_DP_CONNECT_RETRIES", "3")
    calls = {"n": 0}

    class _FakeClient:
        async def request(self, *_a, **_kw):
            calls["n"] += 1
            raise httpx.RemoteProtocolError("disconnect")

    monkeypatch.setattr(qp, "_get_client", lambda _base: _FakeClient())

    async def _run():
        return await qp._proxy("POST", "/v1/query", "ten_x", body=b"{}")

    resp = asyncio.run(_run())
    assert resp.status_code == 502
    assert calls["n"] == 1, "a non-connect transport error must not be retried"


# --- Unknown-tenant ValueError maps to a non-retryable 500 ---------------


def test_proxy_v1_query_unknown_tenant_maps_to_500(monkeypatch):
    """`try_consume_query` raising `ValueError` → non-retryable `500 internal_error`.

    A missing tenant row for an already-authenticated tenant is a server-state
    inconsistency, not a transient routing failure — a retryable 503 would be
    wrong (the row will not reappear on retry).
    """
    import asyncio
    import json

    import services.query_api.query_proxy as qp

    # The quota path is opt-in (`RB_ENABLE_QUOTAS`); turn it on so the proxy
    # actually calls `try_consume_query` and the ValueError can be raised.
    monkeypatch.setenv("RB_ENABLE_QUOTAS", "true")

    def _raise(_tenant):
        raise ValueError("tenant_not_found")

    monkeypatch.setattr(qp, "try_consume_query", _raise)

    # `proxy_v1_query` reads `request.body()` — a minimal fake request.
    class _FakeRequest:
        async def body(self):  # pragma: no cover - never reached
            return b"{}"

    resp = asyncio.run(qp.proxy_v1_query(_FakeRequest(), tenant_id="ten_gone"))
    assert resp.status_code == 500
    body = json.loads(resp.body)
    assert body["error"]["code"] == "internal_error"


# --- Async backoff between connect retries --------------------------------


def test_proxy_sleeps_between_connect_retries(monkeypatch):
    """`_proxy` awaits a backoff between connect retries — it does not spin.

    With 2 retries (3 attempts) there must be exactly 2 backoff sleeps,
    one between each pair of attempts — and none after the final attempt.
    """
    import asyncio

    import httpx
    import services.query_api.query_proxy as qp

    _stub_pool(monkeypatch)
    monkeypatch.setenv("RB_QUERY_DP_CONNECT_RETRIES", "2")
    monkeypatch.setenv("RB_QUERY_DP_CONNECT_BACKOFF_S", "0.001")

    sleeps = []
    real_sleep = asyncio.sleep

    async def _spy_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(qp.asyncio, "sleep", _spy_sleep)

    class _FakeClient:
        async def request(self, *_a, **_kw):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(qp, "_get_client", lambda _base: _FakeClient())

    async def _run():
        return await qp._proxy("POST", "/v1/query", "ten_x", body=b"{}")

    resp = asyncio.run(_run())
    assert resp.status_code == 503
    assert len(sleeps) == 2, (
        f"expected 2 inter-retry backoffs (3 attempts), got {len(sleeps)}"
    )
    assert all(s == 0.001 for s in sleeps)


def test_request_scoped_conn_resolves_inside_offloaded_state_call(monkeypatch):
    """A request connection bound before the offload is visible inside it.

    `asyncio.to_thread` copies the current `contextvars` into the worker
    thread, so a connection bound to `state._REQUEST_CONN` before `_proxy`
    runs must still resolve inside the offloaded `get_tenant_dp_pool` call.
    """
    import asyncio

    import adapters.state.state as state_mod
    import services.query_api.query_proxy as qp

    sentinel = object()
    seen = {}

    def _fake_get_tenant_dp_pool(tenant_id):
        # Runs on the to_thread worker — the contextvar must be the copied
        # value, i.e. the connection bound before the offload.
        seen["conn"] = state_mod._REQUEST_CONN.get()
        return "shared"

    monkeypatch.setattr(qp, "get_tenant_dp_pool", _fake_get_tenant_dp_pool)

    class _FakeResp:
        content = b"{}"
        status_code = 200
        headers = {"content-type": "application/json"}

    class _FakeClient:
        async def request(self, *_a, **_kw):
            return _FakeResp()

    monkeypatch.setattr(qp, "_get_client", lambda _base: _FakeClient())

    async def _run():
        token = state_mod._REQUEST_CONN.set(sentinel)
        try:
            return await qp._proxy("GET", "/v1/query/status/x", "ten_ctx")
        finally:
            state_mod._REQUEST_CONN.reset(token)

    resp = asyncio.run(_run())
    assert resp.status_code == 200
    assert seen.get("conn") is sentinel, (
        "the request-scoped connection contextvar did not resolve inside the "
        "offloaded state call — to_thread must copy contextvars"
    )
