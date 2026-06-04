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


# --- Split connect/read CP→DP timeout (#26) -------------------------------
#
# The CP proxies POST /v1/query to a private Query-DP over a persistent
# httpx.AsyncClient. A single ~5s scalar timeout killed any query that
# resolved a large COLD consolidated shard (first S3 GET + large shard
# deserialise can legitimately run >5s on a healthy DP), surfacing a spurious
# 504. The fix splits the timeout into a SHORT connect (fast-fail a dead DP)
# and a generously LARGER read (let a slow large-cold-shard query on a healthy
# DP finish), both env-tunable. These tests pin the new behaviour.


def test_query_timeout_returns_httpx_timeout_with_split_values(monkeypatch):
    """`_query_timeout()` returns an `httpx.Timeout` with split connect/read.

    Not a single scalar — a short connect plus a generously larger read, so a
    slow large-cold-shard query on a HEALTHY DP isn't killed while a dead DP
    still fast-fails on connect.
    """
    import httpx
    import services.query_api.query_proxy as qp

    for var in (
        "RB_QUERY_DP_READ_TIMEOUT_S",
        "RB_QUERY_DP_CONNECT_TIMEOUT_S",
        "RB_QUERY_DP_TIMEOUT_S",
    ):
        monkeypatch.delenv(var, raising=False)

    t = qp._query_timeout()
    assert isinstance(t, httpx.Timeout)
    # Read default must be clearly > the old 5s so a large cold shard survives.
    assert t.read is not None and t.read >= 30.0, (
        f"read timeout default must be >= 30s (was the old 5s); got {t.read}"
    )
    # Connect must be short (fast-fail a dead DP) and well under the read.
    assert t.connect is not None and t.connect <= 10.0, (
        f"connect timeout default must be short (<=10s); got {t.connect}"
    )
    assert t.connect < t.read, "connect must be shorter than read"
    # write/pool must be set (httpx requires all four, or a default).
    assert t.write is not None and t.pool is not None


def test_query_timeout_read_env_override(monkeypatch):
    """`RB_QUERY_DP_READ_TIMEOUT_S` overrides the read timeout, read live."""
    import services.query_api.query_proxy as qp

    monkeypatch.delenv("RB_QUERY_DP_TIMEOUT_S", raising=False)
    monkeypatch.setenv("RB_QUERY_DP_READ_TIMEOUT_S", "120")
    t = qp._query_timeout()
    assert t.read == 120.0


def test_query_timeout_connect_env_override(monkeypatch):
    """`RB_QUERY_DP_CONNECT_TIMEOUT_S` overrides the connect timeout, read live."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_QUERY_DP_CONNECT_TIMEOUT_S", "1.5")
    t = qp._query_timeout()
    assert t.connect == 1.5


def test_query_timeout_legacy_env_still_sets_read(monkeypatch):
    """The legacy `RB_QUERY_DP_TIMEOUT_S` knob still tunes the READ timeout.

    Backwards-compat: a deployment that set the old single-scalar knob keeps
    tuning the (now read) timeout; the dedicated read knob takes precedence
    when both are set.
    """
    import services.query_api.query_proxy as qp

    monkeypatch.delenv("RB_QUERY_DP_READ_TIMEOUT_S", raising=False)
    monkeypatch.setenv("RB_QUERY_DP_TIMEOUT_S", "45")
    t = qp._query_timeout()
    assert t.read == 45.0


def test_query_timeout_read_knob_precedence_over_legacy(monkeypatch):
    """The dedicated read knob wins over the legacy scalar when both are set."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_QUERY_DP_TIMEOUT_S", "7")
    monkeypatch.setenv("RB_QUERY_DP_READ_TIMEOUT_S", "90")
    t = qp._query_timeout()
    assert t.read == 90.0


def test_query_timeout_bad_env_falls_back_to_defaults(monkeypatch):
    """A non-numeric env value falls back to the sane defaults, not a crash."""
    import httpx
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_QUERY_DP_READ_TIMEOUT_S", "not-a-number")
    monkeypatch.setenv("RB_QUERY_DP_CONNECT_TIMEOUT_S", "garbage")
    t = qp._query_timeout()
    assert isinstance(t, httpx.Timeout)
    assert t.read >= 30.0 and t.connect <= 10.0


@pytest.mark.parametrize("bad_value", ["-5", "0", "-0.5", "0.0"])
def test_query_timeout_read_nonpositive_falls_back_to_default(monkeypatch, bad_value):
    """A negative/zero read knob is INVALID → default, NOT clamped to ~0.

    Clamping `-5`/`0` down to a sub-second budget would 504 essentially every
    real query (the opposite of intent). With no legacy knob set it must
    resolve to the 30s default.
    """
    import services.query_api.query_proxy as qp

    monkeypatch.delenv("RB_QUERY_DP_TIMEOUT_S", raising=False)
    monkeypatch.setenv("RB_QUERY_DP_READ_TIMEOUT_S", bad_value)
    t = qp._query_timeout()
    assert t.read == qp._DEFAULT_READ_TIMEOUT_S
    assert t.read >= 30.0, f"non-positive read knob must not clamp to ~0; got {t.read}"


@pytest.mark.parametrize("bad_value", ["-5", "0", "-0.5", "0.0"])
def test_query_timeout_connect_nonpositive_falls_back_to_default(monkeypatch, bad_value):
    """A negative/zero connect knob is INVALID → default, NOT clamped to ~0."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_QUERY_DP_CONNECT_TIMEOUT_S", bad_value)
    t = qp._query_timeout()
    assert t.connect == qp._DEFAULT_CONNECT_TIMEOUT_S


@pytest.mark.parametrize("bad_value", ["", "   ", "bad", "-5", "0"])
def test_query_timeout_invalid_read_falls_through_to_legacy(monkeypatch, bad_value):
    """An empty/garbage/non-positive NEW read knob must NOT mask a valid legacy.

    `RB_QUERY_DP_READ_TIMEOUT_S=''` (or 'bad'/'-5'/'0') with a valid legacy
    `RB_QUERY_DP_TIMEOUT_S=45` must resolve to 45 (legacy), not 30 (default).
    """
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_QUERY_DP_READ_TIMEOUT_S", bad_value)
    monkeypatch.setenv("RB_QUERY_DP_TIMEOUT_S", "45")
    t = qp._query_timeout()
    assert t.read == 45.0


def test_query_timeout_empty_read_knob_treated_as_unset(monkeypatch):
    """An empty-string new read knob is treated the same as unset → default."""
    import services.query_api.query_proxy as qp

    monkeypatch.delenv("RB_QUERY_DP_TIMEOUT_S", raising=False)
    monkeypatch.setenv("RB_QUERY_DP_READ_TIMEOUT_S", "")
    t = qp._query_timeout()
    assert t.read == qp._DEFAULT_READ_TIMEOUT_S


def test_query_timeout_valid_read_still_wins_over_legacy(monkeypatch):
    """Precedence unchanged: a VALID new read knob still beats the legacy knob."""
    import services.query_api.query_proxy as qp

    monkeypatch.setenv("RB_QUERY_DP_TIMEOUT_S", "45")
    monkeypatch.setenv("RB_QUERY_DP_READ_TIMEOUT_S", "90")
    t = qp._query_timeout()
    assert t.read == 90.0


def test_query_timeout_tiny_positive_read_floored(monkeypatch):
    """A legitimately-tiny POSITIVE value keeps the 0.1s floor (not rejected)."""
    import services.query_api.query_proxy as qp

    monkeypatch.delenv("RB_QUERY_DP_TIMEOUT_S", raising=False)
    monkeypatch.setenv("RB_QUERY_DP_READ_TIMEOUT_S", "0.001")
    t = qp._query_timeout()
    assert t.read == 0.1


def test_build_client_uses_configured_timeout(monkeypatch):
    """`_build_client` bakes the configured `httpx.Timeout` into the client.

    The persistent per-pool client must carry the split connect/read timeout
    so every request through it inherits the generous read budget for large
    cold shards.
    """
    import services.query_api.query_proxy as qp

    monkeypatch.delenv("RB_QUERY_DP_TIMEOUT_S", raising=False)
    monkeypatch.setenv("RB_QUERY_DP_READ_TIMEOUT_S", "33")
    monkeypatch.setenv("RB_QUERY_DP_CONNECT_TIMEOUT_S", "2")

    client = qp._build_client("http://dp.internal:8090")
    try:
        t = client.timeout
        assert t.read == 33.0
        assert t.connect == 2.0
    finally:
        import asyncio

        asyncio.run(client.aclose())


def test_slow_within_read_timeout_succeeds_where_old_5s_failed(monkeypatch):
    """A slow-but-within-read-timeout DP response succeeds.

    Simulates the large-cold-shard case: the DP takes ~0.2s to answer, which
    is well within the (>=30s) read timeout but would have blown the old 5s
    scalar if it had been, say, 6s. We assert the proxy returns the DP's 200
    and that the per-request timeout it passed has a read budget >= 30s — the
    headroom that lets a genuine 6s+ cold-shard query through.
    """
    import asyncio

    import httpx
    import services.query_api.query_proxy as qp

    _stub_pool(monkeypatch)
    monkeypatch.delenv("RB_QUERY_DP_TIMEOUT_S", raising=False)
    monkeypatch.delenv("RB_QUERY_DP_READ_TIMEOUT_S", raising=False)

    seen = {}

    class _FakeResp:
        content = b'{"matches": []}'
        status_code = 200
        headers = {"content-type": "application/json"}

    class _FakeClient:
        async def request(self, *_a, **kw):
            seen["timeout"] = kw.get("timeout")
            # A slow DP — comfortably inside the generous read budget.
            await asyncio.sleep(0.2)
            return _FakeResp()

    monkeypatch.setattr(qp, "_get_client", lambda _base: _FakeClient())

    async def _run():
        return await qp._proxy("POST", "/v1/query", "ten_slow", body=b"{}")

    resp = asyncio.run(_run())
    assert resp.status_code == 200
    t = seen["timeout"]
    assert isinstance(t, httpx.Timeout)
    assert t.read >= 30.0, (
        "the per-request read budget must clear 30s so a 6s+ large-cold-shard "
        f"query is not killed; got {t.read}"
    )


def test_connect_failure_still_fast_fails_to_503(monkeypatch):
    """A connect failure still fast-fails the connect-retry chain → 503.

    The split timeout must not regress the connect path: a dead DP raising
    ConnectTimeout/ConnectError is still retried then mapped to 503
    `query_unavailable` (a query is read-only and safe to retry on connect).
    """
    import asyncio

    import httpx
    import services.query_api.query_proxy as qp

    _stub_pool(monkeypatch)
    monkeypatch.setenv("RB_QUERY_DP_CONNECT_RETRIES", "1")
    monkeypatch.setenv("RB_QUERY_DP_CONNECT_BACKOFF_S", "0")
    calls = {"n": 0}

    class _FakeClient:
        async def request(self, *_a, **_kw):
            calls["n"] += 1
            raise httpx.ConnectTimeout("dead DP")

    monkeypatch.setattr(qp, "_get_client", lambda _base: _FakeClient())

    async def _run():
        return await qp._proxy("POST", "/v1/query", "ten_dead", body=b"{}")

    resp = asyncio.run(_run())
    assert resp.status_code == 503
    import json

    assert json.loads(resp.body)["error"]["code"] == "query_unavailable"
    # 1 initial + 1 retry.
    assert calls["n"] == 2


def test_read_timeout_still_maps_to_504(monkeypatch):
    """The ReadTimeout→504 `query_timeout` mapping is preserved (not retried).

    Even with the generous read budget, a DP that blows the (larger) read
    timeout still maps to 504 `query_timeout` and is NOT retried — the DP may
    have done work.
    """
    import asyncio
    import json

    import httpx
    import services.query_api.query_proxy as qp

    _stub_pool(monkeypatch)
    monkeypatch.setenv("RB_QUERY_DP_CONNECT_RETRIES", "3")
    calls = {"n": 0}

    class _FakeClient:
        async def request(self, *_a, **_kw):
            calls["n"] += 1
            raise httpx.ReadTimeout("still too slow")

    monkeypatch.setattr(qp, "_get_client", lambda _base: _FakeClient())

    async def _run():
        return await qp._proxy("POST", "/v1/query", "ten_to", body=b"{}")

    resp = asyncio.run(_run())
    assert resp.status_code == 504
    assert json.loads(resp.body)["error"]["code"] == "query_timeout"
    assert calls["n"] == 1, "a read timeout must not be retried"
