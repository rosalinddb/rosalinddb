"""Unit coverage for the request-scoped-connection ASGI
middleware and the `PoolCheckoutTimeout` -> 503 mapping.

`RequestScopedConnectionMiddleware` binds ONE pooled Postgres connection per
HTTP request so a request that calls N state functions costs one pool
checkout, not N. These tests are hermetic — they run a tiny FastAPI app with
a `TestClient` and a fake pool, no real Postgres:

  - the middleware is a clean no-op in `memory://` mode;
  - a route inside a request scope sees the bound request connection, and
    every `pooled_conn()` block reuses it;
  - a `PoolCheckoutTimeout` escaping a route handler -> v1 503 envelope;
  - a `PoolCheckoutTimeout` raised by the middleware's own checkout -> v1 503
    envelope (it escapes before the app's handlers run).
"""
from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# --- fakes ----------------------------------------------------------------


class _FakePool:
    """A `ThreadedConnectionPool`-shaped stub with a bounded capacity."""

    def __init__(self, capacity: int = 5):
        self._capacity = capacity
        self._out = 0
        self._pool = [object()]
        self.commits = 0
        self.rollbacks = 0

    def getconn(self):
        import psycopg2.pool as _pp

        if self._out >= self._capacity:
            raise _pp.PoolError("connection pool exhausted")
        self._out += 1
        return _FakeConn(self)

    def putconn(self, conn):
        self._out -= 1


class _FakeConn:
    def __init__(self, pool: "_FakePool"):
        self._pool = pool

    def commit(self):
        self._pool.commits += 1

    def rollback(self):
        self._pool.rollbacks += 1


# --- memory-mode no-op ----------------------------------------------------


def test_middleware_is_noop_in_memory_mode(monkeypatch):
    """In `memory://` mode the middleware adds no behaviour — routes still work.

    Memory mode has no pool; `request_scoped_connection()` is a no-op there, so
    a `TestClient` request through the wrapped app must succeed exactly as if
    the middleware were absent, and nothing is bound to the contextvar.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    import adapters.state.conn_middleware as cm_mod
    importlib.reload(state_mod)
    importlib.reload(cm_mod)
    try:
        app = FastAPI()
        app.add_middleware(cm_mod.RequestScopedConnectionMiddleware)

        @app.get("/ping")
        def ping():
            # No request connection is bound in memory mode.
            return {"bound": state_mod._REQUEST_CONN.get() is not None}

        with TestClient(app) as client:
            resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json() == {"bound": False}
    finally:
        importlib.reload(state_mod)
        importlib.reload(cm_mod)


# --- request-scoped reuse through a real ASGI request ---------------------


def test_request_binds_one_connection_reused_by_pooled_conn(monkeypatch):
    """A route inside the middleware sees ONE bound connection, reused N times.

    With a max-1 fake pool, a handler that opens five `pooled_conn()` blocks
    would deadlock/exhaust under the old per-call-checkout behaviour. Through
    the middleware all five reuse the single request connection — one
    checkout, one commit.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    import adapters.state.conn_middleware as cm_mod
    importlib.reload(state_mod)
    importlib.reload(cm_mod)
    try:
        fake = _FakePool(capacity=1)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)

        app = FastAPI()
        app.add_middleware(cm_mod.RequestScopedConnectionMiddleware)

        @app.get("/multi")
        def multi():
            seen = []
            for _ in range(5):
                with state_mod.pooled_conn() as conn:
                    seen.append(id(conn))
            return {"distinct": len(set(seen))}

        with TestClient(app) as client:
            resp = client.get("/multi")
        assert resp.status_code == 200
        assert resp.json() == {"distinct": 1}, "request used >1 pool connection"
        # Exactly one commit for the whole request; connection returned.
        assert fake.commits == 1
        assert fake._out == 0
    finally:
        importlib.reload(state_mod)
        importlib.reload(cm_mod)


def test_request_scope_commits_on_clean_response(monkeypatch):
    """A clean 2xx response commits the single request transaction once."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    import adapters.state.conn_middleware as cm_mod
    importlib.reload(state_mod)
    importlib.reload(cm_mod)
    try:
        fake = _FakePool(capacity=2)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)

        app = FastAPI()
        app.add_middleware(cm_mod.RequestScopedConnectionMiddleware)

        @app.get("/ok")
        def ok():
            with state_mod.pooled_conn():
                pass
            return {"ok": True}

        with TestClient(app) as client:
            resp = client.get("/ok")
        assert resp.status_code == 200
        assert fake.commits == 1 and fake.rollbacks == 0
        assert fake._out == 0
    finally:
        importlib.reload(state_mod)
        importlib.reload(cm_mod)


def test_request_scope_rolls_back_on_handler_exception(monkeypatch):
    """A handler raising mid-request rolls the request transaction back once."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    import adapters.state.conn_middleware as cm_mod
    importlib.reload(state_mod)
    importlib.reload(cm_mod)
    try:
        fake = _FakePool(capacity=2)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)

        app = FastAPI()
        app.add_middleware(cm_mod.RequestScopedConnectionMiddleware)

        @app.get("/boom")
        def boom():
            with state_mod.pooled_conn():
                pass
            raise RuntimeError("handler blew up")

        # `raise_server_exceptions=False` so the TestClient surfaces the 500
        # response instead of re-raising — we care about the rollback.
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/boom")
        assert resp.status_code == 500
        assert fake.rollbacks == 1, f"expected 1 rollback, got {fake.rollbacks}"
        assert fake.commits == 0
        assert fake._out == 0, "connection leaked on the exception path"
    finally:
        importlib.reload(state_mod)
        importlib.reload(cm_mod)


# --- N4: /healthz skips the request-scoped checkout -----------------------


def test_healthz_skips_request_scoped_checkout(monkeypatch):
    """`/healthz` must do NO pool interaction — no checkout, no commit.

    N4: a pool checkout for the liveness probe risks a cascade (saturated pool
    -> healthz fails -> machine killed -> more load). The middleware must pass
    `/healthz` straight through with the pool untouched.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    import adapters.state.conn_middleware as cm_mod
    importlib.reload(state_mod)
    importlib.reload(cm_mod)
    try:
        fake = _FakePool(capacity=5)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)

        app = FastAPI()
        app.add_middleware(cm_mod.RequestScopedConnectionMiddleware)

        @app.get("/healthz")
        def healthz():
            return {"status": "ok", "bound": state_mod._REQUEST_CONN.get() is not None}

        @app.get("/work")
        def work():
            return {"bound": state_mod._REQUEST_CONN.get() is not None}

        with TestClient(app) as client:
            hz = client.get("/healthz")
            # After /healthz only: no checkout, no commit, nothing bound.
            assert hz.status_code == 200
            assert hz.json()["bound"] is False
            assert fake._out == 0
            assert fake.commits == 0, (
                "/healthz must not commit a request transaction"
            )
            # A normal route still gets the request-scoped connection + commit.
            wk = client.get("/work")
        assert wk.json()["bound"] is True
        assert fake.commits == 1
    finally:
        importlib.reload(state_mod)
        importlib.reload(cm_mod)


# --- query paths skip the request-scoped checkout -------------------------


def test_query_paths_skip_request_scoped_checkout(monkeypatch):
    """`/v1/query` and `/v1/query/status/{id}` skip the request-scoped checkout.

    These requests are long but NOT Postgres-bound — on the CP they are
    dominated by the CP->DP proxy hop, on the DP by the FAISS search / shard
    download. Request-scoping would pin a pooled connection idle across that
    slow part and starve the pool under load (a trace showed a 14.79s query
    holding its connection the whole time). The middleware must pass these
    paths straight through with the pool untouched; their brief DB calls take
    short standalone `pooled_conn()` checkouts instead. A non-query route still
    gets the request-scoped connection.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    import adapters.state.conn_middleware as cm_mod
    importlib.reload(state_mod)
    importlib.reload(cm_mod)
    try:
        fake = _FakePool(capacity=5)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)

        app = FastAPI()
        app.add_middleware(cm_mod.RequestScopedConnectionMiddleware)

        @app.post("/v1/query")
        def v1_query():
            return {"bound": state_mod._REQUEST_CONN.get() is not None}

        @app.get("/v1/query/status/{job_id}")
        def v1_query_status(job_id: str):
            return {"bound": state_mod._REQUEST_CONN.get() is not None}

        @app.get("/work")
        def work():
            return {"bound": state_mod._REQUEST_CONN.get() is not None}

        with TestClient(app) as client:
            q = client.post("/v1/query")
            assert q.status_code == 200
            assert q.json()["bound"] is False, "/v1/query must skip the request scope"
            s = client.get("/v1/query/status/job_abc123")
            assert s.status_code == 200
            assert s.json()["bound"] is False, "status path must skip the request scope"
            # Neither query path touched the pool.
            assert fake._out == 0
            assert fake.commits == 0, "query paths must not commit a request transaction"
            # A non-query route still gets the request-scoped connection + commit.
            w = client.get("/work")
        assert w.json()["bound"] is True
        assert fake.commits == 1
    finally:
        importlib.reload(state_mod)
        importlib.reload(cm_mod)


# --- vector-upload path skips the request-scoped checkout -----------------


def test_vector_upload_path_skips_request_scoped_checkout(monkeypatch):
    """`POST /v1/datasets/{name}/vectors` skips the request-scoped checkout.

    The NDJSON vector-upload request is dominated by reading a multi-MB body
    and writing a ~9 MB landing object to object storage — NOT Postgres work.
    A k6 load sweep at 1536-dim ingest sizes showed concurrent ingests pinning
    every pooled connection idle across their slow S3 writes, starving small
    `POST /v1/datasets` creates into 503s. The middleware must pass the upload
    path straight through with the pool untouched. The skip is POST-only and
    suffix-specific: `POST /v1/datasets` (create) and `GET /v1/datasets/{name}`
    still get the request-scoped connection.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    import adapters.state.conn_middleware as cm_mod
    importlib.reload(state_mod)
    importlib.reload(cm_mod)
    try:
        fake = _FakePool(capacity=5)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)

        app = FastAPI()
        app.add_middleware(cm_mod.RequestScopedConnectionMiddleware)

        @app.post("/v1/datasets/{name}/vectors")
        def upload_vectors(name: str):
            return {"bound": state_mod._REQUEST_CONN.get() is not None}

        @app.post("/v1/datasets")
        def create_dataset():
            return {"bound": state_mod._REQUEST_CONN.get() is not None}

        @app.get("/v1/datasets/{name}")
        def get_dataset(name: str):
            return {"bound": state_mod._REQUEST_CONN.get() is not None}

        with TestClient(app) as client:
            up = client.post("/v1/datasets/ds_abc/vectors")
            assert up.status_code == 200
            assert up.json()["bound"] is False, (
                "vector-upload path must skip the request scope"
            )
            assert fake._out == 0
            assert fake.commits == 0, "upload path must not commit a request transaction"
            # The small create stays request-scoped (it IS Postgres-bound).
            cr = client.post("/v1/datasets")
            assert cr.json()["bound"] is True, "create must keep the request scope"
            # A GET on the same prefix is not the upload path either.
            gd = client.get("/v1/datasets/ds_abc")
        assert gd.json()["bound"] is True, "dataset GET must keep the request scope"
        assert fake.commits == 2
    finally:
        importlib.reload(state_mod)
        importlib.reload(cm_mod)


# --- PoolCheckoutTimeout -> 503 -------------------------------------------


def test_pool_checkout_timeout_in_handler_maps_to_503(monkeypatch):
    """A `PoolCheckoutTimeout` escaping a route handler -> v1 503 envelope.

    `install_pool_exhaustion_handler` must shape it into
    `{"error": {"code": "service_unavailable", ...}}` with status 503 —
    never a 500.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    from services.auth.auth import install_pool_exhaustion_handler
    importlib.reload(state_mod)
    try:
        app = FastAPI()
        install_pool_exhaustion_handler(app)

        @app.get("/explode")
        def explode():
            raise state_mod.PoolCheckoutTimeout("pool exhausted")

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/explode")
        assert resp.status_code == 503
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "service_unavailable"
        assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    finally:
        importlib.reload(state_mod)


def test_pool_checkout_timeout_in_middleware_checkout_maps_to_503(monkeypatch):
    """A `PoolCheckoutTimeout` from the middleware's OWN checkout -> v1 503.

    The middleware checks a connection out BEFORE the app runs, so a timeout
    there escapes above the app's exception handlers. The middleware must emit
    the v1 503 envelope itself rather than let it become a bare ASGI 500.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    import adapters.state.conn_middleware as cm_mod
    importlib.reload(state_mod)
    importlib.reload(cm_mod)
    try:
        # A capacity-0 pool: every checkout is exhausted forever.
        fake = _FakePool(capacity=0)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)
        monkeypatch.setenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", "0.2")

        app = FastAPI()
        app.add_middleware(cm_mod.RequestScopedConnectionMiddleware)

        @app.get("/never-reached")
        def never_reached():  # pragma: no cover - the checkout fails first
            return {"ok": True}

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/never-reached")
        assert resp.status_code == 503
        body = resp.json()
        assert body["error"]["code"] == "service_unavailable"
    finally:
        monkeypatch.delenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", raising=False)
        importlib.reload(state_mod)
        importlib.reload(cm_mod)
