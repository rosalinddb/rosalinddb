"""Unit coverage for the application-side Postgres connection pool.

`adapters/state/state.py` historically opened a brand-new psycopg2 connection
on every `_conn()` call — a full TCP + TLS + auth handshake each time. The pool
opens a small set of connections once and hands them out / takes them back.

These tests are hermetic: they exercise the memory-mode contract (no pool, no
SQL connection) and the pool-sizing / lazy-construction wiring. The actual
checkout-reuse / return-on-exception behaviour against a live server lives in
`tests/integration/test_state_pool.py` (it needs a real Postgres).
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def memory_state(monkeypatch):
    """State module bound to `memory://` (the test default) — no pool."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    assert state_mod._MEMORY_MODE
    yield state_mod
    importlib.reload(state_mod)


def test_memory_mode_has_no_pool(memory_state):
    """Memory mode never builds a pool — there is no SQL server to pool."""
    state = memory_state
    # The lazy getter must refuse in memory mode rather than try to connect.
    with pytest.raises(RuntimeError):
        state._get_pool()
    # And no pool object was constructed as a side effect.
    assert state._POOL is None


def test_memory_mode_dataset_build_lock_is_noop(memory_state):
    """`dataset_build_lock` stays a pure no-op in memory mode (yields True)."""
    state = memory_state
    with state.dataset_build_lock("ten", "ds") as acquired:
        assert acquired is True


def test_pooled_conn_helper_exists(memory_state):
    """The pool-aware context-manager helper is part of the module surface.

    `pooled_conn()` checks a connection out on enter and returns it to the
    pool on exit (commit on success / rollback on exception). Its existence is
    the contract every non-`dataset_build_lock` call site now depends on.
    """
    state = memory_state
    assert hasattr(state, "pooled_conn")
    assert callable(state.pooled_conn)


def test_pool_max_size_env_override(monkeypatch):
    """`RB_PG_POOL_MAX` overrides the default pool ceiling."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("RB_PG_POOL_MAX", "25")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        assert state_mod._pool_max_size() == 25
    finally:
        monkeypatch.delenv("RB_PG_POOL_MAX", raising=False)
        importlib.reload(state_mod)


def test_pool_max_size_default(monkeypatch):
    """With no env override the pool ceiling is the documented default of 10."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.delenv("RB_PG_POOL_MAX", raising=False)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        assert state_mod._pool_max_size() == 10
    finally:
        importlib.reload(state_mod)


def test_pool_max_size_ignores_garbage(monkeypatch):
    """A non-integer `RB_PG_POOL_MAX` falls back to the default, not a crash."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("RB_PG_POOL_MAX", "not-a-number")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        assert state_mod._pool_max_size() == 10
    finally:
        monkeypatch.delenv("RB_PG_POOL_MAX", raising=False)
        importlib.reload(state_mod)


# --- Block-with-timeout pool checkout -------------------------------------
#
# `psycopg2.pool.ThreadedConnectionPool.getconn()` is fail-fast — it raises
# `PoolError` the instant the pool is exhausted. `pooled_conn()` must instead
# poll-retry on `PoolError` until a total deadline; only a sustained
# exhaustion raises `PoolCheckoutTimeout`. These tests drive a fake pool so
# the contract is hermetic — no real Postgres needed (the live-server
# behaviour lives in `tests/integration/test_state_connection_pool.py`).


class _FakePool:
    """A `ThreadedConnectionPool`-shaped stub for the checkout-timeout tests.

    `capacity` connections are available; `getconn()` raises `PoolError` (the
    fail-fast behaviour of the real pool) once they are all checked out, and
    succeeds again after a `putconn()`. A tiny `_pool` attribute keeps the
    `reused` probe in `pooled_conn` happy.
    """

    def __init__(self, capacity: int = 1):
        self._capacity = capacity
        self._out = 0
        self._pool = [object()]  # non-empty -> `reused=True` in pooled_conn
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
    """A psycopg2-connection-shaped stub: just records commit/rollback."""

    def __init__(self, pool: "_FakePool"):
        self._pool = pool

    def commit(self):
        self._pool.commits += 1

    def rollback(self):
        self._pool.rollbacks += 1


def test_pooled_conn_blocks_then_succeeds_when_a_connection_frees(monkeypatch):
    """A checkout against an exhausted pool blocks, then succeeds on a return.

    With a max-1 fake pool primed empty, the first checkout takes the only
    connection. A second checkout (on another thread) finds the pool exhausted
    -> `PoolError` -> poll-retry. Once the first block exits and returns its
    connection, the blocked checkout succeeds — no exception.
    """
    import threading
    import time

    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        fake = _FakePool(capacity=1)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)
        monkeypatch.setenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", "5")

        second_ok = []

        def _second():
            with state_mod.pooled_conn() as conn:
                second_ok.append(conn is not None)

        with state_mod.pooled_conn() as first:
            assert first is not None
            t = threading.Thread(target=_second)
            t.start()
            # Give the second checkout time to hit PoolError and start polling.
            time.sleep(0.2)
            assert not second_ok, "second checkout succeeded before a slot freed"
        # `first` block exited -> connection returned -> second unblocks.
        t.join(timeout=3)
        assert second_ok == [True], "second checkout never unblocked"
    finally:
        monkeypatch.delenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", raising=False)
        importlib.reload(state_mod)


def test_pooled_conn_raises_pool_checkout_timeout_on_sustained_exhaustion(monkeypatch):
    """A checkout while the only connection is held forever times out cleanly.

    The single connection is held for the whole test; the second checkout can
    never get one. After the deadline it must raise `PoolCheckoutTimeout` (the
    apps map that to a 503) — never a bare `PoolError` (a 500).
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        fake = _FakePool(capacity=1)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)
        # A short deadline keeps the test fast.
        monkeypatch.setenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", "0.2")

        with state_mod.pooled_conn():  # holds the only connection
            with pytest.raises(state_mod.PoolCheckoutTimeout):
                with state_mod.pooled_conn():
                    pass
    finally:
        monkeypatch.delenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", raising=False)
        importlib.reload(state_mod)


def test_pool_checkout_timeout_is_env_tunable(monkeypatch):
    """`RB_PG_POOL_CHECKOUT_TIMEOUT_S` overrides the checkout deadline.

    A larger env value must make the checkout wait noticeably longer before
    raising `PoolCheckoutTimeout`. We compare the elapsed time of a short vs a
    longer deadline against the same permanently-exhausted pool.
    """
    import time

    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        # Default when unset.
        monkeypatch.delenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", raising=False)
        assert state_mod._pool_checkout_timeout_s() == 2.5
        # Override is honoured.
        monkeypatch.setenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", "0.75")
        assert state_mod._pool_checkout_timeout_s() == 0.75
        # Garbage falls back to the default rather than crashing.
        monkeypatch.setenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", "not-a-number")
        assert state_mod._pool_checkout_timeout_s() == 2.5

        # And the value actually drives the block-with-timeout deadline.
        fake = _FakePool(capacity=1)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)
        with state_mod.pooled_conn():  # hold the only connection
            monkeypatch.setenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", "0.1")
            t0 = time.monotonic()
            with pytest.raises(state_mod.PoolCheckoutTimeout):
                with state_mod.pooled_conn():
                    pass
            short = time.monotonic() - t0

            monkeypatch.setenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", "0.6")
            t0 = time.monotonic()
            with pytest.raises(state_mod.PoolCheckoutTimeout):
                with state_mod.pooled_conn():
                    pass
            longer = time.monotonic() - t0

        assert short < 0.4, f"short deadline took too long: {short:.2f}s"
        assert longer > short + 0.2, (
            f"longer deadline ({longer:.2f}s) did not exceed the short one "
            f"({short:.2f}s) — env override not honoured"
        )
    finally:
        monkeypatch.delenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", raising=False)
        importlib.reload(state_mod)


def test_transient_exhaustion_is_invisible_no_exception(monkeypatch):
    """A pool that exhausts then frees within the deadline raises nothing.

    The block-with-timeout exists precisely so a brief burst does not become a
    503. The checkout just blocks the short while and then succeeds.
    """
    import threading
    import time

    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        fake = _FakePool(capacity=1)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)
        monkeypatch.setenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", "5")

        holder_release = threading.Event()

        def _holder():
            with state_mod.pooled_conn():
                holder_release.wait(timeout=3)

        t = threading.Thread(target=_holder)
        t.start()
        time.sleep(0.1)  # let the holder grab the connection
        # Schedule the holder to release shortly — well within the 5s deadline.
        threading.Timer(0.3, holder_release.set).start()
        # This checkout finds the pool exhausted, polls, then succeeds.
        with state_mod.pooled_conn() as conn:
            assert conn is not None
        t.join(timeout=3)
    finally:
        monkeypatch.delenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", raising=False)
        importlib.reload(state_mod)


# --- Request-scoped connection --------------------------------------------
#
# A `request_scoped_connection()` binds ONE pooled connection to a contextvar;
# every `pooled_conn()` block inside reuses it (one checkout, not N) and the
# scope owns the single commit/rollback/return.


def test_request_scope_is_noop_in_memory_mode(memory_state):
    """`request_scoped_connection()` is a clean no-op in memory mode.

    Memory mode has no pool — the scope must yield without binding anything,
    so the ASGI middleware degrades to a plain pass-through in `memory://`
    tests.
    """
    state = memory_state
    with state.request_scoped_connection():
        # Nothing is bound — `pooled_conn()` would still take its standalone
        # path (which itself raises in memory mode, but that is unchanged).
        assert state._REQUEST_CONN.get() is None


def test_request_scope_collapses_n_checkouts_to_one(monkeypatch):
    """N `pooled_conn()` blocks inside a request scope share ONE connection.

    Proof: a max-1 fake pool. Under the OLD per-call-checkout behaviour, the
    second `pooled_conn()` block inside one request would find the pool
    exhausted. With the request scope it reuses the single bound connection,
    so five sequential blocks all yield the SAME connection and the pool is
    checked out exactly once.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        fake = _FakePool(capacity=1)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)

        seen = []
        with state_mod.request_scoped_connection():
            for _ in range(5):
                with state_mod.pooled_conn() as conn:
                    seen.append(conn)
        # All five blocks saw the identical connection object.
        assert len(set(id(c) for c in seen)) == 1, "blocks used different conns"
        # The pool was returned to (out == 0) and never exhausted.
        assert fake._out == 0
    finally:
        importlib.reload(state_mod)


def test_request_scope_commits_once_on_clean_exit(monkeypatch):
    """A clean request scope commits the single request transaction exactly once."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        fake = _FakePool(capacity=1)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)

        with state_mod.request_scoped_connection():
            for _ in range(3):
                with state_mod.pooled_conn() as conn:
                    pass
        # ONE commit for the whole request, despite three pooled_conn blocks.
        assert fake.commits == 1, f"expected 1 commit, got {fake.commits}"
        assert fake.rollbacks == 0
        assert fake._out == 0, "connection not returned to the pool"
    finally:
        importlib.reload(state_mod)


def test_request_scope_rolls_back_once_on_exception(monkeypatch):
    """An exception inside a request scope rolls back once and returns the conn."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        fake = _FakePool(capacity=1)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)

        with pytest.raises(RuntimeError):
            with state_mod.request_scoped_connection():
                with state_mod.pooled_conn():
                    pass
                raise RuntimeError("boom")
        assert fake.rollbacks == 1, f"expected 1 rollback, got {fake.rollbacks}"
        assert fake.commits == 0
        assert fake._out == 0, "connection leaked on the exception path"
        # The contextvar was unbound — a later standalone caller is unaffected.
        assert state_mod._REQUEST_CONN.get() is None
    finally:
        importlib.reload(state_mod)


def test_pooled_conn_standalone_unchanged_when_no_request_bound(monkeypatch):
    """With no request scope, `pooled_conn()` keeps its per-call commit+return.

    Non-HTTP callers (workers, scripts, tests) bind nothing — `pooled_conn()`
    must behave EXACTLY as before: its own checkout, its own commit, its own
    return, per block.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        fake = _FakePool(capacity=1)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)
        assert state_mod._REQUEST_CONN.get() is None

        with state_mod.pooled_conn() as conn:
            assert conn is not None
        # Standalone path commits per block and returns the connection.
        assert fake.commits == 1
        assert fake._out == 0
        # A second standalone block also gets a connection (the first returned).
        with state_mod.pooled_conn():
            pass
        assert fake.commits == 2
    finally:
        importlib.reload(state_mod)


def test_request_conn_contextvar_visible_in_worker_thread(monkeypatch):
    """The request connection contextvar is copied into a worker thread.

    Task 3 offloads sync state calls with `asyncio.to_thread`, which copies the
    current `contextvars` into the worker thread. A connection bound by the
    middleware before that offload must therefore still resolve inside the
    offloaded sync call. We model the copy with `contextvars.copy_context()`
    run on a real thread.
    """
    import contextvars
    import threading

    monkeypatch.setenv("DATABASE_URL", "memory://local")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        fake = _FakePool(capacity=1)
        monkeypatch.setattr(state_mod, "_MEMORY_MODE", False)
        monkeypatch.setattr(state_mod, "_POOL", fake)

        with state_mod.request_scoped_connection():
            bound = state_mod._REQUEST_CONN.get()
            assert bound is not None

            seen_in_thread = []

            def _work():
                # Inside the offloaded call, `pooled_conn()` must yield the
                # SAME request connection bound on the originating context.
                with state_mod.pooled_conn() as conn:
                    seen_in_thread.append(conn)

            ctx = contextvars.copy_context()
            t = threading.Thread(target=lambda: ctx.run(_work))
            t.start()
            t.join(timeout=3)

            assert seen_in_thread and seen_in_thread[0] is bound, (
                "the request connection did not resolve inside the worker "
                "thread — contextvar not copied / pooled_conn not aware"
            )
        # Still exactly one checkout for the whole request.
        assert fake.commits == 1
        assert fake._out == 0
    finally:
        importlib.reload(state_mod)
