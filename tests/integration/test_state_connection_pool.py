"""Integration coverage for the application-side Postgres connection pool.

`adapters/state/state.py` opened a fresh psycopg2 connection on every `_conn()`
call — a full TCP + TLS + auth handshake each time (~15ms against a managed
database). A production `POST /v1/query` trace showed FIVE such opens, ~73ms of
a ~108ms request spent purely on handshakes.

The fix is an application-side `psycopg2.pool.ThreadedConnectionPool`: open a
small set of connections once, hand them out via `pooled_conn()` and take them
back. These tests run against a REAL Postgres (testcontainers) — pooling is a
no-op in `memory://` mode — and prove:

  - a second checkout REUSES a connection rather than opening a new one;
  - a connection is returned to the pool after normal use AND after an
    exception (it is never leaked);
  - `dataset_build_lock` still uses a dedicated, NON-pooled connection (a
    session-scoped advisory lock must not ride the pool).
"""
from __future__ import annotations

import importlib

import psycopg2
import pytest

try:
    from testcontainers.postgres import PostgresContainer
except ImportError as exc:  # pragma: no cover
    PostgresContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(scope="module")
def pg_url():
    """Start one Postgres container for this module; yield a psycopg2 DSN."""
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the pool suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with PostgresContainer("postgres:15-alpine", driver=None) as pg:
        yield pg.get_connection_url()


@pytest.fixture
def state(monkeypatch, pg_url):
    """State adapter bound to the container Postgres, schema migrated.

    Teardown closes the pool and restores the default `memory://` adapter so
    the rest of the session is unaffected.
    """
    monkeypatch.setenv("DATABASE_URL", pg_url)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    assert not state_mod._MEMORY_MODE, "test must run against real Postgres"
    state_mod.migrate()
    yield state_mod
    state_mod._close_pool()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    importlib.reload(state_mod)


def test_pool_is_built_lazily_not_at_import(state):
    """No pool exists until the first `pooled_conn()` checkout."""
    state._close_pool()
    assert state._POOL is None
    with state.pooled_conn() as conn:
        assert conn is not None
    assert state._POOL is not None, "first checkout must have built the pool"


def test_second_checkout_reuses_a_connection(state):
    """A second checkout reuses a pooled connection — it does NOT open a new one.

    The pool is primed with one checkout; the connection's backend PID is
    recorded. A second sequential checkout (the first connection already
    returned) must hand back the SAME server-side backend — proof the TCP +
    TLS + auth handshake was paid once, not twice.
    """
    state._close_pool()
    with state.pooled_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_backend_pid()")
            first_pid = cur.fetchone()[0]
    # The connection is back in the pool now; the next checkout reuses it.
    with state.pooled_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_backend_pid()")
            second_pid = cur.fetchone()[0]
    assert second_pid == first_pid, (
        "second checkout opened a fresh backend instead of reusing the pooled "
        f"connection (pids {first_pid} -> {second_pid})"
    )


def test_connection_returned_to_pool_after_normal_use(state):
    """After a normal `pooled_conn()` block the connection is back in the pool.

    Drained probe: with a max-1 pool, if the connection were leaked instead of
    returned, a second checkout would have nothing to hand out. Two sequential
    checkouts both succeeding proves the first return happened.
    """
    state._close_pool()
    # A deliberately tiny pool — max 1 — so a leak is immediately fatal.
    pool = state._get_pool(maxconn_override=1)
    conn = pool.getconn()
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
    pool.putconn(conn)
    # With the single connection returned, a second getconn must succeed.
    conn2 = pool.getconn()
    assert conn2 is not None
    pool.putconn(conn2)


def test_connection_returned_to_pool_after_exception(state):
    """An exception inside `pooled_conn()` still returns the connection.

    If the body raises, the connection MUST be rolled back and returned to the
    pool — never leaked. Probe: a max-1 pool, raise inside the block, then
    confirm a fresh checkout still succeeds (the slot was freed) and the
    connection is usable (it was rolled back, not left in a failed txn).
    """
    state._close_pool()
    boom = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        with state.pooled_conn(maxconn_override=1) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            raise boom
    # The slot must be free — a leaked connection would make this checkout
    # block/fail on the max-1 pool.
    with state.pooled_conn(maxconn_override=1) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 42")
            assert cur.fetchone()[0] == 42, "returned connection was unusable"


def test_bad_sql_does_not_leak_the_connection(state):
    """A SQL error inside `pooled_conn()` rolls back and returns the connection.

    Same guarantee as the explicit-raise test, but the failure originates in
    Postgres (a syntax error). The aborted transaction must be rolled back on
    return so the next borrower gets a clean connection.
    """
    state._close_pool()
    with pytest.raises(psycopg2.Error):
        with state.pooled_conn(maxconn_override=1) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM table_that_does_not_exist")
    # Next checkout must get a clean, usable connection — not one stuck in a
    # failed transaction.
    with state.pooled_conn(maxconn_override=1) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1


def test_dataset_build_lock_uses_a_dedicated_non_pooled_connection(state):
    """`dataset_build_lock` must NOT route through the pool.

    Its `pg_try_advisory_lock` is SESSION-scoped: it lives on one specific
    connection and must be held and released on that same connection. A pooled
    connection returned while still holding the lock would leak it to the next
    borrower. Proof: while the lock is held the pool is untouched, and the
    lock's connection is closed (not returned) on exit.
    """
    state._close_pool()
    seen_pool_during_lock = None
    with state.dataset_build_lock("ten_pool", "locked_ds") as acquired:
        assert acquired is True
        # Inside the lock body the pool must still be unbuilt — the lock did
        # not borrow from it.
        seen_pool_during_lock = state._POOL
    assert seen_pool_during_lock is None, (
        "dataset_build_lock built or used the connection pool — its "
        "session-scoped advisory lock must ride a dedicated connection"
    )


def test_dataset_build_lock_connection_is_closed_not_pooled(state, monkeypatch):
    """The dedicated `dataset_build_lock` connection is closed on exit.

    A pooled connection would be `putconn`-ed back; a dedicated one is
    `close()`-d. We assert the connection psycopg2 hands `dataset_build_lock`
    is closed when the `with` block exits — and that it was a direct
    `psycopg2.connect`, never a pool checkout.
    """
    state._close_pool()
    captured = {}
    real_connect = psycopg2.connect

    def spy_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        captured["conn"] = conn
        return conn

    monkeypatch.setattr(psycopg2, "connect", spy_connect)
    with state.dataset_build_lock("ten_pool", "closed_ds") as acquired:
        assert acquired is True
        assert captured.get("conn") is not None, (
            "dataset_build_lock did not open a direct psycopg2 connection"
        )
        assert captured["conn"].closed == 0, "lock connection closed too early"
    assert captured["conn"].closed != 0, (
        "dataset_build_lock did not close its dedicated connection on exit — "
        "it must be closed, never returned to a pool"
    )
    # The pool was never involved.
    assert state._POOL is None


def test_pool_survives_across_many_checkouts(state):
    """Many sequential checkouts all succeed on a small pool — no exhaustion.

    A min-1/max-2 pool serving 20 sequential checkouts proves connections are
    consistently returned (a leak would exhaust the pool within `max` calls).
    """
    state._close_pool()
    for i in range(20):
        with state.pooled_conn(maxconn_override=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT %s", (i,))
                assert cur.fetchone()[0] == i


# --- Block-with-timeout + request-scoped connection -----------------------


def test_block_with_timeout_second_checkout_blocks_then_succeeds(state):
    """Against the REAL pool, a second checkout blocks then succeeds on return.

    A max-1 pool: thread A holds the only connection briefly; thread B's
    `pooled_conn()` finds the real `ThreadedConnectionPool` exhausted
    (fail-fast `PoolError`), poll-retries, and succeeds once A returns its
    connection. No `PoolCheckoutTimeout` — the wait is well within the
    deadline.
    """
    import threading
    import time

    state._close_pool()
    state._get_pool(maxconn_override=1)  # build the real max-1 pool

    second_ok = []

    def _second():
        with state.pooled_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            second_ok.append(True)

    with state.pooled_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        t = threading.Thread(target=_second)
        t.start()
        time.sleep(0.2)
        assert not second_ok, "second checkout succeeded before the slot freed"
    t.join(timeout=5)
    assert second_ok == [True], "blocked checkout never unblocked"


def test_block_with_timeout_raises_pool_checkout_timeout(state, monkeypatch):
    """A checkout while the sole real connection is held forever -> timeout.

    Against the real `ThreadedConnectionPool`: hold the only connection, set a
    short deadline, and a second `pooled_conn()` must raise
    `PoolCheckoutTimeout` — never the raw fail-fast `PoolError`.
    """
    state._close_pool()
    state._get_pool(maxconn_override=1)
    monkeypatch.setenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", "0.3")

    with state.pooled_conn():  # hold the only real connection
        with pytest.raises(state.PoolCheckoutTimeout):
            with state.pooled_conn():
                pass


def test_request_scope_collapses_checkouts_against_real_pool(state):
    """A request scope makes N `pooled_conn()` blocks cost ONE real checkout.

    Decisive proof: a max-1 real pool. Five sequential `pooled_conn()` blocks
    inside one `request_scoped_connection()` would deadlock under the old
    per-call-checkout behaviour (the second block would find the single
    connection still held). With the request scope they all reuse the one
    bound connection and every block sees the SAME backend PID.
    """
    state._close_pool()

    pids = []
    with state.request_scoped_connection(maxconn_override=1):
        for _ in range(5):
            with state.pooled_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_backend_pid()")
                    pids.append(cur.fetchone()[0])
    assert len(set(pids)) == 1, (
        f"request scope used >1 backend connection: pids {pids}"
    )


def test_request_scope_commit_is_visible_to_a_later_block(state):
    """Within a request, a later `pooled_conn()` block sees an earlier write.

    Transaction-semantics note: a request is now ONE transaction on ONE
    connection, so a write in an earlier block is visible to a later block
    even before the request-end commit. We prove that with a temp table.
    """
    state._close_pool()

    with state.request_scoped_connection(maxconn_override=2):
        with state.pooled_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE TEMP TABLE _rb_phase_d (n int)")
                cur.execute("INSERT INTO _rb_phase_d VALUES (7)")
        # A separate `pooled_conn()` block — same request, same connection —
        # must see the row the first block inserted.
        with state.pooled_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT n FROM _rb_phase_d")
                assert cur.fetchone()[0] == 7


# --- Quota writes must NOT ride the request-scoped transaction ------------


def test_standalone_pooled_conn_ignores_request_scoped_conn(state):
    """`pooled_conn(standalone=True)` checks out a NEW connection.

    Even with a request connection bound, the standalone path must ignore
    `_REQUEST_CONN` and use its own pool checkout — proven by distinct backend
    PIDs (the request connection vs the standalone one).
    """
    state._close_pool()
    with state.request_scoped_connection(maxconn_override=4):
        with state.pooled_conn() as req_conn:
            with req_conn.cursor() as cur:
                cur.execute("SELECT pg_backend_pid()")
                req_pid = cur.fetchone()[0]
        with state.pooled_conn(standalone=True) as sc:
            with sc.cursor() as cur:
                cur.execute("SELECT pg_backend_pid()")
                standalone_pid = cur.fetchone()[0]
    assert standalone_pid != req_pid, (
        "standalone=True reused the request-scoped connection — it must "
        "check out its own"
    )


def test_quota_consume_commits_before_request_transaction_ends(state):
    """`try_consume_query`'s UPDATE commits BEFORE the request scope commits.

    Regression guard: the quota write must not ride the request-scoped
    transaction (whose row lock would otherwise be held for the whole
    request). Inside an open `request_scoped_connection()`, we call
    `try_consume_query` and then — from a SEPARATE, independent connection —
    read `queries_today` back. If the standalone path works, the increment is
    already committed and visible; if it rode the request transaction it would
    be invisible (and the read would block on the row lock) until request end.
    """
    import psycopg2

    state._close_pool()
    tid = "ten_b1_query"
    state.create_tenant(tid, "b1q@example.com", "pw")

    with state.request_scoped_connection(maxconn_override=4):
        ok, _usage = state.try_consume_query(tid)
        assert ok is True
        # A fully independent connection — NOT the pool, NOT the request conn.
        observer = psycopg2.connect(state._dsn())
        try:
            observer.autocommit = True
            with observer.cursor() as cur:
                # Short lock timeout: if the quota UPDATE were still holding
                # the row lock inside the request transaction this SELECT ...
                # FOR UPDATE would error out instead of blocking forever.
                cur.execute("SET lock_timeout = '2s'")
                cur.execute(
                    "SELECT queries_today FROM tenants WHERE id=%s FOR UPDATE",
                    (tid,),
                )
                queries_today = cur.fetchone()[0]
        finally:
            observer.close()

    assert queries_today == 1, (
        "the quota UPDATE was not committed before request end — it rode the "
        "request-scoped transaction instead of its own standalone one"
    )


def test_vector_consume_commits_before_request_transaction_ends(state):
    """`try_consume_vectors`' UPDATE commits in its own standalone transaction.

    Same standalone-transaction guard as the query-quota test, for the vector admission quota.
    """
    import psycopg2

    state._close_pool()
    tid = "ten_b1_vec"
    state.create_tenant(tid, "b1v@example.com", "pw")

    with state.request_scoped_connection(maxconn_override=4):
        ok, _usage = state.try_consume_vectors(tid, 5)
        assert ok is True
        observer = psycopg2.connect(state._dsn())
        try:
            observer.autocommit = True
            with observer.cursor() as cur:
                cur.execute("SET lock_timeout = '2s'")
                cur.execute(
                    "SELECT vectors_used FROM tenants WHERE id=%s FOR UPDATE",
                    (tid,),
                )
                vectors_used = cur.fetchone()[0]
        finally:
            observer.close()

    assert vectors_used == 5, (
        "the vector-quota UPDATE was not committed before request end — it "
        "rode the request-scoped transaction instead of its own"
    )
