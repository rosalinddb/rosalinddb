"""Unit coverage for the recall-tier application-side connection pool.

`adapters/state/state.py` historically opened a brand-new psycopg2 connection on
every recall op (`_recall_conn()` per call — a full TCP + TLS + auth handshake,
~48ms against a managed pgvector instance). The recall pool opens a small set of
connections once and hands them out / takes them back via `recall_pooled_conn()`,
mirroring the control-plane pool (`_POOL` / `pooled_conn()`).

These tests are hermetic (no Docker, no pgvector). They drive a fake recall pool
so the contract — checkout reuse, return-on-exit, the single-connection-per-
transaction guarantee for the LSN-block-allocation + UPSERT, block-with-timeout
checkout, and the DEFAULT-OFF "no pool, no connection" property — is proven
without a real database. The same FakePool shape `tests/unit/test_state_pool.py`
uses for the control-plane pool is reused here.
"""
from __future__ import annotations

import importlib

import pytest


_RECALL_DSN = "postgresql://u:p@recall:5432/recall"


@pytest.fixture
def state_on(monkeypatch):
    """State module with the recall tier configured (DSN set; pool unbuilt)."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("RB_RECALL_DSN", _RECALL_DSN)
    monkeypatch.setenv("RB_RECALL", "true")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    yield state_mod
    monkeypatch.delenv("RB_RECALL", raising=False)
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    monkeypatch.delenv("RB_RECALL_POOL_MAX", raising=False)
    importlib.reload(state_mod)


@pytest.fixture
def state_off(monkeypatch):
    """State module with the recall tier OFF (no DSN) — the default deploy."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.delenv("RB_RECALL", raising=False)
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    yield state_mod
    importlib.reload(state_mod)


# --- Fakes (mirroring tests/unit/test_state_pool.py) -----------------------


class _FakeConn:
    """A psycopg2-connection-shaped stub recording commit/rollback + cursor use."""

    def __init__(self, pool, cur=None):
        self._pool = pool
        self._cur = cur
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, *a, **k):
        return self._cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class _FakePool:
    """A `ThreadedConnectionPool`-shaped stub for the recall checkout tests.

    `capacity` connections are available; `getconn()` raises `PoolError` (the
    fail-fast behaviour of the real pool) once they are all checked out, and
    succeeds again after a `putconn()`. `opens` counts how many distinct
    connections were minted — the proof that checkouts REUSE rather than open a
    fresh connection each time. A non-empty `_pool` keeps the `reused` probe in
    `recall_pooled_conn` happy.
    """

    def __init__(self, capacity: int = 1, cur=None):
        self._capacity = capacity
        self._cur = cur
        self._out = 0
        self._pool = [object()]
        self._free: list[_FakeConn] = []
        self.opens = 0

    def getconn(self):
        import psycopg2.pool as _pp

        if self._out >= self._capacity:
            raise _pp.PoolError("connection pool exhausted")
        self._out += 1
        if self._free:
            return self._free.pop()
        self.opens += 1
        return _FakeConn(self, cur=self._cur)

    def putconn(self, conn):
        self._out -= 1
        self._free.append(conn)


# --- Lazy / flag-off (DEFAULT-OFF: no pool, no connection) -----------------


def test_flag_off_get_recall_pool_raises_and_builds_nothing(state_off):
    """With `RB_RECALL_DSN` unset, the pool getter refuses and builds no pool.

    This is the byte-identical-flag-off property: a flag-off deploy must NEVER
    construct a recall pool or open a recall connection.
    """
    state = state_off
    with pytest.raises(RuntimeError):
        state._get_recall_pool()
    assert state._RECALL_POOL is None
    assert state._RECALL_POOL_DSN is None


def test_flag_off_recall_pooled_conn_opens_nothing(state_off, monkeypatch):
    """Entering `recall_pooled_conn()` with the flag off opens no connection.

    Guard `psycopg2.connect` to blow up if touched — the off path must raise the
    "tier off" RuntimeError BEFORE any connect, never reaching the socket.
    """
    state = state_off

    def _boom(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("flag-off recall path opened a connection")

    monkeypatch.setattr(state.psycopg2, "connect", _boom)
    with pytest.raises(RuntimeError):
        with state.recall_pooled_conn():
            pass
    assert state._RECALL_POOL is None


def test_recall_pool_not_built_at_import(state_on):
    """Configuring the DSN does NOT eagerly build the pool — it stays lazy."""
    assert state_on._RECALL_POOL is None


# --- Pool sizing / env knob (mirrors _pool_max_size) -----------------------


def test_recall_pool_max_size_default(monkeypatch):
    """With no env override the recall pool ceiling is the documented default 10."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.delenv("RB_RECALL_POOL_MAX", raising=False)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        assert state_mod._recall_pool_max_size() == 10
    finally:
        importlib.reload(state_mod)


def test_recall_pool_max_size_env_override(monkeypatch):
    """`RB_RECALL_POOL_MAX` overrides the default recall pool ceiling."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("RB_RECALL_POOL_MAX", "25")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        assert state_mod._recall_pool_max_size() == 25
    finally:
        monkeypatch.delenv("RB_RECALL_POOL_MAX", raising=False)
        importlib.reload(state_mod)


def test_recall_pool_max_size_ignores_garbage(monkeypatch):
    """A non-integer `RB_RECALL_POOL_MAX` falls back to the default, not a crash."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("RB_RECALL_POOL_MAX", "not-a-number")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        assert state_mod._recall_pool_max_size() == 10
    finally:
        monkeypatch.delenv("RB_RECALL_POOL_MAX", raising=False)
        importlib.reload(state_mod)


def test_recall_pool_max_size_independent_from_control_plane(monkeypatch):
    """The recall pool ceiling reads its OWN knob, not `RB_PG_POOL_MAX`.

    The two pools address different instances; their sizes must be tuned
    independently. Setting only the control-plane knob must NOT change the recall
    ceiling, and vice versa.
    """
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("RB_PG_POOL_MAX", "30")
    monkeypatch.delenv("RB_RECALL_POOL_MAX", raising=False)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        assert state_mod._recall_pool_max_size() == 10, "recall ignores RB_PG_POOL_MAX"
        assert state_mod._pool_max_size() == 30
    finally:
        monkeypatch.delenv("RB_PG_POOL_MAX", raising=False)
        importlib.reload(state_mod)


# --- The pool-aware context manager exists ---------------------------------


def test_recall_pooled_conn_helper_exists(state_on):
    """`recall_pooled_conn()` is part of the module surface (every recall op uses it)."""
    assert hasattr(state_on, "recall_pooled_conn")
    assert callable(state_on.recall_pooled_conn)


# --- Checkout reuse: NO fresh connect per call -----------------------------


def test_recall_pooled_conn_reuses_one_connection_across_calls(state_on, monkeypatch):
    """Sequential `recall_pooled_conn()` blocks REUSE the pooled connection.

    The whole point of the pool: a fresh `psycopg2.connect` is NOT paid per op.
    With a capacity-1 fake pool, ten sequential checkouts all succeed (each is
    returned before the next), and only ONE connection is ever minted.
    """
    state = state_on
    fake = _FakePool(capacity=1)
    monkeypatch.setattr(state, "_RECALL_POOL", fake)
    monkeypatch.setattr(state, "_RECALL_POOL_DSN", state._recall_dsn())

    seen = []
    for _ in range(10):
        with state.recall_pooled_conn() as conn:
            seen.append(conn)
    # All ten blocks saw the identical reused connection object...
    assert len(set(id(c) for c in seen)) == 1, "checkouts did not reuse one conn"
    # ...and exactly ONE connection was ever opened (no per-call connect).
    assert fake.opens == 1, f"expected 1 connect, got {fake.opens} (per-call connect!)"
    # The pool was returned to after every block (nothing leaked).
    assert fake._out == 0


def test_recall_pooled_conn_commits_and_returns_on_clean_exit(state_on, monkeypatch):
    """A clean block commits the recall transaction once and returns the conn."""
    state = state_on
    fake = _FakePool(capacity=1)
    monkeypatch.setattr(state, "_RECALL_POOL", fake)
    monkeypatch.setattr(state, "_RECALL_POOL_DSN", state._recall_dsn())

    with state.recall_pooled_conn() as conn:
        assert conn is not None
    assert conn.commits == 1, "clean exit must commit the recall txn exactly once"
    assert conn.rollbacks == 0
    assert fake._out == 0, "connection not returned to the recall pool"


def test_recall_pooled_conn_rolls_back_and_returns_on_exception(state_on, monkeypatch):
    """An exception inside the block rolls back once and STILL returns the conn."""
    state = state_on
    fake = _FakePool(capacity=1)
    monkeypatch.setattr(state, "_RECALL_POOL", fake)
    monkeypatch.setattr(state, "_RECALL_POOL_DSN", state._recall_dsn())

    captured = {}
    with pytest.raises(RuntimeError):
        with state.recall_pooled_conn() as conn:
            captured["conn"] = conn
            raise RuntimeError("boom")
    conn = captured["conn"]
    assert conn.rollbacks == 1, "exception must roll back the recall txn"
    assert conn.commits == 0, "a failed block must never commit"
    assert fake._out == 0, "connection leaked on the exception path"
    # The next checkout reuses the rolled-back connection (returned, not closed).
    with state.recall_pooled_conn() as conn2:
        pass
    assert fake.opens == 1, "the rolled-back conn was reused, not re-opened"


# --- DSN-keyed rebuild -----------------------------------------------------


def test_recall_pool_rebuilds_when_dsn_changes(state_on, monkeypatch):
    """A changed `RB_RECALL_DSN` tears down the old pool and builds a fresh one.

    Keying the pool on the DSN means a reconfigure (or a test rebind) never hands
    out connections to the old recall instance.
    """
    state = state_on
    closed = {"n": 0}

    class _ClosablePool(_FakePool):
        def closeall(self):
            closed["n"] += 1

    old = _ClosablePool(capacity=1)
    monkeypatch.setattr(state, "_RECALL_POOL", old)
    monkeypatch.setattr(state, "_RECALL_POOL_DSN", state._recall_dsn())

    # Point at a DIFFERENT recall instance and force a rebuild via the getter.
    # Stub the constructor so no real connection is attempted.
    built = {}

    def _fake_ctor(minconn, maxconn, dsn):
        built["dsn"] = dsn
        return _FakePool(capacity=1)

    monkeypatch.setattr(state.psycopg2.pool, "ThreadedConnectionPool", _fake_ctor)
    monkeypatch.setenv("RB_RECALL_DSN", "postgresql://u:p@OTHER:5432/recall2")

    pool = state._get_recall_pool()
    assert closed["n"] == 1, "the stale pool (old DSN) must be closed on rebuild"
    assert built["dsn"] == "postgresql://u:p@OTHER:5432/recall2"
    assert pool is not old


# --- Block-with-timeout checkout (mirrors pooled_conn) ---------------------


def test_recall_checkout_blocks_then_succeeds_when_a_connection_frees(state_on, monkeypatch):
    """A checkout against an exhausted recall pool blocks, then succeeds on return.

    A max-1 fake pool: the first checkout takes the only connection. A second
    checkout (another thread) finds the pool exhausted -> `PoolError` ->
    poll-retry. Once the first block exits and returns, the blocked checkout
    succeeds — no exception (the block-with-timeout makes a transient burst
    invisible rather than a 503).
    """
    import threading
    import time

    state = state_on
    fake = _FakePool(capacity=1)
    monkeypatch.setattr(state, "_RECALL_POOL", fake)
    monkeypatch.setattr(state, "_RECALL_POOL_DSN", state._recall_dsn())
    monkeypatch.setenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", "5")

    second_ok = []

    def _second():
        with state.recall_pooled_conn() as conn:
            second_ok.append(conn is not None)

    with state.recall_pooled_conn() as first:
        assert first is not None
        t = threading.Thread(target=_second)
        t.start()
        time.sleep(0.2)  # let the second checkout hit PoolError and start polling
        assert not second_ok, "second checkout succeeded before a slot freed"
    # `first` exited -> connection returned -> second unblocks.
    t.join(timeout=3)
    assert second_ok == [True], "second recall checkout never unblocked"


def test_recall_checkout_raises_pool_checkout_timeout_on_sustained_exhaustion(
    state_on, monkeypatch
):
    """A checkout while the only recall conn is held forever times out cleanly.

    After the deadline it raises `PoolCheckoutTimeout` (the apps map that to a
    503) — never a bare `PoolError` (a 500).
    """
    state = state_on
    fake = _FakePool(capacity=1)
    monkeypatch.setattr(state, "_RECALL_POOL", fake)
    monkeypatch.setattr(state, "_RECALL_POOL_DSN", state._recall_dsn())
    monkeypatch.setenv("RB_PG_POOL_CHECKOUT_TIMEOUT_S", "0.2")

    with state.recall_pooled_conn():  # holds the only connection
        with pytest.raises(state.PoolCheckoutTimeout):
            with state.recall_pooled_conn():
                pass


# --- Upsert txn integrity: LSN allocation + UPSERT on ONE checked-out conn --


class _LSNCursor:
    """Records executed SQL + which connection ran it (proves single-conn txn)."""

    def __init__(self, owner):
        self._owner = owner  # the _FakeConn that owns this cursor
        self._lsn = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self._owner.statements.append(sql)
        if "recall_lsn_seq" in sql and "RETURNING" in sql:
            block = params[2] if params else 1
            self._lsn += block

    def fetchone(self):
        return (self._lsn,)


class _LSNConn:
    """A fake conn whose cursor records the connection it ran on."""

    def __init__(self):
        self.statements: list[str] = []
        self.commits = 0
        self.rollbacks = 0
        self._cur = _LSNCursor(self)

    def cursor(self, *a, **k):
        return self._cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class _SingleConnPool:
    """Hands out ONE connection and asserts the whole txn stays on it.

    `_pool` non-empty for the `reused` probe. Tracks checkouts so the test can
    assert exactly one checkout backed the multi-statement transaction.
    """

    def __init__(self, conn):
        self._conn = conn
        self._pool = [object()]
        self.checkouts = 0
        self.returns = 0

    def getconn(self):
        self.checkouts += 1
        return self._conn

    def putconn(self, conn):
        assert conn is self._conn, "a different connection was returned to the pool"
        self.returns += 1


def test_upsert_lsn_allocation_and_upsert_share_one_checked_out_connection(
    state_on, monkeypatch
):
    """The LSN block allocation and the UPSERT run on the SAME pooled connection.

    CRITICAL CORRECTNESS: `recall_upsert_vectors` allocates an LSN block
    (`UPDATE ... RETURNING`) and applies the multi-row UPSERT in ONE transaction
    on ONE connection. With the pool, that connection is held for the whole
    transaction, committed, then returned. This proves there is exactly ONE
    checkout, both statements ran on the SAME connection, and it committed once.
    """
    state = state_on
    conn = _LSNConn()
    pool = _SingleConnPool(conn)
    monkeypatch.setattr(state, "_RECALL_POOL", pool)
    monkeypatch.setattr(state, "_RECALL_POOL_DSN", state._recall_dsn())

    # Stub execute_values: record onto the SAME conn's statement log so the test
    # can prove the UPSERT ran on the checked-out connection.
    def _fake_execute_values(c, sql, rows, template=None, **kw):
        assert c is conn._cur, "the UPSERT ran on a DIFFERENT cursor/connection"
        conn.statements.append(sql)

    monkeypatch.setattr(state, "execute_values", _fake_execute_values)

    records = [
        {"id": "a", "values": [1.0, 2.0], "metadata": {}},
        {"id": "b", "values": [3.0, 4.0], "metadata": {}},
    ]
    written = state.recall_upsert_vectors("t1", "ds", records)
    assert written == 2

    # EXACTLY ONE checkout backed the whole transaction (LSN alloc + UPSERT).
    assert pool.checkouts == 1, "the upsert txn must use a single pooled checkout"
    assert pool.returns == 1, "the connection must be returned exactly once"

    # BOTH statements ran on the SAME connection, in order: seq-alloc then UPSERT.
    assert any("recall_lsn_seq" in s for s in conn.statements), "LSN alloc missing"
    assert any("INSERT INTO recall_vectors" in s for s in conn.statements), (
        "UPSERT missing"
    )
    seq_idx = next(i for i, s in enumerate(conn.statements) if "recall_lsn_seq" in s)
    ups_idx = next(
        i for i, s in enumerate(conn.statements) if "INSERT INTO recall_vectors" in s
    )
    assert seq_idx < ups_idx, "LSN allocation must precede the UPSERT (one txn)"

    # The single transaction committed exactly once, never rolled back.
    assert conn.commits == 1, "the batch must commit once on the held connection"
    assert conn.rollbacks == 0


def test_upsert_failure_rolls_back_the_single_connection_and_returns_it(
    state_on, monkeypatch
):
    """A mid-transaction failure rolls back the held conn and returns it to the pool.

    The whole batch is one transaction on one connection, so a failure persists
    nothing (rollback) and the connection is returned (not leaked) for reuse.
    """
    state = state_on
    conn = _LSNConn()
    pool = _SingleConnPool(conn)
    monkeypatch.setattr(state, "_RECALL_POOL", pool)
    monkeypatch.setattr(state, "_RECALL_POOL_DSN", state._recall_dsn())

    def _boom_execute_values(c, sql, rows, template=None, **kw):
        raise RuntimeError("recall store dropped mid-UPSERT")

    monkeypatch.setattr(state, "execute_values", _boom_execute_values)

    with pytest.raises(RuntimeError):
        state.recall_upsert_vectors("t1", "ds", [{"id": "a", "values": [1.0], "metadata": {}}])

    assert conn.rollbacks == 1, "a failed batch must roll back the held connection"
    assert conn.commits == 0, "a failed batch must never commit"
    assert pool.returns == 1, "the connection must still be returned to the pool"


# --- _close_recall_pool teardown hook --------------------------------------


def test_close_recall_pool_is_safe_when_no_pool(state_off):
    """`_close_recall_pool()` is a safe no-op when no recall pool exists."""
    state = state_off
    assert state._RECALL_POOL is None
    state._close_recall_pool()  # must not raise
    assert state._RECALL_POOL is None


def test_close_recall_pool_closes_and_clears(state_on, monkeypatch):
    """`_close_recall_pool()` closes the pool and clears the DSN key."""
    state = state_on
    closed = {"n": 0}

    class _ClosablePool(_FakePool):
        def closeall(self):
            closed["n"] += 1

    monkeypatch.setattr(state, "_RECALL_POOL", _ClosablePool(capacity=1))
    monkeypatch.setattr(state, "_RECALL_POOL_DSN", state._recall_dsn())
    state._close_recall_pool()
    assert closed["n"] == 1
    assert state._RECALL_POOL is None
    assert state._RECALL_POOL_DSN is None
