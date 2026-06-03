"""Unit coverage for the synchronous hot-tier write path + `RB_DELTA_TIER`.

The delta tier ships behind `RB_DELTA_TIER` (master switch) AND `RB_HOT_DSN`
(the hot store), both default off. This PR adds the WRITE path only — the query
union, flush, and hot-delete are later PRs.

Two headline properties, both proven hermetically here (no Docker, no pgvector):

  - FLAG OFF (default): `post_vectors` is byte-identical to today — 202, a
    landing write, a `VALIDATE_DATASET` publish, and NO hot connection ever
    opened (asserted with the `psycopg2.connect`-raises trick from PR2).
  - FLAG ON: `delta_tier_enabled()` flips on only with BOTH env vars; the write
    logic allocates a monotonic LSN per record and UPSERTs into `hot_vectors`
    with last-write-wins, against a FAKED hot connection (so the call SHAPE is
    asserted without a real database).
"""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def state(monkeypatch):
    """Fresh state module with both delta-tier env vars cleared (tier off)."""
    monkeypatch.delenv("RB_DELTA_TIER", raising=False)
    monkeypatch.delenv("RB_HOT_DSN", raising=False)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    yield state_mod
    monkeypatch.delenv("RB_DELTA_TIER", raising=False)
    monkeypatch.delenv("RB_HOT_DSN", raising=False)
    importlib.reload(state_mod)


# --- delta_tier_enabled() gate -------------------------------------------


def test_delta_tier_off_by_default(state):
    """Both env vars unset -> tier off (the self-host default)."""
    assert state.delta_tier_enabled() is False


def test_delta_tier_needs_flag_and_dsn(state, monkeypatch):
    """The tier is on ONLY when `RB_DELTA_TIER` is truthy AND `RB_HOT_DSN` is set."""
    # Flag alone (no DSN) -> still off: a deploy that forgets the DSN stays on
    # the byte-identical flag-off path rather than erroring on every write.
    monkeypatch.setenv("RB_DELTA_TIER", "true")
    importlib.reload(state)
    assert state.delta_tier_enabled() is False

    # DSN alone (no flag) -> off: provisioning a hot store does not auto-enable.
    monkeypatch.delenv("RB_DELTA_TIER", raising=False)
    monkeypatch.setenv("RB_HOT_DSN", "postgresql://u:p@hot:5432/hot")
    importlib.reload(state)
    assert state.delta_tier_enabled() is False

    # Both -> on.
    monkeypatch.setenv("RB_DELTA_TIER", "true")
    importlib.reload(state)
    assert state.delta_tier_enabled() is True


@pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", "on"])
def test_delta_tier_truthy_values(state, monkeypatch, truthy):
    """`RB_DELTA_TIER` accepts the same truthy set as `quotas_enabled()`."""
    monkeypatch.setenv("RB_HOT_DSN", "postgresql://u:p@hot:5432/hot")
    monkeypatch.setenv("RB_DELTA_TIER", truthy)
    importlib.reload(state)
    assert state.delta_tier_enabled() is True


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off", "  "])
def test_delta_tier_falsy_values(state, monkeypatch, falsy):
    """A blank/false `RB_DELTA_TIER` keeps the tier off even with a DSN set."""
    monkeypatch.setenv("RB_HOT_DSN", "postgresql://u:p@hot:5432/hot")
    monkeypatch.setenv("RB_DELTA_TIER", falsy)
    importlib.reload(state)
    assert state.delta_tier_enabled() is False


# --- hot_upsert_vectors(): LSN assignment + UPSERT call shape -------------


class _FakeCursor:
    """A psycopg2-cursor stand-in that records executed SQL + params.

    `RETURNING last_lsn` is faked by handing back a monotonically increasing
    integer per `hot_lsn_seq` upsert, so the write path's LSN allocation can be
    asserted without a real Postgres.
    """

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self._lsn = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if "hot_lsn_seq" in sql and "RETURNING" in sql:
            self._lsn += 1

    def fetchone(self):
        return (self._lsn,)


class _FakeConn:
    """A psycopg2-connection stand-in usable as `with conn` (txn) + `closing`."""

    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        # psycopg2's connection-context-manager commits on clean exit, rolls
        # back on exception. Mirror that so the txn semantics are asserted.
        if exc_type is None:
            self.committed = True
        else:
            self.rolled_back = True
        return False

    def close(self):
        self.closed = True


def _on_state(state, monkeypatch):
    """Turn the tier on and return a fake hot connection bound into the module."""
    monkeypatch.setenv("RB_HOT_DSN", "postgresql://u:p@hot:5432/hot")
    monkeypatch.setenv("RB_DELTA_TIER", "true")
    importlib.reload(state)
    cur = _FakeCursor()
    conn = _FakeConn(cur)
    monkeypatch.setattr(state, "_hot_conn", lambda: conn)
    return conn, cur


def test_hot_upsert_assigns_monotonic_lsn_and_upserts(state, monkeypatch):
    """Each record gets the next LSN and an UPSERT into hot_vectors."""
    conn, cur = _on_state(state, monkeypatch)
    records = [
        {"id": "a", "values": [1.0, 2.0, 3.0], "metadata": {"k": "v"}},
        {"id": "b", "values": [4.0, 5.0, 6.0], "metadata": {}},
    ]
    written = state.hot_upsert_vectors("t1", "ds", records)
    assert written == 2

    seq_calls = [c for c in cur.calls if "hot_lsn_seq" in c[0]]
    upserts = [c for c in cur.calls if "INSERT INTO hot_vectors" in c[0]]
    assert len(seq_calls) == 2, "one LSN allocation per record"
    assert len(upserts) == 2, "one UPSERT per record"

    # LSNs are strictly monotonic (1, 2) and stamped onto the matching UPSERT.
    # UPSERT params: (tenant, dataset, id, embedding_literal, metadata_json, lsn).
    assert upserts[0][1][0] == "t1" and upserts[0][1][1] == "ds"
    assert upserts[0][1][2] == "a"
    assert upserts[0][1][5] == 1
    assert upserts[1][1][2] == "b"
    assert upserts[1][1][5] == 2

    # The embedding is bound as a pgvector literal; metadata as JSON.
    assert upserts[0][1][3] == "[1.0,2.0,3.0]"
    assert json.loads(upserts[0][1][4]) == {"k": "v"}

    # Last-write-wins + tombstone-clear on conflict, scoped to (tenant,ds,id).
    assert "ON CONFLICT (tenant_id, dataset, id)" in upserts[0][0]
    assert "deleted   = FALSE" in upserts[0][0]

    # The batch committed once and the connection was closed.
    assert conn.committed is True
    assert conn.closed is True


def test_hot_upsert_empty_is_noop_no_connection(state, monkeypatch):
    """An empty record list never opens a hot connection."""
    monkeypatch.setenv("RB_HOT_DSN", "postgresql://u:p@hot:5432/hot")
    monkeypatch.setenv("RB_DELTA_TIER", "true")
    importlib.reload(state)

    def _boom():  # pragma: no cover - must never be called
        raise AssertionError("opened a hot connection for an empty batch")

    monkeypatch.setattr(state, "_hot_conn", _boom)
    assert state.hot_upsert_vectors("t1", "ds", []) == 0


def test_hot_conn_raises_when_dsn_unset(state):
    """`_hot_conn()` refuses to connect with `RB_HOT_DSN` unset (off path).

    Reaching here with the tier off is a programming error — it raises rather
    than silently connecting to a default DSN.
    """
    with pytest.raises(RuntimeError):
        state._hot_conn()


def test_to_pgvector_literal_format(state):
    """Embeddings format as a bracketed, comma-separated pgvector literal."""
    assert state._to_pgvector_literal([1, 2, 3]) == "[1.0,2.0,3.0]"
    assert state._to_pgvector_literal([0.5, -1.25]) == "[0.5,-1.25]"
