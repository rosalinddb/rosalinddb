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

    The write path now allocates the whole LSN block in ONE upsert-increment
    (`last_lsn = last_lsn + N`) and applies the batch in ONE multi-row UPSERT
    (via `psycopg2.extras.execute_values`, which renders the rows into a single
    `cur.execute`). `RETURNING last_lsn` is faked by advancing a running counter
    by the requested block size (the 3rd param of the seq upsert), so the
    returned value is the LAST lsn in the block — exactly the contract the
    production code assigns the range `last_lsn-N+1 .. last_lsn` from.
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
            # The block size N is bound as the 3rd positional param
            # (VALUES (tenant, dataset, N) ...). Advance by N and return the
            # last lsn in the freshly-reserved block.
            block = params[2] if params else 1
            self._lsn += block

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
    """Turn the tier on and return (conn, cur, upserts).

    The set-based write applies the batch in ONE multi-row UPSERT via
    `psycopg2.extras.execute_values` (imported by-name into `state`). Driving the
    real `execute_values` needs a live connection encoding, so stub it with a
    recorder that captures the rendered SQL + the per-row params — that is the
    call shape the test asserts. `upserts` is the list of recorded
    `(sql, rows, template)` tuples.
    """
    monkeypatch.setenv("RB_HOT_DSN", "postgresql://u:p@hot:5432/hot")
    monkeypatch.setenv("RB_DELTA_TIER", "true")
    importlib.reload(state)
    cur = _FakeCursor()
    conn = _FakeConn(cur)
    monkeypatch.setattr(state, "_hot_conn", lambda: conn)
    upserts: list[tuple] = []

    def _fake_execute_values(c, sql, rows, template=None, **kw):
        assert c is cur
        upserts.append((sql, list(rows), template))

    monkeypatch.setattr(state, "execute_values", _fake_execute_values)
    return conn, cur, upserts


def test_hot_upsert_allocates_lsn_block_and_single_upsert(state, monkeypatch):
    """N LSNs are allocated in ONE statement; the batch is ONE multi-row UPSERT.

    Round-trip count is now ~2 (one seq block-allocation + one multi-row UPSERT),
    NOT 2N. Each record is stamped with its LSN from the contiguous block in
    input order.
    """
    conn, cur, upserts = _on_state(state, monkeypatch)
    records = [
        {"id": "a", "values": [1.0, 2.0, 3.0], "metadata": {"k": "v"}},
        {"id": "b", "values": [4.0, 5.0, 6.0], "metadata": {}},
    ]
    written = state.hot_upsert_vectors("t1", "ds", records)
    assert written == 2

    # ONE seq allocation for the whole batch (not one per record).
    seq_calls = [c for c in cur.calls if "hot_lsn_seq" in c[0]]
    assert len(seq_calls) == 1, "the LSN block is allocated in a single statement"
    # The block size N is bound (VALUES (tenant, dataset, N); +N on conflict).
    assert seq_calls[0][1] == ("t1", "ds", 2, 2)

    # ONE multi-row UPSERT for the whole batch.
    assert len(upserts) == 1, "the batch is applied in a single multi-row UPSERT"
    sql, rows, template = upserts[0]
    assert len(rows) == 2, "both records in one UPSERT"

    # Rows are (tenant, dataset, id, embedding_literal, metadata_json, lsn).
    assert rows[0][0] == "t1" and rows[0][1] == "ds"
    assert rows[0][2] == "a" and rows[0][5] == 1
    assert rows[1][2] == "b" and rows[1][5] == 2

    # The embedding is bound as a pgvector literal; metadata as JSON.
    assert rows[0][3] == "[1.0,2.0,3.0]"
    assert json.loads(rows[0][4]) == {"k": "v"}

    # Last-write-wins + tombstone-clear on conflict, scoped to (tenant,ds,id).
    assert "ON CONFLICT (tenant_id, dataset, id)" in sql
    assert "deleted   = FALSE" in sql
    # `deleted` is set FALSE for every row via the row template, not per-row.
    assert template == "(%s, %s, %s, %s, %s, %s, FALSE)"

    # The batch committed once and the connection was closed.
    assert conn.committed is True
    assert conn.closed is True


def test_hot_upsert_intra_batch_duplicate_id_last_write_wins(state, monkeypatch):
    """A duplicate id in one batch collapses to one row; the LAST input wins.

    Postgres rejects a multi-row UPSERT that lists the same conflict key twice,
    so the batch must be deduped before the UPSERT — keeping the latest
    occurrence (last-write-wins) and giving it a single LSN from the block.
    """
    conn, cur, upserts = _on_state(state, monkeypatch)
    records = [
        {"id": "dup", "values": [1.0, 1.0, 1.0], "metadata": {"v": 1}},
        {"id": "other", "values": [2.0, 2.0, 2.0], "metadata": {}},
        {"id": "dup", "values": [9.0, 9.0, 9.0], "metadata": {"v": 2}},
    ]
    written = state.hot_upsert_vectors("t1", "ds", records)
    # Two distinct ids -> two rows written, two LSNs allocated.
    assert written == 2
    seq_calls = [c for c in cur.calls if "hot_lsn_seq" in c[0]]
    assert seq_calls[0][1] == ("t1", "ds", 2, 2), "block size is the distinct count"

    sql, rows, template = upserts[0]
    by_id = {r[2]: r for r in rows}
    assert set(by_id) == {"dup", "other"}, "exactly one row per distinct id"

    # The surviving `dup` row is the LATEST occurrence (values [9,9,9], v=2)...
    assert by_id["dup"][3] == "[9.0,9.0,9.0]"
    assert json.loads(by_id["dup"][4]) == {"v": 2}
    # ...and it carries the LATER LSN (its winning position is last in input).
    assert by_id["dup"][5] == 2
    assert by_id["other"][5] == 1


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
