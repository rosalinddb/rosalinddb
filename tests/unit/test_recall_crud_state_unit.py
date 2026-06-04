"""Unit coverage for the recall-tier CRUD state helpers (PR6).

Hermetic — no Docker, no pgvector. The recall connection is FAKED (the same
fake-cursor/connection trick the recall write-path suite uses), so these prove
the SQL shape + the read-your-deletes / above-watermark-lsn contract of:

  - `recall_get_vector`  — point lookup above the watermark: live → metadata,
    tombstone → ("tombstone", None), absent → (None, None).
  - `recall_list_rows`   — partition scan above the watermark: live rows + the
    FULL suppress-id set (live AND tombstoned).
  - `recall_delete_vector` — REGRESSION: allocates a FRESH lsn from `recall_lsn_seq`
    (the same upsert-increment writes use) and UPSERTs a `deleted=true` tombstone
    stamped with it. The allocated lsn MUST be strictly greater than the prior
    max (and thus strictly above any watermark `<= max`). Guards the
    below-watermark-tombstone bug the consolidation review flagged.

The integration suite proves the same against a real pgvector container.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def state(monkeypatch):
    """State module with the recall tier ON (faked connection)."""
    monkeypatch.setenv("RB_RECALL_DSN", "postgresql://u:p@recall:5432/recall")
    monkeypatch.setenv("RB_RECALL", "true")
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    yield state_mod
    monkeypatch.delenv("RB_RECALL", raising=False)
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    importlib.reload(state_mod)


class _FakeCur:
    """Records executed SQL/params; returns a scripted row set / fetchone.

    `recall_search` / `recall_list_rows` now issue TWO scans each — a
    `NOT deleted` LIVE/MATCH scan and a separate full SUPPRESS scan — so the fake
    projects the scripted partition rows to the columns the CURRENT statement
    selects: the SUPPRESS scan (`SELECT id ...`) yields `(id,)` for EVERY row, the
    LIVE list scan (`SELECT id, metadata ...`) yields `(id, metadata)` for
    not-deleted rows only. Each scripted row is `(id, metadata, deleted)`.
    """

    def __init__(self, rows=None, seq_start=0):
        self.calls: list[tuple] = []
        self._rows = rows or []
        self._lsn = seq_start
        self._last_sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        self._last_sql = sql
        if "recall_lsn_seq" in sql and "RETURNING" in sql:
            # The delete allocates a SINGLE lsn (bump by 1).
            self._lsn += 1

    def fetchone(self):
        if "recall_lsn_seq" in self._last_sql:
            return (self._lsn,)
        return self._rows[0] if self._rows else None

    def fetchall(self):
        sql = self._last_sql
        # SUPPRESS scan: `SELECT id FROM recall_vectors WHERE ...` — every id.
        if "SELECT id\nFROM recall_vectors" in sql:
            return [(rid,) for rid, _meta, _del in self._rows]
        # LIVE list scan: `SELECT id, metadata ... AND NOT deleted ...` — not-deleted.
        if "SELECT id, metadata" in sql and "NOT deleted" in sql:
            return [(rid, meta) for rid, meta, deleted in self._rows if not deleted]
        # Fallback: the raw scripted rows (point-lookup style helpers).
        return list(self._rows)


class _FakeConn:
    def __init__(self, cur):
        self._cur = cur

    def cursor(self):
        return self._cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass


def _wire(state, monkeypatch, rows=None, seq_start=0):
    cur = _FakeCur(rows=rows, seq_start=seq_start)
    monkeypatch.setattr(state, "_recall_conn", lambda: _FakeConn(cur))
    return cur


# --- recall_get_vector -----------------------------------------------------


def test_get_vector_live(state, monkeypatch):
    cur = _wire(state, monkeypatch, rows=[({"v": "recall"}, False)])
    status, meta = state.recall_get_vector("t", "ds", "id", watermark=5)
    assert status == "live" and meta == {"v": "recall"}
    # The lookup is scoped above the watermark.
    sql, params = cur.calls[-1]
    assert "lsn > %s" in sql
    assert params == ("t", "ds", "id", 5)


def test_get_vector_tombstone(state, monkeypatch):
    _wire(state, monkeypatch, rows=[({"stale": True}, True)])
    status, meta = state.recall_get_vector("t", "ds", "id", watermark=0)
    assert status == "tombstone" and meta is None


def test_get_vector_absent(state, monkeypatch):
    _wire(state, monkeypatch, rows=[])
    status, meta = state.recall_get_vector("t", "ds", "id", watermark=0)
    assert status is None and meta is None


def test_get_vector_null_metadata_coalesced(state, monkeypatch):
    """A SQL NULL metadata (None) on a live row coalesces to {}."""
    _wire(state, monkeypatch, rows=[(None, False)])
    status, meta = state.recall_get_vector("t", "ds", "id", watermark=0)
    assert status == "live" and meta == {}


# --- recall_get_vector_with_embedding (include_values recall path) ----------


def test_get_vector_with_embedding_live_returns_parsed_vector(state, monkeypatch):
    # fetchone fallback returns the scripted row as-is: (metadata, deleted, embedding).
    cur = _wire(
        state,
        monkeypatch,
        rows=[({"v": "recall"}, False, "[1,2,3,4]")],
    )
    status, meta, emb = state.recall_get_vector_with_embedding(
        "t", "ds", "id", watermark=5
    )
    assert status == "live"
    assert meta == {"v": "recall"}
    # The pgvector text literal is parsed to a float list (never zeros).
    assert emb == [1.0, 2.0, 3.0, 4.0]
    sql, params = cur.calls[-1]
    assert "embedding" in sql and "lsn > %s" in sql
    assert params == ("t", "ds", "id", 5)


def test_get_vector_with_embedding_tombstone(state, monkeypatch):
    _wire(state, monkeypatch, rows=[({"stale": True}, True, "[9,9,9,9]")])
    status, meta, emb = state.recall_get_vector_with_embedding(
        "t", "ds", "id", watermark=0
    )
    assert status == "tombstone" and meta is None and emb is None


def test_get_vector_with_embedding_absent(state, monkeypatch):
    _wire(state, monkeypatch, rows=[])
    status, meta, emb = state.recall_get_vector_with_embedding(
        "t", "ds", "id", watermark=0
    )
    assert status is None and meta is None and emb is None


# --- recall_list_rows ------------------------------------------------------


def test_list_rows_live_and_suppress(state, monkeypatch):
    """Live rows return; suppress_ids includes EVERY id above the watermark."""
    cur = _wire(
        state,
        monkeypatch,
        rows=[
            ("live1", {"a": 1}, False),
            ("tomb1", {}, True),       # tombstone: suppress, no live row
            ("live2", None, False),    # NULL metadata coalesces to {}
        ],
    )
    live, suppress = state.recall_list_rows("t", "ds", watermark=9)
    assert {r["id"] for r in live} == {"live1", "live2"}
    by_id = {r["id"]: r for r in live}
    assert by_id["live1"]["metadata"] == {"a": 1}
    assert by_id["live2"]["metadata"] == {}
    # suppress carries the tombstone too (recall-authoritative for all ids).
    assert suppress == {"live1", "tomb1", "live2"}
    sql, params = cur.calls[-1]
    assert "lsn > %s" in sql and params == ("t", "ds", 9)


def test_list_rows_empty_partition(state, monkeypatch):
    _wire(state, monkeypatch, rows=[])
    live, suppress = state.recall_list_rows("t", "ds", watermark=0)
    assert live == [] and suppress == set()


# --- recall_delete_vector (REGRESSION: above-watermark fresh lsn) ----------


def test_delete_allocates_fresh_lsn_above_max(state, monkeypatch):
    """The tombstone lsn is allocated from the seq (prior max 50 → 51), above any watermark.

    The seq returns `max+1`, which is strictly greater than the partition max and
    therefore strictly greater than any watermark (a watermark is always
    `<= max(lsn)`). This is the contract that keeps the tombstone INSIDE the
    union's `lsn > watermark` scan window and applicable by the next
    consolidation — the below-watermark-tombstone regression guard.
    """
    cur = _wire(state, monkeypatch, seq_start=50)
    lsn = state.recall_delete_vector("t", "ds", "doc-1", dimension=4)
    assert lsn == 51, "delete must allocate a fresh lsn = prior max + 1"
    assert lsn > 50, "tombstone lsn MUST be strictly above the partition max/watermark"

    # Statement 1: seq upsert-increment (the SAME mechanism the write path uses).
    seq_calls = [c for c in cur.calls if "recall_lsn_seq" in c[0]]
    assert len(seq_calls) == 1
    assert "last_lsn = recall_lsn_seq.last_lsn + 1" in seq_calls[0][0]

    # Statement 2: tombstone UPSERT with deleted=TRUE, stamped with the fresh lsn.
    tomb = next(c for c in cur.calls if "recall_vectors" in c[0])
    sql, params = tomb
    assert "deleted" in sql.lower() and "TRUE" in sql
    assert "ON CONFLICT (tenant_id, dataset, id)" in sql
    # params: (tenant, dataset, id, embedding_literal, lsn)
    assert params[0] == "t" and params[1] == "ds" and params[2] == "doc-1"
    assert params[4] == 51, "the tombstone is stamped with the freshly-allocated lsn"


def test_delete_never_flips_in_place(state, monkeypatch):
    """The delete must NOT be an in-place UPDATE that reuses an old lsn.

    A pure `UPDATE ... SET deleted=true` (no seq allocation) would leave the
    tombstone at its old, possibly-below-watermark lsn — the exact bug. Assert the
    seq is ALWAYS consulted (a fresh lsn is allocated) and no in-place UPDATE of
    `deleted` without a new lsn is issued.
    """
    cur = _wire(state, monkeypatch, seq_start=0)
    state.recall_delete_vector("t", "ds", "x", dimension=3)
    assert any("recall_lsn_seq" in c[0] for c in cur.calls), (
        "delete MUST allocate a fresh lsn (never flip deleted=true in place)"
    )
    # The tombstone write is an INSERT ... ON CONFLICT (upsert), not a bare UPDATE.
    tomb = next(c for c in cur.calls if "recall_vectors" in c[0])
    assert tomb[0].strip().upper().startswith("INSERT INTO RECALL_VECTORS")


def test_delete_placeholder_embedding_matches_dimension(state, monkeypatch):
    """A cold-only delete's tombstone gets a zero placeholder of the dataset dim.

    `recall_search` now excludes tombstones from its MATCH scan in SQL, so the
    placeholder is never ranked against a query vector — its dimension is a
    belt-and-suspenders match for the `NOT NULL vector` column rather than a
    search-ranking correctness dependency. The helper still writes a zero-vector
    of the dataset dimension.
    """
    cur = _wire(state, monkeypatch, seq_start=0)
    state.recall_delete_vector("t", "ds", "x", dimension=4)
    tomb = next(c for c in cur.calls if "recall_vectors" in c[0])
    embedding_literal = tomb[1][3]
    # A 4-dim zero vector literal.
    assert embedding_literal == "[0.0,0.0,0.0,0.0]", embedding_literal
