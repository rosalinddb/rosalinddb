"""Unit coverage for `recall_search`'s single-snapshot scan (task #17).

`recall_search` (the recall half of the query union) used to run TWO separate
SQL statements on a default READ COMMITTED recall connection:

  (a) a MATCH scan  — live rows, `lsn > W`, ordered by distance;
  (b) a SUPPRESS scan — every id, `lsn > W`.

Under READ COMMITTED each `cur.execute` takes a FRESH MVCC snapshot, so a
re-UPSERT that commits BETWEEN scan (a) and scan (b) makes (b) observe ids that
(a) did not — `suppress_ids ⊋ match_ids` — and the union over-suppresses to a
transient 0-result query (benchmark case b1; root cause:
bench-lab/analysis/b1-rootcause.md).

The fix collapses the two scans into ONE SELECT over `lsn > W` returning
`(id, deleted, score, metadata)`, evaluated against a SINGLE MVCC snapshot even
under READ COMMITTED. `suppress_ids` and `matches` are then derived in Python
from that one consistent row set, so suppression can never see an id the match
split did not.

These tests are hermetic (no Docker, no pgvector): they wire a fake recall pool
+ cursor (mirroring `tests/unit/test_recall_write_path_unit.py`) so the scan
SHAPE and the Python split are asserted without a real database.

The two headline regression properties:

  - **Single statement** — exactly ONE `cur.execute` drives the scan. The old
    two-execute shape WAS the race's structural cause; asserting a single execute
    is the deterministic regression that proves the snapshot is atomic.
  - **Authoritative suppression from the one snapshot** — `suppress_ids` is EVERY
    id the single scan returned (live AND tombstoned), and `matches` is the live
    filter-passing subset; both come from the same rows, so they can never skew.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def state(monkeypatch):
    """Fresh state module with the recall tier ON (flag + DSN set)."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("RB_RECALL_DSN", "postgresql://u:p@recall:5432/recall")
    monkeypatch.setenv("RB_RECALL", "true")
    import adapters.state.state as state_mod

    importlib.reload(state_mod)
    yield state_mod
    monkeypatch.delenv("RB_RECALL", raising=False)
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    importlib.reload(state_mod)


class _FakeCursor:
    """A psycopg2-cursor stand-in for the recall single-scan.

    Records every `execute` (SQL + params) and returns a fixed result row set
    for the scan. Each row is `(id, deleted, score, metadata)` — the exact shape
    the single-statement scan SELECTs.
    """

    def __init__(self, rows):
        self._rows = rows
        self.executes: list[tuple[str, tuple]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executes.append((sql, params))

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    """A psycopg2-connection stand-in yielding one fixed cursor."""

    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class _FakeRecallPool:
    """A `ThreadedConnectionPool`-shaped stub returning a single fake conn."""

    def __init__(self, conn):
        self._conn = conn
        self._pool = [object()]
        self.returned = 0

    def getconn(self):
        return self._conn

    def putconn(self, conn):
        assert conn is self._conn
        self.returned += 1


def _wire(state, monkeypatch, rows):
    """Bind a fake recall pool whose single conn/cursor returns `rows`."""
    cur = _FakeCursor(rows)
    conn = _FakeConn(cur)
    pool = _FakeRecallPool(conn)
    monkeypatch.setattr(state, "_RECALL_POOL", pool)
    monkeypatch.setattr(state, "_RECALL_POOL_DSN", state._recall_dsn())
    return cur, conn, pool


# --- the deterministic single-statement regression ------------------------


def test_recall_search_runs_exactly_one_execute(state, monkeypatch):
    """The scan is ONE `cur.execute` — the structural fix for the b1 race.

    The old two-scan shape ran two `cur.execute`s in two READ COMMITTED
    snapshots, which is exactly what let a between-scan commit over-suppress.
    Asserting a single execute is the deterministic regression: one statement is
    one MVCC snapshot, so `suppress_ids ⊇ match_ids` always holds.
    """
    cur, _conn, _pool = _wire(state, monkeypatch, rows=[])
    state.recall_search("t", "ds", [0.0, 0.0], top_k=5, watermark=0)
    assert len(cur.executes) == 1, (
        f"recall_search must drive the scan with EXACTLY one execute "
        f"(the single-snapshot fix); saw {len(cur.executes)}: "
        f"{[s for s, _ in cur.executes]}"
    )


def test_recall_search_scan_filters_above_watermark_in_one_statement(
    state, monkeypatch
):
    """The single statement scopes to `(tenant, dataset)` AND `lsn > W`.

    The one scan must carry the partition + watermark predicate itself (no
    second statement), and it must return `deleted` so tombstones are split out
    in Python — preserving the "no LIMIT can crowd out a real match with a
    tombstone" property by the explicit flag rather than statement ordering.
    """
    cur, _conn, _pool = _wire(state, monkeypatch, rows=[])
    state.recall_search("ten", "data", [1.0, 2.0], top_k=3, watermark=42)
    (sql, params) = cur.executes[0]
    assert "tenant_id" in sql and "dataset" in sql
    assert "lsn >" in sql, "scan must filter lsn > watermark"
    assert "deleted" in sql, "scan must SELECT `deleted` to split tombstones in Python"
    # The watermark is bound (42 appears in the params), proving the scan is
    # scoped to rows above the resolved shard's watermark.
    assert 42 in params, f"watermark not bound into the single scan: {params}"
    assert "ten" in params and "data" in params


# --- behavioural split (suppress / match) from the one snapshot -----------


def test_reupsert_of_live_id_is_returned(state, monkeypatch):
    """A live row above the watermark is BOTH a match and in suppress_ids."""
    rows = [("a", False, 0.25, {})]
    _wire(state, monkeypatch, rows)
    suppress, matches = state.recall_search("t", "ds", [0.0, 0.0], top_k=5, watermark=0)
    assert "a" in suppress, "every returned id suppresses its cold twin"
    assert [m["id"] for m in matches] == ["a"], "a live row is a match"
    assert matches[0]["deleted"] is False


def test_tombstone_suppresses_but_is_never_a_match(state, monkeypatch):
    """A tombstone (deleted=True) suppresses its cold twin and is NOT a match."""
    rows = [("dead", True, None, {})]
    _wire(state, monkeypatch, rows)
    suppress, matches = state.recall_search("t", "ds", [0.0, 0.0], top_k=5, watermark=0)
    assert suppress == {"dead"}, "a tombstone suppresses its cold copy"
    assert matches == [], "a tombstone is NEVER a match"


def test_filter_failing_live_row_suppresses_but_is_not_a_match(state, monkeypatch):
    """A live row that fails the AND-of-equals filter still suppresses.

    The P1 fix: an authoritative live re-upsert that fails the query filter must
    not let a stale, filter-matching cold copy leak — so its id is in
    suppress_ids even though it is not a match.
    """
    rows = [("x", False, 0.1, {"tag": "blue"})]
    _wire(state, monkeypatch, rows)
    suppress, matches = state.recall_search(
        "t", "ds", [0.0, 0.0], top_k=5, watermark=0, flt={"tag": "red"}
    )
    assert "x" in suppress, "a filter-failing live row still suppresses its cold copy"
    assert matches == [], "a filter-failing live row is not a match"


def test_filter_passing_live_row_is_a_match_and_suppresses(state, monkeypatch):
    """A live row that PASSES the filter is a match AND suppresses."""
    rows = [("x", False, 0.1, {"tag": "red"})]
    _wire(state, monkeypatch, rows)
    suppress, matches = state.recall_search(
        "t", "ds", [0.0, 0.0], top_k=5, watermark=0, flt={"tag": "red"}
    )
    assert "x" in suppress
    assert [m["id"] for m in matches] == ["x"]


def test_suppress_ids_is_every_id_match_is_only_live_filter_passing(state, monkeypatch):
    """One snapshot -> suppress = ALL ids; matches = live filter-passing subset.

    Mixed snapshot: a live match, a tombstone, and a filter-failing live row.
    suppress_ids must be all three; matches must be only the passing live one.
    This is the union-completeness guarantee: suppress_ids ⊇ match_ids always.
    """
    rows = [
        ("keep", False, 0.05, {"k": 1}),   # live, passes filter -> match + suppress
        ("gone", True, None, {}),           # tombstone -> suppress only
        ("drop", False, 0.02, {"k": 9}),    # live, fails filter -> suppress only
    ]
    _wire(state, monkeypatch, rows)
    suppress, matches = state.recall_search(
        "t", "ds", [0.0, 0.0], top_k=5, watermark=0, flt={"k": 1}
    )
    assert suppress == {"keep", "gone", "drop"}, "every returned id must suppress"
    assert [m["id"] for m in matches] == ["keep"], "only the live filter-passing match"
    # The completeness invariant the b1 race violated: match ids ⊆ suppress ids.
    assert {m["id"] for m in matches} <= suppress


def test_matches_ascending_by_score_and_capped_at_top_k(state, monkeypatch):
    """Live matches are ordered ascending by score and capped at top_k.

    A tombstone with a 0.0 placeholder distance must NEVER crowd out a real live
    match even when it sorts first by raw distance — it is filtered out by the
    `deleted` flag in Python, so the no-leak property holds without a SQL LIMIT.
    """
    rows = [
        ("near", False, 0.1, {}),
        ("zero", True, 0.0, {}),   # tombstone with the smallest "distance"
        ("mid", False, 0.2, {}),
        ("far", False, 0.3, {}),
    ]
    _wire(state, monkeypatch, rows)
    suppress, matches = state.recall_search(
        "t", "ds", [0.0, 0.0], top_k=2, watermark=0
    )
    # All four ids suppress; the tombstone is never a match despite distance 0.
    assert suppress == {"near", "zero", "mid", "far"}
    assert [m["id"] for m in matches] == ["near", "mid"], (
        "matches must be the 2 closest LIVE rows, ascending by score, with the "
        "tombstone excluded by the deleted flag (not by SQL LIMIT)"
    )


def test_score_is_l2_squared(state, monkeypatch):
    """`score` is the FAISS-aligned L2-squared distance, returned verbatim.

    The single scan computes `power(embedding <-> q, 2)` (pgvector plain-L2
    squared) so it merges directly with the cold shard's FAISS L2². The Python
    split must pass that score through as a float, unchanged.
    """
    # The fake cursor returns the already-squared value the SQL produced.
    rows = [("a", False, 9.0, {})]  # e.g. an L2 distance of 3.0 squared
    cur, _conn, _pool = _wire(state, monkeypatch, rows)
    _suppress, matches = state.recall_search("t", "ds", [0.0, 0.0], top_k=5, watermark=0)
    assert matches[0]["score"] == 9.0
    assert isinstance(matches[0]["score"], float)
    # And the SQL squares the distance (power(... , 2)) — the alignment is in SQL.
    (sql, _params) = cur.executes[0]
    assert "power(" in sql and ", 2)" in sql, "score must be L2 SQUARED in SQL"


def test_metadata_none_coalesced_to_empty_dict(state, monkeypatch):
    """A SQL NULL metadata decodes to None and is coalesced to `{}` in a match."""
    rows = [("a", False, 0.1, None)]
    _wire(state, monkeypatch, rows)
    _suppress, matches = state.recall_search("t", "ds", [0.0, 0.0], top_k=5, watermark=0)
    assert matches[0]["metadata"] == {}


def test_empty_scan_yields_empty_suppress_and_matches(state, monkeypatch):
    """No recall rows above the watermark -> empty suppress set and no matches."""
    _wire(state, monkeypatch, rows=[])
    suppress, matches = state.recall_search("t", "ds", [0.0, 0.0], top_k=5, watermark=0)
    assert suppress == set()
    assert matches == []
