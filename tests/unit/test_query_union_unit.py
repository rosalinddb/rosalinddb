"""Unit coverage for the Recall + Consolidated query union (PR4).

Hermetic — no Docker, no pgvector, no network. Exercises the merge logic and the
`run_query` union orchestration directly, with the recall connection / cold
search faked. The integration suite (`tests/integration/test_query_union.py`)
proves the same properties end-to-end against a real pgvector container.

Headline properties proven here:

  - **Metric alignment** (correctness-critical): pgvector's plain-L2 `<->`
    SQUARED ranks identically to FAISS L2² over the same un-normalised vectors.
    The squaring is monotonic, so this is really a "we square it, and squaring
    preserves order" proof on a deterministic example — the most likely silent
    ranking bug, per the spec.
  - **Dedup recall-wins**: a recall LIVE row overrides a cold match for the same
    id.
  - **Tombstone suppression**: a recall tombstone (`deleted=true`) drops a cold
    match for that id and contributes no match.
  - **Filter applied to recall**: the AND-of-equals predicate is enforced on
    recall live rows; a tombstone is returned regardless so a delete suppresses.
  - **Watermark partition (I1/I3)**: the recall scan filters `lsn > watermark`,
    and the watermark is the resolved shard's `consolidated_lsn` (0 when no
    shard).
  - **Flag OFF (default)**: byte-identical to the cold-only path AND no recall
    connection is ever opened (the `psycopg2.connect`-raises trick).
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import services.query_api.v1_query as v1q


# --- metric alignment -----------------------------------------------------


def test_pgvector_l2_squared_matches_faiss_l2_squared_ordering():
    """`power(<->, 2)` (the recall SQL) == FAISS L2² over the same vectors.

    FAISS `IndexFlatL2` returns the SQUARED Euclidean distance. pgvector `<->`
    returns the plain (sqrt) Euclidean distance; the recall scan squares it. On
    a deterministic, un-normalised example the two distance vectors are equal
    (to float tolerance) AND induce the identical ascending ranking — so a merge
    that sorts by the recall score and the cold score interleaves correctly.
    """
    import faiss  # type: ignore

    query = np.array([[0.2, 0.5, -0.1, 0.9]], dtype=np.float32)
    vecs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.5, -0.1, 0.9],  # exact match → distance 0
            [1.0, -1.0, 0.5, 0.0],
            [-0.3, 0.2, 0.4, 0.7],
        ],
        dtype=np.float32,
    )

    # FAISS L2² distances.
    index = faiss.IndexFlatL2(4)
    index.add(vecs)
    faiss_d, faiss_i = index.search(query, len(vecs))
    faiss_d = faiss_d[0]
    faiss_order = list(faiss_i[0])

    # pgvector `<->` is plain L2 = sqrt(sum of squared diffs); the recall scan
    # squares it. Compute that here and confirm it equals FAISS L2² and ranks
    # identically.
    plain_l2 = np.linalg.norm(vecs - query[0], axis=1)
    squared = plain_l2 ** 2  # what `power(embedding <-> q, 2)` yields
    pg_order = list(np.argsort(squared, kind="stable"))

    # Same ranking, same distances (squaring aligns the metric).
    assert pg_order == faiss_order, (pg_order, faiss_order)
    # Distances align element-wise (sort both by index to compare).
    np.testing.assert_allclose(
        squared[faiss_order], faiss_d, rtol=1e-5, atol=1e-5
    )
    # The exact-match vector (index 1) has score 0 in both.
    assert squared[1] == pytest.approx(0.0, abs=1e-6)


# --- watermark resolution (I3) --------------------------------------------


def test_watermark_zero_when_no_shard():
    """No resolved shard → watermark 0, so ALL recall rows qualify."""
    assert v1q._watermark_for_shard(None) == 0
    assert v1q._watermark_for_shard({}) == 0


def test_watermark_reads_consolidated_lsn_of_resolved_shard():
    """The watermark is the resolved shard's `consolidated_lsn` (the I3 pairing)."""
    assert v1q._watermark_for_shard({"id": 7, "consolidated_lsn": 42}) == 42
    # A shard predating migration 008 / memory-mode row has no key → 0.
    assert v1q._watermark_for_shard({"id": 1}) == 0
    # Defensive: a non-int value collapses to 0 rather than raising.
    assert v1q._watermark_for_shard({"consolidated_lsn": None}) == 0


# --- merge: dedup recall-wins ---------------------------------------------


def test_merge_recall_wins_on_dedup():
    """A recall LIVE row overrides a cold match for the same id (recall is newer)."""
    recall_rows = [
        {"id": "x", "score": 5.0, "metadata": {"v": "new"}, "deleted": False},
    ]
    cold = [
        {"id": "x", "score": 1.0, "metadata": {"v": "old"}},  # closer, but stale
        {"id": "y", "score": 2.0, "metadata": {}},
    ]
    merged = v1q._merge_recall_and_cold(recall_rows, cold, top_k=10)
    by_id = {m["id"]: m for m in merged}
    # `x` is the RECALL version (metadata "new", score 5.0) — NOT the cold one,
    # even though the cold one ranked closer.
    assert by_id["x"]["metadata"] == {"v": "new"}
    assert by_id["x"]["score"] == 5.0
    # `y` (cold-only) survives.
    assert by_id["y"]["metadata"] == {}
    # Sorted ascending by score: y (2.0) before x (5.0).
    assert [m["id"] for m in merged] == ["y", "x"]


def test_merge_recall_only_rows_when_no_cold():
    """Recall rows alone form the result when there is no cold shard."""
    recall_rows = [
        {"id": "a", "score": 3.0, "metadata": {}, "deleted": False},
        {"id": "b", "score": 1.0, "metadata": {}, "deleted": False},
    ]
    merged = v1q._merge_recall_and_cold(recall_rows, [], top_k=10)
    assert [m["id"] for m in merged] == ["b", "a"]


# --- merge: tombstone suppression -----------------------------------------


def test_merge_recall_tombstone_suppresses_cold_id():
    """A recall tombstone drops the cold match for that id and yields no match."""
    recall_rows = [
        {"id": "gone", "score": 0.0, "metadata": {}, "deleted": True},
    ]
    cold = [
        {"id": "gone", "score": 0.5, "metadata": {"stale": True}},
        {"id": "keep", "score": 1.0, "metadata": {}},
    ]
    merged = v1q._merge_recall_and_cold(recall_rows, cold, top_k=10)
    ids = [m["id"] for m in merged]
    assert "gone" not in ids, "tombstone must suppress the cold id"
    assert ids == ["keep"], merged


def test_merge_tombstone_and_live_mix():
    """A tombstone suppresses while a live recall row for another id wins."""
    recall_rows = [
        {"id": "del", "score": 0.0, "metadata": {}, "deleted": True},
        {"id": "upd", "score": 4.0, "metadata": {"v": 2}, "deleted": False},
    ]
    cold = [
        {"id": "del", "score": 0.1, "metadata": {}},   # suppressed
        {"id": "upd", "score": 0.2, "metadata": {"v": 1}},  # overridden
        {"id": "cold_only", "score": 3.0, "metadata": {}},
    ]
    merged = v1q._merge_recall_and_cold(recall_rows, cold, top_k=10)
    by_id = {m["id"]: m for m in merged}
    assert "del" not in by_id
    assert by_id["upd"]["metadata"] == {"v": 2} and by_id["upd"]["score"] == 4.0
    assert by_id["cold_only"]["score"] == 3.0
    assert [m["id"] for m in merged] == ["cold_only", "upd"]


def test_merge_truncates_to_top_k():
    """The union is sorted ascending and truncated to top_k."""
    recall_rows = [{"id": "r", "score": 2.5, "metadata": {}, "deleted": False}]
    cold = [
        {"id": "c1", "score": 1.0, "metadata": {}},
        {"id": "c2", "score": 2.0, "metadata": {}},
        {"id": "c3", "score": 3.0, "metadata": {}},
    ]
    merged = v1q._merge_recall_and_cold(recall_rows, cold, top_k=2)
    assert [m["id"] for m in merged] == ["c1", "c2"]


# --- run_query union orchestration (faked recall + cold) ------------------


@pytest.fixture
def union_on(monkeypatch):
    """Turn the union on at the `recall_enabled()` gate in `v1_query`."""
    monkeypatch.setattr(v1q, "recall_enabled", lambda: True)


def _parsed(top_k=10, flt=None):
    return v1q._ParsedQuery("ds", [0.0, 0.0, 0.0, 0.0], top_k, None, flt or {})


def test_run_query_union_merges_recall_and_cold(union_on, monkeypatch):
    """Flag on: `run_query` unions the cold matches with the recall rows.

    The cold search resolves a shard (watermark 11) and returns one match; the
    recall scan (faked) returns a live override + a fresh id. The response merges
    them with recall-wins and the cold `mode`.
    """
    def _fake_hot(tenant, dataset, vec, top_k, flt, nprobe, resolved):
        resolved["shard"] = {"id": 1, "consolidated_lsn": 11}
        return ([{"id": "cold", "score": 9.0, "metadata": {"c": 1}}], "hot")

    captured = {}

    def _fake_recall(tenant, dataset, vec, top_k, watermark, flt):
        captured["watermark"] = watermark
        return [
            {"id": "cold", "score": 2.0, "metadata": {"c": 2}, "deleted": False},
            {"id": "fresh", "score": 1.0, "metadata": {}, "deleted": False},
        ]

    monkeypatch.setattr(v1q, "_hot_search", _fake_hot)
    monkeypatch.setattr(v1q, "recall_search", _fake_recall)

    out = v1q.run_query("t1", _parsed())
    assert captured["watermark"] == 11, "recall must be scoped to the shard watermark"
    assert out["mode"] == "hot"
    by_id = {m["id"]: m for m in out["matches"]}
    # recall-wins on `cold`: the recall version (metadata c=2, score 2.0).
    assert by_id["cold"]["metadata"] == {"c": 2}
    assert [m["id"] for m in out["matches"]] == ["fresh", "cold"]


def test_run_query_union_no_shard_returns_recall_synchronously(union_on, monkeypatch):
    """No cold shard + recall has data → SYNCHRONOUS recall result, NOT ephemeral.

    `mode` is `recall` (the cold cache contributed nothing) and there is no
    `job_id` — the read-your-writes property.
    """
    def _fake_hot(tenant, dataset, vec, top_k, flt, nprobe, resolved):
        return None  # no shard yet

    def _fake_recall(tenant, dataset, vec, top_k, watermark, flt):
        assert watermark == 0, "no shard → watermark 0 → all recall rows qualify"
        return [{"id": "just-written", "score": 0.0, "metadata": {}, "deleted": False}]

    monkeypatch.setattr(v1q, "_hot_search", _fake_hot)
    monkeypatch.setattr(v1q, "recall_search", _fake_recall)

    out = v1q.run_query("t1", _parsed())
    assert out["mode"] == "recall"
    assert "job_id" not in out
    assert [m["id"] for m in out["matches"]] == ["just-written"]


def test_run_query_union_no_shard_no_recall_falls_back_to_ephemeral(union_on, monkeypatch):
    """No cold shard AND no recall row → the ephemeral enqueue path (unchanged)."""
    monkeypatch.setattr(
        v1q, "_hot_search",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(v1q, "recall_search", lambda *a, **k: [])
    published = {}
    monkeypatch.setattr(v1q, "publish", lambda topic, msg: published.update({"topic": topic, "msg": msg}))

    out = v1q.run_query("t1", _parsed())
    assert out["mode"] == "ephemeral"
    assert out["matches"] == []
    assert out["job_id"].startswith("job_")
    assert published["topic"] == "RUN_EPHEMERAL_QUERY"


def test_run_query_union_recall_failure_maps_to_503(union_on, monkeypatch):
    """A recall-store failure surfaces as the v1 503 envelope, never a 500/raw."""
    from fastapi.responses import JSONResponse

    def _fake_hot(tenant, dataset, vec, top_k, flt, nprobe, resolved):
        resolved["shard"] = {"id": 1, "consolidated_lsn": 0}
        return ([], "cold")

    def _boom(*a, **k):
        raise OSError("recall store down")

    monkeypatch.setattr(v1q, "_hot_search", _fake_hot)
    monkeypatch.setattr(v1q, "recall_search", _boom)

    out = v1q.run_query("t1", _parsed())
    assert isinstance(out, JSONResponse)
    assert out.status_code == 503


# --- flag OFF: byte-identical, no recall connection -----------------------


def test_run_query_flag_off_is_cold_only_no_recall_call(monkeypatch):
    """Flag off (default): the cold result is returned verbatim and
    `recall_search` is NEVER called."""
    monkeypatch.setattr(v1q, "recall_enabled", lambda: False)

    def _fake_hot(tenant, dataset, vec, top_k, flt, nprobe, resolved):
        # When off, `resolved` is None — the caller never asks for the shard.
        assert resolved is None, "flag-off must not request the I3 shard pairing"
        return ([{"id": "c", "score": 1.0, "metadata": {}}], "hot")

    def _must_not_call(*a, **k):  # pragma: no cover
        raise AssertionError("recall_search called with the flag OFF")

    monkeypatch.setattr(v1q, "_hot_search", _fake_hot)
    monkeypatch.setattr(v1q, "recall_search", _must_not_call)

    out = v1q.run_query("t1", _parsed())
    assert out["mode"] == "hot"
    assert out["matches"] == [{"id": "c", "score": 1.0, "metadata": {}}]


def test_recall_search_opens_no_connection_when_flag_off(monkeypatch):
    """With the tier off, no recall path runs — proven by making
    `psycopg2.connect` raise: a flag-off query must never reach it.

    Mirrors the write-path PR's `psycopg2.connect`-raises trick: if any recall
    code opened a connection, this query would blow up. It must not.
    """
    import adapters.state.state as state_mod

    monkeypatch.delenv("RB_RECALL", raising=False)
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    importlib.reload(state_mod)
    # Reload v1_query so it re-binds `recall_enabled`/`recall_search` to the
    # reloaded state module.
    importlib.reload(v1q)

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("opened a connection with the recall tier OFF")

    monkeypatch.setattr(state_mod.psycopg2, "connect", _boom)

    def _fake_hot(tenant, dataset, vec, top_k, flt, nprobe, resolved=None):
        return ([{"id": "c", "score": 1.0, "metadata": {}}], "cold")

    monkeypatch.setattr(v1q, "_hot_search", _fake_hot)

    out = v1q.run_query("t1", _parsed())
    assert out["mode"] == "cold"
    # Restore a clean module for the rest of the suite.
    importlib.reload(state_mod)
    importlib.reload(v1q)


# --- recall_search row shaping (state adapter) ----------------------------


class _FakeRecallCursor:
    """A cursor stand-in that returns canned `recall_vectors` rows."""

    def __init__(self, rows):
        self._rows = rows
        self.executed = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed = (sql, params)

    def fetchall(self):
        return self._rows


class _FakeRecallConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass


def test_recall_search_applies_filter_to_live_rows_keeps_tombstones(monkeypatch):
    """The AND-of-equals filter is applied to LIVE rows; tombstones pass through.

    A live row failing the filter is dropped; a live row matching it is kept; a
    tombstone is returned regardless (it must still suppress a cold id).
    """
    import adapters.state.state as state_mod

    rows = [
        # (id, score, metadata, deleted) — ordered ascending by distance.
        ("match", 1.0, {"lang": "en"}, False),
        ("nomatch", 2.0, {"lang": "fr"}, False),
        ("tomb", 3.0, {"lang": "fr"}, True),
    ]
    cur = _FakeRecallCursor(rows)
    monkeypatch.setattr(state_mod, "_recall_conn", lambda: _FakeRecallConn(cur))

    out = state_mod.recall_search(
        "t1", "ds", [0.0, 0.0], top_k=10, watermark=5, flt={"lang": "en"}
    )
    by_id = {r["id"]: r for r in out}
    assert "match" in by_id and by_id["match"]["deleted"] is False
    assert "nomatch" not in by_id, "live row failing the filter must be dropped"
    assert "tomb" in by_id and by_id["tomb"]["deleted"] is True, (
        "a tombstone must be returned regardless of the filter so it can suppress"
    )
    # The watermark + partition are bound into the SQL.
    sql, params = cur.executed
    assert "lsn > %s" in sql
    assert params == ("[0.0,0.0]", "t1", "ds", 5, "[0.0,0.0]")


def test_recall_search_scoped_to_tenant_dataset_and_watermark(monkeypatch):
    """The scan is scoped to (tenant, dataset) and `lsn > watermark` (I1/I3)."""
    import adapters.state.state as state_mod

    cur = _FakeRecallCursor([])
    monkeypatch.setattr(state_mod, "_recall_conn", lambda: _FakeRecallConn(cur))
    state_mod.recall_search("tenantA", "datasetB", [1.0], top_k=3, watermark=99)
    sql, params = cur.executed
    assert "tenant_id = %s AND dataset = %s AND lsn > %s" in sql
    assert params[1] == "tenantA" and params[2] == "datasetB" and params[3] == 99


def test_recall_search_squares_pgvector_distance(monkeypatch):
    """The score expression squares pgvector `<->` to align with FAISS L2²."""
    import adapters.state.state as state_mod

    cur = _FakeRecallCursor([])
    monkeypatch.setattr(state_mod, "_recall_conn", lambda: _FakeRecallConn(cur))
    state_mod.recall_search("t", "d", [1.0], top_k=1, watermark=0)
    sql, _ = cur.executed
    assert "power(embedding <-> %s, 2)" in sql, "recall score must be L2-SQUARED"


def test_recall_search_truncates_live_rows_to_top_k(monkeypatch):
    """At most `top_k` LIVE rows are returned (the closest, by distance order)."""
    import adapters.state.state as state_mod

    rows = [
        ("a", 1.0, {}, False),
        ("b", 2.0, {}, False),
        ("c", 3.0, {}, False),  # beyond top_k=2 → dropped
    ]
    cur = _FakeRecallCursor(rows)
    monkeypatch.setattr(state_mod, "_recall_conn", lambda: _FakeRecallConn(cur))
    out = state_mod.recall_search("t", "d", [0.0], top_k=2, watermark=0)
    assert [r["id"] for r in out] == ["a", "b"]


def test_recall_search_returns_tombstone_beyond_top_k_live_rows(monkeypatch):
    """A tombstone further out than `top_k` live rows is STILL returned.

    Correctness: a tombstone's recall embedding can differ from the cold shard's
    consolidated embedding for the same id, so we cannot assume the cold match
    ranks beyond `top_k` just because the tombstone does. The scan must not stop
    after `top_k` live rows — it keeps collecting tombstones for suppression.
    """
    import adapters.state.state as state_mod

    rows = [
        ("live1", 1.0, {}, False),
        ("live2", 2.0, {}, False),
        ("live3", 3.0, {}, False),   # beyond top_k=2 → dropped (not a match)
        ("tomb", 9.0, {}, True),     # far out, but MUST be returned to suppress
    ]
    cur = _FakeRecallCursor(rows)
    monkeypatch.setattr(state_mod, "_recall_conn", lambda: _FakeRecallConn(cur))
    out = state_mod.recall_search("t", "d", [0.0], top_k=2, watermark=0)
    by_id = {r["id"]: r for r in out}
    assert set(by_id) == {"live1", "live2", "tomb"}, out
    assert by_id["tomb"]["deleted"] is True
    assert "live3" not in by_id, "live rows past top_k are not returned"
