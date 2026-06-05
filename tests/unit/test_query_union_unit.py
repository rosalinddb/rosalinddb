"""Unit coverage for the Recall + Consolidated query union (PR4).

Hermetic — no Docker, no pgvector, no network. Exercises the merge logic and the
`run_query` union orchestration directly, with the recall connection / consolidated
search faked. The integration suite (`tests/integration/test_query_union.py`)
proves the same properties end-to-end against a real pgvector container.

Headline properties proven here:

  - **Metric alignment** (correctness-critical): pgvector's plain-L2 `<->`
    SQUARED ranks identically to FAISS L2² over the same un-normalised vectors.
    The squaring is monotonic, so this is really a "we square it, and squaring
    preserves order" proof on a deterministic example — the most likely silent
    ranking bug, per the spec.
  - **Dedup recall-wins**: a recall LIVE row overrides a consolidated match for the same
    id.
  - **Authoritative suppression**: recall suppresses the stale consolidated copy of EVERY
    id above the watermark — tombstoned, filter-failing, OR ranked-past-`top_k` —
    keying on "any recall row for this id", not "a recall row that became a
    match". A live re-upsert that fails the query filter must not let a stale,
    filter-matching consolidated copy leak.
  - **Filter gates matches, not suppression**: the AND-of-equals predicate
    decides whether a live recall row is a MATCH; it never removes an id from the
    suppression set.
  - **Watermark partition (I1/I3)**: the recall scan filters `lsn > watermark`,
    and the watermark is the resolved shard's `consolidated_lsn` (0 when no
    shard).
  - **Flag OFF (default)**: byte-identical to the consolidated-only path AND no recall
    connection is ever opened (the `psycopg2.connect`-raises trick).
"""
from __future__ import annotations

import contextlib
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
    that sorts by the recall score and the consolidated score interleaves correctly.
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
    """A recall LIVE row overrides a consolidated match for the same id (recall is newer)."""
    suppress = {"x"}
    recall_matches = [
        {"id": "x", "score": 5.0, "metadata": {"v": "new"}, "deleted": False},
    ]
    consolidated = [
        {"id": "x", "score": 1.0, "metadata": {"v": "old"}},  # closer, but stale
        {"id": "y", "score": 2.0, "metadata": {}},
    ]
    merged = v1q._merge_recall_and_consolidated(suppress, recall_matches, consolidated, top_k=10)
    by_id = {m["id"]: m for m in merged}
    # `x` is the RECALL version (metadata "new", score 5.0) — NOT the consolidated one,
    # even though the consolidated one ranked closer.
    assert by_id["x"]["metadata"] == {"v": "new"}
    assert by_id["x"]["score"] == 5.0
    # `y` (consolidated-only) survives.
    assert by_id["y"]["metadata"] == {}
    # Sorted ascending by score: y (2.0) before x (5.0).
    assert [m["id"] for m in merged] == ["y", "x"]


def test_merge_recall_only_rows_when_no_consolidated():
    """Recall rows alone form the result when there is no consolidated shard."""
    recall_matches = [
        {"id": "a", "score": 3.0, "metadata": {}, "deleted": False},
        {"id": "b", "score": 1.0, "metadata": {}, "deleted": False},
    ]
    merged = v1q._merge_recall_and_consolidated({"a", "b"}, recall_matches, [], top_k=10)
    assert [m["id"] for m in merged] == ["b", "a"]


# --- merge: tombstone / authoritative suppression -------------------------


def test_merge_recall_tombstone_suppresses_consolidated_id():
    """A recall tombstone drops the consolidated match for that id and yields no match.

    A tombstone is in `suppress_ids` but NOT in `recall_matches`.
    """
    consolidated = [
        {"id": "gone", "score": 0.5, "metadata": {"stale": True}},
        {"id": "keep", "score": 1.0, "metadata": {}},
    ]
    merged = v1q._merge_recall_and_consolidated({"gone"}, [], consolidated, top_k=10)
    ids = [m["id"] for m in merged]
    assert "gone" not in ids, "tombstone must suppress the consolidated id"
    assert ids == ["keep"], merged


def test_merge_filtered_out_live_recall_suppresses_consolidated_id():
    """A live recall row that FAILS the filter still suppresses its stale consolidated id.

    The P1 fix: the live row is in `suppress_ids` (every recall id above the
    watermark) but NOT in `recall_matches` (it failed the filter), so the stale
    consolidated copy whose older metadata DOES match the filter must not leak.
    """
    consolidated = [
        {"id": "x", "score": 0.1, "metadata": {"color": "red"}},  # stale, matches
        {"id": "y", "score": 1.0, "metadata": {}},
    ]
    # `x` was re-upserted into recall with {color: blue} (fails {color: red}) →
    # it is in suppress_ids but contributes no match.
    merged = v1q._merge_recall_and_consolidated({"x"}, [], consolidated, top_k=10)
    ids = [m["id"] for m in merged]
    assert "x" not in ids, "a filter-failing live recall row must suppress its consolidated copy"
    assert ids == ["y"], merged


def test_merge_suppresses_consolidated_for_recall_id_past_top_k():
    """A recall id whose live row ranked past top_k still suppresses the consolidated copy.

    `suppress_ids` carries the id even though no corresponding row is in
    `recall_matches` (it was capped out), so the stale consolidated copy is dropped.
    """
    consolidated = [{"id": "x", "score": 0.1, "metadata": {"stale": True}}]
    merged = v1q._merge_recall_and_consolidated({"x"}, [], consolidated, top_k=10)
    assert [m["id"] for m in merged] == [], merged


def test_merge_tombstone_and_live_mix():
    """A tombstone suppresses while a live recall row for another id wins."""
    suppress = {"del", "upd"}
    recall_matches = [
        {"id": "upd", "score": 4.0, "metadata": {"v": 2}, "deleted": False},
    ]
    consolidated = [
        {"id": "del", "score": 0.1, "metadata": {}},   # suppressed (tombstone)
        {"id": "upd", "score": 0.2, "metadata": {"v": 1}},  # overridden (live)
        {"id": "consolidated_only", "score": 3.0, "metadata": {}},
    ]
    merged = v1q._merge_recall_and_consolidated(suppress, recall_matches, consolidated, top_k=10)
    by_id = {m["id"]: m for m in merged}
    assert "del" not in by_id
    assert by_id["upd"]["metadata"] == {"v": 2} and by_id["upd"]["score"] == 4.0
    assert by_id["consolidated_only"]["score"] == 3.0
    assert [m["id"] for m in merged] == ["consolidated_only", "upd"]


def test_merge_truncates_to_top_k():
    """The union is sorted ascending and truncated to top_k."""
    recall_matches = [{"id": "r", "score": 2.5, "metadata": {}, "deleted": False}]
    consolidated = [
        {"id": "c1", "score": 1.0, "metadata": {}},
        {"id": "c2", "score": 2.0, "metadata": {}},
        {"id": "c3", "score": 3.0, "metadata": {}},
    ]
    merged = v1q._merge_recall_and_consolidated({"r"}, recall_matches, consolidated, top_k=2)
    assert [m["id"] for m in merged] == ["c1", "c2"]


# --- run_query union orchestration (faked recall + consolidated) ----------


@pytest.fixture
def union_on(monkeypatch):
    """Turn the union on at the `recall_enabled()` gate in `v1_query`."""
    monkeypatch.setattr(v1q, "recall_enabled", lambda: True)


def _parsed(top_k=10, flt=None):
    return v1q._ParsedQuery("ds", [0.0, 0.0, 0.0, 0.0], top_k, None, flt or {})


def test_run_query_union_merges_recall_and_consolidated(union_on, monkeypatch):
    """Flag on: `run_query` unions the consolidated matches with the recall rows.

    The consolidated search resolves a shard (watermark 11) and returns one match; the
    recall scan (faked) returns a live override + a fresh id. The response merges
    them with recall-wins and the consolidated shard's cache-state `mode`.
    """
    def _fake_hot(tenant, dataset, vec, top_k, flt, nprobe, resolved):
        resolved["shard"] = {"id": 1, "consolidated_lsn": 11}
        return ([{"id": "shared", "score": 9.0, "metadata": {"c": 1}}], "hot")

    captured = {}

    def _fake_recall(tenant, dataset, vec, top_k, watermark, flt):
        captured["watermark"] = watermark
        return (
            {"shared", "fresh"},
            [
                {"id": "shared", "score": 2.0, "metadata": {"c": 2}, "deleted": False},
                {"id": "fresh", "score": 1.0, "metadata": {}, "deleted": False},
            ],
        )

    monkeypatch.setattr(v1q, "_hot_search", _fake_hot)
    monkeypatch.setattr(v1q, "recall_search", _fake_recall)

    out = v1q.run_query("t1", _parsed())
    assert captured["watermark"] == 11, "recall must be scoped to the shard watermark"
    assert out["mode"] == "hot"
    by_id = {m["id"]: m for m in out["matches"]}
    # recall-wins on `shared`: the recall version (metadata c=2, score 2.0).
    assert by_id["shared"]["metadata"] == {"c": 2}
    assert [m["id"] for m in out["matches"]] == ["fresh", "shared"]


def test_run_query_union_filter_failing_live_recall_suppresses_stale_consolidated(
    union_on, monkeypatch
):
    """P1 regression: a live re-upsert that fails the filter hides its stale consolidated copy.

    consolidated X `{color: red}` (passes filter `{color: red}`); recall live X
    `{color: blue}` (fails the filter), `lsn > watermark`. A query `{color: red}`
    must NOT return X — recall is authoritative and X's current version no longer
    matches the filter, so the stale, filter-matching consolidated copy must not leak.
    """
    def _fake_hot(tenant, dataset, vec, top_k, flt, nprobe, resolved):
        resolved["shard"] = {"id": 1, "consolidated_lsn": 0}
        # Consolidated shard still holds the OLD X (color=red), which DOES pass the filter.
        return (
            [
                {"id": "X", "score": 0.1, "metadata": {"color": "red"}},
                {"id": "Y", "score": 0.5, "metadata": {"color": "red"}},
            ],
            "cold",
        )

    def _fake_recall(tenant, dataset, vec, top_k, watermark, flt):
        # X re-upserted as color=blue → fails {color: red} → NOT a match, but it
        # IS above the watermark so it suppresses the stale consolidated X.
        return ({"X"}, [])

    monkeypatch.setattr(v1q, "_hot_search", _fake_hot)
    monkeypatch.setattr(v1q, "recall_search", _fake_recall)

    out = v1q.run_query("t1", _parsed(flt={"color": "red"}))
    ids = [m["id"] for m in out["matches"]]
    assert "X" not in ids, "stale consolidated X must be suppressed by the live recall re-upsert"
    assert ids == ["Y"], out


def test_run_query_union_live_recall_past_top_k_suppresses_stale_consolidated(
    union_on, monkeypatch
):
    """A filter-passing live recall row ranked past `top_k` still hides stale consolidated X.

    The recall row for X is above the watermark (so it suppresses) but ranked past
    `top_k` within recall, so it is not in the returned matches. The stale consolidated X
    must not surface in its place.
    """
    def _fake_hot(tenant, dataset, vec, top_k, flt, nprobe, resolved):
        resolved["shard"] = {"id": 1, "consolidated_lsn": 0}
        # Stale consolidated X ranks well within the final top_k by consolidated distance.
        return ([{"id": "X", "score": 0.05, "metadata": {}}], "cold")

    def _fake_recall(tenant, dataset, vec, top_k, watermark, flt):
        # X is above the watermark (suppresses) but its live row ranked past
        # top_k within recall, so it contributed no match.
        return ({"X"}, [])

    monkeypatch.setattr(v1q, "_hot_search", _fake_hot)
    monkeypatch.setattr(v1q, "recall_search", _fake_recall)

    out = v1q.run_query("t1", _parsed(top_k=1))
    assert [m["id"] for m in out["matches"]] == [], out


def test_run_query_union_no_shard_returns_recall_synchronously(union_on, monkeypatch):
    """No consolidated shard + recall has data → SYNCHRONOUS recall result, NOT ephemeral.

    `mode` is `recall` (the consolidated tier contributed nothing) and there is no
    `job_id` — the read-your-writes property.
    """
    def _fake_hot(tenant, dataset, vec, top_k, flt, nprobe, resolved):
        return None  # no shard yet

    def _fake_recall(tenant, dataset, vec, top_k, watermark, flt):
        assert watermark == 0, "no shard → watermark 0 → all recall rows qualify"
        return (
            {"just-written"},
            [{"id": "just-written", "score": 0.0, "metadata": {}, "deleted": False}],
        )

    monkeypatch.setattr(v1q, "_hot_search", _fake_hot)
    monkeypatch.setattr(v1q, "recall_search", _fake_recall)

    out = v1q.run_query("t1", _parsed())
    assert out["mode"] == "recall"
    assert "job_id" not in out
    assert [m["id"] for m in out["matches"]] == ["just-written"]


def test_run_query_union_no_shard_no_recall_falls_back_to_ephemeral(union_on, monkeypatch):
    """No consolidated shard AND no recall row → the ephemeral enqueue path (unchanged)."""
    monkeypatch.setattr(
        v1q, "_hot_search",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(v1q, "recall_search", lambda *a, **k: (set(), []))
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


def _envelope_code(resp):
    """Pull the v1 error `code` out of a `JSONResponse` body for assertions."""
    import json

    return json.loads(bytes(resp.body))["error"]["code"]


def test_run_query_union_recall_unavailable_maps_to_503_recall_unavailable(
    union_on, monkeypatch
):
    """C2 (#19): a recall-tier connectivity failure -> 503 `recall_unavailable`.

    The recall search raises the typed `RecallUnavailable` (which `recall_search`
    produces from a psycopg2 OperationalError / sustained pool exhaustion when the
    recall pgvector instance is down). The query path must classify it as a
    DISTINCT, retryable 503 `recall_unavailable` — NOT the opaque
    `ephemeral_error` 500 it used to hard-500 as, and NOT a silent
    consolidated-only 200 (which would drop recent unconsolidated writes without
    signal, breaking read-your-writes).
    """
    from fastapi.responses import JSONResponse

    # Use v1_query's OWN reference to the class — that is exactly what its
    # classifier compares against with `isinstance`. (A sibling test that
    # `importlib.reload`s `adapters.state.state` would otherwise rebind the class
    # object and break identity; in production both come from the one loaded
    # module, so they are the same object.)
    RecallUnavailable = v1q.RecallUnavailable

    def _fake_hot(tenant, dataset, vec, top_k, flt, nprobe, resolved):
        # A healthy consolidated shard WITH a stale match for X — if the code silently
        # degraded to consolidated-only, this would leak as a 200 instead of a 503.
        resolved["shard"] = {"id": 1, "consolidated_lsn": 0}
        return ([{"id": "stale", "score": 0.1, "metadata": {}}], "cold")

    def _recall_down(*a, **k):
        raise RecallUnavailable("recall store unreachable: OperationalError")

    monkeypatch.setattr(v1q, "_hot_search", _fake_hot)
    monkeypatch.setattr(v1q, "recall_search", _recall_down)

    out = v1q.run_query("t1", _parsed())
    assert isinstance(out, JSONResponse), (
        "recall down must NOT silently degrade to a consolidated-only 200"
    )
    assert out.status_code == 503
    assert _envelope_code(out) == "recall_unavailable"


def test_run_query_recall_down_does_not_silently_serve_consolidated(union_on, monkeypatch):
    """No silent degrade: a recall outage never returns the stale consolidated matches.

    The consolidated path resolves a shard and returns a match; recall is down. The
    response must be the 503 error envelope, never a `{matches: [...]}` body
    carrying the stale consolidated copy — the whole point of the recall tier is
    read-your-writes, and a silent consolidated-only answer would lie about it.
    """
    from fastapi.responses import JSONResponse

    RecallUnavailable = v1q.RecallUnavailable

    def _fake_hot(tenant, dataset, vec, top_k, flt, nprobe, resolved):
        resolved["shard"] = {"id": 1, "consolidated_lsn": 0}
        return ([{"id": "stale-consolidated", "score": 0.01, "metadata": {}}], "hot")

    monkeypatch.setattr(v1q, "_hot_search", _fake_hot)
    monkeypatch.setattr(
        v1q, "recall_search",
        lambda *a, **k: (_ for _ in ()).throw(RecallUnavailable("down")),
    )

    out = v1q.run_query("t1", _parsed())
    assert isinstance(out, JSONResponse)
    assert out.status_code == 503
    # No `matches` body — the stale consolidated copy never reaches the client.
    assert b"stale-consolidated" not in bytes(out.body)


def test_run_query_non_recall_operationalerror_is_not_recall_unavailable(
    union_on, monkeypatch
):
    """A non-recall OperationalError (e.g. consolidated/control path) is NOT misclassified.

    The consolidated search itself raises a bare `psycopg2.OperationalError` (a
    control-plane / catalog failure, NOT the recall tier). Because the recall
    wrapper is typed and only raised at the recall boundary, this unwrapped error
    must NOT be classified as `recall_unavailable` — it falls to the generic
    `ephemeral_error` 500, the pre-existing behaviour for an unexpected error on
    the consolidated path.
    """
    import psycopg2
    from fastapi.responses import JSONResponse

    def _consolidated_down(tenant, dataset, vec, top_k, flt, nprobe, resolved):
        raise psycopg2.OperationalError("control-plane catalog unreachable")

    monkeypatch.setattr(v1q, "_hot_search", _consolidated_down)
    # recall_search must never be reached — the consolidated search failed first.
    monkeypatch.setattr(
        v1q, "recall_search",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("unreached")),
    )

    out = v1q.run_query("t1", _parsed())
    assert isinstance(out, JSONResponse)
    assert _envelope_code(out) != "recall_unavailable"
    assert out.status_code == 500
    assert _envelope_code(out) == "ephemeral_error"


# --- flag OFF: byte-identical, no recall connection -----------------------


def test_run_query_flag_off_is_consolidated_only_no_recall_call(monkeypatch):
    """Flag off (default): the consolidated result is returned verbatim and
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
    """A cursor stand-in that returns canned `recall_vectors` rows.

    `recall_search` now issues ONE single-snapshot scan
    (`SELECT id, deleted, power(embedding <-> %s, 2) AS score, metadata`) over
    `lsn > watermark`, and splits suppress/match in Python (task #17). The fake
    projects the scripted `(id, score, metadata, deleted)` rows to the columns the
    single statement selects — `(id, deleted, score, metadata)` for EVERY row —
    and records that one execute in `executed` so the SQL/param assertions inspect
    it. `executes` counts every execute so a test can prove the scan is a SINGLE
    statement (the structural fix for the b1 over-suppression race).
    """

    def __init__(self, rows):
        self._rows = rows
        self.executed = None
        self.executes = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executes.append((sql, params))
        self.executed = (sql, params)

    def fetchall(self):
        # The single-snapshot scan selects (id, deleted, score, metadata) for
        # EVERY row above the watermark (live AND tombstoned).
        return [
            (rid, deleted, score, meta)
            for rid, score, meta, deleted in self._rows
        ]


class _FakeRecallConn:
    """Connection stub for the POOLED recall search path.

    `recall_pooled_conn()` owns the transaction (commit/rollback) and returns the
    conn to the pool, so the connection is no longer used as a `with` context and
    is never closed — it just needs `cursor`, `commit`, `rollback`.
    """

    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        pass


class _FakeRecallPool:
    """A `ThreadedConnectionPool`-shaped stub returning one fake conn.

    Mirrors `tests/unit/test_state_pool.py`'s `_FakePool`. Non-empty `_pool`
    keeps the `reused` probe in `recall_pooled_conn()` happy.
    """

    def __init__(self, conn):
        self._conn = conn
        self._pool = [object()]

    def getconn(self):
        return self._conn

    def putconn(self, conn):
        pass


def _wire_recall(state_mod, monkeypatch, cur):
    """Wire a fake recall pool so `recall_pooled_conn()` yields `cur`'s conn.

    The pooled replacement for `monkeypatch.setattr(state_mod, "_recall_conn",
    ...)`: bind a fake recall pool (`_RECALL_POOL`) keyed to the live DSN, exactly
    as the control-plane pool tests wire `_POOL`. A `RB_RECALL_DSN` is set so the
    pool getter's DSN-key check resolves (and matches `_RECALL_POOL_DSN`) — the
    fake pool is used, no real connection is opened.
    """
    monkeypatch.setenv("RB_RECALL_DSN", "postgresql://u:p@recall:5432/recall")
    pool = _FakeRecallPool(_FakeRecallConn(cur))
    monkeypatch.setattr(state_mod, "_RECALL_POOL", pool)
    monkeypatch.setattr(state_mod, "_RECALL_POOL_DSN", state_mod._recall_dsn())


def test_recall_search_filter_gates_matches_not_suppression(monkeypatch):
    """The filter decides MATCHES; every recall id above the watermark suppresses.

    A live row matching the filter is a MATCH; a live row FAILING the filter is
    NOT a match but is STILL in `suppress_ids`; a tombstone is in `suppress_ids`
    but never a match. This is the P1 fix: suppression keys on "any recall row
    for this id", not "a recall row that became a match".
    """
    import adapters.state.state as state_mod

    rows = [
        # (id, score, metadata, deleted) — ordered ascending by distance.
        ("match", 1.0, {"lang": "en"}, False),
        ("nomatch", 2.0, {"lang": "fr"}, False),
        ("tomb", 3.0, {"lang": "fr"}, True),
    ]
    cur = _FakeRecallCursor(rows)
    _wire_recall(state_mod, monkeypatch, cur)

    suppress_ids, matches = state_mod.recall_search(
        "t1", "ds", [0.0, 0.0], top_k=10, watermark=5, flt={"lang": "en"}
    )
    by_id = {r["id"]: r for r in matches}
    # MATCHES: only the filter-passing live row.
    assert "match" in by_id and by_id["match"]["deleted"] is False
    assert "nomatch" not in by_id, "a live row failing the filter is not a match"
    assert "tomb" not in by_id, "a tombstone is never a match"
    # SUPPRESSION: EVERY id above the watermark, regardless of filter/deleted.
    assert suppress_ids == {"match", "nomatch", "tomb"}, (
        "a filtered-out live row AND a tombstone must both suppress their consolidated id"
    )
    # The watermark + partition are bound into the SQL.
    sql, params = cur.executed
    assert "lsn > %s" in sql
    assert params == ("[0.0,0.0]", "t1", "ds", 5, "[0.0,0.0]")


def test_recall_search_scoped_to_tenant_dataset_and_watermark(monkeypatch):
    """The scan is scoped to (tenant, dataset) and `lsn > watermark` (I1/I3)."""
    import adapters.state.state as state_mod

    cur = _FakeRecallCursor([])
    _wire_recall(state_mod, monkeypatch, cur)
    state_mod.recall_search("tenantA", "datasetB", [1.0], top_k=3, watermark=99)
    sql, params = cur.executed
    # The single-snapshot scan is partition + watermark scoped, and selects
    # `deleted` so tombstones are split out in Python (no `NOT deleted` clause).
    assert "tenant_id = %s AND dataset = %s AND lsn > %s" in sql
    assert "deleted" in sql, "scan must SELECT deleted to split tombstones in Python"
    assert "NOT deleted" not in sql, "the single-snapshot scan returns ALL rows"
    assert params[1] == "tenantA" and params[2] == "datasetB" and params[3] == 99
    # Exactly ONE execute drives the scan — the structural b1 fix.
    assert len(cur.executes) == 1


def test_recall_search_squares_pgvector_distance(monkeypatch):
    """The score expression squares pgvector `<->` to align with FAISS L2²."""
    import adapters.state.state as state_mod

    cur = _FakeRecallCursor([])
    _wire_recall(state_mod, monkeypatch, cur)
    state_mod.recall_search("t", "d", [1.0], top_k=1, watermark=0)
    sql, _ = cur.executed
    assert "power(embedding <-> %s, 2)" in sql, "recall score must be L2-SQUARED"


def test_recall_search_truncates_live_matches_to_top_k(monkeypatch):
    """At most `top_k` LIVE rows are MATCHES (the closest, by distance order)."""
    import adapters.state.state as state_mod

    rows = [
        ("a", 1.0, {}, False),
        ("b", 2.0, {}, False),
        ("c", 3.0, {}, False),  # beyond top_k=2 → not a match (but suppresses)
    ]
    cur = _FakeRecallCursor(rows)
    _wire_recall(state_mod, monkeypatch, cur)
    suppress_ids, matches = state_mod.recall_search("t", "d", [0.0], top_k=2, watermark=0)
    assert [r["id"] for r in matches] == ["a", "b"]
    # `c` ranked past top_k → not a match, but STILL suppresses its stale consolidated id.
    assert suppress_ids == {"a", "b", "c"}


def test_recall_search_returns_match_and_suppresses_past_top_k_and_tombstones(monkeypatch):
    """Matches are capped at `top_k` live rows; suppression covers EVERYTHING.

    Correctness: a live row past `top_k` or a far-out tombstone has a recall
    embedding that can differ from the consolidated shard's consolidated embedding for the
    same id, so its stale consolidated copy could rank within the final `top_k` by consolidated
    distance — it must be suppressed. The scan never stops early; `suppress_ids`
    holds every id above the watermark.
    """
    import adapters.state.state as state_mod

    rows = [
        ("live1", 1.0, {}, False),
        ("live2", 2.0, {}, False),
        ("live3", 3.0, {}, False),   # beyond top_k=2 → not a match, but suppresses
        ("tomb", 9.0, {}, True),     # far out, never a match, but suppresses
    ]
    cur = _FakeRecallCursor(rows)
    _wire_recall(state_mod, monkeypatch, cur)
    suppress_ids, matches = state_mod.recall_search("t", "d", [0.0], top_k=2, watermark=0)
    assert [r["id"] for r in matches] == ["live1", "live2"], matches
    assert suppress_ids == {"live1", "live2", "live3", "tomb"}, suppress_ids


# --- recall_search connectivity boundary: typed RecallUnavailable (C2, #19) -


class _RaisingRecallCursor:
    """A recall cursor whose `execute` raises a scripted exception.

    Stands in for "the recall store is down": the brute-force scan's
    `cur.execute()` raises a psycopg2 connectivity error against a dropped
    backend. The fake is a context manager (the real cursor is used as one).
    """

    def __init__(self, exc):
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        raise self._exc

    def fetchall(self):  # pragma: no cover - never reached, execute() raises first
        return []


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: __import__("psycopg2").OperationalError("server closed the connection"),
        lambda: __import__("psycopg2").InterfaceError("connection already closed"),
    ],
)
def test_recall_search_wraps_connectivity_error_in_recall_unavailable(
    monkeypatch, exc_factory
):
    """A recall-store connectivity error -> typed `RecallUnavailable`.

    `recall_search` is the recall boundary: a psycopg2 OperationalError /
    InterfaceError from the scan (the recall pgvector instance is down) is wrapped
    in `RecallUnavailable`, with the original preserved as `__cause__`. This is the
    typed signal the query path classifies as a 503 `recall_unavailable` — instead
    of the bare psycopg2 error bubbling up unclassified to an `ephemeral_error`
    500 (benchmark finding C2).
    """
    import adapters.state.state as state_mod

    underlying = exc_factory()
    cur = _RaisingRecallCursor(underlying)
    _wire_recall(state_mod, monkeypatch, cur)

    with pytest.raises(state_mod.RecallUnavailable) as ei:
        state_mod.recall_search("t", "d", [0.0], top_k=3, watermark=0)
    # The original psycopg2 error is preserved for logs, never swallowed.
    assert ei.value.__cause__ is underlying


def test_recall_search_wraps_pool_checkout_timeout_in_recall_unavailable(
    monkeypatch,
):
    """A SUSTAINED recall-pool exhaustion -> typed `RecallUnavailable`.

    When the recall pool stays exhausted past the checkout deadline,
    `recall_pooled_conn()` raises `PoolCheckoutTimeout`. That is still "the recall
    tier could not serve this query", so `recall_search` wraps it in
    `RecallUnavailable` -> 503 `recall_unavailable`, not a 500.
    """
    import adapters.state.state as state_mod

    boom = state_mod.PoolCheckoutTimeout("recall pool exhausted")

    @contextlib.contextmanager
    def _raise_on_enter(*a, **k):
        raise boom
        yield  # pragma: no cover

    monkeypatch.setattr(state_mod, "recall_pooled_conn", _raise_on_enter)

    with pytest.raises(state_mod.RecallUnavailable) as ei:
        state_mod.recall_search("t", "d", [0.0], top_k=3, watermark=0)
    assert ei.value.__cause__ is boom


def test_recall_search_does_not_wrap_non_connectivity_error(monkeypatch):
    """A SQL/programming error against a HEALTHY backend is NOT wrapped.

    Only "the recall store is unreachable" becomes `RecallUnavailable`. A
    `psycopg2.ProgrammingError` (e.g. a query bug against a live backend) is a
    real server bug, not a transient outage — it must bubble up unchanged so it
    classifies as the generic `ephemeral_error` 500, never a misleading retryable
    503.
    """
    import psycopg2

    import adapters.state.state as state_mod

    underlying = psycopg2.ProgrammingError("syntax error at or near ...")
    cur = _RaisingRecallCursor(underlying)
    _wire_recall(state_mod, monkeypatch, cur)

    with pytest.raises(psycopg2.ProgrammingError):
        state_mod.recall_search("t", "d", [0.0], top_k=3, watermark=0)
