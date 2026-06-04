"""Unit coverage for the get/list/delete CRUD union with the Recall tier (PR6).

Hermetic — no Docker, no pgvector, no network. The recall connection / recall
helpers are FAKED (stubbed by name on the source_registry module, the same trick
the recall write-path suite uses), so these prove the HANDLER orchestration:

  - get union: recall-wins live; tombstone → 404; fall back to cold; cross-tenant
    404; flag-off cold-only AND no recall connection opened.
  - list union: recall-wins dedup, tombstone-suppress, filter-after-union, stable
    order, pagination; flag-off cold-only.
  - delete recall path: writes an ABOVE-watermark tombstone with a FRESH lsn
    (asserted via the real `recall_delete_vector` against a faked connection —
    the lsn the seq returns is captured and proven > the prior max / watermark);
    returns 204; does NOT publish DELETE_VECTORS; recall-store failure → 503;
    flag-off keeps the 202 + DELETE_VECTORS path; no recall connection when off.

The integration suite (`tests/integration/test_crud_union.py`) proves the same
properties end-to-end against a real pgvector container + MinIO cold shards.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest

os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def env(monkeypatch):
    """Fresh TestClient + helpers with reset in-memory state and storage.

    Mirrors `test_cold_vector_crud.py::env`: a faked cold shard via a sidecar
    written to the memory:// storage adapter. Both recall env vars default off;
    on-tests flip `recall_enabled` and stub the recall helpers on the handler
    module (they are imported by-name into `source_registry.main`).
    """
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.delenv("RB_RECALL", raising=False)
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)

    import adapters.storage.storage as storage_mod
    importlib.reload(storage_mod)
    storage_mod._MEM_OBJECTS.clear()

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_DATASETS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, list):
            obj.clear()
    state_mod._MEM_SHARDS.clear()
    state_mod._MEM_SHARD_ID = 0

    import services.auth.jwt_utils as jwt_utils
    importlib.reload(jwt_utils)
    import services.auth.auth as auth_mod
    importlib.reload(auth_mod)
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)

    from fastapi.testclient import TestClient

    from adapters.landing.parquet_reader import id_to_int64

    def write_sidecar(tenant, dataset, records, consolidated_lsn=0, shard_uri=None):
        """Seed a shard catalog row + its `.meta.json` for (tenant, dataset).

        `records` is an iterable of `(id, metadata)`. `consolidated_lsn` is
        stamped on the in-memory shard row so the union's watermark resolution
        can be exercised. Returns the shard_uri.
        """
        shard_uri = shard_uri or f"memory://rosalinddb/indexes/{tenant}/{dataset}/shard.bin"
        sidecar = {
            str(id_to_int64(rid)): {"id": rid, "metadata": meta}
            for rid, meta in records
        }
        from adapters.storage.storage import write_bytes

        write_bytes(f"{shard_uri}.meta.json", json.dumps(sidecar).encode("utf-8"))
        state_mod.add_shard(
            tenant, dataset, shard_uri,
            checksum="c", vector_count=len(sidecar), index_type="flat",
            indexed_landing_uris=[],
        )
        # Stamp the watermark on the freshly-added (newest) shard row.
        if consolidated_lsn:
            state_mod._MEM_SHARDS[-1]["consolidated_lsn"] = consolidated_lsn
        return shard_uri

    class _Env:
        client = TestClient(main_mod.app)
        state = state_mod
        main = main_mod
        write = staticmethod(write_sidecar)

    return _Env()


def _signup(client, email="alice@example.com", password="password123"):
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    body = r.json()
    body["tenant_id"] = body["tenant"]["id"]
    return body


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _make_dataset(env, token, name="ds", dim=4):
    r = env.client.post("/v1/datasets", headers=_auth(token), json={"name": name, "dimension": dim})
    assert r.status_code == 201, r.text


def _enable_recall(env, monkeypatch):
    """Flip the union on at the handler import site (flag stays default-off env)."""
    monkeypatch.setattr(env.main, "recall_enabled", lambda: True)


# --- GET union -------------------------------------------------------------


def test_get_recall_live_wins(env, monkeypatch):
    """A live recall row above the watermark wins over the cold sidecar copy."""
    _enable_recall(env, monkeypatch)
    monkeypatch.setattr(
        env.main,
        "recall_get_vector",
        lambda t, d, vid, wm: ("live", {"v": "recall"}),
    )
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    env.write(s["tenant_id"], "ds", [("doc-1", {"v": "cold"})])

    r = env.client.get("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "doc-1", "metadata": {"v": "recall"}}


def test_get_recall_tombstone_404(env, monkeypatch):
    """A recall tombstone above the watermark → 404 (never falls back to cold)."""
    _enable_recall(env, monkeypatch)
    monkeypatch.setattr(
        env.main,
        "recall_get_vector",
        lambda t, d, vid, wm: ("tombstone", None),
    )
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    # Cold copy still present, but the tombstone must hide it.
    env.write(s["tenant_id"], "ds", [("doc-1", {"v": "cold"})])

    r = env.client.get("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_get_falls_back_to_cold_when_no_recall_row(env, monkeypatch):
    """No recall row above the watermark → the cold sidecar answers."""
    _enable_recall(env, monkeypatch)
    monkeypatch.setattr(
        env.main, "recall_get_vector", lambda t, d, vid, wm: (None, None)
    )
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    env.write(s["tenant_id"], "ds", [("doc-1", {"v": "cold"})])

    r = env.client.get("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "doc-1", "metadata": {"v": "cold"}}


def test_get_recall_only_no_cold_shard(env, monkeypatch):
    """Recall-wins live row with NO cold shard (watermark 0) → recall answers."""
    _enable_recall(env, monkeypatch)
    monkeypatch.setattr(
        env.main,
        "recall_get_vector",
        lambda t, d, vid, wm: ("live", {"t": "peanuts"}),
    )
    s = _signup(env.client)
    _make_dataset(env, s["token"])  # no shard built

    r = env.client.get("/v1/datasets/ds/vectors/fact-1", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "fact-1", "metadata": {"t": "peanuts"}}


def test_get_union_passes_resolved_watermark(env, monkeypatch):
    """The recall lookup is filtered with the resolved shard's consolidated_lsn (I3)."""
    _enable_recall(env, monkeypatch)
    seen = {}

    def _rec(t, d, vid, wm):
        seen["wm"] = wm
        return (None, None)

    monkeypatch.setattr(env.main, "recall_get_vector", _rec)
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    env.write(s["tenant_id"], "ds", [("doc-1", {})], consolidated_lsn=42)

    env.client.get("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert seen["wm"] == 42, "recall lookup must use the resolved shard's watermark"


def test_get_cross_tenant_404(env, monkeypatch):
    """Cross-tenant get → dataset_not_found before any recall lookup runs."""
    _enable_recall(env, monkeypatch)

    def _boom(*a, **k):  # pragma: no cover - must never be called cross-tenant
        raise AssertionError("recall lookup ran on a cross-tenant dataset")

    monkeypatch.setattr(env.main, "recall_get_vector", _boom)
    a = _signup(env.client, email="a@example.com")
    b = _signup(env.client, email="b@example.com")
    _make_dataset(env, a["token"])
    env.write(a["tenant_id"], "ds", [("doc-1", {})])

    r = env.client.get("/v1/datasets/ds/vectors/doc-1", headers=_auth(b["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "dataset_not_found"


def test_get_flag_off_never_opens_recall_connection(env, monkeypatch):
    """Flag OFF: cold-only AND no recall connection ever opened."""
    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("flag-off get connected to a recall store")

    monkeypatch.setattr(env.state.psycopg2, "connect", _boom)
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    env.write(s["tenant_id"], "ds", [("doc-1", {"v": "cold"})])

    r = env.client.get("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "doc-1", "metadata": {"v": "cold"}}


# --- LIST union ------------------------------------------------------------


def test_list_recall_wins_dedup(env, monkeypatch):
    """A shared id is returned with the RECALL metadata (recall-wins dedup)."""
    _enable_recall(env, monkeypatch)
    # Recall has a live row for `shared` (overrides cold) + suppresses it.
    monkeypatch.setattr(
        env.main,
        "recall_list_rows",
        lambda t, d, wm: ([{"id": "shared", "metadata": {"v": "recall"}}], {"shared"}),
    )
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    env.write(s["tenant_id"], "ds", [
        ("shared", {"v": "cold"}),
        ("cold-only", {"v": "cold"}),
    ])

    r = env.client.get("/v1/datasets/ds/vectors", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    by_id = {v["id"]: v for v in r.json()["vectors"]}
    assert by_id["shared"]["metadata"] == {"v": "recall"}, r.json()
    assert by_id["cold-only"]["metadata"] == {"v": "cold"}
    # No duplicate row for `shared`.
    assert [v["id"] for v in r.json()["vectors"]] == ["cold-only", "shared"]


def test_list_recall_tombstone_suppresses(env, monkeypatch):
    """A recall tombstone (in suppress_ids, NOT in live rows) hides the cold id."""
    _enable_recall(env, monkeypatch)
    monkeypatch.setattr(
        env.main,
        "recall_list_rows",
        lambda t, d, wm: ([], {"doomed"}),  # tombstone-only: suppress, no live row
    )
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    env.write(s["tenant_id"], "ds", [("doomed", {}), ("survivor", {})])

    r = env.client.get("/v1/datasets/ds/vectors", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    ids = [v["id"] for v in r.json()["vectors"]]
    assert "doomed" not in ids, "tombstone must suppress the cold id from the list"
    assert ids == ["survivor"]


def test_list_filter_applied_after_union(env, monkeypatch):
    """The filter sees each id's AUTHORITATIVE (recall-or-cold) metadata.

    Cold `x {color:red}` is re-upserted in recall as `{color:blue}`; a filter
    `{color: red}` must NOT return x (its authoritative version fails the filter).
    """
    _enable_recall(env, monkeypatch)
    monkeypatch.setattr(
        env.main,
        "recall_list_rows",
        lambda t, d, wm: ([{"id": "x", "metadata": {"color": "blue"}}], {"x"}),
    )
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    env.write(s["tenant_id"], "ds", [
        ("x", {"color": "red"}),   # stale cold copy, matches the filter
        ("y", {"color": "red"}),
    ])

    r = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"filter": json.dumps({"color": "red"})},
    )
    assert r.status_code == 200, r.text
    ids = [v["id"] for v in r.json()["vectors"]]
    assert "x" not in ids, "stale cold x must be suppressed by the recall re-upsert"
    assert ids == ["y"]


def test_list_recall_live_passing_filter_included(env, monkeypatch):
    """A recall live row whose metadata passes the filter IS listed."""
    _enable_recall(env, monkeypatch)
    monkeypatch.setattr(
        env.main,
        "recall_list_rows",
        lambda t, d, wm: ([{"id": "x", "metadata": {"color": "red"}}], {"x"}),
    )
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    env.write(s["tenant_id"], "ds", [("x", {"color": "blue"}), ("y", {"color": "red"})])

    r = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"filter": json.dumps({"color": "red"})},
    )
    assert r.status_code == 200, r.text
    assert [v["id"] for v in r.json()["vectors"]] == ["x", "y"]


def test_list_stable_order_and_pagination(env, monkeypatch):
    """The merged list is stable-sorted by id and paginates over the union."""
    _enable_recall(env, monkeypatch)
    # Recall adds two fresh ids (b2, a2) not in cold; cold has a1, b1, c1.
    monkeypatch.setattr(
        env.main,
        "recall_list_rows",
        lambda t, d, wm: (
            [{"id": "a2", "metadata": {}}, {"id": "b2", "metadata": {}}],
            {"a2", "b2"},
        ),
    )
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    env.write(s["tenant_id"], "ds", [("c1", {}), ("a1", {}), ("b1", {})])

    r1 = env.client.get(
        "/v1/datasets/ds/vectors", headers=_auth(s["token"]), params={"limit": 2}
    )
    b1 = r1.json()
    assert [v["id"] for v in b1["vectors"]] == ["a1", "a2"]
    assert b1["next_cursor"] is not None

    r2 = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"limit": 2, "cursor": b1["next_cursor"]},
    )
    b2 = r2.json()
    assert [v["id"] for v in b2["vectors"]] == ["b1", "b2"]

    r3 = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"limit": 2, "cursor": b2["next_cursor"]},
    )
    b3 = r3.json()
    assert [v["id"] for v in b3["vectors"]] == ["c1"]
    assert b3["next_cursor"] is None


def test_list_recall_only_no_cold_shard(env, monkeypatch):
    """Recall live rows with NO cold shard list synchronously (watermark 0)."""
    _enable_recall(env, monkeypatch)
    monkeypatch.setattr(
        env.main,
        "recall_list_rows",
        lambda t, d, wm: ([{"id": "r1", "metadata": {"x": 1}}], {"r1"}),
    )
    s = _signup(env.client)
    _make_dataset(env, s["token"])  # no shard built

    r = env.client.get("/v1/datasets/ds/vectors", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert r.json()["vectors"] == [{"id": "r1", "metadata": {"x": 1}}]


def test_list_flag_off_never_opens_recall_connection(env, monkeypatch):
    """Flag OFF: cold-only list AND no recall connection opened."""
    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("flag-off list connected to a recall store")

    monkeypatch.setattr(env.state.psycopg2, "connect", _boom)
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    env.write(s["tenant_id"], "ds", [("b", {}), ("a", {})])

    r = env.client.get("/v1/datasets/ds/vectors", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert [v["id"] for v in r.json()["vectors"]] == ["a", "b"]


# --- DELETE recall path ----------------------------------------------------


def test_delete_recall_writes_above_watermark_tombstone(env, monkeypatch):
    """Recall delete writes a tombstone with a FRESH lsn strictly above the max.

    Exercises the REAL `recall_delete_vector` against a faked recall connection:
    the LSN sequence is primed at `last_lsn = 50` (the partition's current max,
    i.e. >= any watermark). The delete must allocate `51` — strictly greater than
    the prior max AND strictly greater than the watermark — and UPSERT a
    `deleted=true` row stamped with it. This is the regression guard for the
    below-watermark-tombstone bug the consolidation review flagged.
    """
    _enable_recall(env, monkeypatch)

    captured = {"sql": []}

    class _FakeCur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            captured["sql"].append((sql, params))
            self._last_sql = sql

        def fetchone(self):
            # The seq upsert-increment returns the NEW last_lsn. Prior max = 50,
            # bumped by 1 → 51 (strictly above the max / any watermark <= 50).
            return (51,)

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _FakeCur()

        def close(self):
            pass

    monkeypatch.setattr(env.state, "_recall_conn", lambda: _FakeConn())

    s = _signup(env.client)
    _make_dataset(env, s["token"], dim=4)
    # A cold shard whose watermark is 50 — the tombstone's lsn (51) MUST exceed it.
    env.write(s["tenant_id"], "ds", [("doc-1", {})], consolidated_lsn=50)

    r = env.client.delete("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 204, r.text
    assert r.content == b""

    # Two statements: (1) seq allocate, (2) tombstone UPSERT.
    sqls = [sql for sql, _ in captured["sql"]]
    assert any("recall_lsn_seq" in sql for sql in sqls), "must allocate a fresh lsn"
    upsert = next((p for sql, p in captured["sql"] if "recall_vectors" in sql), None)
    assert upsert is not None, "must UPSERT a tombstone into recall_vectors"
    # The allocated lsn (51) is the 5th positional param of the tombstone insert
    # (tenant, dataset, id, embedding, lsn) and is strictly above the watermark 50.
    allocated_lsn = upsert[4]
    assert allocated_lsn == 51
    assert allocated_lsn > 50, "tombstone lsn MUST be strictly above the watermark"
    # The INSERT stamps deleted=TRUE (in SQL text, not a param).
    tomb_sql = next(sql for sql, _ in captured["sql"] if "recall_vectors" in sql)
    assert "TRUE" in tomb_sql and "deleted" in tomb_sql.lower()


def test_delete_recall_no_publish_and_204(env, monkeypatch):
    """Recall delete returns 204 and does NOT publish DELETE_VECTORS."""
    from adapters.queue.queue import consume

    # Drain any DELETE_VECTORS a prior test left on the process-global queue, so
    # the "nothing published" assertion below is about THIS delete only.
    while consume("DELETE_VECTORS", block=False):
        pass

    _enable_recall(env, monkeypatch)
    recorded = {}

    def _fake_delete(t, d, vid, dim):
        recorded["args"] = (t, d, vid, dim)
        return 7

    monkeypatch.setattr(env.main, "recall_delete_vector", _fake_delete)

    s = _signup(env.client)
    _make_dataset(env, s["token"], dim=4)
    env.write(s["tenant_id"], "ds", [("doc-1", {})])

    r = env.client.delete("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 204, r.text
    assert r.content == b""
    # The recall tombstone got the dataset dimension for the placeholder embedding.
    assert recorded["args"][1] == "ds" and recorded["args"][2] == "doc-1"
    assert recorded["args"][3] == 4
    # NO DELETE_VECTORS published — consolidation applies the tombstone to cold.
    assert consume("DELETE_VECTORS", block=False) is None


def test_delete_recall_failure_503(env, monkeypatch):
    """A recall-store failure on delete → structured 503, not a raw 500."""
    _enable_recall(env, monkeypatch)

    def _boom(t, d, vid, dim):
        raise RuntimeError("recall store dropped")

    monkeypatch.setattr(env.main, "recall_delete_vector", _boom)

    s = _signup(env.client)
    _make_dataset(env, s["token"], dim=4)
    env.write(s["tenant_id"], "ds", [("doc-1", {})])

    r = env.client.delete("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "recall_delete_failed"


def test_delete_recall_cross_tenant_404(env, monkeypatch):
    """Cross-tenant delete → 404 before any recall write runs."""
    _enable_recall(env, monkeypatch)

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("recall delete ran on a cross-tenant dataset")

    monkeypatch.setattr(env.main, "recall_delete_vector", _boom)
    a = _signup(env.client, email="a@example.com")
    b = _signup(env.client, email="b@example.com")
    _make_dataset(env, a["token"])
    env.write(a["tenant_id"], "ds", [("doc-1", {})])

    r = env.client.delete("/v1/datasets/ds/vectors/doc-1", headers=_auth(b["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "dataset_not_found"


def test_delete_flag_off_keeps_builder_path(env, monkeypatch):
    """Flag OFF: 202 + DELETE_VECTORS published + status flip, no recall conn."""
    from adapters.queue.queue import consume

    # Drain leftovers from the process-global queue so we read THIS delete's msg.
    while consume("DELETE_VECTORS", block=False):
        pass

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("flag-off delete connected to a recall store")

    monkeypatch.setattr(env.state.psycopg2, "connect", _boom)

    s = _signup(env.client)
    _make_dataset(env, s["token"])
    env.write(s["tenant_id"], "ds", [("doc-1", {})])

    r = env.client.delete("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 202, r.text
    assert r.json()["job_id"].startswith("job_")

    msg = consume("DELETE_VECTORS", block=False)
    assert msg is not None and msg["id"] == "doc-1"
    # Status flipped to indexing (a shard exists).
    ds = env.client.get("/v1/datasets/ds", headers=_auth(s["token"])).json()
    assert ds["status"] == "indexing"


def test_delete_recall_no_shard_still_204(env, monkeypatch):
    """Recall delete with no cold shard (cold-only id absent) still tombstones → 204.

    Deleting an id present only in COLD (or absent entirely) writes an above-
    watermark tombstone regardless of whether a shard exists yet — the union
    hides it and the next consolidation removes the cold copy.
    """
    _enable_recall(env, monkeypatch)
    monkeypatch.setattr(env.main, "recall_delete_vector", lambda t, d, vid, dim: 1)

    s = _signup(env.client)
    _make_dataset(env, s["token"], dim=4)  # no shard

    r = env.client.delete("/v1/datasets/ds/vectors/ghost", headers=_auth(s["token"]))
    assert r.status_code == 204, r.text
