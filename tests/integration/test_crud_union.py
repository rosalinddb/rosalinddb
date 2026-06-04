"""Integration coverage for the get/list/delete CRUD union with Recall (PR6).

Runs the consolidated-tier CRUD surface end-to-end through the real FastAPI app
with BOTH tiers live:

  - the **recall tier** on a REAL `pgvector/pgvector:pg15` container
    (`RB_RECALL_DSN`);
  - the **consolidated (cold)** tier built by the real ingest pipeline into the
    session MinIO;
  - the control plane on the default `memory://` state adapter (the recall path
    is gated on `RB_RECALL_DSN`, not the control-plane DSN);
  - the **consolidation** (recall→cold flush) driven by the real
    `index_builder.run_consolidate_once`.

Properties proven (docs/architecture/recall-consolidate.md, PR6 + invariants
I1/I2):

  - **get union**: recall-wins live; recall tombstone → 404; fall back to cold.
  - **list union**: recall-wins dedup; tombstone-suppress; filter after union.
  - **read-your-deletes**: a flag-on `POST /vectors` (recall) → DELETE → immediate
    GET = 404 → `POST /query` no longer returns the id.
  - **cold-only delete + consolidation**: deleting an id present only in COLD
    writes an ABOVE-watermark tombstone → the union hides it → a consolidation
    then removes it from cold (the id stays gone after the recall row is trimmed).
  - **above-watermark lsn contract** (REGRESSION): the recall-delete tombstone's
    lsn is strictly greater than the resolved shard's watermark.
  - **flag OFF (default)**: cold-delete-via-builder (202, DELETE_VECTORS),
    byte-identical; recall never consulted by get/list.
"""
from __future__ import annotations

import importlib
import json

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
def recall_url():
    """One pgvector container for this module; yield a psycopg2 DSN."""
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the CRUD-union suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with PostgresContainer("pgvector/pgvector:pg15", driver=None) as pg:
        yield pg.get_connection_url()


def _truncate_recall(dsn: str) -> None:
    """Drop all recall rows + reset the LSN sequence (per-test isolation)."""
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.recall_vectors'), "
                "to_regclass('public.recall_lsn_seq')"
            )
            hv, seq = cur.fetchone()
            if hv is not None:
                cur.execute("TRUNCATE recall_vectors")
            if seq is not None:
                cur.execute("TRUNCATE recall_lsn_seq")
    finally:
        conn.close()


def _recall_row(dsn, tenant, dataset, vid):
    """Return `(lsn, deleted)` for one recall row, or None if absent."""
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT lsn, deleted FROM recall_vectors "
                "WHERE tenant_id=%s AND dataset=%s AND id=%s",
                (tenant, dataset, vid),
            )
            return cur.fetchone()
    finally:
        conn.close()


def _build_client(monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on):
    """Build a TestClient + reloaded state/builder/query modules (both tiers)."""
    if recall_on:
        monkeypatch.setenv("RB_RECALL", "true")
    else:
        monkeypatch.delenv("RB_RECALL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "memory://test")
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")

    from adapters.queue.queue import consume as _consume
    for _topic in (
        "VALIDATE_DATASET", "DATASET_READY", "DELETE_VECTORS", "CONSOLIDATE",
        "RUN_EPHEMERAL_QUERY", "RESULT_READY",
    ):
        while _consume(_topic, block=False):
            pass

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._RECALL_MIGRATED = False
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_DATASETS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, list):
            obj.clear()
    state_mod._MEM_SHARDS.clear()

    import services.auth.jwt_utils as jwt_utils
    importlib.reload(jwt_utils)
    import services.auth.auth as auth_mod
    importlib.reload(auth_mod)
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)
    import services.validator_worker.run as validator
    importlib.reload(validator)
    import services.index_builder.run as builder
    importlib.reload(builder)
    import services.ephemeral_runner.run as ephemeral
    importlib.reload(ephemeral)
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    v1_query.cache_clear()
    v1_query._RESULTS.clear()

    main_mod.app.include_router(v1_query.router)

    from fastapi.testclient import TestClient
    return TestClient(main_mod.app), state_mod, main_mod, builder


def _signup(client, email="alice@example.com"):
    r = client.post("/auth/signup", json={"email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _tenant_of(client, signup):
    r = client.get("/auth/me", headers=_auth(signup["token"]))
    return r.json()["tenant"]["id"]


def _migrate_recall(recall_url):
    import os

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    os.environ["RB_RECALL_DSN"] = recall_url
    state_mod._RECALL_MIGRATED = False
    state_mod.migrate_recall(force=True)


def _run_pipeline_once():
    """Drain VALIDATE_DATASET and run the builder synchronously (cold shard)."""
    from adapters.queue.queue import consume
    from services.validator_worker.run import process_uri
    from services.index_builder.run import run_once

    pending = []
    while True:
        msg = consume("VALIDATE_DATASET", block=False)
        if not msg:
            break
        try:
            process_uri(msg["dataset"], msg["tenant"], msg["uri"], msg.get("file_type"))
            pending.append(msg)
        except Exception:
            pass
    for msg in pending:
        run_once(msg["dataset"], msg["tenant"])


def _make_cold_dataset(client, token, name, dimension, records):
    """Create + populate + index a dataset into a real cold FAISS shard.

    Runs with the recall flag OFF so the upload lands + builds a shard.
    """
    r = client.post(
        "/v1/datasets", headers=_auth(token), json={"name": name, "dimension": dimension}
    )
    assert r.status_code == 201, r.text
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        f"/v1/datasets/{name}/vectors",
        headers={**_auth(token), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    _run_pipeline_once()
    ds = client.get(f"/v1/datasets/{name}", headers=_auth(token)).json()
    assert ds["status"] == "indexed", ds


def _post_recall(client, token, name, records):
    """Flag-on `POST /vectors` → synchronous recall write (200)."""
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        f"/v1/datasets/{name}/vectors",
        headers={**_auth(token), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 200, r.text
    return r.json()


def _flip_recall_on(recall_url, app):
    """Flip the union ON in place + reload only the query/handler modules.

    Used after building a cold shard with the flag OFF: re-reloads v1_query so its
    `recall_enabled()` gate flips on, preserving the live cold shard + tenant
    (`_build_client` would clear the memory state).
    """
    import os

    os.environ["RB_RECALL"] = "true"
    os.environ["RB_RECALL_DSN"] = recall_url
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    app.include_router(v1_query.router)
    return v1_query


# --- get / list union -----------------------------------------------------


def test_get_recall_live_wins_over_cold(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """GET: a live recall re-upsert wins over the cold sidecar copy."""
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, state_mod, main_mod, _b = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=False
    )
    s = _signup(client, email="getwin@example.com")
    _make_cold_dataset(client, s["token"], "g", 4, [
        {"id": "shared", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": "cold"}},
    ])
    tenant = _tenant_of(client, s)
    _flip_recall_on(recall_url, client.app)

    state_mod.recall_upsert_vectors(tenant, "g", [
        {"id": "shared", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": "recall"}},
    ])

    r = client.get("/v1/datasets/g/vectors/shared", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "shared", "metadata": {"v": "recall"}}


def test_get_falls_back_to_cold(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """GET: an id only in cold (no recall row) is served from the sidecar."""
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, _main, _b = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=False
    )
    s = _signup(client, email="getcold@example.com")
    _make_cold_dataset(client, s["token"], "gc", 4, [
        {"id": "only-cold", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": "cold"}},
    ])
    _flip_recall_on(recall_url, client.app)

    r = client.get("/v1/datasets/gc/vectors/only-cold", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "only-cold", "metadata": {"v": "cold"}}


def test_list_recall_wins_and_tombstone_suppress(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """LIST: recall-wins dedup + a recall tombstone hides a cold id."""
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, state_mod, main_mod, _b = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=False
    )
    s = _signup(client, email="list@example.com")
    _make_cold_dataset(client, s["token"], "l", 4, [
        {"id": "a", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": "cold"}},
        {"id": "b", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"v": "cold"}},
        {"id": "doomed", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {"v": "cold"}},
    ])
    tenant = _tenant_of(client, s)
    _flip_recall_on(recall_url, client.app)

    # Re-upsert `a` (recall-wins) and DELETE `doomed` (tombstone-suppress) via the
    # endpoint so the real delete write path is exercised.
    state_mod.recall_upsert_vectors(tenant, "l", [
        {"id": "a", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": "recall"}},
    ])
    rd = client.delete("/v1/datasets/l/vectors/doomed", headers=_auth(s["token"]))
    assert rd.status_code == 204, rd.text

    r = client.get("/v1/datasets/l/vectors", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    by_id = {v["id"]: v for v in r.json()["vectors"]}
    assert "doomed" not in by_id, "tombstone must suppress the cold id from the list"
    assert by_id["a"]["metadata"] == {"v": "recall"}, "recall-wins on the shared id"
    assert by_id["b"]["metadata"] == {"v": "cold"}
    assert [v["id"] for v in r.json()["vectors"]] == ["a", "b"]  # stable id order


# --- read-your-deletes ----------------------------------------------------


def test_read_your_deletes_recall(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Flag-on write → DELETE → immediate GET 404 → query no longer returns it."""
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    monkeypatch.setenv("RB_RECALL", "true")
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, _main, _b = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=True
    )
    s = _signup(client, email="ryd@example.com")
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "d", "dimension": 4})

    _post_recall(client, s["token"], "d", [
        {"id": "fact-1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"t": "peanuts"}},
        {"id": "fact-2", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"t": "shellfish"}},
    ])
    # Present before the delete.
    assert client.get("/v1/datasets/d/vectors/fact-1", headers=_auth(s["token"])).status_code == 200

    # DELETE → 204 synchronous.
    rd = client.delete("/v1/datasets/d/vectors/fact-1", headers=_auth(s["token"]))
    assert rd.status_code == 204, rd.text
    assert rd.content == b""

    # Immediate GET → 404 (read-your-deletes).
    rg = client.get("/v1/datasets/d/vectors/fact-1", headers=_auth(s["token"]))
    assert rg.status_code == 404, rg.text
    assert rg.json()["error"]["code"] == "not_found"

    # The query union no longer returns it.
    rq = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "d", "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 10},
    )
    assert rq.status_code == 200, rq.text
    ids = [m["id"] for m in rq.json()["matches"]]
    assert "fact-1" not in ids, "deleted id must not appear in the query union"
    assert "fact-2" in ids

    # List union also hides it.
    rl = client.get("/v1/datasets/d/vectors", headers=_auth(s["token"]))
    assert "fact-1" not in [v["id"] for v in rl.json()["vectors"]]


# --- cold-only delete → above-watermark tombstone → consolidation ----------


def test_cold_only_delete_tombstone_above_watermark_then_consolidated(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Delete an id present only in COLD: tombstone > watermark, union hides it,
    consolidation removes it from cold, and the id stays gone.

    Also the REGRESSION assertion: the tombstone's lsn is strictly greater than
    the resolved shard's watermark (guards the below-watermark-tombstone bug).
    """
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, state_mod, main_mod, builder = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=False
    )
    s = _signup(client, email="coldonly@example.com")
    _make_cold_dataset(client, s["token"], "c", 4, [
        {"id": "gone", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "keep", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ])
    tenant = _tenant_of(client, s)
    _flip_recall_on(recall_url, client.app)

    # The cold shard's watermark (consolidated_lsn). A fresh cold-only shard has
    # never consolidated recall, so its watermark is 0; the tombstone lsn (>=1)
    # is strictly above it. We assert the relationship generically.
    shard = state_mod.get_latest_shard(tenant, "c")
    watermark = int(shard.get("consolidated_lsn", 0) or 0)

    # DELETE a COLD-ONLY id (no prior recall row) via the endpoint.
    rd = client.delete("/v1/datasets/c/vectors/gone", headers=_auth(s["token"]))
    assert rd.status_code == 204, rd.text

    # A tombstone row now exists with lsn STRICTLY above the watermark (contract).
    row = _recall_row(recall_url, tenant, "c", "gone")
    assert row is not None, "the cold-only delete must write a recall tombstone"
    lsn, deleted = row
    assert deleted is True, "the row is a tombstone"
    assert lsn > watermark, (
        f"tombstone lsn ({lsn}) MUST be strictly above the watermark ({watermark})"
    )

    # The union hides the cold id immediately (still in cold, tombstone suppresses).
    rg = client.get("/v1/datasets/c/vectors/gone", headers=_auth(s["token"]))
    assert rg.status_code == 404, "tombstone must hide the still-present cold id"
    rl = client.get("/v1/datasets/c/vectors", headers=_auth(s["token"]))
    assert [v["id"] for v in rl.json()["vectors"]] == ["keep"]

    # Now CONSOLIDATE: fold the recall partition (the tombstone) into a new cold
    # shard. The tombstoned id is `_remove_ids`'d from the consolidated shard.
    builder.run_consolidate_once("c", tenant)

    # The id is gone from COLD too. To prove it is not just hidden by a still-
    # present recall tombstone, drop the recall partition and re-check the cold
    # sidecar directly via a flag-OFF get (cold-only path).
    _truncate_recall(recall_url)
    import os
    os.environ.pop("RB_RECALL", None)  # flag OFF → pure cold get
    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)

    rg2 = client.get("/v1/datasets/c/vectors/gone", headers=_auth(s["token"]))
    assert rg2.status_code == 404, "consolidation must have removed `gone` from cold"
    rg3 = client.get("/v1/datasets/c/vectors/keep", headers=_auth(s["token"]))
    assert rg3.status_code == 200, "the surviving id is still in the consolidated shard"


# --- flag OFF: cold-delete-via-builder unchanged --------------------------


def test_flag_off_delete_uses_builder_path(
    monkeypatch, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path
):
    """Flag OFF (default): DELETE → 202 + DELETE_VECTORS published, no recall row."""
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)  # DSN set, flag OFF
    _migrate_recall(recall_url)
    _truncate_recall(recall_url)

    client, _state, _main, _b = _build_client(
        monkeypatch, s3_landing_prefix, s3_indexes_prefix, tmp_path, recall_on=False
    )
    s = _signup(client, email="offdel@example.com")
    _make_cold_dataset(client, s["token"], "od", 4, [
        {"id": "x", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
    ])
    tenant = _tenant_of(client, s)

    from adapters.queue.queue import consume
    rd = client.delete("/v1/datasets/od/vectors/x", headers=_auth(s["token"]))
    assert rd.status_code == 202, rd.text
    assert rd.json()["job_id"].startswith("job_")

    msg = consume("DELETE_VECTORS", block=False)
    assert msg is not None and msg["id"] == "x", "flag-off must publish DELETE_VECTORS"
    # No recall tombstone written (flag off → builder path only).
    assert _recall_row(recall_url, tenant, "od", "x") is None

    # The dataset flips to `indexing`.
    ds = client.get("/v1/datasets/od", headers=_auth(s["token"])).json()
    assert ds["status"] == "indexing"
