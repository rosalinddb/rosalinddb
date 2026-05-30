"""Ghost-row regression: delete + same-name re-create.

The single sequential trace (no race) showed:

    create _stress_idem dim=4
    ingest 5 vectors (ids i1_0..i1_4)
    delete _stress_idem            -> deleted:true
    create _stress_idem dim=4      -> row_count:0, status:empty
    query _stress_idem             -> 5 ghost rows from the deleted dataset

The dataset row was correctly forgotten (`row_count:0` confirms a fresh
catalog row), but the `shard_catalog` rows for the deleted dataset were
NOT cleaned up — `list_shards(tenant, dataset)` returned them, the query
path resolved the latest one, hit a still-warm FAISS shard cache entry,
and served stale vectors as `mode:"hot"`.

The fix removes shard catalog rows on `delete_dataset` and emits a notify
so any per-`(tenant, dataset)` catalog cache on a DP gets evicted within
the same transaction. After the fix this test asserts the new dataset
returns zero matches.

These tests run inline (validator + builder driven synchronously) over
real MinIO via the shared `s3_landing_prefix` / `s3_indexes_prefix`
fixtures from `tests/integration/conftest.py` — the standard integration
posture.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def client(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Fresh TestClient: source_registry app + v1_query router mounted.

    Same recipe as `tests/integration/test_query_api.py:client`: reload the
    pipeline modules so the per-test MinIO prefixes win, reset the in-process
    shard cache and the per-dataset catalog cache, and drain any queue
    leftovers from a previous test.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")

    # Drain process-global in-proc queue topics so a fresh test never picks
    # up stale work from a prior test.
    from adapters.queue.queue import consume as _consume
    for _topic in (
        "VALIDATE_DATASET", "DATASET_READY", "RUN_EPHEMERAL_QUERY", "RESULT_READY",
    ):
        while _consume(_topic, block=False):
            pass

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS",
                 "_MEM_DATASETS"):
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
    if hasattr(v1_query, "_catalog_cache_clear"):
        v1_query._catalog_cache_clear()

    main_mod.app.include_router(v1_query.router)

    from fastapi.testclient import TestClient
    return TestClient(main_mod.app)


def _signup(client, email="bug1@example.com", password="password123"):
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _run_pipeline_once():
    """Drain VALIDATE_DATASET and run the builder synchronously."""
    from adapters.queue.queue import consume
    from services.validator_worker.run import process_uri
    from services.index_builder.run import run_once

    pending = []
    while True:
        msg = consume("VALIDATE_DATASET", block=False)
        if not msg:
            break
        try:
            process_uri(msg["dataset"], msg["tenant"], msg["uri"],
                        msg.get("file_type"))
            pending.append(msg)
        except Exception:
            pass
    for msg in pending:
        run_once(msg["dataset"], msg["tenant"])


def _create_and_ingest(client, token, name, dim, records):
    """Create `name`, ingest `records`, drive the pipeline to `indexed`."""
    r = client.post(
        "/v1/datasets",
        headers=_auth(token),
        json={"name": name, "dimension": dim},
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


# --- the bug --------------------------------------------------------------


def test_delete_then_recreate_does_not_serve_ghost_rows(client):
    """The headline regression: the delete/recreate sequence.

    Reproduces the exact sequence and asserts the second-create's query
    returns zero matches (the catalog says `row_count:0`; the query path
    must agree). Before the fix, the query returned 5 ghost rows from
    the deleted dataset — its shard rows were never removed, so
    `list_shards` happily resurfaced them post-recreate.
    """
    s = _signup(client)
    token = s["token"]
    name = "_stress_idem"

    records = [
        {"id": f"i1_{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {}}
        for i in range(5)
    ]
    _create_and_ingest(client, token, name, 4, records)

    # Sanity: query the indexed dataset, expect the 5 records to come back.
    r = client.post(
        "/v1/query",
        headers=_auth(token),
        json={"dataset": name, "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 10},
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["matches"]) == 5, r.json()

    # --- the bug repro ----------------------------------------------------
    # Soft-delete.
    r = client.delete(f"/v1/datasets/{name}", headers=_auth(token))
    assert r.status_code == 204, r.text

    # Re-create with the same name + dimension.
    r = client.post(
        "/v1/datasets",
        headers=_auth(token),
        json={"name": name, "dimension": 4},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["row_count"] == 0, body
    assert body["status"] == "empty", body

    # Query the freshly-recreated, empty dataset. Must return zero matches
    # — not the 5 ghost rows from the deleted predecessor.
    r = client.post(
        "/v1/query",
        headers=_auth(token),
        json={"dataset": name, "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 10},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matches"] == [], (
        f"ghost rows resurfaced from deleted dataset; got: {body['matches']}"
    )

    # And the dataset's reported row_count must still agree with the query
    # — `row_count:0` and the query returning 0 must remain consistent
    # (the original bug had row_count:0 but 5 query matches).
    ds = client.get(f"/v1/datasets/{name}", headers=_auth(token)).json()
    assert ds["row_count"] == 0, ds
    assert ds["row_count"] == len(body["matches"]), (ds, body)


def test_delete_dataset_drops_shard_catalog_rows(client):
    """`delete_dataset` must purge `shard_catalog` for the dataset.

    Lower-level pin of the same fix: the catalog adapter is the source of
    truth for the engine. After a soft-delete, `list_shards` must return
    an empty list for that `(tenant, dataset)` — otherwise the query
    path's `latest = shards[0]` would resolve to a now-orphaned shard.
    """
    from adapters.state import state as state_mod

    s = _signup(client, email="shards@example.com")
    token = s["token"]
    tenant_id = state_mod.get_tenant_by_email("shards@example.com")["id"]

    records = [
        {"id": f"x{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {}}
        for i in range(3)
    ]
    _create_and_ingest(client, token, "purge_me", 4, records)

    # Pre-delete: at least one shard row exists.
    assert len(state_mod.list_shards(tenant_id, "purge_me")) >= 1

    r = client.delete("/v1/datasets/purge_me", headers=_auth(token))
    assert r.status_code == 204, r.text

    # Post-delete: zero shard rows for the deleted dataset.
    assert state_mod.list_shards(tenant_id, "purge_me") == []


def test_delete_does_not_touch_a_sibling_dataset(client):
    """Tenant + dataset isolation: deleting `A` must not purge `B`'s shards.

    The fix scopes the shard purge by `(tenant_id, dataset_name)`; this
    test guards against a sloppy `DELETE FROM shard_catalog WHERE
    tenant_id=...` that would wipe every dataset for the tenant.
    """
    from adapters.state import state as state_mod

    s = _signup(client, email="sibling@example.com")
    token = s["token"]
    tenant_id = state_mod.get_tenant_by_email("sibling@example.com")["id"]

    a_records = [
        {"id": f"a{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {}}
        for i in range(3)
    ]
    b_records = [
        {"id": f"b{i}", "values": [0.0, float(i), 0.0, 0.0], "metadata": {}}
        for i in range(3)
    ]
    _create_and_ingest(client, token, "ds_a", 4, a_records)
    _create_and_ingest(client, token, "ds_b", 4, b_records)

    assert len(state_mod.list_shards(tenant_id, "ds_a")) >= 1
    assert len(state_mod.list_shards(tenant_id, "ds_b")) >= 1

    r = client.delete("/v1/datasets/ds_a", headers=_auth(token))
    assert r.status_code == 204, r.text

    # ds_a is purged, ds_b is intact.
    assert state_mod.list_shards(tenant_id, "ds_a") == []
    assert len(state_mod.list_shards(tenant_id, "ds_b")) >= 1

    # And the sibling can still serve its vectors.
    r = client.post(
        "/v1/query",
        headers=_auth(token),
        json={"dataset": "ds_b", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 10},
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["matches"]) == 3, r.json()
