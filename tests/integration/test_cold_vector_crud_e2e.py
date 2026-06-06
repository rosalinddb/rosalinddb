"""End-to-end integration test for consolidated-tier vector get/list/delete-by-id.

Drives the full pipeline against real MinIO (testcontainers): ingest -> index
-> get/list -> delete -> drive the builder's DELETE_VECTORS consumer -> poll
status -> confirm the id is gone from get, list, AND query. This is the
flag-independent consolidated CRUD surface (no `RB_RECALL`), so it exercises the
shard rewrite + sidecar drop + sweep path the builder's delete handler runs.
"""
from __future__ import annotations

import importlib
import json

import pytest

import os

os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def client(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """TestClient over source_registry + the mounted v1_query router."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")  # tiny test fixtures use flat

    from adapters.queue.queue import consume as _consume
    for _topic in (
        "VALIDATE_DATASET", "DATASET_READY", "DELETE_VECTORS",
        "RUN_EPHEMERAL_QUERY", "RESULT_READY",
    ):
        while _consume(_topic, block=False):
            pass

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
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
    return TestClient(main_mod.app)


def _signup(client, email="alice@example.com", password="password123"):
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _run_ingest_pipeline():
    """Drain VALIDATE_DATASET and run the builder build path synchronously."""
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


def _run_delete_pipeline():
    """Drain DELETE_VECTORS and run the builder delete handler synchronously."""
    from adapters.queue.queue import consume
    from services.index_builder.run import _handle_delete_vectors

    while True:
        msg = consume("DELETE_VECTORS", block=False)
        if not msg:
            break
        _handle_delete_vectors(msg)


def _make_indexed_dataset(client, token, name="mem", records=None):
    r = client.post("/v1/datasets", headers=_auth(token), json={"name": name, "dimension": 4})
    assert r.status_code == 201, r.text
    if records is None:
        records = [
            {"id": f"doc-{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"title": f"t{i}"}}
            for i in range(6)
        ]
    body = "\n".join(json.dumps(rec) for rec in records)
    r = client.post(
        f"/v1/datasets/{name}/vectors",
        headers={**_auth(token), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    _run_ingest_pipeline()
    ds = client.get(f"/v1/datasets/{name}", headers=_auth(token)).json()
    assert ds["status"] == "indexed", ds
    return records


def test_get_list_delete_end_to_end(client):
    s = _signup(client)
    token = s["token"]
    _make_indexed_dataset(client, token)

    # get-by-id returns the customer's metadata.
    r = client.get("/v1/datasets/mem/vectors/doc-2", headers=_auth(token))
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "doc-2", "metadata": {"title": "t2"}}

    # list returns all six, stably sorted by id.
    r = client.get("/v1/datasets/mem/vectors", headers=_auth(token))
    assert r.status_code == 200
    body = r.json()
    assert [v["id"] for v in body["vectors"]] == [f"doc-{i}" for i in range(6)]
    assert body["next_cursor"] is None

    # delete doc-2.
    r = client.delete("/v1/datasets/mem/vectors/doc-2", headers=_auth(token))
    assert r.status_code == 202, r.text
    assert r.json()["job_id"].startswith("job_")

    # The dataset reflects the in-flight delete.
    ds = client.get("/v1/datasets/mem", headers=_auth(token)).json()
    assert ds["status"] == "indexing"

    # Drive the builder's DELETE_VECTORS consumer.
    _run_delete_pipeline()

    # Poll: status back to indexed, row_count dropped by one.
    ds = client.get("/v1/datasets/mem", headers=_auth(token)).json()
    assert ds["status"] == "indexed", ds
    assert ds["row_count"] == 5, ds

    # The deleted vector is gone from get and list.
    assert client.get("/v1/datasets/mem/vectors/doc-2", headers=_auth(token)).status_code == 404
    remaining = client.get("/v1/datasets/mem/vectors", headers=_auth(token)).json()["vectors"]
    assert "doc-2" not in {v["id"] for v in remaining}
    assert len(remaining) == 5

    # And a query no longer returns it. The shard cache must reflect the
    # superseded shard — eviction happens in the builder's sweep.
    import services.query_api.v1_query as v1_query
    v1_query.cache_clear()
    r = client.post(
        "/v1/query",
        headers=_auth(token),
        json={"dataset": "mem", "vector": [2.0, 0.0, 0.0, 0.0], "top_k": 10},
    )
    assert r.status_code == 200, r.text
    matched = {m["id"] for m in r.json()["matches"]}
    assert "doc-2" not in matched
    assert "doc-1" in matched  # a surviving neighbour is still returned


def test_list_filter_and_pagination_e2e(client):
    s = _signup(client)
    token = s["token"]
    records = [
        {"id": f"doc-{i}", "values": [float(i), 0.0, 0.0, 0.0],
         "metadata": {"kind": "even" if i % 2 == 0 else "odd"}}
        for i in range(6)
    ]
    _make_indexed_dataset(client, token, records=records)

    # filter to evens.
    r = client.get(
        "/v1/datasets/mem/vectors",
        headers=_auth(token),
        params={"filter": json.dumps({"kind": "even"})},
    )
    assert r.status_code == 200, r.text
    assert [v["id"] for v in r.json()["vectors"]] == ["doc-0", "doc-2", "doc-4"]

    # paginate the full set.
    r1 = client.get("/v1/datasets/mem/vectors", headers=_auth(token), params={"limit": 4})
    b1 = r1.json()
    assert [v["id"] for v in b1["vectors"]] == ["doc-0", "doc-1", "doc-2", "doc-3"]
    assert b1["next_cursor"] is not None
    r2 = client.get(
        "/v1/datasets/mem/vectors",
        headers=_auth(token),
        params={"limit": 4, "cursor": b1["next_cursor"]},
    )
    b2 = r2.json()
    assert [v["id"] for v in b2["vectors"]] == ["doc-4", "doc-5"]
    assert b2["next_cursor"] is None
