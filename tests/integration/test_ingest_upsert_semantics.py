"""Upsert semantics on the customer-facing ingest path.

Re-ingest upsert correctness regression.
Re-ingesting the same `id` twice was correctly upserted by FAISS at the
shard level (a query for the new values returned the row at score=0) but
the `dataset_catalog.row_count` doubled on every re-ingest of an
existing id. Storage / billing / sharding decisions that trust `row_count`
were poisoned.

The repro sequence:

    ingest id="x" values=[1,0,0,0]   -> row_count == 1   (OK)
    ingest id="x" values=[0,0,0,1]   (SAME id)
    get_dataset.row_count == 2                           (BUG: expected 1)

These tests pin the post-fix invariant: a re-ingest of an existing id
overwrites the stored vector (last-write-wins) AND `row_count` reflects
the count of *unique* live ids in the dataset.

Same bug FAMILY as I-01 (purge `shard_catalog` on delete_dataset, commit
``faee23d``) but on the INGEST path instead of DELETE.
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
    """Fresh FastAPI TestClient with reset in-memory state + MinIO landing.

    Mirrors the fixture in ``test_ingest_api.py``: each test gets a unique
    MinIO landing/indexes prefix and a freshly-reloaded pipeline so module-
    level constants pick up the patched env vars.
    """
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")

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

    from fastapi.testclient import TestClient
    return TestClient(main_mod.app)


def _signup(client, email="alice@example.com", password="password123"):
    r = client.post("/auth/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _run_pipeline_once():
    """Drain VALIDATE_DATASET and run the builder synchronously.

    Identical to ``test_ingest_api.py::_run_pipeline_once`` — the HTTP-
    level test bypasses ``validator.main_loop`` and ``builder.main_loop``.
    """
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


def _ingest(client, token, dataset, records):
    body = "\n".join(json.dumps(r) for r in records)
    r = client.post(
        f"/v1/datasets/{dataset}/vectors",
        headers={**_auth(token), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    return r.json()


def _get_dataset(client, token, dataset):
    r = client.get(f"/v1/datasets/{dataset}", headers=_auth(token))
    assert r.status_code == 200, r.text
    return r.json()


def test_reingest_same_ids_does_not_double_row_count(client):
    """U-01/U-03 invariant: re-ingest of the SAME id set keeps row_count stable.

    Ingest 5 distinct ids, then re-ingest the same 5 ids with different
    vector values. `row_count` MUST remain 5, not become 10.
    """
    s = _signup(client)
    client.post(
        "/v1/datasets",
        headers=_auth(s["token"]),
        json={"name": "upsert5", "dimension": 4},
    )

    first = [
        {"id": f"id-{i}", "values": [float(i), 0.0, 0.0, 0.0]}
        for i in range(5)
    ]
    _ingest(client, s["token"], "upsert5", first)
    _run_pipeline_once()

    ds = _get_dataset(client, s["token"], "upsert5")
    assert ds["row_count"] == 5, ds

    # Re-ingest the same 5 ids with NEW values (last-write-wins upsert).
    second = [
        {"id": f"id-{i}", "values": [0.0, 0.0, 0.0, float(i)]}
        for i in range(5)
    ]
    _ingest(client, s["token"], "upsert5", second)
    _run_pipeline_once()

    ds = _get_dataset(client, s["token"], "upsert5")
    assert ds["row_count"] == 5, (
        f"row_count must stay at 5 across an idempotent re-ingest; "
        f"got {ds['row_count']}."
    )


def test_repeated_reingest_single_id_stays_one(client):
    """U-04 invariant: hammering the same id 10 times still ends at row_count=1.

    Storage grows unbounded on idempotent re-writes if this fails, and
    any quota / billing / sharding decision that trusts `row_count` is
    poisoned.
    """
    s = _signup(client)
    client.post(
        "/v1/datasets",
        headers=_auth(s["token"]),
        json={"name": "upsert1", "dimension": 4},
    )

    _ingest(client, s["token"], "upsert1", [{"id": "x", "values": [1.0, 0.0, 0.0, 0.0]}])
    _run_pipeline_once()
    ds = _get_dataset(client, s["token"], "upsert1")
    assert ds["row_count"] == 1, ds

    # Re-ingest "x" 10 times in a loop, each time with a different vector.
    for k in range(10):
        _ingest(
            client,
            s["token"],
            "upsert1",
            [{"id": "x", "values": [0.0, float(k + 1), 0.0, 0.0]}],
        )
        _run_pipeline_once()

    ds = _get_dataset(client, s["token"], "upsert1")
    assert ds["row_count"] == 1, (
        f"row_count must stay at 1 across 11 ingests of the same id; "
        f"got {ds['row_count']}."
    )


def test_mixed_batch_partial_overlap_counts_unique_ids(client):
    """U-05 invariant: mixed-overlap batches end with the unique-id total.

    Ingest ``[a, b, c]`` then ``[b, c, d]`` — `b` and `c` overlap, `d` is
    new. `row_count` MUST end at 4 (a/b/c/d), not 6.
    """
    s = _signup(client)
    client.post(
        "/v1/datasets",
        headers=_auth(s["token"]),
        json={"name": "upsertmix", "dimension": 4},
    )

    _ingest(client, s["token"], "upsertmix", [
        {"id": "a", "values": [1.0, 0.0, 0.0, 0.0]},
        {"id": "b", "values": [0.0, 1.0, 0.0, 0.0]},
        {"id": "c", "values": [0.0, 0.0, 1.0, 0.0]},
    ])
    _run_pipeline_once()
    ds = _get_dataset(client, s["token"], "upsertmix")
    assert ds["row_count"] == 3, ds

    _ingest(client, s["token"], "upsertmix", [
        {"id": "b", "values": [9.0, 0.0, 0.0, 0.0]},
        {"id": "c", "values": [0.0, 9.0, 0.0, 0.0]},
        {"id": "d", "values": [0.0, 0.0, 0.0, 1.0]},
    ])
    _run_pipeline_once()
    ds = _get_dataset(client, s["token"], "upsertmix")
    assert ds["row_count"] == 4, (
        f"row_count must end at 4 unique ids (a/b/c/d) after a partial-overlap "
        f"re-ingest; got {ds['row_count']}."
    )
