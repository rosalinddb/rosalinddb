"""Tests for cross-tenant isolation at the dataset/shard layer.

Covers the isolation guarantees that:
  - Tenant A's datasets are invisible to tenant B in list/get/delete.
  - Tenant A cannot delete tenant B's dataset.
  - Two tenants can run validations concurrently without polluting each
    other's catalog or landing area.
"""
from __future__ import annotations

import importlib
import os
import time
from pathlib import Path

import pytest


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def two_tenants(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """A test client + a fresh in-memory state with two signed-up tenants.

    Landing + index shards live in real MinIO; only the FAISS shard cache is
    a local tmp dir.
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
    # Validator/builder cache LANDING_PREFIX at import time; reload so the
    # tmp_path-scoped value wins. Same for INDEXES_PREFIX in the builder.
    import services.validator_worker.run as validator
    importlib.reload(validator)
    import services.index_builder.run as builder
    importlib.reload(builder)

    from fastapi.testclient import TestClient
    client = TestClient(main_mod.app)
    a = client.post("/auth/signup", json={"email": "a@example.com", "password": "password123"}).json()
    b = client.post("/auth/signup", json={"email": "b@example.com", "password": "password123"}).json()
    return client, a, b, s3_landing_prefix


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_list_isolated_per_tenant(two_tenants):
    client, a, b, _ = two_tenants
    client.post("/v1/datasets", headers=_auth(a["token"]), json={"name": "a-private", "dimension": 4})
    rb = client.get("/v1/datasets", headers=_auth(b["token"]))
    names = [d["name"] for d in rb.json()["datasets"]]
    assert "a-private" not in names


def test_get_other_tenant_dataset_returns_404(two_tenants):
    client, a, b, _ = two_tenants
    client.post("/v1/datasets", headers=_auth(a["token"]), json={"name": "secret", "dimension": 4})
    r = client.get("/v1/datasets/secret", headers=_auth(b["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "dataset_not_found"


def test_delete_other_tenant_dataset_returns_404(two_tenants):
    client, a, b, _ = two_tenants
    client.post("/v1/datasets", headers=_auth(a["token"]), json={"name": "secret", "dimension": 4})
    r = client.delete("/v1/datasets/secret", headers=_auth(b["token"]))
    assert r.status_code == 404
    # A's dataset should still be there
    r2 = client.get("/v1/datasets/secret", headers=_auth(a["token"]))
    assert r2.status_code == 200


def test_same_dataset_name_per_tenant_allowed(two_tenants):
    client, a, b, _ = two_tenants
    # Both tenants can create a dataset named "products" — the PK is
    # (tenant_id, dataset_name), so this is permitted.
    ra = client.post("/v1/datasets", headers=_auth(a["token"]), json={"name": "products", "dimension": 4})
    rb = client.post("/v1/datasets", headers=_auth(b["token"]), json={"name": "products", "dimension": 4})
    assert ra.status_code == 201
    assert rb.status_code == 201


def test_validator_does_not_pollute_other_tenant_state(two_tenants):
    """Run the validator against two tenants' uploads and assert state stays scoped."""
    client, a, b, landing = two_tenants

    # Each tenant creates the same-named dataset
    client.post("/v1/datasets", headers=_auth(a["token"]), json={"name": "shared", "dimension": 4})
    client.post("/v1/datasets", headers=_auth(b["token"]), json={"name": "shared", "dimension": 4})

    # Upload distinct records to each tenant
    body_a = '\n'.join([
        '{"id":"a1","values":[1,0,0,0]}',
        '{"id":"a2","values":[0,1,0,0]}',
    ])
    body_b = '\n'.join([
        '{"id":"b1","values":[0,0,1,0]}',
        '{"id":"b2","values":[0,0,0,1]}',
        '{"id":"b3","values":[1,1,0,0]}',
    ])
    ra = client.post(
        "/v1/datasets/shared/vectors",
        headers={**_auth(a["token"]), "Content-Type": "application/x-ndjson"},
        data=body_a,
    )
    rb = client.post(
        "/v1/datasets/shared/vectors",
        headers={**_auth(b["token"]), "Content-Type": "application/x-ndjson"},
        data=body_b,
    )
    assert ra.status_code == 202 and rb.status_code == 202

    # Drain the validator and builder queues synchronously.
    from adapters.queue.queue import consume
    import services.validator_worker.run as validator
    import services.index_builder.run as builder
    pending = []
    while True:
        msg = consume("VALIDATE_DATASET", block=False)
        if not msg:
            break
        validator.process_uri(msg["dataset"], msg["tenant"], msg["uri"], msg.get("file_type"))
        pending.append(msg)
    # The HTTP-level test bypasses validator.main_loop, so we manually fire
    # the DATASET_READY message that main_loop would otherwise publish.
    for msg in pending:
        builder.run_once(msg["dataset"], msg["tenant"])

    # Each tenant's row_count is their own
    da = client.get("/v1/datasets/shared", headers=_auth(a["token"])).json()
    db = client.get("/v1/datasets/shared", headers=_auth(b["token"])).json()
    assert da["row_count"] == 2, da
    assert db["row_count"] == 3, db
    # Each tenant only sees their own shard catalog
    from adapters.state import state as state_mod
    sa = state_mod.list_shards(a["tenant"]["id"], "shared")
    sb = state_mod.list_shards(b["tenant"]["id"], "shared")
    assert len(sa) == 1 and sa[0]["vector_count"] == 2
    assert len(sb) == 1 and sb[0]["vector_count"] == 3
