"""Tests for the customer-facing dataset/vectors HTTP surface.

Covers `POST /v1/datasets`, `POST /v1/datasets/{name}/vectors`,
`GET /v1/datasets`, `GET /v1/datasets/{name}`, and
`DELETE /v1/datasets/{name}` per the v1 contract.

All tests run with `DATABASE_URL=memory://test` and a tmp landing dir so no
Postgres or shared filesystem is required.
"""
from __future__ import annotations

import importlib
import json
import os
import tempfile
import threading
import time

import pytest


os.environ["DATABASE_URL"] = "memory://test"
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


@pytest.fixture
def client(tmp_path, monkeypatch, s3_landing_prefix, s3_indexes_prefix):
    """Fresh FastAPI TestClient with reset in-memory state + MinIO landing.

    Each test gets its own unique MinIO landing prefix so successive uploads
    do not leak across tests. Modules are reloaded so the `_LANDING_PREFIX`
    module constant picks up the patched env var.
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
    # Reload validator/builder so their module-level LANDING_PREFIX picks
    # up the patched tmp_path value rather than a stale earlier-test path.
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


# --- POST /v1/datasets ----------------------------------------------------


def test_create_dataset_happy_path(client):
    s = _signup(client)
    r = client.post(
        "/v1/datasets",
        headers=_auth(s["token"]),
        json={"name": "products", "dimension": 4},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "products"
    assert body["dimension"] == 4
    assert body["status"] == "empty"
    assert body["row_count"] == 0
    assert body["created_at"]
    assert body["last_indexed_at"] is None
    assert body["error_message"] is None


def test_create_dataset_duplicate_409(client):
    s = _signup(client)
    payload = {"name": "dup", "dimension": 4}
    r1 = client.post("/v1/datasets", headers=_auth(s["token"]), json=payload)
    assert r1.status_code == 201
    r2 = client.post("/v1/datasets", headers=_auth(s["token"]), json=payload)
    assert r2.status_code == 409, r2.text
    assert r2.json()["error"]["code"] == "dataset_exists"


@pytest.mark.parametrize(
    "bad_name",
    [
        "Products",  # uppercase not allowed
        "with space",
        "x" * 65,  # too long
        "",
        "name!",  # special char
    ],
)
def test_create_dataset_invalid_name(client, bad_name):
    s = _signup(client)
    r = client.post(
        "/v1/datasets",
        headers=_auth(s["token"]),
        json={"name": bad_name, "dimension": 4},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_name"


@pytest.mark.parametrize("bad_dim", [0, -1, "foo", 1.5, None])
def test_create_dataset_invalid_dimension(client, bad_dim):
    s = _signup(client)
    r = client.post(
        "/v1/datasets",
        headers=_auth(s["token"]),
        json={"name": "ok-name", "dimension": bad_dim},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_dimension"


# --- GET /v1/datasets / GET /v1/datasets/{name} --------------------------


def test_list_returns_created_dataset(client):
    s = _signup(client)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "a", "dimension": 4})
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "b", "dimension": 8})
    r = client.get("/v1/datasets", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    names = [d["name"] for d in r.json()["datasets"]]
    assert names == ["a", "b"]


def test_list_filters_by_tenant(client):
    a = _signup(client, email="a@example.com")
    b = _signup(client, email="b@example.com")
    client.post("/v1/datasets", headers=_auth(a["token"]), json={"name": "a-only", "dimension": 4})
    client.post("/v1/datasets", headers=_auth(b["token"]), json={"name": "b-only", "dimension": 4})
    ra = client.get("/v1/datasets", headers=_auth(a["token"])).json()["datasets"]
    rb = client.get("/v1/datasets", headers=_auth(b["token"])).json()["datasets"]
    assert {d["name"] for d in ra} == {"a-only"}
    assert {d["name"] for d in rb} == {"b-only"}


def test_get_returns_dataset(client):
    s = _signup(client)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "products", "dimension": 4})
    r = client.get("/v1/datasets/products", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "products"


def test_get_cross_tenant_404(client):
    a = _signup(client, email="a@example.com")
    b = _signup(client, email="b@example.com")
    client.post("/v1/datasets", headers=_auth(a["token"]), json={"name": "a-only", "dimension": 4})
    r = client.get("/v1/datasets/a-only", headers=_auth(b["token"]))
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "dataset_not_found"


# --- DELETE /v1/datasets/{name} -------------------------------------------


def test_delete_soft_removes(client):
    s = _signup(client)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "to-delete", "dimension": 4})
    r = client.delete("/v1/datasets/to-delete", headers=_auth(s["token"]))
    assert r.status_code == 204, r.text
    r = client.get("/v1/datasets/to-delete", headers=_auth(s["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "dataset_not_found"
    # list also excludes it
    r = client.get("/v1/datasets", headers=_auth(s["token"]))
    assert r.json()["datasets"] == []


# --- POST /v1/datasets/{name}/vectors -------------------------------------


def _run_pipeline_once():
    """Drain VALIDATE_DATASET and run the builder synchronously.

    The HTTP-level test bypasses validator.main_loop, so we manually
    forward each validated dataset into the builder (the loop would
    otherwise publish a DATASET_READY message that main_loop consumes).
    Keeps tests deterministic without spawning worker threads.
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


def test_post_vectors_happy_path(client):
    s = _signup(client)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4})
    body = "\n".join(json.dumps({"id": f"r{i}", "values": [0.1 * i, 0.2, 0.3, 0.4]}) for i in range(3))
    r = client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    out = r.json()
    assert out["accepted"] == 3
    assert out["rejected"] == 0
    assert out["errors"] == []
    assert out["job_id"].startswith("job_")

    # Drain the queue: validator + builder pick up the message.
    _run_pipeline_once()

    r = client.get("/v1/datasets/v", headers=_auth(s["token"]))
    assert r.status_code == 200
    ds = r.json()
    assert ds["row_count"] == 3, ds
    # status should have moved past empty
    assert ds["status"] in ("indexed", "indexing"), ds


def test_post_vectors_nonexistent_404(client):
    s = _signup(client)
    r = client.post(
        "/v1/datasets/does-not-exist/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data='{"id":"a","values":[1,2,3,4]}\n',
    )
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "dataset_not_found"


def test_post_vectors_malformed_ndjson_partial(client):
    """Bad lines are reported in `errors` but good lines are accepted."""
    s = _signup(client)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4})
    body = "\n".join([
        '{"id":"ok1","values":[0.1,0.2,0.3,0.4]}',
        '{not valid json',  # bad
        '{"id":"ok2","values":[0.5,0.6,0.7,0.8]}',
        '{"id":"bad_dim","values":[1,2,3]}',
    ])
    r = client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    out = r.json()
    assert out["accepted"] == 2
    assert out["rejected"] == 2
    reasons = [e["reason"] for e in out["errors"]]
    assert any("invalid json" in r for r in reasons)
    assert any("dimension mismatch" in r for r in reasons)


def test_post_vectors_oversized_413(client, monkeypatch):
    s = _signup(client)
    client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "big", "dimension": 4})
    # Build a body larger than the cap (10 MiB default). Use a smaller cap
    # for the test to avoid actually allocating 10 MiB.
    import services.source_registry.main as main_mod
    monkeypatch.setattr(main_mod, "_INGEST_MAX_BYTES", 1024)
    big = ("a" * 2000).encode("utf-8")  # 2 KiB > 1 KiB cap
    r = client.post(
        "/v1/datasets/big/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=big,
    )
    assert r.status_code == 413, r.text
    assert r.json()["error"]["code"] == "payload_too_large"


# --- Auth -----------------------------------------------------------------


def test_endpoints_reject_missing_auth(client):
    # No headers
    assert client.get("/v1/datasets").status_code == 401
    assert client.post("/v1/datasets", json={"name": "x", "dimension": 4}).status_code == 401
    assert client.get("/v1/datasets/x").status_code == 401
    assert client.delete("/v1/datasets/x").status_code == 401
    r = client.post(
        "/v1/datasets/x/vectors",
        headers={"Content-Type": "application/x-ndjson"},
        data='{"id":"a","values":[1,2,3,4]}\n',
    )
    assert r.status_code == 401


def test_endpoints_reject_bad_auth(client):
    headers = {"Authorization": "Bearer not-a-real-token"}
    assert client.get("/v1/datasets", headers=headers).status_code == 401
