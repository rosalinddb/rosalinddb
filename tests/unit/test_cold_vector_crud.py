"""Unit tests for the cold-tier vector get/list/delete-by-id surface.

These cover the `source_registry` HTTP handlers for:

  - `GET    /v1/datasets/{name}/vectors/{id}`   get-by-id
  - `GET    /v1/datasets/{name}/vectors`        list (filter + pagination)
  - `DELETE /v1/datasets/{name}/vectors/{id}`   tombstone (publish + 202)

All hermetic: `memory://` state, no FAISS, no real object storage. A shard is
faked by seeding `_MEM_SHARDS` with a catalog row and writing a `.meta.json`
sidecar to the `memory://` storage adapter (the exact bytes the builder
produces). This isolates the read/list/delete handlers from the FAISS build.
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

    Returns a small namespace: `client`, `state`, `main` (the reloaded
    source_registry module), and `write_sidecar` (seeds a shard catalog row
    plus its `.meta.json` sidecar so the read/list handlers have something to
    resolve).
    """
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("TENANT_PREFIX", "true")

    import adapters.storage.storage as storage_mod
    importlib.reload(storage_mod)
    # The memory:// storage adapter keeps a process-global dict; clear it so a
    # prior test's sidecar can never leak into this one.
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

    def write_sidecar(tenant, dataset, records, shard_uri=None):
        """Seed a shard catalog row + its `.meta.json` for (tenant, dataset).

        `records` is an iterable of `(id, metadata)`. Returns the shard_uri.
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
    # Flatten the tenant id to the top level for convenient test access.
    body["tenant_id"] = body["tenant"]["id"]
    return body


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _make_dataset(env, token, name="ds", dim=4):
    r = env.client.post("/v1/datasets", headers=_auth(token), json={"name": name, "dimension": dim})
    assert r.status_code == 201, r.text


# --- GET /v1/datasets/{name}/vectors/{id} ---------------------------------


def test_get_vector_hit(env):
    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    env.write(tenant, "ds", [("doc-1", {"title": "hello"}), ("doc-2", {"title": "world"})])

    r = env.client.get("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"id": "doc-1", "metadata": {"title": "hello"}}


def test_get_vector_miss_404(env):
    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    env.write(tenant, "ds", [("doc-1", {"title": "hello"})])

    r = env.client.get("/v1/datasets/ds/vectors/nope", headers=_auth(s["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_get_vector_no_shard_404(env):
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    # Dataset exists but no shard has been built yet.
    r = env.client.get("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_get_vector_missing_dataset_404(env):
    s = _signup(env.client)
    r = env.client.get("/v1/datasets/ghost/vectors/x", headers=_auth(s["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "dataset_not_found"


def test_get_vector_cross_tenant_404(env):
    a = _signup(env.client, email="a@example.com")
    b = _signup(env.client, email="b@example.com")
    _make_dataset(env, a["token"])
    env.write(a["tenant_id"], "ds", [("doc-1", {"title": "secret"})])

    # Tenant B has no "ds" dataset at all -> dataset_not_found, never leaks A's.
    r = env.client.get("/v1/datasets/ds/vectors/doc-1", headers=_auth(b["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "dataset_not_found"


# --- GET /v1/datasets/{name}/vectors (list) -------------------------------


def test_list_vectors_basic_stable_order(env):
    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    # Insert out of sorted order; the list must come back sorted by id.
    env.write(tenant, "ds", [("b", {}), ("a", {}), ("c", {})])

    r = env.client.get("/v1/datasets/ds/vectors", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [v["id"] for v in body["vectors"]] == ["a", "b", "c"]
    assert body["next_cursor"] is None


def test_list_vectors_empty_when_no_shard(env):
    s = _signup(env.client)
    _make_dataset(env, s["token"])
    r = env.client.get("/v1/datasets/ds/vectors", headers=_auth(s["token"]))
    assert r.status_code == 200
    assert r.json() == {"vectors": [], "next_cursor": None}


def test_list_vectors_filter(env):
    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    env.write(tenant, "ds", [
        ("a", {"kind": "fruit"}),
        ("b", {"kind": "veg"}),
        ("c", {"kind": "fruit"}),
    ])
    r = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"filter": json.dumps({"kind": "fruit"})},
    )
    assert r.status_code == 200, r.text
    assert [v["id"] for v in r.json()["vectors"]] == ["a", "c"]


def test_list_vectors_pagination(env):
    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    env.write(tenant, "ds", [(f"doc-{i:02d}", {}) for i in range(5)])

    # Page 1: limit 2.
    r1 = env.client.get(
        "/v1/datasets/ds/vectors", headers=_auth(s["token"]), params={"limit": 2}
    )
    assert r1.status_code == 200
    b1 = r1.json()
    assert [v["id"] for v in b1["vectors"]] == ["doc-00", "doc-01"]
    assert b1["next_cursor"] is not None

    # Page 2: follow the cursor.
    r2 = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"limit": 2, "cursor": b1["next_cursor"]},
    )
    b2 = r2.json()
    assert [v["id"] for v in b2["vectors"]] == ["doc-02", "doc-03"]
    assert b2["next_cursor"] is not None

    # Page 3: final partial page, no further cursor.
    r3 = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"limit": 2, "cursor": b2["next_cursor"]},
    )
    b3 = r3.json()
    assert [v["id"] for v in b3["vectors"]] == ["doc-04"]
    assert b3["next_cursor"] is None


def test_list_vectors_bad_cursor_400(env):
    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    env.write(tenant, "ds", [("a", {})])
    r = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"cursor": "!!!not-base64!!!"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_cursor"


def test_list_vectors_negative_offset_cursor_400(env):
    """A well-formed base64 cursor whose decoded offset is negative is rejected.

    The decode path explicitly refuses `offset < 0` (a hand-edited cursor)
    rather than treating it as offset 0 — that would silently restart
    pagination from the beginning.
    """
    import base64
    import json as _json

    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    env.write(tenant, "ds", [("a", {})])

    neg = base64.urlsafe_b64encode(_json.dumps({"o": -1}).encode("utf-8")).decode("ascii")
    r = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"cursor": neg},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_cursor"


def test_list_vectors_invalid_limit_400(env):
    """A non-integer `?limit` returns the v1 envelope (400 invalid_limit), not 422."""
    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    env.write(tenant, "ds", [("a", {})])

    r = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"limit": "abc"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_limit"


def test_list_vectors_limit_below_one_400(env):
    """A `limit < 1` is rejected as invalid_limit."""
    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    env.write(tenant, "ds", [("a", {})])

    r = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"limit": "0"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_limit"


def test_list_vectors_limit_cap_enforced(env):
    """`limit` above the 1000 cap is silently clamped to 1000, not rejected.

    Seed more than the cap so a request asking for a huge limit returns at
    most `_VECTORS_LIST_MAX_LIMIT` rows and exposes a `next_cursor` for the
    remainder.
    """
    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    cap = env.main._VECTORS_LIST_MAX_LIMIT
    env.write(tenant, "ds", [(f"doc-{i:05d}", {}) for i in range(cap + 5)])

    r = env.client.get(
        "/v1/datasets/ds/vectors",
        headers=_auth(s["token"]),
        params={"limit": str(cap * 10)},  # way above the cap
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["vectors"]) == cap
    # More rows remain than the cap returned, so a continuation cursor exists.
    assert body["next_cursor"] is not None


def test_list_vectors_cross_tenant_404(env):
    a = _signup(env.client, email="a@example.com")
    b = _signup(env.client, email="b@example.com")
    _make_dataset(env, a["token"])
    env.write(a["tenant_id"], "ds", [("a", {})])
    r = env.client.get("/v1/datasets/ds/vectors", headers=_auth(b["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "dataset_not_found"


# --- DELETE /v1/datasets/{name}/vectors/{id} ------------------------------


def test_delete_vector_publishes_and_202(env):
    from adapters.queue.queue import consume

    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    env.write(tenant, "ds", [("doc-1", {})])

    r = env.client.delete("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["job_id"].startswith("job_")

    # The dataset flips to `indexing` so a poll reflects the in-flight delete.
    ds = env.client.get("/v1/datasets/ds", headers=_auth(s["token"])).json()
    assert ds["status"] == "indexing"

    # A DELETE_VECTORS message was published carrying the id + tenant/dataset.
    msg = consume("DELETE_VECTORS", block=False)
    assert msg is not None
    assert msg["dataset"] == "ds"
    assert msg["tenant"] == tenant
    assert msg["id"] == "doc-1"
    assert msg["job_id"] == body["job_id"]


def test_delete_vector_no_shard_keeps_empty_status(env):
    """Delete on a never-ingested (`empty`) dataset must NOT flip the status.

    With no shard the delete is a guaranteed no-op against the cold tier, so
    the CP leaves the status untouched. Flipping to `indexing`/`indexed` here
    would falsely report a vacant dataset as built (`status=indexed,
    row_count=0`) and mask its real `empty` state.
    """
    s = _signup(env.client)
    _make_dataset(env, s["token"])

    # Freshly-created dataset, no ingest → status is `empty`, no shard.
    before = env.client.get("/v1/datasets/ds", headers=_auth(s["token"])).json()
    assert before["status"] == "empty"

    r = env.client.delete("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    # Still a 202 (the delete is accepted as a no-op).
    assert r.status_code == 202, r.text

    after = env.client.get("/v1/datasets/ds", headers=_auth(s["token"])).json()
    assert after["status"] == "empty"  # unchanged — NOT indexing/indexed.


def test_delete_vector_error_status_not_clobbered(env):
    """Delete on an `error` dataset (no shard) must NOT overwrite the status."""
    s = _signup(env.client)
    tenant = s["tenant_id"]
    _make_dataset(env, s["token"])
    # Simulate a failed pipeline: error status, still no shard.
    env.state.update_dataset_status(tenant, "ds", "error", error_message="boom")

    r = env.client.delete("/v1/datasets/ds/vectors/doc-1", headers=_auth(s["token"]))
    assert r.status_code == 202, r.text

    after = env.client.get("/v1/datasets/ds", headers=_auth(s["token"])).json()
    assert after["status"] == "error"  # preserved, not rewritten to indexing.


def test_delete_vector_missing_dataset_404(env):
    s = _signup(env.client)
    r = env.client.delete("/v1/datasets/ghost/vectors/x", headers=_auth(s["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "dataset_not_found"


def test_delete_vector_cross_tenant_404(env):
    a = _signup(env.client, email="a@example.com")
    b = _signup(env.client, email="b@example.com")
    _make_dataset(env, a["token"])
    env.write(a["tenant_id"], "ds", [("doc-1", {})])
    r = env.client.delete("/v1/datasets/ds/vectors/doc-1", headers=_auth(b["token"]))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "dataset_not_found"


# --- Auth -----------------------------------------------------------------


def test_cold_crud_rejects_missing_auth(env):
    assert env.client.get("/v1/datasets/ds/vectors/x").status_code == 401
    assert env.client.get("/v1/datasets/ds/vectors").status_code == 401
    assert env.client.delete("/v1/datasets/ds/vectors/x").status_code == 401
