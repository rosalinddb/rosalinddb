"""Unit coverage for `POST /vectors` under the `RB_DELTA_TIER` flag.

Hermetic (memory:// state + storage + queue; no Docker, no pgvector). Asserts
the two flag modes at the HTTP-handler level:

  - FLAG OFF (default): 202, a landing object is written, a `VALIDATE_DATASET`
    message is published, and NO hot connection is EVER attempted (the
    `psycopg2.connect`-raises trick from PR2 — if the off path touched the hot
    store the test would fail loudly).
  - FLAG ON: 200, body `{accepted, rejected, errors}` (no `job_id`), NO landing
    object, NO `VALIDATE_DATASET`, and the hot UPSERT was called with the
    accepted records. The real `hot_upsert_vectors` is replaced by a recorder so
    the test needs no database.
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
    """Fresh TestClient with reset memory:// state, storage, and queue."""
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("TENANT_PREFIX", "true")
    # Default both delta-tier env vars off; on-tests set them explicitly.
    monkeypatch.delenv("RB_DELTA_TIER", raising=False)
    monkeypatch.delenv("RB_HOT_DSN", raising=False)

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

    import services.auth.jwt_utils as jwt_utils
    importlib.reload(jwt_utils)
    import services.auth.auth as auth_mod
    importlib.reload(auth_mod)
    import services.source_registry.main as main_mod
    importlib.reload(main_mod)

    from fastapi.testclient import TestClient

    class _Env:
        client = TestClient(main_mod.app)
        state = state_mod
        main = main_mod
        storage = storage_mod

    return _Env()


def _signup(client, email="alice@example.com"):
    r = client.post("/auth/signup", json={"email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    return r.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _drain_validate():
    """Pop every queued `VALIDATE_DATASET` message; return the list."""
    from adapters.queue.queue import consume

    msgs = []
    while True:
        m = consume("VALIDATE_DATASET", block=False)
        if not m:
            break
        msgs.append(m)
    return msgs


_BODY = "\n".join(
    json.dumps({"id": f"r{i}", "values": [0.1 * i, 0.2, 0.3, 0.4]}) for i in range(3)
)


# --- flag OFF (default): byte-identical to today --------------------------


def test_flag_off_returns_202_lands_and_publishes(env):
    """Default path: 202, landing object written, VALIDATE_DATASET published."""
    _drain_validate()  # clear anything a prior test left
    s = _signup(env.client)
    env.client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4}
    )
    r = env.client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=_BODY,
    )
    assert r.status_code == 202, r.text
    out = r.json()
    assert out["accepted"] == 3 and out["rejected"] == 0
    assert out["job_id"].startswith("job_")

    # A landing object was written (the validator's input).
    landing = [k for k in env.storage._MEM_OBJECTS if "/landing/" in k and k.endswith(".jsonl")]
    assert landing, "flag-off path must write a landing object"

    # And a VALIDATE_DATASET was published.
    msgs = _drain_validate()
    assert len(msgs) == 1 and msgs[0]["dataset"] == "v"


def test_flag_off_never_opens_hot_connection(env, monkeypatch):
    """The default path must NEVER connect to a hot store (PR2 trick)."""
    def _boom(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("flag-off path connected to a hot store")

    monkeypatch.setattr(env.state.psycopg2, "connect", _boom)

    s = _signup(env.client)
    env.client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4}
    )
    r = env.client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=_BODY,
    )
    assert r.status_code == 202, r.text


# --- flag ON: synchronous hot write --------------------------------------


def _enable_delta(env, monkeypatch, recorder):
    """Flip the flag on at the handler import site and stub the hot UPSERT.

    `delta_tier_enabled` / `hot_upsert_vectors` are imported by-name into the
    source_registry module, so patch the names ON THAT MODULE.
    """
    monkeypatch.setattr(env.main, "delta_tier_enabled", lambda: True)
    monkeypatch.setattr(env.main, "hot_upsert_vectors", recorder)


def test_flag_on_returns_200_no_landing_no_publish(env, monkeypatch):
    """Flag on: 200, body has no job_id, no landing object, no VALIDATE_DATASET."""
    _drain_validate()
    captured = {}

    def _recorder(tenant, dataset, records):
        captured["args"] = (tenant, dataset, records)
        return len(records)

    _enable_delta(env, monkeypatch, _recorder)

    s = _signup(env.client)
    env.client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4}
    )
    before = set(env.storage._MEM_OBJECTS)
    r = env.client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=_BODY,
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["accepted"] == 3 and out["rejected"] == 0 and out["errors"] == []
    assert "job_id" not in out, "flag-on body omits job_id (no async job)"

    # No landing object written, no VALIDATE_DATASET published.
    new_objs = set(env.storage._MEM_OBJECTS) - before
    assert not any("/landing/" in k for k in new_objs), "flag-on must not land"
    assert _drain_validate() == [], "flag-on must not publish VALIDATE_DATASET"

    # The hot UPSERT got the accepted records, tenant-/dataset-scoped.
    tenant, dataset, records = captured["args"]
    assert dataset == "v"
    assert [rec["id"] for rec in records] == ["r0", "r1", "r2"]
    assert all(len(rec["values"]) == 4 for rec in records)


def test_flag_on_validation_unchanged(env, monkeypatch):
    """Flag on: per-line validation is identical — bad lines still rejected."""
    _enable_delta(env, monkeypatch, lambda t, d, recs: len(recs))
    s = _signup(env.client)
    env.client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4}
    )
    body = "\n".join([
        '{"id":"ok1","values":[0.1,0.2,0.3,0.4]}',
        "{not valid json",
        '{"id":"bad_dim","values":[1,2,3]}',
    ])
    r = env.client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["accepted"] == 1 and out["rejected"] == 2
    reasons = [e["reason"] for e in out["errors"]]
    assert any("invalid json" in x for x in reasons)
    assert any("dimension mismatch" in x for x in reasons)


def test_flag_on_nonexistent_dataset_404(env, monkeypatch):
    """Flag on does not change the 404 contract for a missing dataset."""
    _enable_delta(env, monkeypatch, lambda t, d, recs: len(recs))
    s = _signup(env.client)
    r = env.client.post(
        "/v1/datasets/missing/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=_BODY,
    )
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "dataset_not_found"
