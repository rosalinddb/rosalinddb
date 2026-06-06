"""Unit coverage for `POST /vectors` under the `RB_RECALL` flag.

Hermetic (memory:// state + storage + queue; no Docker, no pgvector). Asserts
the two flag modes at the HTTP-handler level:

  - FLAG OFF (default): 202, a landing object is written, a `VALIDATE_DATASET`
    message is published, and NO recall connection is EVER attempted (the
    `psycopg2.connect`-raises trick from PR2 — if the off path touched the recall
    store the test would fail loudly).
  - FLAG ON: 200, body `{accepted, rejected, errors}` (no `job_id`), NO landing
    object, NO `VALIDATE_DATASET`, and the recall UPSERT was called with the
    accepted records. The real `recall_upsert_vectors` is replaced by a recorder so
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
    # Default both recall-tier env vars off; on-tests set them explicitly.
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


def test_flag_off_never_opens_recall_connection(env, monkeypatch):
    """The default path must NEVER connect to a recall store (PR2 trick)."""
    def _boom(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("flag-off path connected to a recall store")

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


# --- flag ON: synchronous recall write --------------------------------------


def _enable_recall(env, monkeypatch, recorder):
    """Flip the flag on at the handler import site and stub the recall UPSERT.

    `recall_enabled` / `recall_upsert_vectors` are imported by-name into the
    source_registry module, so patch the names ON THAT MODULE.
    """
    monkeypatch.setattr(env.main, "recall_enabled", lambda: True)
    monkeypatch.setattr(env.main, "recall_upsert_vectors", recorder)


def test_flag_on_returns_200_no_landing_no_publish(env, monkeypatch):
    """Flag on: 200, body has no job_id, no landing object, no VALIDATE_DATASET."""
    _drain_validate()
    captured = {}

    def _recorder(tenant, dataset, records):
        captured["args"] = (tenant, dataset, records)
        return len(records)

    _enable_recall(env, monkeypatch, _recorder)

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

    # The recall UPSERT got the accepted records, tenant-/dataset-scoped.
    tenant, dataset, records = captured["args"]
    assert dataset == "v"
    assert [rec["id"] for rec in records] == ["r0", "r1", "r2"]
    assert all(len(rec["values"]) == 4 for rec in records)


def test_flag_on_validation_unchanged(env, monkeypatch):
    """Flag on: per-line validation is identical — bad lines still rejected."""
    _enable_recall(env, monkeypatch, lambda t, d, recs: len(recs))
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
    _enable_recall(env, monkeypatch, lambda t, d, recs: len(recs))
    s = _signup(env.client)
    r = env.client.post(
        "/v1/datasets/missing/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=_BODY,
    )
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "dataset_not_found"


# --- non-finite / float4-overflow rejection (both flag modes, identical) ---
#
# `json.loads` accepts the JS literals `NaN`/`Infinity`/`-Infinity` (allow_nan
# default) so they reach validation as Python floats; an out-of-float4-range
# magnitude is finite Python but overflows pgvector's float4 column. Before the
# fix these passed source-registry validation: flag-ON then failed the all-or-
# nothing recall transaction (bare 500, every valid row rolled back), flag-OFF
# silently cast to float32 (Inf/NaN garbage in a consolidated shard, 202). Both modes
# must now reject them PER-LINE — not a 500, not a silent accept.

# (id, line-json) pairs whose `values` must be rejected per-line.
_BAD_VALUE_BODIES = {
    "nan": '{"id":"x","values":[NaN,0.2,0.3,0.4]}',
    "inf": '{"id":"x","values":[Infinity,0.2,0.3,0.4]}',
    "neg_inf": '{"id":"x","values":[-Infinity,0.2,0.3,0.4]}',
    "overflow": '{"id":"x","values":[1e40,0.2,0.3,0.4]}',
}


@pytest.mark.parametrize("kind", list(_BAD_VALUE_BODIES))
def test_flag_off_rejects_non_finite_and_overflow_per_line(env, kind):
    """Flag OFF: NaN/Inf/overflow are rejected per-line (not silently accepted)."""
    s = _signup(env.client, email=f"off-{kind}@example.com")
    env.client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4}
    )
    # One bad line plus one good line: the good one is accepted, the bad one is
    # rejected per-line (partial success), and the whole upload is NOT a 500.
    good = '{"id":"ok","values":[0.1,0.2,0.3,0.4]}'
    body = "\n".join([_BAD_VALUE_BODIES[kind], good])
    r = env.client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    out = r.json()
    assert out["accepted"] == 1 and out["rejected"] == 1
    reasons = [e["reason"] for e in out["errors"]]
    assert any("finite" in x for x in reasons), reasons


@pytest.mark.parametrize("kind", list(_BAD_VALUE_BODIES))
def test_flag_on_rejects_non_finite_and_overflow_per_line(env, monkeypatch, kind):
    """Flag ON: same rejection — per-line, never reaching the recall write as a 500.

    The recall UPSERT is stubbed to BLOW UP if it is ever handed a bad value, so
    the test proves rejection happens in validation (before the recall write), not
    via a recall-store transaction failure.
    """
    def _recorder(tenant, dataset, records):
        for rec in records:
            for v in rec["values"]:
                assert v == v and abs(v) != float("inf"), "bad value reached recall write"
        return len(records)

    _enable_recall(env, monkeypatch, _recorder)
    s = _signup(env.client, email=f"on-{kind}@example.com")
    env.client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4}
    )
    good = '{"id":"ok","values":[0.1,0.2,0.3,0.4]}'
    body = "\n".join([_BAD_VALUE_BODIES[kind], good])
    r = env.client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["accepted"] == 1 and out["rejected"] == 1
    reasons = [e["reason"] for e in out["errors"]]
    assert any("finite" in x for x in reasons), reasons


# --- recall-write failure stays in the v1 error envelope ---------------------


def test_flag_on_recall_write_failure_returns_503_envelope(env, monkeypatch):
    """A recall-store failure maps to a structured 503, not a raw 500.

    The whole batch is one transaction, so a failure persists nothing; the
    response must be the v1 envelope `{error:{code,message}}` with code
    `recall_write_failed`.
    """
    def _boom(tenant, dataset, records):
        raise RuntimeError("connection to recall store dropped")

    _enable_recall(env, monkeypatch, _boom)
    s = _signup(env.client, email="boom@example.com")
    env.client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4}
    )
    r = env.client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=_BODY,
    )
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["error"]["code"] == "recall_write_failed"
    assert "message" in body["error"]


# --- per-tenant recall cap → CONSOLIDATE enqueue --------------------------


def _drain_consolidate():
    """Pop every queued `CONSOLIDATE` message; return the list."""
    from adapters.queue.queue import consume

    msgs = []
    while True:
        m = consume("CONSOLIDATE", block=False)
        if not m:
            break
        msgs.append(m)
    return msgs


def test_cap_exceeded_enqueues_consolidate(env, monkeypatch):
    """Recall partition over `RB_RECALL_MAX_ROWS` → a CONSOLIDATE is enqueued."""
    _drain_consolidate()
    _enable_recall(env, monkeypatch, lambda t, d, recs: len(recs))
    # Tiny cap so 3 accepted rows trip it; the count is stubbed above the cap.
    monkeypatch.setenv("RB_RECALL_MAX_ROWS", "2")
    monkeypatch.setattr(env.main, "recall_partition_count", lambda t, d: 5)

    s = _signup(env.client, email="cap@example.com")
    env.client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4}
    )
    r = env.client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=_BODY,
    )
    assert r.status_code == 200, r.text
    msgs = _drain_consolidate()
    assert len(msgs) == 1, f"cap exceeded must enqueue one CONSOLIDATE: {msgs}"
    assert msgs[0]["dataset"] == "v"


def test_cap_not_exceeded_no_consolidate(env, monkeypatch):
    """Recall partition under the cap → NO CONSOLIDATE enqueued."""
    _drain_consolidate()
    _enable_recall(env, monkeypatch, lambda t, d, recs: len(recs))
    monkeypatch.setenv("RB_RECALL_MAX_ROWS", "2000")
    monkeypatch.setattr(env.main, "recall_partition_count", lambda t, d: 3)

    s = _signup(env.client, email="undercap@example.com")
    env.client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4}
    )
    r = env.client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=_BODY,
    )
    assert r.status_code == 200, r.text
    assert _drain_consolidate() == [], "under-cap write must not enqueue CONSOLIDATE"


def test_cap_check_failure_does_not_fail_write(env, monkeypatch):
    """A cap-count failure (after the durable write) must NOT turn 200 into error."""
    _drain_consolidate()
    _enable_recall(env, monkeypatch, lambda t, d, recs: len(recs))

    def _boom(t, d):
        raise RuntimeError("recall count query failed")

    monkeypatch.setattr(env.main, "recall_partition_count", _boom)

    s = _signup(env.client, email="capfail@example.com")
    env.client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4}
    )
    r = env.client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=_BODY,
    )
    assert r.status_code == 200, "cap-check failure must not fail the committed write"
    assert _drain_consolidate() == []


def test_flag_off_never_enqueues_consolidate(env, monkeypatch):
    """Flag OFF: the cap path is never reached → no CONSOLIDATE, no recall count."""
    _drain_consolidate()

    def _boom(t, d):  # pragma: no cover - must never be called
        raise AssertionError("cap check ran with the flag off")

    monkeypatch.setattr(env.main, "recall_partition_count", _boom)

    s = _signup(env.client, email="capoff@example.com")
    env.client.post(
        "/v1/datasets", headers=_auth(s["token"]), json={"name": "v", "dimension": 4}
    )
    r = env.client.post(
        "/v1/datasets/v/vectors",
        headers={**_auth(s["token"]), "Content-Type": "application/x-ndjson"},
        data=_BODY,
    )
    assert r.status_code == 202, r.text
    assert _drain_consolidate() == []
