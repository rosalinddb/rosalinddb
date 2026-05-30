"""Tests for the customer-facing query surface.

Covers `POST /v1/query` and `GET /v1/query/status/{job_id}` per the v1
contract. The core assertion is the id/metadata bridge: FAISS only stores
SHA1-derived int64 hashes, so without the shard sidecar a query could only
return opaque integers. These tests prove the *original uploaded string ids*
and their `metadata` round-trip back through `POST /v1/query`.

All tests run with `DATABASE_URL=memory://test` and tmp landing/index dirs
so no Postgres or shared filesystem is required. The full ingest pipeline
(validator + builder) is driven inline to produce a real FAISS shard.
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
    """Fresh TestClient over the source_registry app + mounted v1_query router.

    Mounting the shared `v1_query` router on the source_registry app mirrors
    the dev/prod single-origin setup and lets one TestClient drive both the
    dataset setup endpoints and the query endpoints. State is reset and the
    pipeline modules reloaded so module-level env reads pick up the per-test
    MinIO prefixes. Landing + index shards live in real MinIO; only the FAISS
    shard cache (`CACHE_DIR`) is a local tmp dir.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(cache))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")  # tiny test fixtures use flat

    # The in-proc queue adapter holds process-global queues; drain any
    # messages left behind by a prior test so a fresh test never consumes
    # stale `RUN_EPHEMERAL_QUERY` / `RESULT_READY` / `VALIDATE_DATASET` work.
    from adapters.queue.queue import consume as _consume
    for _topic in ("VALIDATE_DATASET", "DATASET_READY", "RUN_EPHEMERAL_QUERY", "RESULT_READY"):
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
    # Reset the per-process shard cache / result store.
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
            process_uri(msg["dataset"], msg["tenant"], msg["uri"], msg.get("file_type"))
            pending.append(msg)
        except Exception:
            pass
    for msg in pending:
        run_once(msg["dataset"], msg["tenant"])


def _upload(client, token, name, records):
    """Upload NDJSON records and drive the pipeline to `indexed`."""
    body = "\n".join(json.dumps(r) for r in records)
    r = client.post(
        f"/v1/datasets/{name}/vectors",
        headers={**_auth(token), "Content-Type": "application/x-ndjson"},
        data=body,
    )
    assert r.status_code == 202, r.text
    _run_pipeline_once()


def _make_indexed_dataset(client, token, name="test", dimension=4, records=None):
    """Create + populate + index a dataset, returning the uploaded records."""
    r = client.post("/v1/datasets", headers=_auth(token), json={"name": name, "dimension": dimension})
    assert r.status_code == 201, r.text
    if records is None:
        records = [
            {"id": f"doc-{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"title": f"t{i}"}}
            for i in range(10)
        ]
    _upload(client, token, name, records)
    ds = client.get(f"/v1/datasets/{name}", headers=_auth(token)).json()
    assert ds["status"] == "indexed", ds
    return records


# --- happy path -----------------------------------------------------------


def test_query_indexed_dataset_returns_matches(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] in ("hot", "cold")
    assert isinstance(body["latency_ms"], int)
    assert len(body["matches"]) > 0
    for m in body["matches"]:
        assert isinstance(m["id"], str)
        assert isinstance(m["score"], (int, float))
        assert isinstance(m["metadata"], dict)


def test_returned_ids_are_original_uploaded_strings(client):
    """The whole point: matches carry the customer's string ids, not hashes."""
    s = _signup(client)
    records = _make_indexed_dataset(client, s["token"])
    uploaded_ids = {r["id"] for r in records}
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [3.0, 0.0, 0.0, 0.0], "top_k": 10},
    )
    assert r.status_code == 200, r.text
    matched_ids = {m["id"] for m in r.json()["matches"]}
    assert matched_ids
    # Every returned id must be one of the original uploaded string ids.
    assert matched_ids.issubset(uploaded_ids), matched_ids
    # And it must literally look like our `doc-N` ids, not a bare integer.
    for mid in matched_ids:
        assert mid.startswith("doc-"), mid


def test_metadata_round_trips(client):
    """A metadata object uploaded with a record comes back on query."""
    s = _signup(client)
    records = [
        {"id": "only", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"title": "x"}},
        {"id": "other", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"title": "y"}},
    ]
    _make_indexed_dataset(client, s["token"], records=records)
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1},
    )
    assert r.status_code == 200, r.text
    matches = r.json()["matches"]
    assert matches[0]["id"] == "only"
    assert matches[0]["metadata"] == {"title": "x"}


def test_top_k_respected(client):
    s = _signup(client)
    records = [
        {"id": f"v{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {}}
        for i in range(20)
    ]
    _make_indexed_dataset(client, s["token"], records=records)
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 5},
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["matches"]) == 5


def test_filter_applies_and_of_equals(client):
    """`filter` is honoured: AND-of-equals over record metadata.

    The default fixture's metadata is `{"title": "t<N>"}`; filtering on
    `title=t0` must return exactly the single matching record. Detailed
    filter coverage lives in `test_query_filter.py`.
    """
    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={
            "dataset": "test",
            "vector": [0.0, 0.0, 0.0, 0.0],
            "top_k": 10,
            "filter": {"title": "t0"},
        },
    )
    assert r.status_code == 200, r.text
    matches = r.json()["matches"]
    assert [m["id"] for m in matches] == ["doc-0"]


def test_filter_empty_object_unchanged(client):
    """An empty `filter` object leaves results unchanged, not an error."""
    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={
            "dataset": "test",
            "vector": [0.0, 0.0, 0.0, 0.0],
            "top_k": 3,
            "filter": {},
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["matches"]) == 3


# --- IVF nprobe is set and tunable ----------------------------------------


def test_nprobe_applied_to_ivf_shard(client, monkeypatch):
    """An IVF shard search runs with the configured `nprobe`, not the
    FAISS default of 1. The dataset is built IVFFlat (>= 64 vectors) and
    `index.search` is wrapped to capture the per-search `SearchParametersIVF`
    `nprobe` — the index itself is NEVER mutated (no cross-query race).

    IVFFlat is an IVF index, so the `nprobe` query-path machinery
    (`SearchParametersIVF`, the shard cache) works unchanged from IVF+PQ —
    that is exactly what this test confirms."""
    import faiss
    import services.query_api.v1_query as v1_query

    monkeypatch.setenv("RB_QUERY_NPROBE", "12")
    monkeypatch.setenv("INDEX_TYPE", "ivfflat")
    importlib.reload(__import__("services.index_builder.run", fromlist=["run"]))

    s = _signup(client)
    # 300 dim-8 vectors → above the IVF training floor (64) → ivfflat.
    records = [
        {"id": f"v{i}", "values": [float(i % 16)] + [float(i)] + [0.0] * 6,
         "metadata": {}}
        for i in range(300)
    ]
    _make_indexed_dataset(client, s["token"], name="ivf", dimension=8, records=records)

    seen = {}
    real_search = faiss.IndexIDMap2.search

    def _capturing_search(self, x, k, *a, **kw):
        # `nprobe` is delivered via the per-search `params` kwarg, not by
        # mutating the index — capture it from there.
        params = kw.get("params")
        seen["nprobe"] = getattr(params, "nprobe", None) if params is not None else None
        # The shared index object must remain at the FAISS default.
        try:
            seen["index_nprobe"] = faiss.extract_index_ivf(self).nprobe
        except Exception:  # noqa: BLE001
            seen["index_nprobe"] = None
        return real_search(self, x, k, *a, **kw)

    monkeypatch.setattr(faiss.IndexIDMap2, "search", _capturing_search)
    v1_query.cache_clear()

    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "ivf", "vector": [0.0] * 8, "top_k": 5},
    )
    assert r.status_code == 200, r.text
    assert seen.get("nprobe") == 12, seen
    # The shared cached index is never mutated — still the FAISS default.
    assert seen.get("index_nprobe") == 1, seen

    # A per-query `nprobe` overrides the server default for that query only.
    r2 = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "ivf", "vector": [0.0] * 8, "top_k": 5, "nprobe": 3},
    )
    assert r2.status_code == 200, r2.text
    assert seen.get("nprobe") == 3, seen
    assert seen.get("index_nprobe") == 1, seen
    # A bad `nprobe` is a 400, not a silent default.
    bad = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "ivf", "vector": [0.0] * 8, "nprobe": 0},
    )
    assert bad.status_code == 400, bad.text
    # An absurdly large `nprobe` is rejected (clamped ceiling), not unbounded.
    huge = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "ivf", "vector": [0.0] * 8, "nprobe": v1_query.MAX_NPROBE + 1},
    )
    assert huge.status_code == 400, huge.text
    assert huge.json()["error"]["code"] == "nprobe_out_of_range"


# --- in-memory shard cache ------------------------------------------------


def test_cache_hit_reuses_index_object(client, monkeypatch):
    """The second query for a shard reuses the cached index object.

    A genuine cold load deserialises the FAISS index; a subsequent query must
    NOT re-deserialise — `faiss.read_index` is monkeypatched to count calls
    and assert it fires exactly once across two queries against one shard.
    """
    import faiss
    import services.query_api.v1_query as v1_query

    s = _signup(client)
    _make_indexed_dataset(client, s["token"])

    v1_query.cache_clear()
    real_read = faiss.read_index
    calls = {"n": 0}

    def _counting_read(path, *a, **kw):
        calls["n"] += 1
        return real_read(path, *a, **kw)

    monkeypatch.setattr(v1_query.faiss, "read_index", _counting_read)

    body = {"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 5}
    r1 = client.post("/v1/query", headers=_auth(s["token"]), json=body)
    assert r1.status_code == 200, r1.text
    assert r1.json()["mode"] == "cold"  # first query: genuine cold load

    r2 = client.post("/v1/query", headers=_auth(s["token"]), json=body)
    assert r2.status_code == 200, r2.text
    assert r2.json()["mode"] == "hot"  # second query: cache hit

    # The index was deserialised exactly once for the two queries.
    assert calls["n"] == 1, f"index re-deserialised {calls['n']} times"


def test_swept_shard_is_evicted_from_cache(client):
    """When the builder sweeps a superseded shard the query cache entry
    for it is dropped, so a stale index can never be served."""
    import services.query_api.v1_query as v1_query

    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    tenant_id = _signup_tenant_id(client, s)

    # Warm the cache with the current (soon-to-be-superseded) shard.
    body = {"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 3}
    client.post("/v1/query", headers=_auth(s["token"]), json=body)
    from adapters.state.state import list_shards
    old_shard_id = list_shards(tenant_id, "test")[0]["id"]
    assert v1_query._cache_get(old_shard_id) is not None

    # An incremental ingest writes a new shard; SHARD_KEEP=2 keeps it cached
    # until enough builds supersede it. Force the sweep by lowering keep to 1.
    # Restore `_SHARDS_TO_KEEP` afterwards so test ordering can't be broken by
    # this module-global mutation leaking into other tests.
    import services.index_builder.run as builder
    saved_keep = builder._SHARDS_TO_KEEP
    builder._SHARDS_TO_KEEP = 1
    try:
        _upload(client, s["token"], "test", [
            {"id": "new-1", "values": [9.0, 0.0, 0.0, 0.0], "metadata": {"title": "n"}},
        ])
        # The superseded-shard sweep must have evicted the old shard's entry.
        assert v1_query._cache_get(old_shard_id) is None
    finally:
        builder._SHARDS_TO_KEEP = saved_keep


def test_query_mode_label_reflects_cache(client):
    """The `mode` label is honest: first query `cold`, repeat `hot`."""
    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    body = {"dataset": "test", "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 2}
    first = client.post("/v1/query", headers=_auth(s["token"]), json=body).json()
    second = client.post("/v1/query", headers=_auth(s["token"]), json=body).json()
    assert first["mode"] == "cold"
    assert second["mode"] == "hot"


# --- error cases ----------------------------------------------------------


def test_query_cross_tenant_dataset_404(client):
    a = _signup(client, email="a@example.com")
    b = _signup(client, email="b@example.com")
    _make_indexed_dataset(client, a["token"], name="a-only")
    r = client.post(
        "/v1/query",
        headers=_auth(b["token"]),
        json={"dataset": "a-only", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "dataset_not_found"


def test_query_nonexistent_dataset_404(client):
    s = _signup(client)
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "does-not-exist", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "dataset_not_found"


def test_dimension_mismatch_400(client):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0]},  # 3 != 4
    )
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["code"] == "dimension_mismatch"
    assert err["details"]["expected"] == 4
    assert err["details"]["got"] == 3


@pytest.mark.parametrize("bad_top_k", [0, 1001, -5])
def test_top_k_out_of_range_400(client, bad_top_k):
    s = _signup(client)
    _make_indexed_dataset(client, s["token"])
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0], "top_k": bad_top_k},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "top_k_out_of_range"


def test_query_rejects_missing_auth(client):
    r = client.post("/v1/query", json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]})
    assert r.status_code == 401, r.text


def test_query_rejects_bad_auth(client):
    r = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer not-a-real-token"},
        json={"dataset": "test", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 401, r.text


# --- no-shard / ephemeral graceful path -----------------------------------


def test_query_dataset_with_no_shard_is_graceful(client):
    """A dataset created but never indexed must not 500.

    Falls back to the ephemeral path: empty matches + a job_id to poll.
    """
    s = _signup(client)
    r = client.post("/v1/datasets", headers=_auth(s["token"]), json={"name": "empty", "dimension": 4})
    assert r.status_code == 201, r.text
    r = client.post(
        "/v1/query",
        headers=_auth(s["token"]),
        json={"dataset": "empty", "vector": [0.0, 0.0, 0.0, 0.0]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "ephemeral"
    assert body["matches"] == []
    assert body["job_id"].startswith("job_")


def test_query_status_unknown_job_not_ready(client):
    s = _signup(client)
    r = client.get("/v1/query/status/job_does_not_exist", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    assert r.json() == {"ready": False}


def test_ephemeral_result_carries_original_ids(client):
    """End-to-end ephemeral path: enqueue, run inline, poll, assert ids.

    A dataset whose only shard is removed from the catalog forces the
    ephemeral fallback. Driving the ephemeral runner inline and draining
    RESULT_READY proves the runner's sidecar lookup also yields string ids.
    """
    import services.query_api.v1_query as v1_query
    from services.ephemeral_runner.run import handle as ephemeral_handle
    from adapters.queue.queue import consume, publish

    s = _signup(client)
    _make_indexed_dataset(client, s["token"])

    # Enqueue an ephemeral job directly against the indexed dataset.
    job_id = "job_ephemeral_test"
    publish(
        "RUN_EPHEMERAL_QUERY",
        {
            "dataset": "test",
            "tenant": _signup_tenant_id(client, s),
            "vector": [3.0, 0.0, 0.0, 0.0],
            "top_k": 3,
            "correlation_id": job_id,
            "reply_to": "RESULT_READY",
        },
    )
    # Run the ephemeral runner's consume+handle inline.
    msg = consume("RUN_EPHEMERAL_QUERY", block=False)
    assert msg is not None
    matches = ephemeral_handle(msg["tenant"], msg["dataset"], msg["vector"], msg["top_k"])
    publish(
        "RESULT_READY",
        {"correlation_id": msg["correlation_id"], "matches": matches, "latency_ms": 1},
    )
    v1_query.drain_result_queue_once()

    r = client.get(f"/v1/query/status/{job_id}", headers=_auth(s["token"]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ready"] is True
    assert len(body["matches"]) > 0
    for m in body["matches"]:
        assert isinstance(m["id"], str) and m["id"].startswith("doc-")
        assert isinstance(m["metadata"], dict)


def _signup_tenant_id(client, signup_body):
    """Resolve the tenant_id for a signed-up user via /auth/me."""
    r = client.get("/auth/me", headers=_auth(signup_body["token"]))
    return r.json()["tenant"]["id"]
