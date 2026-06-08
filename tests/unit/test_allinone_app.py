"""Hermetic tests for the single-process all-in-one app.

Drives the all-in-one composition (`services.allinone.run`) via a FastAPI
TestClient against memory:// catalog+storage + the embedded numpy recall
backend + the in-process queue — NO docker, NO Postgres, NO Redis, NO pgvector.
Proves the eval-defaults contract end to end: boot+healthz, the in-process query
router (not the CP->DP proxy), read-your-writes, the recall/cold union, delete
read-your-deletes, and the in-process consolidation fold.

Auth is turned OFF (`RB_REQUIRE_AUTH=false`) so every request resolves to the
bootstrap "default" tenant — single-tenant mode, no JWT plumbing in the test.
"""
from __future__ import annotations

import importlib
import json
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A TestClient over the all-in-one app with the eval defaults active.

    Auth off (every request -> "default" tenant); embedded recall on; memory://
    everything; a writable CACHE_DIR. The allinone module sets most of these via
    `os.environ.setdefault`; we set them explicitly here too so the values are
    pinned regardless of import order, then reload the module so the app + the
    captured env are consistent.
    """
    monkeypatch.setenv("RB_REQUIRE_AUTH", "false")
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("STAGING_PREFIX", "memory://rosalinddb/staging")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("RB_RECALL", "true")
    monkeypatch.setenv("RB_RECALL_BACKEND", "memory")
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))

    # Reset the embedded memtable so a partition from another test does not leak.
    import adapters.recall.memtable as memtable
    importlib.reload(memtable)
    memtable._reset()

    import services.allinone.run as run_mod
    importlib.reload(run_mod)

    # v1_query captures CACHE_DIR at IMPORT time; in the full suite it may have
    # been imported earlier with a stale/unwritable dir. Pin it to a writable
    # per-test dir so the in-process FAISS load of a memory:// shard works (the
    # established pattern other query unit tests use). No-op in a fresh process,
    # where the allinone env default already points CACHE_DIR at a writable tmp.
    import services.query_api.v1_query as v1q
    monkeypatch.setattr(v1q, "CACHE_DIR", str(tmp_path / "cache"))

    with TestClient(run_mod.app) as c:  # context triggers the startup hook
        c._run_mod = run_mod  # stash for tests that need the module
        yield c
    memtable._reset()


def _create_dataset(client, name, dim=4):
    r = client.post("/v1/datasets", json={"name": name, "dimension": dim})
    assert r.status_code == 201, r.text
    return r


def _ingest(client, name, records):
    body = "\n".join(json.dumps(rec) for rec in records).encode("utf-8")
    return client.post(f"/v1/datasets/{name}/vectors", content=body)


# --- boot + composition ----------------------------------------------------


def test_boot_and_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_router_is_in_process_not_proxy(client):
    routes = {r.path for r in client.app.routes}
    # the in-process query router is mounted
    assert "/v1/query" in routes
    assert "/v1/query/status/{job_id}" in routes
    # the CP->DP proxy module is NOT what answers /v1/query here: assert the
    # actual endpoint function comes from v1_query, not query_proxy.
    query_routes = [r for r in client.app.routes if getattr(r, "path", None) == "/v1/query"]
    assert query_routes
    endpoint_modules = {r.endpoint.__module__ for r in query_routes}
    assert any("v1_query" in m for m in endpoint_modules)
    assert not any("query_proxy" in m for m in endpoint_modules)


def test_no_external_infra_required(client):
    # in-process queue (no Redis), memory:// state, recall on with no DSN.
    from adapters.queue import queue as queue_mod
    from adapters.state import state as state_mod
    from adapters import config, recall as recall_pkg

    assert queue_mod._redis is None  # in-process queue.Queue fallback
    assert state_mod._MEMORY_MODE is True  # memory:// catalog
    assert config.recall() is True
    assert config.recall_dsn() is None  # no recall DSN
    assert recall_pkg._use_memory_backend() is True  # embedded numpy backend


# --- read-your-writes ------------------------------------------------------


def test_ingest_then_immediate_query_ryw(client):
    _create_dataset(client, "ryw", dim=4)
    r = _ingest(client, "ryw", [{"id": "v1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"k": "a"}}])
    # recall on => synchronous 200 (not 202)
    assert r.status_code == 200, r.text
    assert r.json()["accepted"] == 1

    q = client.post("/v1/query", json={"dataset": "ryw", "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 5})
    assert q.status_code == 200, q.text
    body = q.json()
    ids = {m["id"] for m in body["matches"]}
    assert "v1" in ids  # read-your-writes out of the box
    # only recall could answer (no consolidated shard yet) -> mode "recall"
    assert body["mode"] == "recall"


def test_delete_read_your_deletes(client):
    _create_dataset(client, "del", dim=4)
    assert _ingest(client, "del", [
        {"id": "keep", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
        {"id": "gone", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
    ]).status_code == 200

    d = client.delete("/v1/datasets/del/vectors/gone")
    assert d.status_code in (200, 202, 204), d.text

    # immediate query no longer returns the deleted id
    q = client.post("/v1/query", json={"dataset": "del", "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 5})
    assert q.status_code == 200, q.text
    ids = {m["id"] for m in q.json()["matches"]}
    assert "gone" not in ids
    assert "keep" in ids

    # GET the deleted vector -> 404
    g = client.get("/v1/datasets/del/vectors/gone")
    assert g.status_code == 404, g.text


def test_union_recall_authoritative(client):
    """A re-upserted id's fresh recall copy beats its stale (consolidated) twin.

    Without a full fold here we still prove the recall-wins suppression: ingest a
    vector, then re-ingest the SAME id with different values; the query returns
    the NEW values' ranking (recall authoritative), exactly once (no stale twin).
    """
    _create_dataset(client, "uni", dim=4)
    assert _ingest(client, "uni", [{"id": "x", "values": [0.0, 0.0, 0.0, 9.0], "metadata": {"v": 1}}]).status_code == 200
    # re-upsert same id, now close to the query point
    assert _ingest(client, "uni", [{"id": "x", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": 2}}]).status_code == 200
    assert _ingest(client, "uni", [{"id": "y", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}}]).status_code == 200

    q = client.post("/v1/query", json={"dataset": "uni", "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 5})
    assert q.status_code == 200, q.text
    matches = q.json()["matches"]
    xs = [m for m in matches if m["id"] == "x"]
    assert len(xs) == 1  # authoritative recall copy, no stale twin
    assert xs[0]["metadata"] == {"v": 2}  # the LATEST version
    assert {m["id"] for m in matches} >= {"x", "y"}  # union returns both


def test_consolidation_fold_in_process(client, monkeypatch):
    """Upsert past a low RB_RECALL_MAX_ROWS to trigger CONSOLIDATE; the builder
    daemon thread folds the partition into a memory:// shard and the recall set
    drains. Polls (no docker) for the folded shard.

    The worker daemon threads this starts are STOPPED in teardown (request_stop +
    join with a deadline, then reset) so they do not consume the shared
    in-process queue in later tests (otherwise a leaked validator thread would
    swallow another test's VALIDATE_DATASET message)."""
    import threading

    from adapters.storage import storage
    from adapters.state import state as state_mod
    from adapters import config
    from adapters.queue import shutdown as queue_shutdown

    run_mod = client._run_mod
    # Low cap so a handful of upserts trips CONSOLIDATE.
    monkeypatch.setenv("RB_RECALL_MAX_ROWS", "3")
    # Skip the workers' metrics HTTP servers (ports 9100-9102) so the only new
    # threads are the stoppable consume loops + the reaper (no leaked listeners).
    monkeypatch.setenv("RB_ALLINONE_DISABLE_METRICS", "1")

    # The worker run modules capture INDEXES_PREFIX / LANDING_PREFIX at IMPORT
    # time. In a fresh process `services.allinone.run` sets memory:// defaults
    # BEFORE importing them, so they capture memory://. But the unit suite may
    # have already imported them with the conftest's s3:// defaults; reload them
    # NOW (env is memory://) so the in-process fold writes to the memory:// store
    # this test polls. (Pure test-isolation concern — not a production path.)
    import services.validator_worker.run as _vw
    import services.index_builder.run as _ib
    import services.ephemeral_runner.run as _er
    importlib.reload(_vw)
    importlib.reload(_ib)
    importlib.reload(_er)

    # Make sure the loops start un-stopped.
    queue_shutdown.reset()
    before = set(threading.enumerate())
    run_mod._start_workers()

    try:
        _create_dataset(client, "fold", dim=4)
        for i in range(8):
            r = _ingest(client, "fold", [
                {"id": f"f{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {}}
            ])
            assert r.status_code == 200, r.text

        indexes_prefix = config.indexes_prefix()
        shard_prefix = f"{indexes_prefix}/default/fold"

        # Bounded deadline poll for a folded shard .bin (builder runs async on a
        # daemon thread).
        deadline = time.time() + 20.0
        shard_seen = False
        while time.time() < deadline:
            keys = storage.list(shard_prefix)
            if any(k.endswith(".bin") for k in keys):
                shard_seen = True
                break
            time.sleep(0.05)

        assert shard_seen, (
            f"no folded shard appeared under {shard_prefix}; "
            f"keys={storage.list(shard_prefix)}"
        )
        # The fold drains the recall partition by watermark — count drops from 8.
        drain_deadline = time.time() + 10.0
        while time.time() < drain_deadline:
            if state_mod.recall_partition_count("default", "fold") < 8:
                break
            time.sleep(0.05)
        assert state_mod.recall_partition_count("default", "fold") < 8

        # The shard exists; recall + cold union still answers.
        q = client.post("/v1/query", json={"dataset": "fold", "vector": [7.0, 0.0, 0.0, 0.0], "top_k": 5})
        assert q.status_code == 200, q.text
        assert {m["id"] for m in q.json()["matches"]}  # non-empty union
    finally:
        # Stop the daemon worker loops so they do not consume the shared
        # in-process queue in later tests, then clear the stop flag.
        queue_shutdown.request_stop()
        stop_deadline = time.time() + 5.0
        new_threads = set(threading.enumerate()) - before
        for th in new_threads:
            remaining = max(0.0, stop_deadline - time.time())
            th.join(timeout=remaining)
        queue_shutdown.reset()
