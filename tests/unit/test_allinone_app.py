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


import contextlib


@contextlib.contextmanager
def _running_workers(client):
    """Start the all-in-one worker daemon threads for the body, then STOP them.

    The CONSOLIDATE/DELETE_VECTORS fold runs on the index_builder daemon thread,
    so a cross-tier test must have the workers running. Teardown trips the shared
    stop event and joins the new threads (with a deadline) then resets the flag,
    so a leaked worker does not swallow a later test's queue message. Mirrors the
    setup/teardown in `test_consolidation_fold_in_process`.
    """
    import threading

    from adapters.queue import shutdown as queue_shutdown

    run_mod = client._run_mod
    # Worker run modules capture INDEXES/LANDING prefixes at import; the full
    # suite may have imported them with the conftest's s3:// defaults. Reload
    # under the memory:// env so the in-process fold writes to the memory store.
    import services.validator_worker.run as _vw
    import services.index_builder.run as _ib
    import services.ephemeral_runner.run as _er
    importlib.reload(_vw)
    importlib.reload(_ib)
    importlib.reload(_er)

    queue_shutdown.reset()
    before = set(threading.enumerate())
    run_mod._start_workers()
    try:
        yield run_mod
    finally:
        queue_shutdown.request_stop()
        stop_deadline = time.time() + 5.0
        for th in set(threading.enumerate()) - before:
            th.join(timeout=max(0.0, stop_deadline - time.time()))
        queue_shutdown.reset()


def _wait_for_fold(name, expect_below):
    """Poll until the recall partition for `name` drains below `expect_below`
    (the CONSOLIDATE fold has advanced the watermark + trimmed recall) AND a
    cold shard .bin exists. Returns once both hold or raises on timeout."""
    from adapters.storage import storage
    from adapters.state import state as state_mod
    from adapters import config

    shard_prefix = f"{config.indexes_prefix()}/default/{name}"
    deadline = time.time() + 20.0
    shard_seen = False
    while time.time() < deadline:
        if any(k.endswith(".bin") for k in storage.list(shard_prefix)):
            shard_seen = True
            if state_mod.recall_partition_count("default", name) < expect_below:
                return
        time.sleep(0.05)
    raise AssertionError(
        f"fold did not complete for {name}: shard_seen={shard_seen}, "
        f"recall_count={state_mod.recall_partition_count('default', name)}"
    )


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


# --- TRUE cross-tier read-your-writes (union across the watermark) ----------


def test_cross_tier_ryw_recall_wins_over_cold_twin(client, monkeypatch):
    """Ingest an id, drive a CONSOLIDATE fold so it lands in a COLD shard, then
    re-upsert the SAME id with a NEW vector and assert the query returns the NEW
    vector — recall (above the watermark) beats the stale consolidated twin.

    This is the union-across-the-watermark contract, NOT recall-only: after the
    fold the id's old copy is in the cold shard (below the watermark); the
    re-upsert writes a fresh recall row (above the watermark) that must win and
    appear exactly once (no stale twin)."""
    monkeypatch.setenv("RB_RECALL_MAX_ROWS", "3")
    monkeypatch.setenv("RB_ALLINONE_DISABLE_METRICS", "1")

    with _running_workers(client):
        _create_dataset(client, "xtier", dim=4)
        # Ingest enough distinct ids (with the target id "x" far from the query)
        # to trip CONSOLIDATE (cap=3) and fold "x" into a cold shard.
        recs = [{"id": "x", "values": [0.0, 0.0, 0.0, 9.0], "metadata": {"v": 1}}]
        recs += [{"id": f"p{i}", "values": [float(i + 1), 0.0, 0.0, 0.0], "metadata": {}} for i in range(7)]
        for rec in recs:
            assert _ingest(client, "xtier", [rec]).status_code == 200

        _wait_for_fold("xtier", expect_below=len(recs))

        # "x" is now in the cold shard (its old, far vector). Confirm a query near
        # its OLD position can still find it via cold (sanity that it was folded).
        # Re-upsert the SAME id with a NEW vector close to the query point — this
        # writes a fresh recall row ABOVE the new watermark.
        assert _ingest(client, "xtier", [
            {"id": "x", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"v": 2}}
        ]).status_code == 200

        q = client.post("/v1/query", json={"dataset": "xtier", "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 10})
        assert q.status_code == 200, q.text
        matches = q.json()["matches"]
        xs = [m for m in matches if m["id"] == "x"]
        assert len(xs) == 1, f"stale cold twin not suppressed: {matches}"
        # the AUTHORITATIVE recall copy (v:2), not the folded cold one (v:1)
        assert xs[0]["metadata"] == {"v": 2}


def test_cross_tier_delete_after_fold_is_404(client, monkeypatch):
    """Delete an id AFTER it has been folded past the watermark, then assert GET
    -> 404 and the query omits it. This is the fresh-lsn-above-max tombstone
    contract spanning tiers: the cold shard still physically holds the id, but a
    recall tombstone above the watermark suppresses it everywhere."""
    monkeypatch.setenv("RB_RECALL_MAX_ROWS", "3")
    monkeypatch.setenv("RB_ALLINONE_DISABLE_METRICS", "1")

    with _running_workers(client):
        _create_dataset(client, "deltier", dim=4)
        recs = [{"id": "doomed", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}}]
        recs += [{"id": f"q{i}", "values": [0.0, float(i + 1), 0.0, 0.0], "metadata": {}} for i in range(7)]
        for rec in recs:
            assert _ingest(client, "deltier", [rec]).status_code == 200

        _wait_for_fold("deltier", expect_below=len(recs))

        # "doomed" is now in the cold shard (below the watermark). Confirm it is
        # reachable BEFORE the delete (so the 404 after is meaningful).
        g_before = client.get("/v1/datasets/deltier/vectors/doomed")
        assert g_before.status_code == 200, g_before.text

        # Delete it: writes an above-watermark recall tombstone (fresh lsn).
        d = client.delete("/v1/datasets/deltier/vectors/doomed")
        assert d.status_code in (200, 202, 204), d.text

        # GET -> 404 even though the cold shard still holds the id.
        g = client.get("/v1/datasets/deltier/vectors/doomed")
        assert g.status_code == 404, g.text

        # Query omits it (the cold twin is suppressed by the recall tombstone).
        q = client.post("/v1/query", json={"dataset": "deltier", "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 10})
        assert q.status_code == 200, q.text
        assert "doomed" not in {m["id"] for m in q.json()["matches"]}


def test_fold_with_delta_tier_produces_delta_shard(client, monkeypatch):
    """With RB_DELTA_TIER on, a CONSOLIDATE fold over an existing IVF base produces
    a level=1 DELTA shard (the LSM delta-on-base layering) rather than a fresh
    base — proving the delta-tier path is reachable from the all-in-one.

    The delta path only engages when the generation's base is an IVF index built
    with a frozen quantizer (`quantizer_version>=1`). That requires the FIRST fold
    to carry >= IVF_TRAINING_FLOOR rows and `nlist>=4` (=> N//8>=4, i.e. >=32
    rows), so we lower the training floor and fold a single large batch."""
    from adapters.state import state as state_mod

    # A single batch crossing this cap folds ~all rows at once (so the first
    # fold is IVF-eligible). Low IVF floor so 40 rows clears it.
    monkeypatch.setenv("RB_RECALL_MAX_ROWS", "31")
    monkeypatch.setenv("IVF_TRAINING_FLOOR", "16")
    monkeypatch.setenv("RB_ALLINONE_DISABLE_METRICS", "1")
    monkeypatch.setenv("RB_DELTA_TIER", "true")

    with _running_workers(client):
        _create_dataset(client, "deltatest", dim=4)
        # First fold -> IVF base (level 0, quantizer_version 1). One batch of 40
        # distinct vectors crosses the cap once and snapshots all 40.
        batch_a = [
            {"id": f"a{i}", "values": [float(i), float(i % 3), 0.0, 0.0], "metadata": {}}
            for i in range(40)
        ]
        assert _ingest(client, "deltatest", batch_a).status_code == 200

        # Wait for the IVF base to appear.
        deadline = time.time() + 20.0
        base_ivf = False
        while time.time() < deadline:
            shards = state_mod.list_shards("default", "deltatest")
            bases = [s for s in shards if int(s.get("level", 0) or 0) == 0]
            if bases and str(bases[0].get("index_type")) == "ivfflat" \
                    and int(bases[0].get("quantizer_version", 0) or 0) >= 1:
                base_ivf = True
                break
            time.sleep(0.05)
        assert base_ivf, (
            "first fold did not produce an IVF base with a frozen quantizer; "
            f"shards={[(s['id'], s.get('level'), s.get('index_type'), s.get('quantizer_version')) for s in state_mod.list_shards('default', 'deltatest')]}"
        )

        # Second batch crossing the cap -> with the delta tier on, this layers a
        # level=1 delta on the existing base.
        batch_b = [
            {"id": f"b{i}", "values": [0.0, float(i), float(i % 3), 0.0], "metadata": {}}
            for i in range(40)
        ]
        assert _ingest(client, "deltatest", batch_b).status_code == 200

        deadline = time.time() + 20.0
        delta_seen = False
        while time.time() < deadline:
            shards = state_mod.list_shards("default", "deltatest")
            if any(int(s.get("level", 0) or 0) == 1 for s in shards):
                delta_seen = True
                break
            time.sleep(0.05)
        assert delta_seen, (
            "no level=1 delta shard appeared with RB_DELTA_TIER on; "
            f"shards={[(s['id'], s.get('level'), s.get('build_type')) for s in state_mod.list_shards('default', 'deltatest')]}"
        )


def test_recall_off_async_202_path(monkeypatch, tmp_path):
    """With RB_RECALL=false the all-in-one boots and POST /vectors returns 202
    (the ASYNC ingest path: a job_id, NOT the synchronous-200 recall path)."""
    monkeypatch.setenv("RB_REQUIRE_AUTH", "false")
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("LANDING_PREFIX", "memory://rosalinddb/landing")
    monkeypatch.setenv("INDEXES_PREFIX", "memory://rosalinddb/indexes")
    monkeypatch.setenv("STAGING_PREFIX", "memory://rosalinddb/staging")
    monkeypatch.delenv("REDIS_URL", raising=False)
    # Recall OFF -> the synchronous recall write path is not engaged.
    monkeypatch.setenv("RB_RECALL", "false")
    # Set RB_RECALL_BACKEND via monkeypatch (even though recall is off) so the
    # `os.environ.setdefault("RB_RECALL_BACKEND", "memory")` at `run.py` import
    # does NOT permanently leak it into the real environ for later pgvector-path
    # recall tests. monkeypatch restores it at teardown; `setdefault` is then a
    # no-op against this tracked value.
    monkeypatch.setenv("RB_RECALL_BACKEND", "memory")
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))

    import adapters.recall.memtable as memtable
    importlib.reload(memtable)
    memtable._reset()
    import services.allinone.run as run_mod
    importlib.reload(run_mod)

    # `source_registry.main` captures LANDING_PREFIX at IMPORT time into the
    # module global `_LANDING_PREFIX`; the full suite may have imported it with
    # the conftest's s3:// default, so the recall-OFF async landing write would
    # hit S3 (NoSuchBucket). Pin it to the memory:// prefix for this test (the
    # established pattern for import-time-captured config). Recall-ON tests never
    # reach this path (their write is synchronous to the recall memtable).
    import services.source_registry.main as sr_main
    monkeypatch.setattr(sr_main, "_LANDING_PREFIX", "memory://rosalinddb/landing")

    with TestClient(run_mod.app) as c:
        # boots fine
        assert c.get("/healthz").status_code == 200
        _create_dataset(c, "asyncds", dim=4)
        r = _ingest(c, "asyncds", [{"id": "v1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}}])
        # async path: 202 Accepted with a job_id, NOT a synchronous 200.
        assert r.status_code == 202, r.text
        assert "job_id" in r.json()
    memtable._reset()
