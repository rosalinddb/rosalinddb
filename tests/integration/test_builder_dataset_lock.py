"""Integration coverage for the per-dataset builder advisory lock (Change 3).

Multi-worker safety. With `index_builder` replicated, two builder replicas can
pick up two `DATASET_READY` messages for the SAME dataset concurrently (or a
redelivered message races the original). Without serialisation both read the
same landing parts and both fold them in — double-indexing the vectors.

The fix serialises builds per dataset with a Postgres advisory lock
(`pg_try_advisory_lock`, non-blocking): the second concurrent build for a
dataset fails to take the lock and SKIPS (the queue redelivers; the shard
manifest makes a duplicate a no-op). Builds of *different* datasets get
distinct locks and still run in parallel.

These tests run against a REAL Postgres (testcontainers) — the advisory lock
is a no-op in `memory://` mode, so the locking behaviour can only be exercised
here. Landing/index objects live in real MinIO (the autouse `minio_env`
fixture). They prove:

  - two concurrent `run_once` calls for the SAME dataset do not double-index
    (final `ntotal` is correct, exactly one build actually ran);
  - two concurrent `run_once` calls for DIFFERENT datasets both proceed.
"""
from __future__ import annotations

import importlib
import threading

import faiss  # type: ignore
import numpy as np
import pytest

try:
    from testcontainers.postgres import PostgresContainer
except ImportError as exc:  # pragma: no cover
    PostgresContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    from testcontainers.redis import RedisContainer
except ImportError as exc:  # pragma: no cover
    RedisContainer = None  # type: ignore
    _REDIS_IMPORT_ERROR = exc
else:
    _REDIS_IMPORT_ERROR = None


@pytest.fixture(scope="module")
def pg_url():
    """Start one Postgres container for this module; yield a psycopg2 DSN."""
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the builder-lock suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with PostgresContainer("postgres:15-alpine", driver=None) as pg:
        yield pg.get_connection_url()


@pytest.fixture
def env(monkeypatch, pg_url, s3_landing_prefix, s3_indexes_prefix, tmp_path):
    """State bound to real Postgres + MinIO landing/index prefixes.

    The schema is migrated up front; a tenant is seeded. Teardown restores the
    default `memory://` adapter for the rest of the session.
    """
    monkeypatch.setenv("DATABASE_URL", pg_url)
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    assert not state_mod._MEMORY_MODE, "test must run against real Postgres"
    state_mod.migrate()
    # The Postgres container is shared module-scope; seed the tenant only if a
    # prior test in this module has not already created it.
    if state_mod.get_tenant_by_id("ten_lock") is None:
        state_mod.create_tenant("ten_lock", "lock@example.com", "x")

    import services.index_builder.run as builder
    importlib.reload(builder)
    yield state_mod, builder, s3_landing_prefix

    monkeypatch.delenv("DATABASE_URL", raising=False)
    importlib.reload(state_mod)


def _write_landing(landing: str, tenant: str, dataset: str, upload: str, records):
    """Write one upload's parquet into its own sub-prefix (mirrors validator)."""
    from adapters.landing.parquet_writer import write_parquet

    write_parquet(f"{landing}/{tenant}/{dataset}/upload-{upload}", records)


def _read_shard(shard_uri):
    """Download a MinIO-resident FAISS shard and load it via faiss."""
    import tempfile

    from adapters.storage.storage import read_bytes

    blob = read_bytes(shard_uri)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(blob)
        path = fh.name
    return faiss.read_index(path)


def test_concurrent_same_dataset_build_does_not_double_index(env):
    """Two concurrent `run_once` for the SAME dataset → no double-indexing.

    Both threads target one dataset's landing parts. The per-dataset advisory
    lock lets exactly one build run; the other fails the try-lock and skips.
    The final shard must hold each vector once — `ntotal == 5`, not 10.
    """
    state_mod, builder, landing = env
    state_mod.create_dataset("ten_lock", "racer", 4)
    records = [
        {"id": f"r{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"_": "1"}}
        for i in range(5)
    ]
    _write_landing(landing, "ten_lock", "racer", "a", records)

    results: dict[int, int] = {}
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def build_worker(idx: int) -> None:
        barrier.wait()  # both threads enter run_once at the same instant
        added = builder.run_once("racer", "ten_lock")
        with results_lock:
            results[idx] = added

    threads = [threading.Thread(target=build_worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one shard, holding each of the 5 vectors once — never doubled.
    shards = state_mod.list_shards("ten_lock", "racer")
    assert len(shards) == 1, f"expected one shard, got {len(shards)}"
    assert shards[0]["vector_count"] == 5, shards[0]
    index = _read_shard(shards[0]["shard_uri"])
    assert index.ntotal == 5, f"double-indexed: ntotal={index.ntotal}"

    # One build folded in the 5 vectors (returns 5); the other either lost the
    # advisory lock (returns the `BUILD_SKIPPED` sentinel) or ran after the
    # winner committed and found every part already indexed (a manifest no-op,
    # returns 0). Either way exactly 5 vectors were folded in, exactly once.
    winner, loser = sorted(results.values(), reverse=True)
    assert winner == 5, results
    assert loser in (0, builder.BUILD_SKIPPED), results


def test_concurrent_different_datasets_both_build(env):
    """Two concurrent `run_once` for DIFFERENT datasets both proceed.

    Distinct datasets hash to distinct advisory-lock ids, so neither build
    blocks the other — both must produce a shard.
    """
    state_mod, builder, landing = env
    for name in ("alpha", "beta"):
        state_mod.create_dataset("ten_lock", name, 4)
        _write_landing(landing, "ten_lock", name, "a", [
            {"id": f"{name}-{i}", "values": [float(i), 1.0, 0.0, 0.0],
             "metadata": {"_": "1"}}
            for i in range(4)
        ])

    results: dict[str, int] = {}
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def build_worker(name: str) -> None:
        barrier.wait()
        added = builder.run_once(name, "ten_lock")
        with results_lock:
            results[name] = added

    threads = [
        threading.Thread(target=build_worker, args=(n,))
        for n in ("alpha", "beta")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Both datasets were indexed in parallel — neither was skipped.
    assert results == {"alpha": 4, "beta": 4}, results
    for name in ("alpha", "beta"):
        shards = state_mod.list_shards("ten_lock", name)
        assert len(shards) == 1, f"{name}: {shards}"
        assert shards[0]["vector_count"] == 4, shards[0]


@pytest.fixture(scope="module")
def redis_url():
    """Start one Redis container for this module; yield its URL."""
    if RedisContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers[redis] is required for the builder-skip test. "
            f"Import error: {_REDIS_IMPORT_ERROR}"
        )
    with RedisContainer("redis:7-alpine") as rc:
        host = rc.get_container_host_ip()
        port = rc.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


def test_skipped_build_nacks_message_instead_of_acking_it(env, redis_url, monkeypatch):
    """A builder that SKIPS on a held per-dataset lock must NOT ack the message.

    Regression test for the per-dataset advisory-lock nack bug. When the per-dataset advisory lock is held by
    another replica, `run_once` returns `BUILD_SKIPPED` and the build does NOT
    run. The skipped `DATASET_READY` message may carry a newer upload than the
    in-progress build, so it must be REDELIVERED — `nack(requeue=True)` — not
    `ack`-ed away. This drives the real consume-loop logic against a real Redis
    reliable queue.

    Without the fix the consume loop treats the `0`-return skip as success and
    `ack`s the message: after one loop iteration the queue is EMPTY and the
    build is lost. With the fix the message is nack'd and redelivered, so it
    is still consumable.
    """
    state_mod, builder, landing = env

    # Bind the queue adapter to the container Redis (reliable at-least-once).
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("QUEUE_MAX_ATTEMPTS", "5")
    import adapters.queue.queue as queue_mod
    importlib.reload(queue_mod)
    queue_mod._redis.flushdb()
    # The builder imported `consume/ack/nack` by name at module load; rebind
    # them to the freshly-reloaded (Redis-backed) queue functions.
    monkeypatch.setattr(builder, "consume", queue_mod.consume)
    monkeypatch.setattr(builder, "ack", queue_mod.ack)
    monkeypatch.setattr(builder, "nack", queue_mod.nack)

    state_mod.create_dataset("ten_lock", "skipme", 4)
    _write_landing(landing, "ten_lock", "skipme", "a", [
        {"id": f"s{i}", "values": [float(i), 0.0, 0.0, 0.0], "metadata": {"_": "1"}}
        for i in range(4)
    ])

    # Hold the per-dataset advisory lock on a SEPARATE connection so the
    # builder's `run_once` loses the try-lock and must skip — exactly the
    # "another replica is already building this dataset" scenario.
    objid = state_mod._dataset_lock_objid("ten_lock", "skipme")
    holder = state_mod._conn()
    holder.autocommit = True
    try:
        with holder.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(%s, %s)",
                (state_mod._BUILD_LOCK_CLASS, objid),
            )
            assert cur.fetchone()[0] is True, "could not pre-acquire the lock"

        # Publish a genuine DATASET_READY, then run ONE consume-loop iteration
        # exactly as `main_loop` does.
        queue_mod.publish("DATASET_READY", {"dataset": "skipme", "tenant": "ten_lock"})
        msg = queue_mod.consume("DATASET_READY", block=True, timeout=2.0)
        assert msg is not None
        done = builder._handle_dataset_ready(msg)
        assert done is False, "a skipped build must report not-done"
        if done:
            queue_mod.ack(msg)
        else:
            queue_mod.nack(msg, requeue=True)

        # The message MUST still be deliverable — it was redelivered, not
        # acked away. Without the fix it would have been acked and lost.
        assert queue_mod.processing_size("DATASET_READY") == 0
        redelivered = queue_mod.consume("DATASET_READY", block=True, timeout=2.0)
        assert redelivered is not None, (
            "skipped build's message was lost — it was acked instead of "
            "nack'd for redelivery"
        )
        assert redelivered["dataset"] == "skipme"
    finally:
        holder.close()
        queue_mod._redis.flushdb()
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.delenv("QUEUE_MAX_ATTEMPTS", raising=False)
        importlib.reload(queue_mod)
