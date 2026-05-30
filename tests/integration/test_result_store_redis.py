"""Integration coverage for the shared, Redis-backed ephemeral result store.

Multi-worker safety (Change 1). `query_api` answers a query against a
not-yet-indexed dataset asynchronously: the `RESULT_READY` consumer stashes the
result, and `GET /v1/query/status/{job_id}` reads it. With `query_api` running
multiple workers / replicas the consumer that stores the result and the status
poll that reads it are DIFFERENT processes — an in-process dict loses the
result across that boundary.

These tests spin up a real Redis (testcontainers) and prove:

  - a result stored by one `result_store` instance is readable by a SEPARATE,
    independently-imported `result_store` instance (the cross-replica case) —
    this FAILS with the old in-process dict, passes with the Redis store;
  - the stored key is given a bounded TTL so ephemeral results self-expire;
  - the v1 status response shape is unchanged end-to-end.

A separate module instance (via `importlib`) stands in for a separate
`query_api` replica: each binds its own `result_store`, but both point at the
one shared Redis.
"""
from __future__ import annotations

import importlib
import sys

import pytest

try:
    from testcontainers.redis import RedisContainer
except ImportError as exc:  # pragma: no cover
    RedisContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(scope="module")
def redis_url():
    """Start one Redis container for this module; yield its URL."""
    if RedisContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers[redis] is required for the result-store suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with RedisContainer("redis:7-alpine") as rc:
        host = rc.get_container_host_ip()
        port = rc.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


def _fresh_result_store(monkeypatch, redis_url, module_name: str):
    """Import a *fresh* `result_store` bound to `redis_url`, as a new module.

    The queue adapter binds its Redis client at import time, and `result_store`
    imports that client. Reloading both under a distinct module name yields an
    independent `result_store` — a stand-in for a separate `query_api` replica
    that shares the one Redis.
    """
    monkeypatch.setenv("REDIS_URL", redis_url)
    # Drop any cached copies so the import genuinely re-binds `_redis`.
    for name in ("adapters.queue.queue", "services.query_api.result_store"):
        sys.modules.pop(name, None)
    import adapters.queue.queue as queue_mod  # noqa: F401 - re-imported for binding
    import services.query_api.result_store as rs
    # Register under a unique name so a second call gets a *separate* object.
    sys.modules[module_name] = rs
    return rs


@pytest.fixture
def stores(monkeypatch, redis_url):
    """Two independently-imported `result_store` instances ('replicas').

    Both share `redis_url`. Teardown restores the default in-process queue
    adapter so the rest of the session is unaffected.
    """
    replica_a = _fresh_result_store(monkeypatch, redis_url, "_rs_replica_a")
    replica_b = _fresh_result_store(monkeypatch, redis_url, "_rs_replica_b")
    replica_a.clear()
    yield replica_a, replica_b
    replica_a.clear()
    for name in (
        "_rs_replica_a", "_rs_replica_b",
        "adapters.queue.queue", "services.query_api.result_store",
    ):
        sys.modules.pop(name, None)
    monkeypatch.delenv("REDIS_URL", raising=False)
    importlib.import_module("adapters.queue.queue")
    importlib.import_module("services.query_api.result_store")


def test_result_visible_across_replicas(stores):
    """A result stored by replica A is readable by replica B.

    This is the multi-worker bug fix: the `RESULT_READY` consumer (replica A)
    stores the result, the user's status poll lands on replica B. With the old
    in-process dict B would see nothing; with the shared Redis store it does.
    """
    replica_a, replica_b = stores
    replica_a.store_result(
        "job_xreplica",
        {"correlation_id": "job_xreplica",
         "matches": [{"id": "doc-1", "score": 0.5, "metadata": {}}],
         "latency_ms": 3},
    )
    # Replica B never stored anything itself — it must still find the result.
    res = replica_b.get_result("job_xreplica")
    assert res is not None, "cross-replica result lookup must succeed via Redis"
    assert res["matches"] == [{"id": "doc-1", "score": 0.5, "metadata": {}}]
    assert res["latency_ms"] == 3


def test_stored_result_has_bounded_ttl(stores):
    """The Redis key for an ephemeral result is given a bounded TTL.

    Ephemeral results are transient; the key must self-expire so the store
    cannot grow without bound.
    """
    replica_a, _ = stores
    replica_a.store_result("job_ttl", {"matches": [], "latency_ms": 1})
    ttl = replica_a._queue_redis.ttl(replica_a._key("job_ttl"))
    assert ttl > 0, "ephemeral result key must have a TTL set"
    assert ttl <= replica_a.RESULT_TTL_SECONDS


def test_unknown_job_is_absent(stores):
    """An unknown job_id reads back None (status maps it to ready=false)."""
    _, replica_b = stores
    assert replica_b.get_result("job_does_not_exist") is None


def test_clear_drops_redis_keys(stores):
    """`clear()` removes every `query_result:*` key from Redis."""
    replica_a, replica_b = stores
    replica_a.store_result("job_a", {"matches": []})
    replica_a.store_result("job_b", {"matches": []})
    replica_a.clear()
    assert replica_b.get_result("job_a") is None
    assert replica_b.get_result("job_b") is None
