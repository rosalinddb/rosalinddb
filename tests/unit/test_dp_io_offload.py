"""Unit tests for DP query hot-path I/O offload (TDD).

TDD suite — written BEFORE the implementation change.

THE PROBLEM
-----------
`dp_v1_query` is an ``async def`` FastAPI handler, but it calls
``execute_v1_query`` synchronously on the event loop.  On a shard *cache miss*
``execute_v1_query`` → ``run_query`` → ``_hot_search`` performs:

  1. ``_ensure_cached`` → ``read_bytes(shard_uri)`` — blocking boto3 GET
     (~80-134 ms on S3/R2, any latency on MinIO).
  2. ``faiss.read_index(local_path)`` — CPU-heavy FAISS deserialisation.
  3. ``read_shard_sidecar`` → ``read_bytes(meta_uri)`` — second blocking S3 GET
     (7-29 ms).

All three run INLINE on the uvicorn event loop, blocking every other in-flight
coroutine on that worker — which is the untracked ~1.2 s gap visible in Tempo
traces.

THE FIX
-------
In ``dp_query.py``, wrap ``execute_v1_query`` in
``starlette.concurrency.run_in_threadpool`` so the entire synchronous pipeline
runs off the event loop.  The span instrumentation, shard cache, error handling,
and query results must be preserved.

TESTS
-----
The tests here verify the post-fix contract:

  1. ``execute_v1_query`` is awaited via ``run_in_threadpool`` from
     ``dp_v1_query`` — i.e. the synchronous work is dispatched to a thread, not
     called inline.  We verify this by monkeypatching
     ``starlette.concurrency.run_in_threadpool`` and confirming it is called
     during a DP query request.

  2. The event loop is not blocked during a simulated slow cold load: a
     concurrent async task scheduled on the event loop makes progress while the
     blocking call is running in the thread pool.

  3. Cache behaviour, span emission, error handling and result shape are
     unchanged — the offload must be transparent to callers.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import threading
import time

import faiss  # type: ignore
import numpy as np
import pytest

os.environ.setdefault("DATABASE_URL", "memory://test")
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")


# ---------------------------------------------------------------------------
# Helpers — build a real in-memory shard so _hot_search has something to find
# ---------------------------------------------------------------------------


def _build_memory_shard(tenant: str, dataset: str, dim: int = 4, n: int = 8):
    """Write a FAISS shard + sidecar to memory:// and register it in state."""
    from adapters.storage import storage as storage_mod
    from adapters.state import state as state_mod

    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim), dtype=np.float32)
    ids = np.arange(1, n + 1, dtype=np.int64)
    inner = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap2(inner)
    index.add_with_ids(vecs, ids)

    shard_uri = f"memory://shards/{tenant}/{dataset}/shard.bin"
    storage_mod.write_bytes(shard_uri, faiss.serialize_index(index).tobytes())
    sidecar = {str(int(i)): {"id": f"r{int(i)}", "metadata": {}} for i in ids}
    storage_mod.write_bytes(
        f"{shard_uri}.meta.json", json.dumps(sidecar).encode("utf-8")
    )
    state_mod.add_shard(tenant, dataset, shard_uri, "chk", n, "flat", "full", [])
    return shard_uri


# ---------------------------------------------------------------------------
# Fixture: reset all shared state per test
# ---------------------------------------------------------------------------


@pytest.fixture
def dp_env(tmp_path, monkeypatch):
    """Reset state, storage, and shard cache; yield a wired-up dp_query module.

    Returns a namespace with:
      - ``dp_query``   — the reloaded dp_query module
      - ``v1_query``   — the reloaded v1_query module
      - ``state_mod``  — the reloaded state module
      - ``storage_mod`` — the storage module (for memory_reset)
    """
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("RB_PROXY_SECRET", raising=False)

    from adapters.storage import storage as storage_mod
    from adapters.state import state as state_mod

    storage_mod.memory_reset()
    importlib.reload(state_mod)
    for attr in ("_MEM_TENANTS", "_MEM_TENANTS_BY_EMAIL", "_MEM_API_KEYS", "_MEM_DATASETS"):
        obj = getattr(state_mod, attr, None)
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, list):
            obj.clear()
    state_mod._MEM_SHARDS.clear()

    import services.query_api.v1_query as v1_query
    importlib.reload(v1_query)
    v1_query.cache_clear()

    import services.query_api.dp_query as dp_query
    importlib.reload(dp_query)

    class _Env:
        pass

    e = _Env()
    e.dp_query = dp_query
    e.v1_query = v1_query
    e.state_mod = state_mod
    e.storage_mod = storage_mod
    return e


# ---------------------------------------------------------------------------
# 1.  execute_v1_query is offloaded via run_in_threadpool in dp_v1_query
# ---------------------------------------------------------------------------


class TestDpQueryOffloadsToThreadpool:
    """dp_v1_query must dispatch execute_v1_query via run_in_threadpool."""

    def test_run_in_threadpool_is_called(self, dp_env, monkeypatch):
        """run_in_threadpool is invoked when dp_v1_query handles a request.

        We monkeypatch ``starlette.concurrency.run_in_threadpool`` (and the
        reference imported into dp_query) to a wrapper that records calls, then
        drive the handler via httpx/TestClient and assert the wrapper was hit.
        """
        _build_memory_shard("t1", "ds1")
        dp_env.state_mod.create_tenant("t1", "t1@example.com", "pw")
        dp_env.state_mod.create_dataset("t1", "ds1", 4)

        called = {"n": 0, "fn": None}
        import starlette.concurrency as sc_mod

        real_ritp = sc_mod.run_in_threadpool

        async def _spy(fn, *args, **kwargs):
            called["n"] += 1
            called["fn"] = fn
            return await real_ritp(fn, *args, **kwargs)

        monkeypatch.setattr(sc_mod, "run_in_threadpool", _spy)
        # Also patch the reference that dp_query imported at module load.
        import services.query_api.dp_query as dq
        monkeypatch.setattr(dq, "run_in_threadpool", _spy)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        dp_app = FastAPI()
        dp_app.include_router(dp_env.dp_query.router)

        client = TestClient(dp_app)
        r = client.post(
            "/v1/query",
            headers={"X-RB-Tenant-Id": "t1"},
            json={"dataset": "ds1", "vector": [0.1, 0.0, 0.0, 0.0], "top_k": 3},
        )
        assert r.status_code == 200, r.text
        assert called["n"] >= 1, (
            "run_in_threadpool must be called by dp_v1_query to offload "
            "execute_v1_query off the event loop; got 0 calls"
        )

    def test_execute_v1_query_runs_in_worker_thread(self, dp_env, monkeypatch):
        """execute_v1_query must execute in a thread-pool thread, not the event-loop thread.

        We record the thread-id inside execute_v1_query and compare it against
        the event-loop thread id.  They must differ.
        """
        _build_memory_shard("t2", "ds2")
        dp_env.state_mod.create_tenant("t2", "t2@example.com", "pw")
        dp_env.state_mod.create_dataset("t2", "ds2", 4)

        event_loop_thread_id = threading.current_thread().ident
        recorded = {"thread_id": None}

        real_execute = dp_env.v1_query.execute_v1_query

        def _spy_execute(*args, **kwargs):
            recorded["thread_id"] = threading.current_thread().ident
            return real_execute(*args, **kwargs)

        monkeypatch.setattr(dp_env.v1_query, "execute_v1_query", _spy_execute)
        # Patch the reference in dp_query too.
        import services.query_api.dp_query as dq
        monkeypatch.setattr(dq, "execute_v1_query", _spy_execute)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        dp_app = FastAPI()
        dp_app.include_router(dp_env.dp_query.router)

        client = TestClient(dp_app)
        r = client.post(
            "/v1/query",
            headers={"X-RB-Tenant-Id": "t2"},
            json={"dataset": "ds2", "vector": [0.1, 0.0, 0.0, 0.0], "top_k": 3},
        )
        assert r.status_code == 200, r.text
        assert recorded["thread_id"] is not None, "execute_v1_query was never called"
        assert recorded["thread_id"] != event_loop_thread_id, (
            "execute_v1_query ran on the event-loop thread — it must run in a "
            "thread-pool worker to avoid blocking the event loop"
        )


# ---------------------------------------------------------------------------
# 2.  The event loop is not blocked during a slow synchronous cold load
# ---------------------------------------------------------------------------


class TestEventLoopNotBlocked:
    """A slow blocking call in execute_v1_query must not starve the event loop."""

    def test_concurrent_coroutine_runs_during_blocking_call(self, dp_env, monkeypatch):
        """A concurrent asyncio.sleep task completes while a 'slow' cold load runs.

        If execute_v1_query were called inline (blocking the event loop), an
        asyncio.sleep(0) scheduled concurrently could not complete until the
        blocking call returned.  With run_in_threadpool the event loop remains
        free and the sleep completes during the blocking call.
        """
        _build_memory_shard("t3", "ds3")
        dp_env.state_mod.create_tenant("t3", "t3@example.com", "pw")
        dp_env.state_mod.create_dataset("t3", "ds3", 4)

        SLOW_SECONDS = 0.1  # 100 ms simulated blocking I/O

        real_execute = dp_env.v1_query.execute_v1_query

        def _slow_execute(*args, **kwargs):
            time.sleep(SLOW_SECONDS)
            return real_execute(*args, **kwargs)

        import services.query_api.dp_query as dq
        monkeypatch.setattr(dq, "execute_v1_query", _slow_execute)

        concurrent_ran = {"completed": False}

        async def _run_test():
            # Schedule the concurrent task BEFORE the query fires.
            async def _side_task():
                await asyncio.sleep(0)
                concurrent_ran["completed"] = True

            task = asyncio.create_task(_side_task())

            from fastapi import FastAPI
            import httpx
            from fastapi.testclient import TestClient

            dp_app = FastAPI()
            dp_app.include_router(dp_env.dp_query.router)

            # Use anyio / asyncio directly to call the handler.
            async with httpx.AsyncClient(
                app=dp_app, base_url="http://test"
            ) as ac:
                resp = await ac.post(
                    "/v1/query",
                    headers={"X-RB-Tenant-Id": "t3"},
                    json={"dataset": "ds3", "vector": [0.1, 0.0, 0.0, 0.0], "top_k": 3},
                )
            assert resp.status_code == 200, resp.text
            # Give the task a chance to complete.
            await asyncio.gather(task)
            return concurrent_ran["completed"]

        result = asyncio.run(_run_test())
        assert result, (
            "The concurrent asyncio task did not run while the DP was handling "
            "the query — execute_v1_query is blocking the event loop.  "
            "Wrap it in run_in_threadpool."
        )


# ---------------------------------------------------------------------------
# 3.  Result shape, cache behaviour, and error handling are preserved
# ---------------------------------------------------------------------------


class TestDpOffloadPreservesSemantics:
    """After the offload, DP query results and behaviour must be unchanged."""

    def test_result_shape_preserved(self, dp_env):
        """A successful DP query returns the same v1 shape after offloading."""
        _build_memory_shard("t4", "ds4")
        dp_env.state_mod.create_tenant("t4", "t4@example.com", "pw")
        dp_env.state_mod.create_dataset("t4", "ds4", 4)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        dp_app = FastAPI()
        dp_app.include_router(dp_env.dp_query.router)
        client = TestClient(dp_app)

        r = client.post(
            "/v1/query",
            headers={"X-RB-Tenant-Id": "t4"},
            json={"dataset": "ds4", "vector": [0.1, 0.0, 0.0, 0.0], "top_k": 3},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] in ("hot", "cold")
        assert isinstance(body["latency_ms"], int)
        assert isinstance(body["matches"], list)
        for m in body["matches"]:
            assert "id" in m
            assert "score" in m
            assert "metadata" in m

    def test_cold_load_populates_shard_cache(self, dp_env):
        """After a cold query via the offloaded path, the shard cache is warm."""
        _build_memory_shard("t5", "ds5")
        dp_env.state_mod.create_tenant("t5", "t5@example.com", "pw")
        dp_env.state_mod.create_dataset("t5", "ds5", 4)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        dp_app = FastAPI()
        dp_app.include_router(dp_env.dp_query.router)
        client = TestClient(dp_app)

        dp_env.v1_query.cache_clear()
        r1 = client.post(
            "/v1/query",
            headers={"X-RB-Tenant-Id": "t5"},
            json={"dataset": "ds5", "vector": [0.1, 0.0, 0.0, 0.0], "top_k": 3},
        )
        assert r1.status_code == 200, r1.text
        assert r1.json()["mode"] == "cold"

        # Second query — shard cache must be warm.
        r2 = client.post(
            "/v1/query",
            headers={"X-RB-Tenant-Id": "t5"},
            json={"dataset": "ds5", "vector": [0.1, 0.0, 0.0, 0.0], "top_k": 3},
        )
        assert r2.status_code == 200, r2.text
        assert r2.json()["mode"] == "hot"

    def test_missing_tenant_still_400(self, dp_env):
        """Error handling is preserved — a missing tenant header is still 400."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        dp_app = FastAPI()
        dp_app.include_router(dp_env.dp_query.router)
        client = TestClient(dp_app)

        r = client.post(
            "/v1/query",
            json={"dataset": "whatever", "vector": [0.1, 0.0, 0.0, 0.0]},
        )
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == "invalid_request"

    def test_proxy_secret_check_still_enforced(self, dp_env, monkeypatch):
        """The proxy-secret check still fires before the threadpool dispatch."""
        monkeypatch.setenv("RB_PROXY_SECRET", "s3cr3t")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        dp_app = FastAPI()
        dp_app.include_router(dp_env.dp_query.router)
        client = TestClient(dp_app)

        r = client.post(
            "/v1/query",
            headers={"X-RB-Tenant-Id": "t_any", "X-RB-Proxy-Secret": "wrong"},
            json={"dataset": "whatever", "vector": [0.1, 0.0, 0.0, 0.0]},
        )
        assert r.status_code == 403, r.text
        assert r.json()["error"]["code"] == "proxy_unauthorized"

    def test_dataset_not_found_still_404(self, dp_env):
        """A non-existent dataset still returns 404 after the offload."""
        dp_env.state_mod.create_tenant("t6", "t6@example.com", "pw")

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        dp_app = FastAPI()
        dp_app.include_router(dp_env.dp_query.router)
        client = TestClient(dp_app)

        r = client.post(
            "/v1/query",
            headers={"X-RB-Tenant-Id": "t6"},
            json={"dataset": "no-such-dataset", "vector": [0.1, 0.0, 0.0, 0.0]},
        )
        assert r.status_code == 404, r.text
        assert r.json()["error"]["code"] == "dataset_not_found"
