from __future__ import annotations

"""Query API service.

Exposes vector search functionality with a hot cache path backed by FAISS and a
fallback that enqueues ephemeral jobs for on-demand compute. Also surfaces
dataset listings, health, and metrics.
"""

import os
import time
from typing import Any, Dict, List, Optional
import uuid
import threading

import faiss  # type: ignore
import numpy as np
from fastapi import FastAPI, Depends
from pydantic import BaseModel

from adapters.observability import init_observability
from adapters.observability.otel import instrument_fastapi
from adapters.state.conn_middleware import RequestScopedConnectionMiddleware
from adapters.state.state import migrate, list_datasets, list_shards
from adapters.metrics.metrics import snapshot, counter, timer
from adapters.queue.queue import publish, consume
from services.auth.jwt_utils import current_tenant_id
from services.auth.auth import install_exception_handlers, install_pool_exhaustion_handler
from services.auth.quota import install_rate_limit_handler
from services.query_api.v1_query import (
    router as v1_query_router,
    start_result_consumer,
    v1_query_status,
)

# Observability bootstrap at import. Idempotent — first caller wins;
# OTEL_SERVICE_NAME overrides.
init_observability("rosalinddb-query-api")

app = FastAPI(title="Query API")
# FastAPI HTTP server traces + metrics.
instrument_fastapi(app)
# Bind ONE pooled Postgres connection per HTTP request — a request that calls
# N state functions costs one pool checkout, not N. A pure no-op in
# `memory://` mode (no pool).
app.add_middleware(RequestScopedConnectionMiddleware)
# Rewrite HTTPException payloads from the auth dependency into the v1
# `{"error": {"code", "message"}}` envelope so unauthorized callers see the
# contract-spec'd shape rather than `{"detail": ...}`.
install_exception_handlers(app)
# A sustained pool exhaustion -> v1 503 envelope, not a 500.
install_pool_exhaustion_handler(app)
# Turn a `RateLimited` raised by the `/v1/query` rate-limit dependency into
# the v1 `rate_limited` 429 envelope.
install_rate_limit_handler(app)
# Mount the customer-facing `POST /v1/query` + `GET /v1/query/status/{job_id}`
# surface. The handlers live in the shared `v1_query` module so a single app
# can mount them on a combined origin if the CP/DP split is collapsed into one
# process.
app.include_router(v1_query_router)

CACHE_DIR = os.getenv("CACHE_DIR", "/var/cache/shards")


class QueryRequest(BaseModel):
    """Request schema for /query API."""
    dataset: str
    tenant: Optional[str] = None
    vector: Optional[List[float]] = None
    id_lookup: Optional[str] = None
    top_k: Optional[int] = None
    filter: Optional[Dict[str, Any]] = None
    rerank: Optional[Dict[str, Any]] = None
    mode: Optional[str] = None


def _ensure_cached(shard_uri: str) -> str:
    """Ensure an index shard is available locally and return its path.

    FAISS's `read_index` needs a filesystem path, so an object-store shard
    (`s3://` or `memory://`) is fetched once into `CACHE_DIR`. There is no
    `file://` branch — RosalindDB is object-storage-first.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if shard_uri.startswith("s3://") or shard_uri.startswith("memory://"):
        from adapters.storage.storage import read_bytes

        cache_key = shard_uri.split("://", 1)[1].replace("/", "_")
        path = os.path.join(CACHE_DIR, cache_key)
        if not os.path.exists(path):
            # Atomic publish: write to a unique temp file then rename, so a
            # concurrent reader in another process sharing CACHE_DIR never
            # sees a partial file (a plain write races — see v1_query._ensure_cached).
            tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(read_bytes(shard_uri))
                os.replace(tmp, path)
            except BaseException:
                # On any failure before the rename, remove the leftover temp
                # file so a crash mid-write does not leak `.tmp` files into
                # CACHE_DIR. Best-effort: the file may already be gone.
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        return path
    raise ValueError("Unsupported shard uri")


def _load_latest_index(tenant: str, dataset: str):
    """Load the newest shard's FAISS index for a dataset, if present."""
    shards = list_shards(tenant, dataset)
    if not shards:
        return None
    latest = shards[0]
    local_path = _ensure_cached(latest["shard_uri"])
    return faiss.read_index(local_path), latest


@app.on_event("startup")
def on_start():
    """Initialize state and local cache directories on service startup.

    The RESULT_READY consumer lives in the shared `v1_query` module
    (`start_result_consumer`). Both the legacy `/query` route and the v1
    surface read ephemeral results from that single shared store — running two
    consumer threads would split messages between them.
    """
    migrate()
    os.makedirs(CACHE_DIR, exist_ok=True)
    start_result_consumer()


@app.on_event("shutdown")
def on_stop():
    """Graceful shutdown: stop the RESULT_READY consumer thread.

    uvicorn already drains in-flight HTTP requests on `SIGTERM`. The query_api
    additionally runs a background RESULT_READY consumer thread; tripping the
    shared shutdown event lets that loop exit cleanly at the top of its next
    iteration rather than being killed mid-consume (which, on the Redis path,
    would leave its message reclaimable but delay the result).
    """
    from adapters.queue.shutdown import request_stop

    request_stop()


@app.get("/healthz", include_in_schema=False)
def healthz():
    """Unauthenticated liveness probe.

    Cheap — no DB/storage round-trip; it only proves the process is up and
    routing. The `{"status": "ok", "service": ...}` shape is shared across
    every RosalindDB HTTP service so a health gate can treat them uniformly.
    """
    return {"status": "ok", "service": "query_api"}


@app.get("/datasets")
def datasets(tenant_id: str = Depends(current_tenant_id)):
    """List datasets owned by the calling tenant.

    The legacy `/datasets` endpoint is scoped to the caller's tenant. The
    customer-facing `GET /v1/datasets` lives in
    `services/source_registry/main.py`; this remains for internal callers and
    existing dashboard wiring.
    """
    return list_datasets(tenant_id)


@app.get("/metrics")
def metrics():
    """Return a snapshot of in-process metrics."""
    return snapshot()


@app.post("/query")
def query(req: QueryRequest, tenant_id: str = Depends(current_tenant_id)):
    """Perform a vector search over the newest shard or enqueue ephemeral work.

    The service prefers the hot path when possible. If the shard is unavailable
    or `mode=ephemeral`, a message is enqueued and an empty result is returned
    immediately. The public `POST /v1/query` route is the canonical surface;
    this internal `/query` endpoint stays for backwards compatibility with the
    dashboard and existing tests.
    """
    start = time.time()
    top_k = req.top_k or int(os.getenv("TOP_K", "10"))
    mode = req.mode or os.getenv("QUERY_MODE", "auto")

    if mode in ("hot", "auto"):
        try:
            loaded = _load_latest_index(tenant_id, req.dataset)
            idx, shard_meta = (loaded if loaded is not None else (None, None))
            if idx is not None and req.vector is not None:
                x = np.array([req.vector], dtype=np.float32)
                distances, ids = idx.search(x, top_k)
                counter("query_reads", 1)
                counter("cache_hit", 1)
                timer("latency_ms", (time.time() - start) * 1000.0)
                return {
                    "matches": [
                        {"id": int(ids[0][i]), "score": float(distances[0][i])}
                        for i in range(min(top_k, len(ids[0])))
                    ],
                    "latency_ms": int((time.time() - start) * 1000.0),
                    "mode": "hot",
                    "shard_version": shard_meta.get("created_at"),
                }
        except Exception:  # noqa: BLE001
            pass

    # Fallback to ephemeral
    correlation_id = str(uuid.uuid4())
    payload = {
        "dataset": req.dataset,
        "tenant": tenant_id,
        "vector": req.vector,
        "top_k": top_k,
        "correlation_id": correlation_id,
        "reply_to": os.getenv("RESULT_TOPIC", "RESULT_READY"),
    }
    publish("RUN_EPHEMERAL_QUERY", payload)
    counter("ephemeral_queries", 1)
    return {
        "matches": [],
        "latency_ms": int((time.time() - start) * 1000.0),
        "mode": "ephemeral",
        "shard_version": None,
        "job_id": correlation_id,
    }


@app.get("/query-status/{job_id}")
def query_status(job_id: str):
    """Return the result for a previously enqueued ephemeral query, if ready.

    Delegates to the shared `v1_query` status handler so the legacy `/query`
    route and `POST /v1/query` read the same ephemeral result store.
    """
    return v1_query_status(job_id)

