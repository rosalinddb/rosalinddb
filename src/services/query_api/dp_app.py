from __future__ import annotations

"""The Query Data Plane (Query-DP) ASGI app.

This is the lean FastAPI app that runs as a **private** process group
reachable only on a private network (e.g. Docker network, k8s ClusterIP,
VPC, Tailscale) — never exposed publicly. It hosts the DP-trust query
surface from `dp_query.py`: `POST /v1/query` and `GET /v1/query/status/
{job_id}` that trust the Control Plane's verified `X-RB-Tenant-Id` header
and do no auth / no query-quota of their own.

Compared to the full `query_api` app (`main.py`), this app deliberately omits
the legacy `/query`, `/datasets`, `/metrics` surface and the
`Authorization`-based dependencies — the DP's only job is the trusted query
path. It keeps `/healthz` (the shared liveness shape) and the RESULT_READY
result consumer (the ephemeral fallback still publishes results that a status
poll must find).
"""

import logging
import os

from fastapi import FastAPI, Request

from adapters.config import truthy as _truthy
from adapters.observability import init_observability
from adapters.observability.otel import instrument_fastapi
from adapters.state.conn_middleware import RequestScopedConnectionMiddleware
from adapters.state.state import migrate
from adapters.storage import shard_tier
from services.auth.auth import install_exception_handlers, install_pool_exhaustion_handler
from services.query_api.dp_query import router as dp_query_router
from services.query_api.v1_query import _err, start_result_consumer


# Observability bootstrap at import. The OTEL_SERVICE_NAME secret overrides the
# default in production (`rosalinddb-query-dp`, per the CP↔DP contract).
init_observability("rosalinddb-query-dp")

app = FastAPI(title="Query DP")
# FastAPI HTTP server traces + metrics.
instrument_fastapi(app)
# One pooled Postgres connection per HTTP request (no-op in `memory://` mode).
# The DP's query path touches state several times per request; this collapses
# that to a single pool checkout.
app.add_middleware(RequestScopedConnectionMiddleware)
# Rewrite any HTTPException payloads into the v1 `{"error": {...}}` envelope so
# the DP always speaks the v1 error shape (the CP forwards it verbatim).
install_exception_handlers(app)
# Map a sustained pool exhaustion to a v1 503 envelope, not a 500.
install_pool_exhaustion_handler(app)
# Mount the DP-trust query router. There is intentionally no `rate_limit`
# handler and no `Authorization` dependency — the CP owns auth and rate
# limiting at the edge.
app.include_router(dp_query_router)

CACHE_DIR = os.getenv("CACHE_DIR", "/var/cache/shards")

_log = logging.getLogger("rosalinddb.query_dp")


@app.on_event("startup")
def on_start():
    """Initialize state + cache dir and start the RESULT_READY consumer.

    The DP still runs the ephemeral fallback (`run_query` enqueues a
    `RUN_EPHEMERAL_QUERY` when a dataset has no shard), so it must consume
    `RESULT_READY` to land results in the shared `result_store` for a later
    `GET /v1/query/status/{job_id}` poll.

    If `RB_PROXY_SECRET` is unset the DP `/v1/query` shared-secret check is
    skipped — the DP is then an unauthenticated FAISS endpoint relying on
    private-network isolation (e.g. Docker network, k8s ClusterIP,
    Tailscale) alone. That is acceptable for dev/tests but a misconfiguration
    in production, so emit a loud WARNING. Startup is NOT hard-failed — that
    would break dev and the test suite.
    """
    if not os.getenv("RB_PROXY_SECRET"):
        _log.warning(
            "RB_PROXY_SECRET is unset — the Query-DP /v1/query shared-secret "
            "check is DISABLED. The DP is an unauthenticated FAISS endpoint "
            "relying on private-network isolation alone. Set RB_PROXY_SECRET "
            "on both the CP and query-dp groups in production."
        )
    migrate()
    os.makedirs(CACHE_DIR, exist_ok=True)
    start_result_consumer()
    # Opt-in PREWARM_SHARD consumer. The builder publishes when
    # `RB_PREWARM_ON_BUILD=true`; the DP consumes when this env opts in.
    # Defaults preserve current behaviour (no consumer thread, no admit
    # path engaged) so the rollback contract holds.
    if _truthy(os.getenv("RB_PREWARM_CONSUMER")):
        from services._common import prewarm_consumer

        prewarm_consumer.start_if_needed()
    # Opt-in DP residency writer. Periodically reconciles
    # `shard_tier.residency()` to the `dp_shard_residency` table so the
    # CP routing layer can prefer the DP that already has the shard warm.
    # The module's own gate check (`_gate_enabled`) is also internal to
    # `start_if_needed`; the env check here mirrors the prewarm-consumer
    # wiring pattern so an operator scanning startup code sees the
    # activation surface in one place. Defaults preserve current behaviour
    # (no thread, no DB writes).
    if _truthy(os.getenv("RB_DP_RESIDENCY_REGISTRY")):
        from services._common import residency_writer

        residency_writer.start_if_needed()


@app.on_event("shutdown")
def on_stop():
    """Graceful shutdown: trip the shared stop event so the RESULT_READY
    consumer thread exits cleanly at the top of its next loop iteration
    rather than being killed mid-consume."""
    from adapters.queue.shutdown import request_stop

    request_stop()
    # Stop the PREWARM_SHARD consumer if it was started. Idempotent so
    # calling stop on a never-started consumer is a clean no-op.
    if _truthy(os.getenv("RB_PREWARM_CONSUMER")):
        try:
            from services._common import prewarm_consumer

            prewarm_consumer.stop()
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            _log.exception("dp_app: prewarm_consumer.stop failed")
    # Stop the residency writer if it was started. Idempotent so calling
    # stop on a never-started writer is a clean no-op. Wrapped so a stuck
    # writer cannot block the shutdown of its sibling subsystems (the
    # daemon thread is left to die with the process if the join times out).
    if _truthy(os.getenv("RB_DP_RESIDENCY_REGISTRY")):
        try:
            from services._common import residency_writer

            residency_writer.stop()
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            _log.exception("dp_app: residency_writer.stop failed")


@app.get("/healthz", include_in_schema=False)
def healthz():
    """Unauthenticated liveness probe.

    Cheap — no DB/storage round-trip. The `{"status": "ok", "service": ...}`
    shape is shared across every RosalindDB HTTP service so a health gate can
    treat them uniformly.
    """
    return {"status": "ok", "service": "query_dp"}


# --- Opt-in admin surface -------------------------------------------------
#
# `POST /admin/prewarm` is the manual / smoke-test entry point for the SSD
# tier's prewarm contract. Gated on `RB_ADMIN_ENDPOINTS=true` so the surface
# is opt-in — a deployment that has not enabled it sees a stock FastAPI 404
# on the route (the handler is never registered). This is the default-off
# rollback contract: every new env flag preserves current behaviour when
# unset.
#
# Wire shape:
#   - 200 + `{"shard_uri": ..., "local_path": ...}` on success
#   - 503 + `cache_capacity_exceeded` envelope when the admission floor
#     rejects (mirrors the hot-path classifier so dashboards see one signal)
#   - 404 + `shard_not_found` envelope when the object store has no key
#   - 400 + `invalid_request` envelope on a malformed body
#
# The consumer path (`services/_common/prewarm_consumer.py`) is the bulk
# producer; this endpoint is the operator's manual hook.
if _truthy(os.getenv("RB_ADMIN_ENDPOINTS")):

    @app.post("/admin/prewarm", include_in_schema=False)
    async def admin_prewarm(request: Request):
        """Manually prewarm a shard URI into the SSD tier.

        Body: `{"shard_uri": "memory://bucket/path.bin"}`.

        On `CacheCapacityExceeded` the response is 503 +
        `cache_capacity_exceeded` so an operator dashboard can distinguish
        capacity pressure (raise `RB_SHARD_TIER_BYTES`) from a storage
        outage. On `FileNotFoundError` the response is 404 +
        `shard_not_found` — the shard URI does not resolve and the operator
        should re-check the URI (typo, deleted shard, etc.).
        """
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _err(400, "invalid_request", "Request body must be JSON")
        if not isinstance(body, dict):
            return _err(400, "invalid_request", "Request body must be a JSON object")
        shard_uri = body.get("shard_uri")
        if not shard_uri or not isinstance(shard_uri, str):
            return _err(
                400, "invalid_request",
                "shard_uri is required and must be a string",
            )

        try:
            local_path = shard_tier.prewarm(shard_uri)
        except shard_tier.CacheCapacityExceeded as exc:
            _log.info(
                "admin_prewarm: capacity rejection for %s: %s", shard_uri, exc,
            )
            return _err(
                503, "cache_capacity_exceeded",
                "SSD cache tier is at capacity",
            )
        except FileNotFoundError:
            return _err(404, "shard_not_found", f"shard URI not found: {shard_uri}")
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "admin_prewarm: dispatch failed for %s (%s)",
                shard_uri, type(exc).__name__,
                exc_info=True,
            )
            return _err(
                500, "internal_error",
                f"prewarm failed: {type(exc).__name__}",
            )

        return {"shard_uri": shard_uri, "local_path": local_path}
