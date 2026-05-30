from __future__ import annotations

"""The Control Plane (CP) ASGI app.

The CP is the **only public surface** of the CP/DP split. It mounts the
**reverse-proxy** query router (`query_proxy.router`) and forwards `/v1/query`
to a private Query Data Plane (DP) node; FAISS does not run in this process.

Composition:

  - It reuses the existing `source_registry` app — that app already carries
    auth (`/auth/*`), the `/v1/datasets*` catalog, the vector-upload + bulk
    import surface, the CORS middleware, the request-scoped-connection
    middleware, the v1 exception handlers (including the
    `PoolCheckoutTimeout` -> 503 mapping) and the rate-limit handler. The CP
    needs all of that unchanged.
  - It mounts `query_proxy.router` on top — the proxied `POST /v1/query` +
    `GET /v1/query/status/{job_id}`.
  - It deliberately does NOT mount `v1_query.router` (the in-process search
    router). On the CP a query is proxied, never run locally.

The matching DP-side app is `services.query_api.dp_app:app`, which mounts
`dp_query_router` (the DP-trust router) and trusts the verified
`X-RB-Tenant-Id` header the CP sends. The DP reuses `v1_query`'s core
functions but mounts the DP-trust router, not `v1_query.router` directly.
Each router has exactly one quota call and exactly one of them is live on a
given process, so there is no double-charge across the hop.
"""

# Bootstrap observability as the Control Plane FIRST, before importing
# source_registry. `init_observability` is idempotent — the first call wins —
# and importing `source_registry.main` triggers ITS `init_observability(
# "rosalinddb-source-registry")`. Calling it here first pins the resolved
# service name to `rosalinddb-control-plane` so the CP reports under its own
# name in traces/metrics (the `OTEL_SERVICE_NAME` env var still overrides, but
# no per-process env wiring is needed in the deployment config).
from adapters.observability import init_observability

init_observability("rosalinddb-control-plane")

# Importing the source_registry module builds the FastAPI `app`, installs
# CORS + the auth router + the v1 exception handlers + the rate-limit handler +
# the request-scoped-connection middleware, and registers the `/v1/datasets*` /
# ingest / import routes. The CP reuses that app wholesale. Its own
# `init_observability` call is now a no-op (the CP bootstrapped it above).
from services.source_registry.main import app
from services.query_api.query_proxy import router as query_proxy_router

# Mount the CP reverse-proxy query router. This is the ONLY query surface on
# the CP — `/v1/query` and `/v1/query/status/{job_id}` are authenticated +
# rate-limited + quota-checked here, then proxied to a private Query-DP node.
# `v1_query.router` (the in-process search router) is intentionally NOT
# mounted: the CP proxies queries, it does not run them.
app.include_router(query_proxy_router)

__all__ = ["app"]
