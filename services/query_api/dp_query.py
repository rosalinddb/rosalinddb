from __future__ import annotations

"""The DP-trust `/v1/query` router.

This router is the Query **Data Plane** surface. It exposes the same paths as
the customer-facing `v1_query` router — `POST /v1/query` and
`GET /v1/query/status/{job_id}` — but with a different trust model:

  - **No `Authorization` parsing.** The Control Plane (CP) authenticates the
    customer at the edge and resolves the `tenant_id`. The CP→DP call carries
    that already-verified tenant in the trusted `X-RB-Tenant-Id` header; the DP
    `/v1/query` handler reads the tenant from **that header only**. A
    missing/empty header is a `400` (the CP must always send it).
  - **Shared-secret defense-in-depth.** When the DP's `RB_PROXY_SECRET` env var
    is set, the request must carry a matching `X-RB-Proxy-Secret` header or it
    is rejected with `403 proxy_unauthorized`. When `RB_PROXY_SECRET` is unset
    (local dev / unit tests / a single-process dev/test harness) the check is
    skipped and private-network isolation (e.g. Docker network, k8s ClusterIP,
    Tailscale) is the sole control.
  - **No query quota.** `try_consume_query` has moved to the CP proxy handler;
    the DP `/v1/query` handler consumes no quota and has no `rate_limit`
    dependency. A tenant who is over their daily quota can still be served by
    the DP — the CP rejects them before the request ever reaches here.

The DP **re-validates** the request body (it must never trust an unvalidated
body, even from the CP) and runs the *same* search core as the authenticated
route — `execute_v1_query` with `consume_quota=None`. Responses are the
customer-facing v1 shapes, byte-identical to the authenticated route, so the
CP can stream them back verbatim.

This router is mounted on `dp_app.py` (the deployed Query Data Plane).
"""

import hmac
import os
from typing import Optional

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from services.query_api.v1_query import (
    _err,
    execute_v1_query,
    v1_query_status,
)

# Header names from the CP↔DP internal contract.
TENANT_HEADER = "X-RB-Tenant-Id"
PROXY_SECRET_HEADER = "X-RB-Proxy-Secret"

router = APIRouter()


def _check_proxy_secret(provided: Optional[str]) -> Optional[JSONResponse]:
    """Enforce the `X-RB-Proxy-Secret` shared-secret check.

    Returns `None` when the request is allowed (either the secret matches, or
    `RB_PROXY_SECRET` is unset so the check is skipped), or a
    `403 proxy_unauthorized` v1 envelope when the DP has a secret configured
    and the request's secret is missing or wrong.

    `RB_PROXY_SECRET` is read live (not cached at import) so a test can set or
    clear it per-case without re-importing the module.
    """
    expected = os.getenv("RB_PROXY_SECRET")
    if not expected:
        # Unset → skip the check; private-network isolation (Docker network,
        # k8s ClusterIP, Tailscale, etc.) is the sole control.
        return None
    # Constant-time comparison — `!=` on `str` short-circuits on the first
    # differing byte, leaking the secret's length/prefix via timing.
    # `hmac.compare_digest` compares in time independent of the content.
    if provided is None or not hmac.compare_digest(provided, expected):
        return _err(
            403,
            "proxy_unauthorized",
            "Invalid or missing proxy secret",
        )
    return None


@router.post("/v1/query")
async def dp_v1_query(
    request: Request,
    x_rb_tenant_id: Optional[str] = Header(default=None),
    x_rb_proxy_secret: Optional[str] = Header(default=None),
):
    """DP-trust vector search — tenant from `X-RB-Tenant-Id`, no quota.

    The CP has already authenticated the customer and resolved the tenant; the
    DP trusts that and reads the tenant from the `X-RB-Tenant-Id` header. The
    body is re-validated and the search runs via the shared `execute_v1_query`
    core with `consume_quota=None` — query quota is enforced on the CP, not
    here. The response is the customer-facing v1 shape, byte-identical to the
    authenticated route.

    DP I/O offload (query-latency fix): ``execute_v1_query`` calls into the
    synchronous hot path (``run_query`` → ``_hot_search``).  On a shard *cache
    miss* that path performs three blocking operations inline:

      1. ``_ensure_cached`` → ``read_bytes(shard_uri)`` — a boto3 ``GetObject``
         (~80-134 ms cold on S3/R2).
      2. ``faiss.read_index(local_path)`` — CPU-heavy FAISS deserialisation.
      3. ``read_shard_sidecar`` → a second ``read_bytes`` call for the
         ``.meta.json`` sidecar (7-29 ms).

    Calling any of these inline in an ``async def`` handler freezes the
    uvicorn worker's event loop for the duration — every other in-flight query
    on that worker stalls behind it, producing the ~1.2 s untracked gap visible
    in Tempo traces.  ``run_in_threadpool`` dispatches the entire synchronous
    pipeline to the thread-pool executor so the event loop stays free.

    The proxy-secret check and the tenant-header check are deliberately kept
    *before* the threadpool dispatch: they are cheap (env-var read + hmac) and
    must reject invalid requests without ever touching the thread pool.

    Span instrumentation, the shard cache (``_SHARD_CACHE``), error handling,
    and query results are unchanged — the offload is transparent to callers.
    """
    # Defense-in-depth: the shared-secret check runs first, before any work.
    rejection = _check_proxy_secret(x_rb_proxy_secret)
    if rejection is not None:
        return rejection

    # Tenant comes from the trusted header ONLY — never from `Authorization`.
    if not x_rb_tenant_id:
        return _err(
            400,
            "invalid_request",
            f"Missing required {TENANT_HEADER} header",
        )

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _err(400, "invalid_request", "Request body must be JSON")

    # Offload the synchronous pipeline to the thread-pool executor.
    # `consume_quota=None`: the DP never consumes query quota (it moved to the
    # CP proxy handler — the CP consumes quota before forwarding).
    return await run_in_threadpool(
        execute_v1_query, x_rb_tenant_id, body, consume_quota=None
    )


@router.get("/v1/query/status/{job_id}")
def dp_v1_query_status(
    job_id: str,
    x_rb_tenant_id: Optional[str] = Header(default=None),
    x_rb_proxy_secret: Optional[str] = Header(default=None),
):
    """DP-trust ephemeral-result poll.

    Reuses the shared `v1_query_status` logic, which reads from the shared
    Redis `result_store` so the poll works against any Query-DP node. Per the
    contract, the status handler does not scope by tenant (`X-RB-Tenant-Id` is
    still accepted for parity but unused) — that matches today's behaviour. The
    shared-secret check still applies.
    """
    rejection = _check_proxy_secret(x_rb_proxy_secret)
    if rejection is not None:
        return rejection
    return v1_query_status(job_id)
