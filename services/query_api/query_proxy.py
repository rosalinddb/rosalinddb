from __future__ import annotations

"""The Control Plane (CP) reverse-proxy query router.

The CP mounts this router and proxies `/v1/query` to a private Query-DP.
It exposes the customer-facing query paths — `POST /v1/query` and
`GET /v1/query/status/{job_id}` — but instead of running the FAISS search
in-process (the legacy `v1_query.router` does that) it **authenticates the
customer at the edge, then reverse-proxies the request to a private Query-DP
node**.

Per-request shape, `POST /v1/query`:

  1. `Depends(current_tenant_id)` — resolve the customer's `Authorization`
     (JWT or `rb_live_…` key) to a verified `tenant_id`. A bad/missing key is
     a 401/403 here; the request never leaves the CP.
  2. `Depends(rate_limit)` — the per-key token-bucket rate limit stays on the
     CP edge.
  3. Consume one unit of daily query quota via `try_consume_query`. A tenant
     over quota gets `429 query_quota_exceeded` and the request never reaches
     the DP. (This is a deliberate relocation of the quota call out of
     `v1_query.py`.)
  4. Resolve `tenant_id -> dp_pool` (`get_tenant_dp_pool`) -> DP base URL
     (`resolve_dp_base_url`).
  5. Forward the raw request body byte-for-byte to `<dp>/v1/query` with the
     trusted headers `X-RB-Tenant-Id` (the verified tenant) and, when set,
     `X-RB-Proxy-Secret`. The customer's `Authorization` header is NOT
     forwarded.
  6. Stream the DP response back verbatim — status code + JSON body.

`GET /v1/query/status/{job_id}` authenticates the poll at the edge
(`current_tenant_id`) but consumes NO quota and is NOT rate-limited — it just
resolves the pool and proxies.

QUOTA:
  Query quota is enforced here on the CP via the existing Postgres-backed
  `try_consume_query`. A Redis INCR+TTL counter is not yet implemented;
  swapping to one is a future improvement.

BODY VALIDATION — accepted v1 behaviour change:
  The CP does NOT validate the request body — the DP re-validates it. So a
  malformed query that passes auth burns one quota unit before the DP returns
  a 400. Today (the monolith) validation precedes quota, so a malformed body
  burns nothing. This is a minor, accepted v1 behaviour change; there is no
  refund path.
"""

import asyncio
import hashlib
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

from adapters.observability import metrics as obs_metrics
from adapters.state.state import get_tenant_dp_pool, try_consume_query
from services.auth.jwt_utils import current_tenant_id
from services.auth.quota import query_quota_429, quotas_enabled, rate_limit

# Trusted header names — must match `services/query_api/dp_query.py` and the
# CP↔DP internal contract.
TENANT_HEADER = "X-RB-Tenant-Id"
PROXY_SECRET_HEADER = "X-RB-Proxy-Secret"

# The shared Query-DP pool name (matches `state._DEFAULT_DP_POOL`). Kept here
# as a constant so the resolver does not import a private state symbol.
SHARED_POOL = "shared"

# --- DP-pool in-process cache ---------------------------------------------
#
# A tenant's dp_pool column changes rarely (only when a tenant is migrated to
# a dedicated pool). Fetching it from Postgres on every query adds a DB call
# on the hot path. A small in-process TTL cache eliminates this lookup on
# warm requests.
#
# Cache key: tenant_id (str). Value: (pool_name, expiry_monotonic).
# Thread-safe: a threading.Lock guards mutations.
#
# NOTE: like the auth cache, this is process-local. If a tenant's dp_pool is
# updated in the DB (a rare operational event), it takes effect on each CP
# worker within _DP_POOL_CACHE_TTL_S (default 60 s). That is an accepted
# bounded tradeoff — pool migrations are not latency-sensitive events.

_DP_POOL_CACHE_TTL_S: float = 60.0  # seconds; overridable in tests
_DP_POOL_CACHE: Dict[str, Tuple[str, float]] = {}  # tenant_id -> (pool, expiry)
_DP_POOL_CACHE_LOCK = threading.Lock()


def get_tenant_dp_pool_cached(tenant_id: str) -> str:
    """Return the DP pool for `tenant_id`, using a short TTL in-process cache.

    Calls the real `get_tenant_dp_pool` (Postgres) only on a cold miss or
    after TTL expiry. This eliminates the dp_pool lookup Postgres call from
    the hot query path on warm requests.
    """
    now = time.monotonic()
    with _DP_POOL_CACHE_LOCK:
        cached = _DP_POOL_CACHE.get(tenant_id)
    if cached is not None:
        pool, expiry = cached
        if now < expiry:
            return pool
        with _DP_POOL_CACHE_LOCK:
            _DP_POOL_CACHE.pop(tenant_id, None)

    pool = get_tenant_dp_pool(tenant_id)
    with _DP_POOL_CACHE_LOCK:
        _DP_POOL_CACHE[tenant_id] = (pool, now + _DP_POOL_CACHE_TTL_S)
    return pool


router = APIRouter()


# --- DP-pool -> base-URL resolver -----------------------------------------


def resolve_dp_base_url(dp_pool: str) -> str:
    """Resolve a `dp_pool` name to the base URL of a Query-DP node.

    Routing convention:

      - `'shared'` -> the shared Query-DP pool, from env `QUERY_DP_URL`
        (dev default `http://localhost:8090`). In production this is the
        private-network address (e.g. Docker network, k8s Service) of the
        shared private Query-DP.

      - `'dedicated-<tenant>'` -> that tenant's dedicated Query-DP pool. No
        dedicated pool is deployed yet, so this is a **documented best-effort
        convention**: the per-tenant base URL is read from a per-tenant env
        var `QUERY_DP_URL_<TENANT>` (the tenant id upper-cased, non-alphanumeric
        chars replaced with `_`). When that env var is unset the resolver falls
        back to the shared pool — a tenant flagged for a dedicated pool that has
        not actually been provisioned transparently keeps using `shared` rather
        than routing to an unreachable host. The dedicated-pool addressing
        scheme is finalised when the first dedicated pool is provisioned.

    The returned URL has no trailing slash; callers append `/v1/query` etc.
    """
    shared = os.getenv("QUERY_DP_URL", "http://localhost:8090").rstrip("/")
    if not dp_pool or dp_pool == SHARED_POOL:
        return shared
    if dp_pool.startswith("dedicated-"):
        tenant = dp_pool[len("dedicated-") :]
        env_key = "QUERY_DP_URL_" + "".join(
            c if c.isalnum() else "_" for c in tenant
        ).upper()
        dedicated = os.getenv(env_key)
        if dedicated:
            return dedicated.rstrip("/")
        # Dedicated pool not provisioned yet -> fall back to shared, never to
        # an unroutable host.
        return shared
    # Unrecognised pool name -> shared, same fail-safe principle.
    return shared


# --- Rendezvous-hash (HRW) routing ----------------------------------------
#
# Rendezvous hashing is shipped additively, behind `RB_ROUTING_RENDEZVOUS=true`.
# The default CP routing (see `resolve_dp_base_url` above) is a
# single-DP-per-pool static map: each pool's env var (`QUERY_DP_URL`,
# `QUERY_DP_URL_<TENANT>`) resolves to one URL. Rendezvous earns its keep
# only when ≥2 DPs serve the same pool, so the gate flag + a
# comma-separated env var (`QUERY_DP_URL=http://dp-1,http://dp-2`) opt in
# to HRW. With the flag unset, behaviour is identical to the static map; a
# misconfigured comma-separated value emits a one-shot WARNING and
# gracefully degrades to the first URL so queries keep flowing while the
# operator fixes the config.
#
# Routing key shape: `f"{tenant}|{dataset}"`. Using `(tenant, dataset)`
# (not `shard_uri`) keeps same-dataset queries stable per-DP across version
# changes. Residency-hint routing (preferring the DP that already holds the
# shard warm) is not yet implemented.
#
# Identity for HRW: the URL string itself is the HRW key. The host:port
# pair is stable per deployment slot, so there is no need for an explicit
# `dp_id` env on the CP side — one less moving part.

# Process-local set of pool/env tuples we have already warned about, so a
# misconfigured operator does not flood the logs.
_MULTI_URL_WARNED: set = set()
_MULTI_URL_WARNED_LOCK = threading.Lock()


def _routing_enabled() -> bool:
    """`True` when `RB_ROUTING_RENDEZVOUS` is set to a truthy value.

    Read live (per-request) so a test can flip the gate without re-importing
    the module — matches the style of `_query_timeout` / `_connect_retries`.
    """
    raw = os.getenv("RB_ROUTING_RENDEZVOUS", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _routing_key(tenant_id: str, dataset: str) -> str:
    """Build the HRW routing key from `(tenant, dataset)`.

    Plain string concat with a `|` separator — the same dataset always maps
    to the same key regardless of version, so the same DP keeps serving its
    queries across rebuilds. Whitespace and other special chars in the
    dataset name are preserved verbatim; HRW only cares that the byte
    string is stable.
    """
    return f"{tenant_id}|{dataset}"


def _hrw_pick(routing_key: str, dp_urls: List[str]) -> str:
    """Return the URL that wins `argmax_url hash(routing_key, url)` — HRW.

    Highest Random Weight (rendezvous) hashing: for every URL in the pool,
    score it by `sha256(f"{key}|{url}")` and pick the maximum. The function
    is pure, order-independent, and (for stable inputs) deterministic.

    Properties this enforces:
      - same `(key, urls)` -> same URL on every call;
      - rough uniform distribution across the pool over many keys;
      - minimal disruption — adding/removing one DP only moves the
        `1/N`-share of keys that the change touches.
    """
    if not dp_urls:
        raise ValueError("_hrw_pick: dp_urls is empty")
    return max(
        dp_urls,
        key=lambda u: hashlib.sha256(f"{routing_key}|{u}".encode("utf-8")).digest(),
    )


def _hrw_rank(routing_key: str, dp_urls: List[str]) -> List[str]:
    """Return every URL ordered by descending HRW weight (rank 0 is the pick).

    The unhealthy-DP fallback chain walks this list: try rank 0; if it is
    sick, try rank 1; etc. The returned list is a permutation of `dp_urls`
    — every URL appears exactly once.
    """
    if not dp_urls:
        return []
    return sorted(
        dp_urls,
        key=lambda u: hashlib.sha256(f"{routing_key}|{u}".encode("utf-8")).digest(),
        reverse=True,
    )


def _is_dp_healthy(url: str) -> bool:  # noqa: ARG001 - hook signature is the contract
    """DP-health hook for the unhealthy-DP fallback chain.

    Residency-aware routing is not yet wired; an active `/healthz` poller
    that feeds this hook is not yet implemented. This defaults `True` so
    HRW picks are honoured. Tests monkeypatch this to drive the
    fallback-chain assertions.
    """
    return True


def _parse_dp_urls(raw: str) -> List[str]:
    """Split a comma-separated DP URL env var into a normalised URL list.

    Each segment is stripped of whitespace and trailing slashes (matching
    `resolve_dp_base_url`'s `rstrip("/")` rule). An empty/whitespace-only
    segment is dropped. The result preserves the operator's order — that
    order is the all-DPs-unhealthy fallback choice, so it must be stable.
    """
    if not raw:
        return []
    parts = [seg.strip().rstrip("/") for seg in raw.split(",")]
    return [p for p in parts if p]


def _resolve_pool_urls(dp_pool: str) -> Tuple[str, List[str]]:
    """Resolve a `dp_pool` name to its (env-var-name, [url, ...]) configuration.

    Mirrors the env-var-name choices that `resolve_dp_base_url` makes
    (`QUERY_DP_URL` for the shared pool, `QUERY_DP_URL_<TENANT>` for a
    dedicated pool, with the same shared-pool fallback when the dedicated
    env var is unset) and returns the parsed URL list. Returning the env
    var name alongside the URLs lets the misconfiguration WARNING name the
    exact variable the operator must fix.
    """
    shared_env = "QUERY_DP_URL"
    shared_raw = os.getenv(shared_env, "http://localhost:8090")
    if not dp_pool or dp_pool == SHARED_POOL:
        return shared_env, _parse_dp_urls(shared_raw)
    if dp_pool.startswith("dedicated-"):
        tenant = dp_pool[len("dedicated-") :]
        env_key = "QUERY_DP_URL_" + "".join(
            c if c.isalnum() else "_" for c in tenant
        ).upper()
        dedicated_raw = os.getenv(env_key)
        if dedicated_raw:
            return env_key, _parse_dp_urls(dedicated_raw)
        # Dedicated pool not provisioned -> fall back to shared (parity with
        # `resolve_dp_base_url`'s same fail-safe).
        return shared_env, _parse_dp_urls(shared_raw)
    # Unrecognised pool name -> shared.
    return shared_env, _parse_dp_urls(shared_raw)


def _warn_misconfigured_multi_url(dp_pool: str, env_key: str, n_urls: int) -> None:
    """One-shot WARNING (per pool) when a comma-separated URL is set without the flag.

    Operator-misconfiguration safety net: if `QUERY_DP_URL=a,b,c` is set
    but `RB_ROUTING_RENDEZVOUS` is unset, we surface the mismatch
    immediately rather than silently routing every query to the first URL.

    The dedup key includes `dp_pool` so two pools with distinct env vars
    cannot accidentally share a slot. Today every pool maps 1:1 to a
    distinct env var (`shared`->`QUERY_DP_URL`, `dedicated-X`->`QUERY_DP_URL_X`),
    but pinning the contract here means a future pool-naming refactor
    cannot silently suppress one pool's WARNING.
    """
    key = (dp_pool, env_key)
    with _MULTI_URL_WARNED_LOCK:
        if key in _MULTI_URL_WARNED:
            return
        _MULTI_URL_WARNED.add(key)
    logger.warning(
        "Configuration mismatch: %s carries %d comma-separated URLs but "
        "RB_ROUTING_RENDEZVOUS is unset. Falling back to the first URL for "
        "this pool. Set RB_ROUTING_RENDEZVOUS=true to enable rendezvous "
        "(HRW) routing across all %d URLs.",
        env_key,
        n_urls,
        n_urls,
    )


def pick_dp_url(dp_pool: str, routing_key: str) -> str:
    """Resolve `dp_pool` to a concrete DP URL, gated on `RB_ROUTING_RENDEZVOUS`.

    Flag off (default):
      - Single URL -> return it (identical to `resolve_dp_base_url`).
      - Multi URL -> log a one-shot WARNING and return the first URL so
        queries keep flowing through the misconfiguration.

    Flag on:
      - HRW-rank the URL list by `routing_key`.
      - Return the highest-rank URL whose `_is_dp_healthy` returns True.
      - If every URL is unhealthy, fall back to the first URL in
        operator-configured order (a deterministic, predictable choice
        — NOT an HRW pick that would change as keys change). This
        preserves today's "best-effort even when sick" pattern: the
        static map returns the configured URL regardless of health.

    A single-URL pool with the flag on still works — HRW of N=1 is
    trivial, and the health-fallback degenerates to "return that one URL".
    """
    env_key, urls = _resolve_pool_urls(dp_pool)
    if not urls:
        # Empty env var — preserve `resolve_dp_base_url`'s dev default so
        # the caller still gets a URL it can attempt a request against.
        return resolve_dp_base_url(dp_pool)

    if not _routing_enabled():
        if len(urls) > 1:
            _warn_misconfigured_multi_url(dp_pool, env_key, len(urls))
        return urls[0]

    # Flag on — HRW + health-fallback.
    ranked = _hrw_rank(routing_key, urls)
    for url in ranked:
        if _is_dp_healthy(url):
            return url
    # Every DP is unhealthy — best-effort fallback to the first
    # operator-configured URL (config order, not HRW order, so the choice
    # is predictable to the operator regardless of the routing key).
    return urls[0]


# --- per-pool httpx.AsyncClient registry ----------------------------------
#
# One persistent `httpx.AsyncClient` per DP base URL, built lazily on first
# use and reused for the process lifetime (HTTP/2 keep-alive — a fresh client
# per request would defeat connection pooling and the keep-alive the contract
# asks for). A small dict keyed by base URL is the registry; a lock guards
# construction so two coroutines racing the first request for a pool do not
# build two clients.

_CLIENTS: Dict[str, httpx.AsyncClient] = {}
_CLIENTS_LOCK = threading.Lock()


# --- CP→DP timeout defaults (rationale) -----------------------------------
#
# A single ~5s scalar timeout was wrong for this proxy: it conflated two very
# different waits. A query that resolves a large COLD consolidated shard does
# the first S3 GET of that shard plus a large-shard deserialise on the DP, and
# that work legitimately runs WELL past 5s on a perfectly healthy DP — the
# 6 GB-shard first-GET measured ~100s in the mmap bench (see
# docs/architecture/ssd-cache.md). Under the old scalar the CP's
# `httpx.ReadTimeout` fired ~95s before the DP would have answered, turning a
# slow-but-correct query into a spurious 504.
#
# The fix splits the budget by what each phase actually means:
#
#   connect (default 3s): a TCP/TLS handshake to a healthy DP on the private
#     network is sub-second; 3s is generous headroom for a transient blip yet
#     still FAST-FAILS a dead/unreachable DP, so the connect-retry chain
#     (_connect_retries) and the 503 `query_unavailable` mapping stay snappy
#     instead of waiting out a multi-second read budget on a host that will
#     never answer.
#
#   read (default 30s): the time the DP is allowed to spend producing the
#     response body. 30s is ~6x the old scalar — comfortably clears a genuine
#     large-cold-shard fetch+deserialise on a healthy DP — while still bounding
#     a truly stuck DP so a client is not held forever. We deliberately do NOT
#     ship the bench's 120s as the prod default: 120s was an overlay value to
#     stop the bench's 6 GB-shard Docker-network GET from being killed, and a
#     2-minute prod default would mask real DP pathologies and pin a CP worker
#     for two minutes per stuck request. Operators with genuinely huge cold
#     shards raise RB_QUERY_DP_READ_TIMEOUT_S to taste (the bench sets 120s).
#
#   write/pool (default 5s): small request body (a query + vector) and a
#     bounded local connection pool — neither phase is the large-cold-shard
#     bottleneck, so a short, sane bound is fine.
_DEFAULT_CONNECT_TIMEOUT_S = 3.0
_DEFAULT_READ_TIMEOUT_S = 30.0
_DEFAULT_WRITE_TIMEOUT_S = 5.0
_DEFAULT_POOL_TIMEOUT_S = 5.0


def _read_timeout_s() -> float:
    """Resolve the CP→DP READ timeout in seconds (env-tunable).

    Precedence: the dedicated `RB_QUERY_DP_READ_TIMEOUT_S` knob wins; falling
    back to the LEGACY single-scalar `RB_QUERY_DP_TIMEOUT_S` (which used to be
    the whole timeout and is, conceptually, the read budget) so an existing
    deployment that tuned the old knob keeps tuning the read timeout; finally
    the 30s default. Read live so an operator or a test can retune without
    re-importing.
    """
    raw = os.getenv("RB_QUERY_DP_READ_TIMEOUT_S")
    if raw is None:
        # Backwards-compat: the legacy scalar knob now tunes the read budget.
        raw = os.getenv("RB_QUERY_DP_TIMEOUT_S")
    if raw is None:
        return _DEFAULT_READ_TIMEOUT_S
    try:
        return max(0.1, float(raw))
    except (TypeError, ValueError):
        return _DEFAULT_READ_TIMEOUT_S


def _connect_timeout_s() -> float:
    """Resolve the CP→DP CONNECT timeout in seconds (env-tunable, default 3s).

    Short so a dead/unreachable DP fast-fails the connect-retry chain instead
    of waiting out the (much larger) read budget. Read live.
    """
    try:
        return max(0.1, float(os.getenv("RB_QUERY_DP_CONNECT_TIMEOUT_S", str(_DEFAULT_CONNECT_TIMEOUT_S))))
    except (TypeError, ValueError):
        return _DEFAULT_CONNECT_TIMEOUT_S


def _query_timeout() -> httpx.Timeout:
    """CP→DP query timeout as a SPLIT `httpx.Timeout` (env-tunable).

    Separate connect vs read (plus write/pool) instead of one scalar: a SHORT
    connect fast-fails a dead DP, while a generously LARGER read lets a slow
    large-cold-shard query on a HEALTHY DP finish. See the rationale block and
    the per-knob helpers above. Read live so an operator or a test can retune
    it without re-importing.

    Env knobs (all `RB_QUERY_DP_*`, matching the existing convention here):
      - `RB_QUERY_DP_READ_TIMEOUT_S` — read budget (default 30s); the legacy
        `RB_QUERY_DP_TIMEOUT_S` still works as a fallback for the read budget.
      - `RB_QUERY_DP_CONNECT_TIMEOUT_S` — connect budget (default 3s).
    """
    return httpx.Timeout(
        connect=_connect_timeout_s(),
        read=_read_timeout_s(),
        write=_DEFAULT_WRITE_TIMEOUT_S,
        pool=_DEFAULT_POOL_TIMEOUT_S,
    )


def _connect_retries() -> int:
    """Number of CP→DP connect retries (env-tunable, default 2).

    A query is read-only and safe to retry on a *connect* failure (DP node
    unreachable / connection refused). A request that reached the DP and
    produced a 5xx is NOT retried — see `_proxy`.
    """
    try:
        return max(0, int(os.getenv("RB_QUERY_DP_CONNECT_RETRIES", "2")))
    except (TypeError, ValueError):
        return 2


def _connect_retry_backoff_s() -> float:
    """Async backoff between CP→DP connect retries, in seconds.

    Env-tunable (`RB_QUERY_DP_CONNECT_BACKOFF_S`), default 25ms. Without a
    pause the connect-retry loop spins with zero delay, hammering an
    unreachable DP. A short sleep gives a transiently-restarting DP node a
    moment to come back.
    """
    try:
        return max(0.0, float(os.getenv("RB_QUERY_DP_CONNECT_BACKOFF_S", "0.025")))
    except (TypeError, ValueError):
        return 0.025


def _build_client(base_url: str) -> httpx.AsyncClient:
    """Construct the persistent `httpx.AsyncClient` for one DP pool.

    HTTP/2 keep-alive enabled and a bounded connection pool, per the contract.
    The split connect/read `httpx.Timeout` (see `_query_timeout`) is baked in
    as the client default so every request through this persistent client
    inherits the generous read budget that a large COLD-shard query needs. It
    is ALSO passed per-request in `_proxy` (read live) so an operator can
    retune the timeout at runtime without rebuilding the client, and so a
    test-registered client (`register_dp_client`) — built without this default
    — still gets the configured split timeout on each call.
    """
    return httpx.AsyncClient(
        base_url=base_url,
        http2=True,
        timeout=_query_timeout(),
        limits=httpx.Limits(
            max_connections=int(os.getenv("RB_QUERY_DP_MAX_CONNECTIONS", "100")),
            max_keepalive_connections=int(
                os.getenv("RB_QUERY_DP_MAX_KEEPALIVE", "20")
            ),
        ),
    )


def _get_client(base_url: str) -> httpx.AsyncClient:
    """Return the persistent client for `base_url`, building it on first use."""
    client = _CLIENTS.get(base_url)
    if client is not None:
        return client
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(base_url)
        if client is None:
            client = _build_client(base_url)
            _CLIENTS[base_url] = client
        return client


def register_dp_client(base_url: str, client: httpx.AsyncClient) -> None:
    """Register a pre-built `httpx.AsyncClient` for a DP base URL.

    Test hook: the proxy is exercised in-process against `dp_app.py` via an
    `httpx.ASGITransport`, so a test builds a client wired to the DP ASGI app
    and registers it under the base URL `resolve_dp_base_url` will produce.
    Never called in production — there the registry builds real network
    clients lazily.
    """
    with _CLIENTS_LOCK:
        _CLIENTS[base_url] = client


async def reset_dp_clients() -> None:
    """Close and drop every registered DP client (test teardown / reset)."""
    with _CLIENTS_LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
    for client in clients:
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


def _err(status_code: int, code: str, message: str) -> JSONResponse:
    """Build a v1 error envelope response for a CP-originated error."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _trusted_headers(tenant_id: str) -> Dict[str, str]:
    """Build the trusted CP→DP request headers for `tenant_id`.

    Always carries `X-RB-Tenant-Id` (the CP-verified tenant) and
    `Content-Type: application/json`. `X-RB-Proxy-Secret` is added only when
    `RB_PROXY_SECRET` is set on the CP — the DP skips the check when its own
    `RB_PROXY_SECRET` is unset. The customer's `Authorization` header is never
    included.
    """
    headers = {
        TENANT_HEADER: tenant_id,
        "Content-Type": "application/json",
    }
    secret = os.getenv("RB_PROXY_SECRET")
    if secret:
        headers[PROXY_SECRET_HEADER] = secret
    return headers


async def _proxy(
    method: str,
    path: str,
    tenant_id: str,
    *,
    body: Optional[bytes] = None,
) -> Response:
    """Reverse-proxy one request to the tenant's Query-DP node.

    Resolves `tenant_id -> dp_pool -> base URL`, sends the request with the
    trusted headers, and maps the outcome per the CP↔DP contract's failure
    table:

      - DP 2xx/4xx/5xx -> forwarded verbatim (status code + body);
      - connect failure (`ConnectError`/`ConnectTimeout`), after
        `_connect_retries()` retries -> 503 `query_unavailable` (a query is
        read-only, so a connect failure is safe to retry; a DP that *answered*
        with a 5xx is NOT retried);
      - CP→DP read timeout (`ReadTimeout`/`TimeoutException`) -> 504
        `query_timeout`;
      - any OTHER `httpx` transport error — `RemoteProtocolError`, `ReadError`,
        `NetworkError`, `WriteError`, `PoolTimeout`, or any other
        `httpx.HTTPError` — means the request reached the DP but the response
        broke / was incomplete. That is NOT safe to retry (the DP may have done
        work), so it maps to `502 bad_gateway`. An `httpx.HTTPError` never
        escapes this function as a bare ASGI 500.

    All CP-originated errors use the v1 `{"error": {"code","message"}}`
    envelope.
    """
    # `get_tenant_dp_pool_cached` checks the in-process TTL cache first; only
    # on a cold miss does it make a Postgres call. It is still a SYNC call —
    # offload it to a worker thread so the event loop is never blocked.
    # `contextvars` are copied into the `to_thread` thread, so a
    # request-scoped connection bound by `RequestScopedConnectionMiddleware`
    # before this call is still visible to `pooled_conn()` inside it.
    dp_pool = await asyncio.to_thread(get_tenant_dp_pool_cached, tenant_id)
    base_url = resolve_dp_base_url(dp_pool)
    client = _get_client(base_url)
    headers = _trusted_headers(tenant_id)
    timeout = _query_timeout()

    attempts = _connect_retries() + 1
    last_connect_error: Optional[Exception] = None
    backoff = _connect_retry_backoff_s()
    for _attempt in range(attempts):
        try:
            resp = await client.request(
                method,
                path,
                content=body,
                headers=headers,
                timeout=timeout,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Connect failure: the request never reached the DP, so it is safe
            # to retry. Loop and try again (up to `attempts`). A small async
            # backoff between attempts prevents spinning with zero delay
            # against an unreachable / restarting DP.
            last_connect_error = exc
            if _attempt < attempts - 1 and backoff > 0:
                await asyncio.sleep(backoff)
            continue
        except httpx.PoolTimeout:
            # The CP-side httpx connection pool had no slot free in time. The
            # request never left the CP, but it is a saturation signal, not a
            # DP-reached condition — bucket it with the other non-retryable
            # transport errors as a 502 (it is NOT a DP read timeout).
            # Listed BEFORE `TimeoutException` because `PoolTimeout` subclasses
            # it and would otherwise be caught as `query_timeout`.
            return _err(
                502,
                "bad_gateway",
                "Query service returned an invalid or incomplete response",
            )
        except (httpx.ReadTimeout, httpx.TimeoutException):
            # The request reached the DP but did not answer in time. NOT
            # retried — the DP may have done work; surface a 504.
            return _err(504, "query_timeout", "Query timed out")
        except httpx.HTTPError:
            # Any other transport/protocol error — `RemoteProtocolError`
            # (DP closed the connection / HTTP/2 GOAWAY mid-response),
            # `ReadError`, `NetworkError`, `WriteError`, etc. The request
            # reached the DP but the response broke / was incomplete; the DP
            # may have done work, so this is NOT safe to retry. Surface a
            # v1-envelope 502 rather than letting the `httpx.HTTPError` escape
            # as a bare ASGI 500.
            return _err(
                502,
                "bad_gateway",
                "Query service returned an invalid or incomplete response",
            )
        # Reached the DP — forward its response verbatim (any status code).
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    # Exhausted connect retries — the DP pool is unreachable.
    return _err(
        503,
        "query_unavailable",
        "Query service is temporarily unavailable",
    )


@router.post("/v1/query")
async def proxy_v1_query(
    request: Request,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """Authenticate, consume query quota, then reverse-proxy to a Query-DP node.

    The customer-observable contract (`docs/api/v1.md`) is unchanged — the CP
    is a transparent reverse proxy. Auth + rate limit run as dependencies; the
    quota consume + proxy run in the handler body.
    """
    # Quota is consumed AFTER auth/rate-limit and BEFORE the proxy hop. Note
    # (per the CP↔DP contract): unlike the monolith, the CP does NOT validate
    # the body first — the DP re-validates it — so a malformed query that
    # passes auth burns one quota unit before the DP 400s it. Accepted minor
    # v1 behaviour change; there is no refund path.
    #
    # `try_consume_query` is a SYNC Postgres write (a pool checkout + an
    # `UPDATE ... RETURNING`). Offload it so the blocking DB I/O — including
    # the block-with-timeout pool checkout — never stalls the event loop. The
    # request-scoped connection contextvar is copied into the worker thread,
    # so `pooled_conn()` inside the call still resolves the request connection.
    #
    # OSS opt-in: skipped entirely when `RB_ENABLE_QUOTAS` is unset/false (the
    # self-host default). The proxy hop then carries no quota cost.
    if quotas_enabled():
        try:
            ok, usage = await asyncio.to_thread(try_consume_query, tenant_id)
        except ValueError:
            # `try_consume_query` raised — the tenant row is missing for an
            # ALREADY-authenticated tenant. That is a server-state inconsistency,
            # not a transient routing failure: a retryable 503 would be wrong (the
            # row will not reappear on retry). Map it to a non-retryable
            # `500 internal_error` v1 envelope.
            return _err(500, "internal_error", "Internal error resolving tenant")
        if not ok:
            obs_metrics.record_quota_rejection("query")
            return query_quota_429(usage)

    # Forward the raw request body byte-for-byte — the DP re-validates it.
    body = await request.body()
    return await _proxy("POST", "/v1/query", tenant_id, body=body)


@router.get("/v1/query/status/{job_id}")
async def proxy_v1_query_status(
    job_id: str,
    tenant_id: str = Depends(current_tenant_id),
):
    """Authenticate the poll at the edge, then reverse-proxy to the Query-DP node.

    No quota and no `rate_limit` — a status poll consumes neither (parity with
    the legacy in-process status route). `current_tenant_id` still runs so the
    poll is authenticated and the tenant picks the DP pool.
    """
    return await _proxy("GET", f"/v1/query/status/{job_id}", tenant_id)
