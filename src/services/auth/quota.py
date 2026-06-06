"""Per-API-key rate limiting + a shared dependency surface for quota
enforcement.

Two concerns live here so both the ingest service (`source_registry`) and the
query service (`query_api`) can import one module:

  - `rate_limit` — a FastAPI dependency that enforces an in-memory token-bucket
    rate limit (50 req/s sustained, burst 100) per API key, or per tenant when
    the request authenticated with a JWT. Exhaustion raises a 429 `rate_limited`.
  - `quota_429` — a small helper that turns a `state.try_consume_*` failure into
    the contract's 429 JSONResponse, so the ingest/query handlers stay terse.

Design notes / MVP limitations:

  - The token buckets live in a process-local dict. They are NOT shared across
    workers/pods and NOT persisted across restarts. A restart resets every
    bucket to full; with N pods the effective limit is N x the configured rate.
    This is an accepted MVP tradeoff — the v1 contract documents it explicitly.
    The upgrade path is a shared Redis token bucket.
  - The limiter keys on the raw bearer token: an `rb_live_...` key buckets per
    key, a JWT buckets per tenant (we hash the JWT down to its `sub`). Keying a
    JWT on the tenant is intentional — dashboard traffic is low-volume and we do
    not want each page load minting a fresh bucket.
  - Bucket math is a lock + two floats (tokens, last-refill timestamp); there
    is no background sweeper thread. Idle buckets are left in the dict; at MVP
    scale (<1000 keys) the memory cost is negligible.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Dict, Optional

from fastapi import Header
from fastapi.responses import JSONResponse

from adapters.errors import RateLimited, set_rate_limited_response_factory
from adapters.observability import metrics as obs_metrics
from services.auth.jwt_utils import (
    API_KEY_PREFIX,
    DEFAULT_TENANT_ID,
    _parse_bearer,
    _unauthorized,
    auth_required,
    decode_jwt,
)


# --- OSS opt-in switch ----------------------------------------------------

# `RB_ENABLE_QUOTAS` is the single env switch that turns the per-tenant quota
# and rate-limit subsystem on. It defaults OFF so a self-hoster running their
# own database (the headline `docker compose up` path) is never throttled by
# their own queries — a 10k-queries/day cap on a self-host install would be a
# footgun. A deployment turns it on by exporting `RB_ENABLE_QUOTAS=true` in its environment.
#
# When OFF:
#   - the `rate_limit` FastAPI dependency is a no-op (no token-bucket math).
#   - the ingest / query handlers skip `try_consume_vectors` /
#     `try_consume_query` entirely — the runtime check is what's gated, not
#     the schema (`tenants.vectors_used`, `tenants.queries_today` etc. stay).
#   - `GET /auth/usage` returns an honest `{"enabled": false}` payload (see
#     `services/auth/auth.py`).
#
# Read fresh on every call (not module-level) so a test can flip it via
# monkeypatch without re-importing this module.
def quotas_enabled() -> bool:
    """Whether per-tenant quotas + the rate limiter are active.

    Defaults to OFF (self-host friendly). Set `RB_ENABLE_QUOTAS=true` to
    turn full quota enforcement back on.
    Truthy values: `1`, `true`, `yes`, `on` (case-insensitive).
    """
    return os.getenv("RB_ENABLE_QUOTAS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# --- rate limiter ---------------------------------------------------------

# MVP defaults from the v1 contract "Rate limits" section. Overridable via env
# so the E2E harness can dial them down to force a `rate_limited` 429 cheaply.
RATE_LIMIT_RPS = float(os.getenv("RB_RATE_LIMIT_RPS", "50"))
RATE_LIMIT_BURST = float(os.getenv("RB_RATE_LIMIT_BURST", "100"))


class _TokenBucket:
    """A classic token bucket: `tokens` refills at `rate`/s, capped at `burst`."""

    __slots__ = ("tokens", "last", "rate", "burst")

    def __init__(self, rate: float, burst: float) -> None:
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last = time.monotonic()

    def take(self) -> bool:
        """Refill based on elapsed time, then try to spend one token.

        Returns True if a token was available (request allowed), False if the
        bucket is empty (request should be 429'd).
        """
        now = time.monotonic()
        elapsed = now - self.last
        self.last = now
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


_BUCKETS: Dict[str, _TokenBucket] = {}
_BUCKETS_LOCK = threading.Lock()


def _bucket_key(token: str) -> str:
    """Derive a stable bucket key from a bearer token.

    An `rb_live_...` API key buckets per key (the raw token is the key). A JWT
    buckets per tenant: we decode it and key on `tenant:<sub>` so all of a
    tenant's dashboard sessions share one bucket. A JWT we cannot decode falls
    back to keying on the raw token — it will fail auth downstream anyway.
    """
    if token.startswith(API_KEY_PREFIX):
        return f"key:{token}"
    sub = decode_jwt(token)
    if sub is not None:
        return f"tenant:{sub}"
    return f"raw:{token}"


def _take_token(bucket_key: str) -> bool:
    """Spend one token from `bucket_key`'s bucket, creating it on first use."""
    with _BUCKETS_LOCK:
        bucket = _BUCKETS.get(bucket_key)
        if bucket is None:
            bucket = _TokenBucket(RATE_LIMIT_RPS, RATE_LIMIT_BURST)
            _BUCKETS[bucket_key] = bucket
        return bucket.take()


def reset_rate_limiter() -> None:
    """Drop every token bucket. Test hook — never called in production."""
    with _BUCKETS_LOCK:
        _BUCKETS.clear()


def _rate_limited_response() -> JSONResponse:
    """Build the v1 `rate_limited` 429 envelope."""
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "rate_limited",
                "message": "Rate limit exceeded; slow down and retry",
                "details": {
                    "limit_rps": int(RATE_LIMIT_RPS),
                    "burst": int(RATE_LIMIT_BURST),
                },
            }
        },
    )


# `RateLimited` is defined once in `adapters.errors` (so its class identity is
# shared and the registered exception handler / isinstance checks keep working)
# and re-exported here under its original `services.auth.quota.RateLimited` name.
# Its constructor builds `self.response` from `_rate_limited_response`, which we
# register as the factory below — keeping the response-shaping concern local to
# this module while the class lives in the stdlib-only errors leaf.
set_rate_limited_response_factory(_rate_limited_response)


def rate_limit(authorization: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency: enforce the per-key token bucket.

    Apply to customer-facing v1 endpoints (`/v1/datasets*`, `/v1/query`). It
    parses the bearer token, derives the bucket key, and spends one token. An
    empty bucket raises `RateLimited` → 429 `rate_limited`. A missing/malformed
    Authorization header raises the standard 401 (auth runs anyway, but failing
    fast here keeps the behaviour obvious).

    This dependency does NOT itself resolve the tenant — the endpoint still
    depends on `current_tenant_id` separately. Ordering does not matter; both
    just read the same header.

    OSS opt-in: when `RB_ENABLE_QUOTAS` is unset/false (the self-host default)
    the dependency is a no-op — no header parsing, no bucket math, no 429.

    When `RB_REQUIRE_AUTH` is unset/false but quotas are on, there is no
    bearer token to parse — `current_tenant_id` short-circuits to the
    bootstrap "default" tenant for every caller. We key the rate-limit
    bucket on that single tenant id so the limiter still throttles (the
    quotas-on operator wanted that) without 401'ing requests that, by
    deployment policy, are exempt from carrying credentials.
    """
    if not quotas_enabled():
        return
    if not auth_required():
        # OSS single-tenant mode: bucket the whole process under the default
        # tenant. The bucket math is unchanged — every request shares one
        # bucket, which is the correct semantic for "single principal".
        if not _take_token(f"tenant:{DEFAULT_TENANT_ID}"):
            obs_metrics.record_quota_rejection("rate_limit")
            raise RateLimited()
        return
    token = _parse_bearer(authorization)  # raises 401 on missing/garbled
    if not _take_token(_bucket_key(token)):
        # rosalinddb.quota.rejections{kind=rate_limit}. No per-key label —
        # the bucket key embeds the raw token and must never become a metric
        # attribute (cardinality rule).
        obs_metrics.record_quota_rejection("rate_limit")
        raise RateLimited()


def install_rate_limit_handler(app) -> None:
    """Register an app-level handler that turns `RateLimited` into its 429.

    Call once per app at startup (alongside `install_exception_handlers`).
    """

    @app.exception_handler(RateLimited)
    async def _rate_limited_handler(_req, exc: RateLimited):  # noqa: F811
        return exc.response


# --- quota 429 helper -----------------------------------------------------


def query_quota_429(usage: dict) -> JSONResponse:
    """Build the 429 `query_quota_exceeded` envelope from a usage snapshot."""
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "query_quota_exceeded",
                "message": "Daily query quota exceeded for this tenant",
                "details": {
                    "limit": usage.get("daily_query_quota"),
                    "reset_at": usage.get("queries_reset_at"),
                },
            }
        },
    )


def vector_quota_429(usage: dict) -> JSONResponse:
    """Build the 429 `vector_quota_exceeded` envelope from a usage snapshot."""
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "vector_quota_exceeded",
                "message": "Vector storage quota exceeded for this tenant",
                "details": {
                    "limit": usage.get("vector_quota"),
                    "used": usage.get("vectors_used"),
                },
            }
        },
    )
