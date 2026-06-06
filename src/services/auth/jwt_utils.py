"""JWT encode/decode + the `current_tenant_id` FastAPI dependency.

The token payload is intentionally minimal: `sub` (tenant id), `iat`,
`exp`. The dependency parses `Authorization: Bearer <token>` and returns
the resolved tenant_id, or raises a 401 with the canonical error envelope
from the v1 API contract.

`current_tenant_id` accepts API keys of the form `rb_live_<32 url-safe
chars>`. The same `Authorization: Bearer` header carries both kinds of
token; downstream handlers do not learn which was used. A second dependency,
`current_tenant_id_jwt_only`, is exported for the small set of endpoints
(currently `POST /auth/keys`) that must refuse API-key auth — see the v1
contract under "Auth".

API-key auth cache
------------------
Under concurrent load the CP was making a Postgres lookup on EVERY request
to resolve an API key to a tenant. With 10 concurrent VUs this exhausted
the per-worker connection pool and forced expensive cold TCP+auth opens.

_resolve_api_key now maintains a small in-process LRU-style cache keyed by
sha256(raw_key) -> (tenant_id, expiry_monotonic). A cache hit skips both
the Postgres lookup AND the fire-and-forget touch update (since the TTL
naturally throttles last_used_at drift).

CORRECTNESS REQUIREMENTS:
  - Cache key is sha256(raw_key), never the raw secret.
  - Only successful resolutions are cached (never failed/unknown/revoked).
  - A revoked key stops working quickly via TWO mechanisms:
      (a) 30 s TTL bounds staleness on THIS process;
      (b) DELETE /auth/keys/{id} calls bust_api_key_cache(raw_key) to
          evict the entry immediately on THIS worker.
  - IMPORTANT: a revoked key may still authenticate for up to TTL on OTHER
    CP worker processes / machines that did not handle the DELETE request.
    Cross-process cache invalidation via Redis is a deliberate non-goal here
    (follow-up: add a Redis pub/sub bust channel when cross-worker
    invalidation becomes required). The 30 s window is an accepted bounded
    tradeoff for a massive reduction in per-query Postgres calls.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from typing import Dict, Optional, Tuple

import jwt as pyjwt
from fastapi import Header, HTTPException, status

from adapters import config
from adapters.state import state as state_mod


logger = logging.getLogger(__name__)


# --- OSS opt-in switch: auth + tenancy ------------------------------------
#
# `RB_REQUIRE_AUTH` mirrors `RB_ENABLE_QUOTAS` (see services/auth/quota.py).
# It is the single env switch that turns the auth + tenancy stack on. It
# defaults OFF so the headline self-host path (`docker compose up`) works
# without anyone first having to call `/auth/signup` and ferry JWTs / API
# keys around — a self-hoster running a single tenant on their own box is
# the principal, and forcing them through a sign-up flow is a footgun.
#
# When OFF:
#   - `current_tenant_id` short-circuits to `DEFAULT_TENANT_ID` and never
#     reads the `Authorization` header — all calls resolve to the bootstrap
#     "default" tenant.
#   - `/auth/signup`, `/auth/login`, `/auth/me`, `/auth/keys*` return 404
#     (the surface is hidden, not just neutered — see services/auth/auth.py).
#   - `/auth/usage` returns `{"enabled": false}` with no tenant context.
#   - A loud warning is logged once at startup so a misconfigured public
#     deploy is obvious in the logs (see services/source_registry/main.py).
#
# When ON (set via environment / k8s manifest):
#   - Behaviour is unchanged: every request must carry a valid JWT or
#     `rb_live_...` API key, the principal is resolved per-request, and
#     `/auth/*` is fully wired.
#
# Read fresh on every call (not module-level) so a test can flip it via
# monkeypatch without re-importing this module.
DEFAULT_TENANT_ID = "default"


def auth_required() -> bool:
    """Whether auth/tenancy is enforced. Default OFF for OSS self-hosters.

    Truthy values: `1`, `true`, `yes`, `on` (case-insensitive). Anything else
    — including an unset env var — counts as OFF. Production self-host
    deploys export `RB_REQUIRE_AUTH=true` (see docs/deploy/self-host.md).
    """
    return config.require_auth()


# --- API-key auth resolution cache ----------------------------------------
#
# Small in-process dict: sha256(raw_key) -> (tenant_id, expiry_monotonic).
# Only successful resolutions are stored. The cache is process-local; a
# revoked key may still hit this cache for up to _AUTH_CACHE_TTL_S on
# processes that did not handle the DELETE (see module docstring).
#
# Thread safety: a plain dict with a threading.Lock is sufficient. The lock
# guards only the dict mutation; bcrypt (for passwords) and Postgres calls
# happen OUTSIDE the lock to avoid holding it during I/O.

_AUTH_CACHE_TTL_S: float = 30.0  # seconds; overridable in tests via monkeypatch
_AUTH_CACHE: Dict[str, Tuple[str, float]] = {}  # key_hash -> (tenant_id, expiry)
_AUTH_CACHE_LOCK = threading.Lock()


def _clear_auth_cache() -> None:
    """Clear the auth cache. Exposed for test teardown / module reload."""
    with _AUTH_CACHE_LOCK:
        _AUTH_CACHE.clear()


def bust_api_key_cache(raw_key: str) -> None:
    """Immediately evict `raw_key`'s cache entry on THIS process.

    Called by the DELETE /auth/keys/{id} handler after a successful revocation
    so the revoked key stops working instantly on this worker (not waiting for
    the TTL to expire). Other worker processes are NOT notified — see the module
    docstring for the cross-process staleness note.
    """
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    with _AUTH_CACHE_LOCK:
        _AUTH_CACHE.pop(key_hash, None)


def bust_api_key_cache_by_hash(key_hash: str) -> None:
    """Evict a cache entry by its stored SHA-256 key_hash (not the raw key).

    Used by the DELETE /auth/keys/{id} handler when only the persisted
    key_hash is available (the raw key is never re-surfaced after creation).
    """
    with _AUTH_CACHE_LOCK:
        _AUTH_CACHE.pop(key_hash, None)

# Token lifetime: 24h. Long enough that a UI client does not need a
# refresh-token flow for the MVP; short enough that a leaked token does
# not stay valid forever.
TOKEN_TTL_SECONDS = 60 * 60 * 24
JWT_ALGORITHM = "HS256"


def _resolve_secret() -> str:
    """Return the HS256 signing secret.

    Production MUST set `JWT_SECRET` in the environment. For local dev /
    tests, we fall back to a random 32-byte secret generated once per
    process import. This makes tokens unforgeable across runs but is
    obviously not durable across restarts — a WARNING is logged so any
    accidental prod misconfig is loud.
    """
    env = config.jwt_secret()
    if env:
        return env
    if not getattr(_resolve_secret, "_warned", False):
        logger.warning(
            "JWT_SECRET is not set; using an ephemeral per-process dev "
            "secret. Tokens will be invalidated on restart. DO NOT run "
            "this configuration in production."
        )
        _resolve_secret._warned = True  # type: ignore[attr-defined]
    if not hasattr(_resolve_secret, "_fallback"):
        _resolve_secret._fallback = secrets.token_urlsafe(32)  # type: ignore[attr-defined]
    return _resolve_secret._fallback  # type: ignore[attr-defined]


def encode_jwt(tenant_id: str, ttl_seconds: int = TOKEN_TTL_SECONDS) -> str:
    """Encode a JWT for `tenant_id` valid for `ttl_seconds`."""
    now = int(time.time())
    payload = {"sub": tenant_id, "iat": now, "exp": now + ttl_seconds}
    return pyjwt.encode(payload, _resolve_secret(), algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> Optional[str]:
    """Decode `token` and return the `sub` (tenant_id), or None on any failure.

    Failures collapse to None on purpose: callers should respond with the
    same generic 401 regardless of whether the token was malformed,
    tampered with, expired, or signed with a stale secret. Leaking the
    distinction is a small but real auth-side-channel.
    """
    try:
        payload = pyjwt.decode(token, _resolve_secret(), algorithms=[JWT_ALGORITHM])
    except pyjwt.PyJWTError:
        return None
    sub = payload.get("sub")
    if not isinstance(sub, str):
        return None
    return sub


def _unauthorized() -> HTTPException:
    """Build the canonical 401 with the v1 error envelope as `detail`.

    The auth router translates this into a JSONResponse with the
    `{"error": {"code", "message"}}` body via the FastAPI exception
    handler installed in `auth.py`.
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "unauthorized", "message": "Missing or invalid credentials"},
    )


API_KEY_PREFIX = "rb_live_"


def _parse_bearer(authorization: Optional[str]) -> str:
    """Return the raw bearer token or raise 401.

    Factored out so JWT-only and dual-auth dependencies share the same
    header-parsing rules (and the same 401 envelope on malformed input).
    """
    if not authorization:
        raise _unauthorized()
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise _unauthorized()
    token = parts[1].strip()
    if not token:
        raise _unauthorized()
    return token


def _resolve_api_key(token: str) -> Optional[str]:
    """Resolve `token` against `api_keys` and return the owning tenant_id.

    `rb_live_` keys are stored as `SHA-256(raw_key)` — a deterministic
    digest — so resolution is a single indexed lookup: compute the
    digest, fetch the one matching row (O(1), independent of how many
    keys exist system-wide), and validate it. SHA-256 is the right hash
    here because the token is a 190-bit random value, not a low-entropy
    password — there is no brute-force surface for bcrypt to slow down.

    Returns the tenant_id on a successful match; returns None — so the
    caller maps to 401 — when the token is unknown (no row) or the
    matching key has been revoked.

    The result is cached in `_AUTH_CACHE` for `_AUTH_CACHE_TTL_S` seconds
    (default 30 s). On a cache hit neither the Postgres lookup nor the touch
    UPDATE runs. touch_api_key_last_used is fired on a background daemon
    thread on a cold (cache-miss) hit so the response path does not block on
    the UPDATE.
    """
    key_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    # --- cache lookup -------------------------------------------------------
    now = time.monotonic()
    with _AUTH_CACHE_LOCK:
        cached = _AUTH_CACHE.get(key_hash)
    if cached is not None:
        tenant_id, expiry = cached
        if now < expiry:
            # Cache hit: skip DB lookup and touch entirely.
            return tenant_id
        # Expired entry — fall through to DB lookup.
        with _AUTH_CACHE_LOCK:
            _AUTH_CACHE.pop(key_hash, None)

    # --- cold path: DB lookup -----------------------------------------------
    row = state_mod.get_api_key_by_hash(key_hash)
    if row is None:
        return None
    if row.get("revoked_at") is not None:
        return None

    tenant_id = row["tenant_id"]
    key_id = row["id"]

    # Store in cache BEFORE firing the touch so a concurrent request for the
    # same key sees the entry immediately.
    with _AUTH_CACHE_LOCK:
        _AUTH_CACHE[key_hash] = (tenant_id, now + _AUTH_CACHE_TTL_S)

    # touch_api_key_last_used is fire-and-forget: run it on a background daemon
    # thread so the response path is not blocked by this UPDATE. The TTL on the
    # cache entry naturally throttles how often this fires — at most once per
    # _AUTH_CACHE_TTL_S per key per worker process.
    _fire_touch(key_id)

    return tenant_id


def _fire_touch(key_id: str) -> None:
    """Fire touch_api_key_last_used on a daemon background thread.

    Daemon threads are cleaned up automatically when the process exits so
    there is no need to join them. This must NOT block the response path.
    """
    t = threading.Thread(
        target=state_mod.touch_api_key_last_used,
        args=(key_id,),
        daemon=True,
        name=f"rb-touch-{key_id[:8]}",
    )
    t.start()


def current_tenant_id(authorization: Optional[str] = Header(default=None)) -> str:
    """FastAPI dependency that resolves the caller's `tenant_id` from the
    `Authorization` header.

    Accepts either a JWT (issued by signup/login) or an API key of the
    form `rb_live_<32 url-safe chars>`. Tokens starting with `rb_live_`
    are routed to the API-key path; everything else is decoded as a JWT.
    Any failure on either path collapses to the same 401 with the v1
    `unauthorized` error code.

    OSS opt-in: when `RB_REQUIRE_AUTH` is unset/false (the self-host default)
    this dependency does NOT read the `Authorization` header at all and
    instead returns `DEFAULT_TENANT_ID`. Every request therefore resolves
    to the bootstrap "default" tenant — single-tenant mode is trivially
    correct because every call gets the same tenant_id. Set
    `RB_REQUIRE_AUTH=true` to enable auth enforcement; behaviour is unchanged.
    """
    if not auth_required():
        return DEFAULT_TENANT_ID

    token = _parse_bearer(authorization)

    if token.startswith(API_KEY_PREFIX):
        tenant_id = _resolve_api_key(token)
        if tenant_id is None:
            raise _unauthorized()
        return tenant_id

    tenant_id = decode_jwt(token)
    if tenant_id is None:
        raise _unauthorized()
    return tenant_id


def current_tenant_id_jwt_only(authorization: Optional[str] = Header(default=None)) -> str:
    """Dependency variant that REJECTS API keys.

    Used by `POST /auth/keys` so customer code cannot bootstrap more
    keys from an existing key — the contract is explicit that key
    issuance is a dashboard (JWT) operation. From the caller's
    perspective the failure mode is indistinguishable from any other
    bad token: 401 `unauthorized`.

    OSS opt-in: when `RB_REQUIRE_AUTH` is unset/false this dependency also
    short-circuits to `DEFAULT_TENANT_ID`. In practice it is unreachable in
    OSS mode — the `/auth/keys*` routes 404 before the dependency runs (see
    services/auth/auth.py) — but the short-circuit keeps the two
    dependencies in lockstep so any future caller using this variant gets
    the same single-tenant behaviour.
    """
    if not auth_required():
        return DEFAULT_TENANT_ID
    token = _parse_bearer(authorization)
    if token.startswith(API_KEY_PREFIX):
        raise _unauthorized()
    tenant_id = decode_jwt(token)
    if tenant_id is None:
        raise _unauthorized()
    return tenant_id
