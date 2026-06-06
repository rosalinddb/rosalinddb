"""FastAPI router for `/auth/signup`, `/auth/login`, `/auth/me`,
and the `/auth/keys` surface.

Conforms to the v1 API contract at `docs/api/v1.md`. Signup
auto-issues a "Default" API key — the raw `rb_live_...` value is
returned exactly once in `first_api_key.key` and never re-surfaced.
Subsequent keys are minted via `POST /auth/keys` (JWT-only), listed
via `GET /auth/keys`, and revoked via `DELETE /auth/keys/{id}`.
"""
from __future__ import annotations

import hashlib
import secrets
from typing import Optional
from uuid import uuid4

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, ValidationError

from adapters.errors import error_envelope
from adapters.observability import metrics as obs_metrics
from adapters.state import state as state_mod
from services.auth.jwt_utils import (
    auth_required,
    bust_api_key_cache,
    bust_api_key_cache_by_hash,
    current_tenant_id,
    current_tenant_id_jwt_only,
    encode_jwt,
)
from services.auth.quota import quotas_enabled


router = APIRouter()


# --- request models -------------------------------------------------------


class _CredentialsBase(BaseModel):
    """Shared signup/login payload shape.

    `EmailStr` triggers pydantic's email-validator dependency (already in
    requirements.txt). We deliberately keep `password` as plain `str` —
    length and other policy checks happen in the handler so we can return
    the contract-specific `weak_password` error code rather than pydantic's
    generic validation envelope.
    """
    email: EmailStr
    password: str


class SignupRequest(_CredentialsBase):
    pass


class LoginRequest(_CredentialsBase):
    pass


# --- helpers --------------------------------------------------------------


def _err(status_code: int, code: str, message: str, details: Optional[dict] = None) -> JSONResponse:
    """Build a JSONResponse matching the v1 error envelope.

    Centralising this prevents the router from drifting from the contract
    when new error codes are added; every handler funnels through here.
    Delegates to the canonical `adapters.errors.error_envelope` (same
    byte-for-byte body).
    """
    return error_envelope(status_code, code, message, details)


def _auth_disabled_404() -> JSONResponse:
    """Build the 404 served by `/auth/*` mutation endpoints when auth is OFF.

    In OSS mode (`RB_REQUIRE_AUTH` unset/false) the signup / login / me /
    keys surface is intentionally hidden — there are no per-tenant principals
    to issue tokens for, and surfacing a 200 here would be a confusing lie.
    We return 404 (rather than 503) so the routes look like they simply do
    not exist on this deployment; the error envelope explains how to turn
    them on. Self-hosters who want the full auth stack flip
    `RB_REQUIRE_AUTH=true` and these routes come back.
    """
    return _err(
        404,
        "auth_disabled",
        "Auth is disabled on this deployment. Set RB_REQUIRE_AUTH=true to enable signup/login and per-tenant API keys.",
    )


def _tenant_response(row: dict) -> dict:
    """Project the internal tenant row down to the v1 `tenant` shape.

    Internal rows carry `password_hash` and quota counters that customers
    never see; keep the response surface tight.
    """
    return {
        "id": row["id"],
        "email": row["email"],
        "plan": row.get("plan", "free"),
        "created_at": _stringify_created_at(row.get("created_at")),
    }


def _api_key_response(row: dict, raw_key: Optional[str] = None) -> dict:
    """Project an `api_keys` row down to the list/listing v1 shape.

    The raw bearer is only included when `raw_key` is provided (i.e.
    immediately after creation). Stored rows carry only `key_hash`;
    we never leak it.
    """
    out = {
        "id": row["id"],
        "name": row["name"],
        "created_at": _stringify_created_at(row.get("created_at")),
        "last_used_at": _stringify_created_at(row["last_used_at"]) if row.get("last_used_at") else None,
        "revoked_at": _stringify_created_at(row["revoked_at"]) if row.get("revoked_at") else None,
    }
    if raw_key is not None:
        out["key"] = raw_key
    return out


def _generate_api_key() -> str:
    """Generate an `rb_live_...` token per the v1 contract.

    `secrets.token_urlsafe(24)` yields a 32-character URL-safe base64
    string (24 raw bytes → 32 chars). Combined with the 8-character
    `rb_live_` prefix the total length is 40 characters. The character
    set is `[A-Za-z0-9_-]` which matches the regex in the contract.
    """
    body = secrets.token_urlsafe(24)
    assert len(body) == 32, f"unexpected token_urlsafe length: {len(body)}"
    return f"rb_live_{body}"


def _issue_api_key(tenant_id: str, name: str) -> dict:
    """Mint a new API key for `tenant_id` and return the v1 response payload.

    The raw key is generated here, SHA-256-hashed, persisted, and
    returned alongside the row metadata. The raw value MUST be surfaced
    to the caller exactly once and then discarded; subsequent reads only
    ever see the hash.

    SHA-256 (not bcrypt) is the correct hash for an `rb_live_` token:
    the token is 32 url-safe random chars (~190 bits of entropy), so
    there is nothing for bcrypt's deliberate slowness to defend — and a
    deterministic digest is directly indexable, making auth-time
    resolution an O(1) lookup instead of a linear bcrypt scan. Bcrypt is
    still used for human passwords (`tenants.password_hash`).
    """
    raw = _generate_api_key()
    key_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    key_id = "key_" + uuid4().hex
    row = state_mod.create_api_key(key_id, tenant_id, key_hash, name)
    return _api_key_response(row, raw_key=raw)


def _stringify_created_at(value) -> str:
    """Coerce a created_at value (datetime or str) to ISO 8601 UTC.

    Postgres returns a `datetime`; the memory adapter already stores ISO
    strings. Normalising here keeps the response shape consistent.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # datetime / date
    try:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(value)


# --- routes ---------------------------------------------------------------


@router.post("/signup", status_code=201)
async def signup(request: Request):
    """Create a new tenant. Returns `{token, tenant}`.

    We parse the body manually instead of using pydantic's automatic
    binding so we can map (a) malformed JSON, (b) pydantic email errors,
    and (c) weak passwords to the three distinct v1 error codes
    (`invalid_email`, `weak_password`) the contract requires.

    OSS opt-in: returns 404 when `RB_REQUIRE_AUTH` is unset/false — there
    is no per-tenant signup flow in OSS mode (the bootstrap "default" tenant
    is the only principal). See _auth_disabled_404 for the response shape.
    """
    if not auth_required():
        return _auth_disabled_404()
    try:
        raw = await _read_json(request)
    except ValueError:
        return _err(400, "invalid_email", "Request body must be JSON with email and password")

    # Validate email shape via pydantic; map errors back to the v1 codes.
    try:
        email = _CredentialsBase.model_validate({"email": raw.get("email"), "password": raw.get("password") or "x" * 8}).email
    except ValidationError:
        return _err(400, "invalid_email", "Email address is malformed")

    password = raw.get("password")
    if not isinstance(password, str) or len(password) < 8:
        return _err(400, "weak_password", "Password must be at least 8 characters")

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    tenant_id = "ten_" + uuid4().hex
    try:
        row = state_mod.create_tenant(tenant_id, str(email), password_hash)
    except ValueError as exc:
        if str(exc) == "duplicate_email":
            return _err(409, "email_taken", "An account with that email already exists")
        raise

    token = encode_jwt(row["id"])
    # rosalinddb.auth.signups — no attributes (no email/tenant labels).
    obs_metrics.record_signup()
    # Auto-issue a "Default" API key. The raw bearer is returned exactly once
    # here; a UI client should surface it once and never re-fetch.
    first_api_key = _issue_api_key(row["id"], name="Default")
    return JSONResponse(
        status_code=201,
        content={
            "token": token,
            "tenant": _tenant_response(row),
            "first_api_key": first_api_key,
        },
    )


@router.post("/login")
async def login(request: Request):
    """Verify credentials and return a fresh JWT.

    Wrong-password and unknown-email collapse to the same 401 with code
    `invalid_credentials`: leaking which side was wrong is a classic
    enumeration vector and the v1 contract codifies the merge.

    OSS opt-in: returns 404 when `RB_REQUIRE_AUTH` is unset/false (auth
    disabled — no JWTs to issue).
    """
    if not auth_required():
        return _auth_disabled_404()
    try:
        raw = await _read_json(request)
    except ValueError:
        obs_metrics.record_login("failure")
        return _err(401, "invalid_credentials", "Invalid email or password")

    email = raw.get("email")
    password = raw.get("password")
    if not isinstance(email, str) or not isinstance(password, str):
        obs_metrics.record_login("failure")
        return _err(401, "invalid_credentials", "Invalid email or password")

    row = state_mod.get_tenant_by_email(email)
    if row is None:
        obs_metrics.record_login("failure")
        return _err(401, "invalid_credentials", "Invalid email or password")

    try:
        ok = bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8"))
    except (ValueError, TypeError):
        ok = False
    if not ok:
        obs_metrics.record_login("failure")
        return _err(401, "invalid_credentials", "Invalid email or password")

    token = encode_jwt(row["id"])
    # rosalinddb.auth.logins{outcome=success}. `outcome` is the only label —
    # never email/tenant (cardinality rule).
    obs_metrics.record_login("success")
    return {"token": token, "tenant": _tenant_response(row)}


@router.get("/me")
def me(tenant_id: str = Depends(current_tenant_id)):
    """Return the calling tenant. Accepts a JWT or an API key.

    OSS opt-in: returns 404 when `RB_REQUIRE_AUTH` is unset/false. In
    single-tenant OSS mode there is no per-caller principal to describe —
    `current_tenant_id` would short-circuit to `DEFAULT_TENANT_ID` for
    everyone, which is more confusing than useful.
    """
    if not auth_required():
        return _auth_disabled_404()
    row = state_mod.get_tenant_by_id(tenant_id)
    if row is None:
        # Token signature was valid but the tenant has been deleted.
        # Treat as unauthorized — the principal no longer exists.
        return _err(401, "unauthorized", "Missing or invalid credentials")
    return {"tenant": _tenant_response(row)}


@router.get("/usage")
def usage(tenant_id: str = Depends(current_tenant_id)):
    """Return the calling tenant's current-period usage and quotas.

    Performs a lazy daily reset (via `state.get_usage`) so a stale
    `queries_today` from a previous day is never reported. Response shape is
    fixed by the v1 contract — UI clients bind to this shape.

    OSS opt-in: when `RB_ENABLE_QUOTAS` is unset/false (the self-host default)
    nothing meaningful is being tracked, so the endpoint returns the honest
    `{"enabled": false}` envelope instead of the full v1 shape. A dashboard
    implementation should read `enabled` first and fall back to the legacy keys.

    Auth opt-in: when `RB_REQUIRE_AUTH` is also unset/false there is no
    per-caller principal — `current_tenant_id` short-circuits to
    `DEFAULT_TENANT_ID` for everyone — so we skip the per-tenant lookup
    entirely and return the same `{"enabled": false}` envelope without any
    tenant context. This keeps the endpoint reachable and non-noisy in OSS
    mode (a client can probe it to discover its own deployment shape).
    """
    if not auth_required():
        # OSS single-tenant mode: no per-tenant accounting, no tenant lookup.
        return {"enabled": False}
    if not quotas_enabled():
        # Cheap probe: confirm the principal still exists (so a JWT for a
        # deleted tenant still 401s) but don't bother with the lazy reset or
        # the counter projection — they are no-ops in disabled mode.
        if state_mod.get_tenant_by_id(tenant_id) is None:
            return _err(401, "unauthorized", "Missing or invalid credentials")
        return {"enabled": False}
    try:
        return state_mod.get_usage(tenant_id)
    except ValueError:
        # Token signature valid but the tenant has been deleted.
        return _err(401, "unauthorized", "Missing or invalid credentials")


# --- API keys ---------------------------------------------------------------


class _CreateKeyRequest(BaseModel):
    """Pydantic shape for `POST /auth/keys`.

    Validated manually in the handler so name failures map to the
    contract code `invalid_name` rather than pydantic's envelope.
    """
    name: str


@router.post("/keys", status_code=201)
async def create_key(
    request: Request,
    tenant_id: str = Depends(current_tenant_id_jwt_only),
):
    """Issue a new API key for the calling tenant. JWT-only.

    Customers cannot bootstrap more keys with an API key — see the
    v1 contract "Auth / POST /auth/keys" note. The raw `rb_live_...`
    value is in the response body once; it is never queryable later.

    OSS opt-in: returns 404 when `RB_REQUIRE_AUTH` is unset/false. There
    are no per-tenant API keys in OSS mode — the bootstrap "default" tenant
    is unauthenticated and there is nothing to key.
    """
    if not auth_required():
        return _auth_disabled_404()
    try:
        raw = await _read_json(request)
    except ValueError:
        return _err(400, "invalid_name", "Body must be JSON with a `name` field")

    name = raw.get("name")
    if not isinstance(name, str) or not (1 <= len(name) <= 64):
        return _err(400, "invalid_name", "name must be 1-64 characters")

    payload = _issue_api_key(tenant_id, name=name)
    return JSONResponse(status_code=201, content=payload)


@router.get("/keys")
def list_keys(tenant_id: str = Depends(current_tenant_id)):
    """List the caller's API keys. Raw keys are never returned.

    OSS opt-in: returns 404 when `RB_REQUIRE_AUTH` is unset/false (no
    per-tenant API keys in OSS mode).
    """
    if not auth_required():
        return _auth_disabled_404()
    rows = state_mod.list_api_keys(tenant_id)
    return {"keys": [_api_key_response(r) for r in rows]}


@router.delete("/keys/{key_id}", status_code=204)
def revoke_key(key_id: str, tenant_id: str = Depends(current_tenant_id)):
    """Revoke an API key. Sets `revoked_at`; the row is kept for audit.

    Cross-tenant requests return 404 — we never leak existence. An
    already-revoked key also returns 404 (idempotency at the HTTP
    level isn't part of the contract, and the row's `revoked_at`
    is already terminal).

    After revoking the key the in-process auth cache entry is immediately busted
    so the revoked key stops working on THIS worker process without waiting for
    the TTL. The cache entry is keyed by sha256(raw_key); the persisted row
    carries `key_hash` which IS that sha256 digest, so we can bust directly
    without knowing the raw key.
    Note: other worker processes are NOT invalidated here — they will continue
    accepting the key for up to _AUTH_CACHE_TTL_S (default 30 s).
    Cross-process invalidation via Redis pub/sub is a tracked follow-up.

    OSS opt-in: returns 404 when `RB_REQUIRE_AUTH` is unset/false (no
    per-tenant API keys in OSS mode).
    """
    if not auth_required():
        return _auth_disabled_404()
    row = state_mod.get_api_key(key_id, tenant_id)
    if row is None or row.get("revoked_at") is not None:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": "API key not found"},
        )
    state_mod.revoke_api_key(key_id, tenant_id)
    # Bust the in-process auth cache so this worker rejects the key immediately.
    # The raw key is unavailable here (never re-surfaced); bust by the stored
    # key_hash which is sha256(raw_key), the same value used as the cache key.
    bust_api_key_cache_by_hash(row["key_hash"])
    return Response(status_code=204)


# --- shared helpers -------------------------------------------------------


async def _read_json(request: Request) -> dict:
    """Read and parse the JSON body, raising ValueError on bad shapes.

    FastAPI ordinarily wires this up through pydantic, but the auth
    endpoints need fine-grained control over which error code maps to
    which failure mode (see the comment in `signup`).
    """
    import json

    body = await request.body()
    if not body:
        raise ValueError("empty body")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid json") from exc
    if not isinstance(data, dict):
        raise ValueError("body must be an object")
    return data


# --- exception handler hookup --------------------------------------------


def install_exception_handlers(app) -> None:
    """Register a FastAPI exception handler that rewrites HTTPException
    payloads into the v1 error envelope.

    FastAPI's default 401/403/404/etc bodies are `{"detail": ...}`. Our
    contract requires `{"error": {"code", "message"}}`. When the
    `current_tenant_id` dependency raises with
    `detail={"code", "message"}`, we lift those fields into the envelope.
    Any other HTTPException (raw string detail) becomes a generic
    `internal_error`-shaped envelope with the original status.

    The handler is installed on the *app* (not the router) because
    FastAPI dispatches exception handlers app-wide.
    """
    from fastapi import HTTPException as _HTTPException
    from fastapi.exceptions import RequestValidationError as _RVE
    from starlette.requests import Request as _Req

    @app.exception_handler(_HTTPException)
    async def _http_exc_handler(_req: _Req, exc: _HTTPException):  # noqa: F811
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": detail.get("code"),
                        "message": detail.get("message", ""),
                    }
                },
            )
        # Fallback: best-effort envelope for non-auth callers that still
        # raise plain HTTPException (e.g. starlette internals).
        code = {
            400: "invalid_request",
            401: "unauthorized",
            404: "not_found",
            409: "conflict",
        }.get(exc.status_code, "internal_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": str(detail)}},
        )

    @app.exception_handler(_RVE)
    async def _validation_handler(_req: _Req, exc: _RVE):  # noqa: F811
        # Auth handlers parse manually, so this only fires for non-auth
        # routes. Map pydantic errors to a generic 400 envelope so the
        # response still parses on the client side.
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "Request validation failed",
                    "details": {"errors": exc.errors()},
                }
            },
        )


def install_pool_exhaustion_handler(app) -> None:
    """Register a handler mapping `PoolCheckoutTimeout` to a v1 503 envelope.

    `psycopg2.pool.ThreadedConnectionPool.getconn()` is fail-fast;
    `state.pooled_conn()` wraps it in a block-with-timeout that, on a
    *sustained* pool exhaustion, raises `state.PoolCheckoutTimeout`. That is
    genuine overload — it must surface as HTTP **503**, never the bare 500 a
    raw `PoolError` would become.

    When a `PoolCheckoutTimeout` escapes a route handler this handler shapes
    it into `{"error": {"code": "service_unavailable", "message": ...}}` with
    status 503 — the same v1 envelope `install_exception_handlers` produces
    for HTTPExceptions. (A `PoolCheckoutTimeout` raised by the request-scoped
    connection middleware's OWN pool checkout — before the app runs — is
    shaped by the middleware itself, since it escapes above the app.)

    Installed on the *app* alongside `install_exception_handlers`.
    """
    from adapters.state.state import PoolCheckoutTimeout
    from starlette.requests import Request as _Req

    @app.exception_handler(PoolCheckoutTimeout)
    async def _pool_exhausted_handler(_req: _Req, _exc: PoolCheckoutTimeout):  # noqa: F811
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "service_unavailable",
                    "message": "Service temporarily unavailable, please retry",
                }
            },
        )


@router.get("/_healthz", include_in_schema=False)
def _healthz():
    """Lightweight liveness probe — distinct path so it doesn't shadow /me."""
    return {"ok": True}
