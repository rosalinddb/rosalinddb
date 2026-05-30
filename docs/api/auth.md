# Auth quickstart

A usage cheat-sheet for the auth surface. The full contract is in
[`v1.md`](./v1.md) under "Auth"; refer to it for authoritative response
shapes and error codes.

All examples below assume the Control Plane is running on
`http://localhost:8080`. Implementation:
`services/auth/auth.py` (the router) + `services/auth/jwt_utils.py` (the
`Authorization` resolver).

## OSS opt-in: auth is off by default

`RB_REQUIRE_AUTH` is the single env switch that enables the per-tenant auth
stack. It defaults **OFF** so the headline `docker compose up` self-host
works without anyone first signing up and ferrying tokens around.

When `RB_REQUIRE_AUTH` is unset or false:

- Every `/v1/*` request resolves to a single built-in `default` tenant. The
  `Authorization` header is not read at all (see
  `services/auth/jwt_utils.py:current_tenant_id`).
- All `/auth/*` mutation endpoints (`/signup`, `/login`, `/me`, `/keys`,
  `/keys/{id}`) return `404 auth_disabled` — the surface is hidden, not just
  neutered. The body suggests setting `RB_REQUIRE_AUTH=true` to enable it.
- `/auth/usage` is reachable and returns `{"enabled": false}` so the
  dashboard can probe its deployment shape.

Set `RB_REQUIRE_AUTH=true` (truthy values: `1`, `true`, `yes`, `on`) to turn
the full stack on. It is a runtime env var, set however your platform injects
secrets. The rest of this document assumes auth is enabled.

## Tokens: JWTs and API keys

The `Authorization: Bearer <token>` header accepts **two** kinds of token;
downstream handlers don't know or care which was used.

| Kind     | Format                            | Issued by                       | Use case                |
|----------|-----------------------------------|---------------------------------|-------------------------|
| JWT      | `eyJhbG...` (HS256, 24h)          | `POST /auth/signup`, `/login`   | Dashboard / browser     |
| API key  | `rb_live_<32 url-safe chars>`     | `POST /auth/keys`, `/signup`    | Customer code / servers |

Both resolve to the same `tenant_id` via the `current_tenant_id` FastAPI
dependency in `services/auth/jwt_utils.py`.

A second dependency, `current_tenant_id_jwt_only`, is exported for the
narrow case (currently `POST /auth/keys`) where API-key auth must be
rejected: customer code cannot bootstrap more keys from an existing key —
that is a dashboard operation by design.

### API-key resolution

API keys are stored as **SHA-256 digests** of the raw `rb_live_...` token in
the `api_keys.key_hash` column. On a request,
`services/auth/jwt_utils.py:_resolve_api_key` computes
`sha256(raw_key)` and does a single indexed lookup
(`state.get_api_key_by_hash`) — `O(1)` regardless of how many keys exist
system-wide. Successful resolutions are cached in-process for 30 seconds
(`_AUTH_CACHE`) to keep query-path latency low; a revoked key is busted
immediately on the worker that handled the `DELETE` and stops working
within at most 30 s on other workers.

SHA-256 is the correct hash here. An `rb_live_` token is a 24-byte
url-safe random value (~190 bits of entropy), so there is no brute-force
surface for bcrypt's deliberate slowness to defend, and a deterministic
digest is directly indexable. Bcrypt is still used for **human passwords**
(`tenants.password_hash`), where the slowness is the point.

## Environment

The JWT signing secret is read from `JWT_SECRET`. **You must set it in
production.** If unset, the service falls back to an ephemeral per-process
secret and logs a WARNING; tokens issued under the fallback are invalidated
whenever the service restarts.

```bash
export JWT_SECRET=$(openssl rand -base64 48)
```

API keys are not signed — they are SHA-256-hashed at creation and verified
by hash on every request. There is no separate secret to configure.

## Signup

Create a new tenant. Returns a JWT, the tenant row, and an auto-issued
"Default" API key. The raw key is in `first_api_key.key` and is the **only**
time it is ever returned.

```bash
curl -s -X POST http://localhost:8080/auth/signup \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"password123"}'
```

Successful response (HTTP 201):

```json
{
  "token": "eyJhbGc...",
  "tenant": {
    "id": "ten_61e1cfc7416e4741a8c1c4aeb8fd4a44",
    "email": "you@example.com",
    "plan": "free",
    "created_at": "2026-05-14T12:34:56Z"
  },
  "first_api_key": {
    "id": "key_1954eb30ee25486c8ebe3b2b0ab5c26f",
    "key": "rb_live_d9EiFXSL_r7jKpxJvvbyVnURq91ZrMsE",
    "name": "Default",
    "created_at": "2026-05-14T12:34:56Z",
    "last_used_at": null,
    "revoked_at": null
  }
}
```

Error codes:

| HTTP | code               | when                                     |
|------|--------------------|------------------------------------------|
| 400  | `invalid_email`    | email failed RFC validation              |
| 400  | `weak_password`    | password < 8 characters                  |
| 409  | `email_taken`      | email already registered                 |
| 404  | `auth_disabled`    | `RB_REQUIRE_AUTH` is unset/false         |

## Login

```bash
curl -s -X POST http://localhost:8080/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"password123"}'
```

Successful response (HTTP 200): same shape as signup, **without**
`first_api_key`. Login never re-emits raw keys; clients fetch the
list (without raw values) via `GET /auth/keys`.

Errors collapse to a single 401 — wrong password and unknown email both
return `invalid_credentials` (account-enumeration defence).

## /auth/me

Echo the current tenant. Accepts either a JWT or an API key.

```bash
TOKEN=...  # JWT or rb_live_... key
curl -s http://localhost:8080/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

Any missing / malformed / expired / revoked token returns:

```json
{ "error": { "code": "unauthorized", "message": "Missing or invalid credentials" } }
```

with HTTP 401.

## POST /auth/keys

Issue a new API key. **JWT-only.**

```bash
curl -s -X POST http://localhost:8080/auth/keys \
  -H "Authorization: Bearer $JWT" \
  -H 'Content-Type: application/json' \
  -d '{"name":"Production server"}'
```

Successful response (HTTP 201):

```json
{
  "id": "key_e304fba319664616b819c39dfa80ae1d",
  "key": "rb_live_ay5m3VJOwqDMmtHvIyqJp-_xQNT7qpIf",
  "name": "Production server",
  "created_at": "2026-05-14T12:34:56Z",
  "last_used_at": null,
  "revoked_at": null
}
```

The raw `key` is returned exactly once. Treat it as a password: store it in
a secret manager and reference it from your application code.

Errors:

| HTTP | code            | when                                      |
|------|-----------------|-------------------------------------------|
| 400  | `invalid_name`  | name empty, missing, or > 64 characters   |
| 401  | `unauthorized`  | missing JWT, expired JWT, or API-key auth |
| 404  | `auth_disabled` | `RB_REQUIRE_AUTH` is unset/false          |

## GET /auth/keys

List the caller's keys. Accepts JWT or API key. Raw keys are **never**
returned.

```bash
curl -s -H "Authorization: Bearer $JWT" http://localhost:8080/auth/keys
```

```json
{
  "keys": [
    {
      "id": "key_...",
      "name": "Default",
      "created_at": "2026-05-14T12:34:56Z",
      "last_used_at": "2026-05-14T13:45:00Z",
      "revoked_at": null
    }
  ]
}
```

`last_used_at` updates on every successful API-key auth (fire-and-forget,
on a background thread; the response path doesn't block on the UPDATE). JWT
auth does not touch any key row.

## DELETE /auth/keys/{id}

Revoke a key. Sets `revoked_at`; the row is kept for audit.

```bash
curl -s -X DELETE \
  -H "Authorization: Bearer $JWT" \
  http://localhost:8080/auth/keys/$KEY_ID
```

Response: HTTP 204 (no body).

Cross-tenant and already-revoked requests both return `404 not_found` — we
deliberately do not leak whether a key id belongs to another tenant.

## Using the dependency from another service

Internal routers resolve the caller's tenant via the shared FastAPI
dependency:

```python
from fastapi import Depends
from services.auth.jwt_utils import current_tenant_id

@app.get("/v1/datasets")
def list_datasets(tenant_id: str = Depends(current_tenant_id)):
    ...
```

If you have an endpoint that must refuse API-key auth (e.g. a future
"rotate password" or "delete account" flow), depend on
`current_tenant_id_jwt_only` instead. The failure mode is
indistinguishable from any other bad token (401 `unauthorized`) so callers
cannot probe which dependency a route uses.

Either dependency raises a contract-shaped 401 on auth failure; you do not
need to wrap it. When `RB_REQUIRE_AUTH` is off, both dependencies
short-circuit to the `default` tenant id and never touch the
`Authorization` header.
