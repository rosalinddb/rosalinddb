# Quotas & rate limiting

This page documents how RosalindDB enforces the default per-tenant limits from the
[`v1` contract](./v1.md) "Rate limits" section. It covers per-tenant quotas,
the `GET /auth/usage` surface, and the per-API-key rate limiter.
Implementation: `services/auth/quota.py` (rate-limit dependency + 429
helpers) and `adapters/state/state.py` (the lazy daily reset +
`try_consume_*` atomic update).

> **OSS opt-in:** when `RB_ENABLE_QUOTAS` is unset/false (the default), every
> quota and rate-limit on this page is a **no-op**. The `rate_limit`
> dependency returns without parsing the header; the `try_consume_*` calls
> are skipped entirely in the request handlers; `/auth/usage` returns the
> honest `{"enabled": false}` envelope. Set `RB_ENABLE_QUOTAS=true` (truthy
> values: `1`, `true`, `yes`, `on`) to turn full quota enforcement on.
> The rest of this page assumes quotas are
> enabled.

## Limits (defaults)

| Limit | Default | Scope |
|---|---|---|
| Vectors stored | 100,000 | per tenant, lifetime (`vector_quota`) |
| Queries | 10,000 / day | per tenant, resets at UTC midnight (`daily_query_quota`) |
| Request rate | 50 req/s, burst 100 | per API key (token bucket) |

Limits are configurable per tenant via the `tenants` table.

## `GET /auth/usage`

Returns the calling tenant's current usage and quotas. Requires a JWT or API
key. Performs a lazy daily reset before reading, so `queries_today` is never a
stale value from a previous day.

```json
{
  "vectors_used": 12500,
  "vector_quota": 100000,
  "queries_today": 342,
  "daily_query_quota": 10000,
  "queries_reset_at": "2026-05-16"
}
```

`queries_reset_at` is a `YYYY-MM-DD` date string — the day the daily counter
was last reset (effectively "today" once any usage call has run).

## Query quota enforcement (`POST /v1/query`)

A query consumes **one unit** of `daily_query_quota`. Quota is consumed on
the **Control Plane** (`services/query_api/query_proxy.py`) — after auth
and rate-limit succeed, before the request is proxied to the Query Data
Plane. The DP itself never consumes query quota; a tenant who is over
their daily cap is rejected at the CP and the request never reaches the DP.

Note: unlike the in-process monolith, the CP does not validate the request
body before consuming quota — the DP re-validates it. So a malformed query
that passes auth burns one quota unit before the DP returns a 400. There
is no refund path. This is an accepted trade for keeping the CP a thin
trust-checking proxy.

When the cap is hit the request is rejected with `429`:

```json
{
  "error": {
    "code": "query_quota_exceeded",
    "message": "Daily query quota exceeded for this tenant",
    "details": { "limit": 10000, "reset_at": "2026-05-16" }
  }
}
```

The daily counter resets lazily: the first usage/consume call on a new UTC day
sees `queries_reset_at < CURRENT_DATE`, zeroes `queries_today`, and bumps the
date. The reset and the consume run inside the same lock (memory mode) or
transaction (Postgres), so a request landing exactly on the day boundary is
counted correctly.

## Vector quota enforcement (`POST /v1/datasets/{name}/vectors`)

After the per-line NDJSON validation produces an `accepted_count`, the upload
consumes `accepted_count` units of `vector_quota`. The check happens **before**
anything is persisted to the landing area or published to the validator.

**All-or-nothing on the cap boundary.** If `vectors_used + accepted_count`
would exceed `vector_quota`, the *entire* upload is rejected — there is no
partial acceptance up to the cap. Nothing is written or published:

```json
{
  "error": {
    "code": "vector_quota_exceeded",
    "message": "Vector storage quota exceeded for this tenant",
    "details": { "limit": 100000, "used": 99980 }
  }
}
```

### `vectors_used` accounting caveat

`vectors_used` is incremented at **upload time** by `accepted_count` — the
count of records that passed the source_registry's per-line validation (valid
JSON, id present, `values` length matches the dataset dimension, metadata is an
object). The canonical validator worker runs downstream and *may* reject a few
more records (e.g. NaN values, duplicate ids). Because the validator is not
hooked into the quota path, `vectors_used` can therefore **slightly overcount**
relative to the rows that actually land in an index.

This is an accepted tradeoff: a default cap that errs on the side of
counting a few too many is safe (it can only make a tenant hit the cap
marginally sooner), and it avoids coupling the quota counter to the
asynchronous validator. The clean fix is to move the increment to the
point where the validator commits `row_count`.

## Rate limiter (per API key)

An in-memory token bucket per API key: 50 tokens/s refill, capacity 100. A
request authenticated with a JWT (dashboard traffic) is bucketed per *tenant*
instead of per key. Exhaustion → `429`:

```json
{
  "error": {
    "code": "rate_limited",
    "message": "Rate limit exceeded; slow down and retry",
    "details": { "limit_rps": 50, "burst": 100 }
  }
}
```

Applied to the customer-facing v1 endpoints only (`/v1/datasets*`,
`/v1/query`). The `/auth/*` surface is **not** rate-limited.

### Limitations

- **Process-local.** Buckets live in a per-process dict. They are not
  shared across workers/pods and not persisted across restarts. A restart
  resets every bucket to full; with N pods the effective limit is roughly
  N x the configured rate. The upgrade path is a shared Redis token
  bucket.
- No background sweeper — idle buckets stay in the dict. Negligible at
  current scale (<1000 keys).

## Atomicity

| Mode | Mechanism |
|---|---|
| Memory | A single process-wide `threading.Lock` (`_MEM_QUOTA_LOCK`) guards the lazy-reset + check + increment as one critical section. |
| Postgres | A single conditional `UPDATE ... WHERE <cap not hit> RETURNING ...`. The DB serialises concurrent callers, so two requests can never both slip past the cap. The lazy daily reset runs as a preceding `UPDATE` in the same transaction. |

## Test / E2E hooks (NOT for production)

Two env vars lower the default per-tenant limits so a `429` can be triggered without
issuing thousands of requests. They are read **fresh on every `POST /auth/signup`**
(`state._quota_defaults`), so set them before signing up the tenant under test:

| Env var | Effect |
|---|---|
| `RB_TEST_QUERY_QUOTA` | `daily_query_quota` stamped on new tenants (e.g. `1`) |
| `RB_TEST_VECTOR_QUOTA` | `vector_quota` stamped on new tenants (e.g. `3`) |

Two more env vars tune the rate limiter (read at module import):

| Env var | Effect |
|---|---|
| `RB_RATE_LIMIT_RPS` | token refill rate (default `50`) |
| `RB_RATE_LIMIT_BURST` | bucket capacity (default `100`) |

Leaving all four unset yields the contract defaults. To force a
`query_quota_exceeded` 429: start the service with `RB_TEST_QUERY_QUOTA=1`,
sign up, run one query (succeeds), run a second (429).
