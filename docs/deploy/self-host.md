# RosalindDB — Self-Host Guide

The canonical guide for running RosalindDB on your own infrastructure. The
companion to the quickstart in the root `README.md`: this is what you read
after `docker compose up` works, when you want to put it behind a real domain.

---

## 1. The headline path

```bash
git clone https://github.com/desquaredp/rosalinddb
cd rosalinddb
docker compose up
```

That's it. You have:

- `http://localhost:8080` — the full RosalindDB v1 API.
- Auth disabled. Every request resolves to a bootstrap `default` tenant.
- Quotas / rate-limits disabled.
- Local MinIO + Postgres + Redis bundled.
- Schema migrated once via a one-shot `migrator` service before any app
  container boots.

If you only want to play with it, you're done. Run `python -m scripts.smoke`
against `http://localhost:8080` for an end-to-end happy-path check. Read on if
you're putting it behind a real domain.

---

## 2. Required environment

`docker-compose.yml` already sets all of these — the table is here so a
non-compose deploy (k8s manifest, systemd unit, Nomad job, …) has a single
authoritative reference.

### Always required

| Variable | Purpose | Default | Notes |
|---|---|---|---|
| `DATABASE_URL` | Postgres DSN — the catalog | `postgresql://postgres:postgres@localhost:5432/vectors` | Any Postgres 14+. See [§3d](#d-postgres) for the **PgBouncer caveat**. |
| `REDIS_URL` | Redis connection — queue + ephemeral result store | unset → in-process queue (test only) | Any Redis 6.2+. AOF strongly recommended — see [§3e](#e-redis). |
| `S3_ENDPOINT_URL` | S3-compatible endpoint | unset → real AWS | Set to MinIO/R2/B2/Wasabi for non-AWS. |
| `S3_ACCESS_KEY` | S3 access key | — | |
| `S3_SECRET_KEY` | S3 secret key | — | |
| `S3_REGION` | S3 region | `us-east-1` | `auto` for R2. |
| `LANDING_PREFIX` | URI prefix for raw landing data | `s3://rosalinddb/landing` | `s3://yourbucket/landing` |
| `INDEXES_PREFIX` | URI prefix for FAISS index shards | `s3://rosalinddb/indexes` | `s3://yourbucket/indexes` |
| `CACHE_DIR` | Local FS path for shard read-through cache | `/var/cache/shards` | Mount a volume here on the query-DP / ephemeral-runner containers. |

### Required for production

Anything past localhost MUST set these. Without them you have an
unauthenticated public origin and a JWT secret that resets on every restart.

| Variable | Purpose | Default | Notes |
|---|---|---|---|
| `JWT_SECRET` | HS256 signing secret | ephemeral per-process random | `openssl rand -hex 32`. Must be identical across every CP / DP / worker process. |
| `RB_REQUIRE_AUTH` | Turn on the JWT + API-key auth stack | `false` (OSS-friendly) | Set to `true` in production. See [§3b](#b-authentication). |

### Optional / tunable

The ~10 most useful knobs. Every other env var lives in the source — grep
`os.getenv` under `services/` and `adapters/` for the full inventory.

| Variable | Purpose | Default |
|---|---|---|
| `RB_ENABLE_QUOTAS` | Per-tenant vector cap + token-bucket rate limiter | `false` |
| `RB_RATE_LIMIT_RPS` | Sustained rate-limit ceiling (when quotas enabled) | `50` |
| `RB_RATE_LIMIT_BURST` | Burst capacity | `100` |
| `RB_QUERY_NPROBE` | FAISS IVF nprobe — recall vs. latency knob | `64` |
| `RB_SHARD_CACHE_BYTES` | Per-DP shard cache byte budget | `512 MiB` |
| `IMPORT_MAX_BYTES` | Bulk-import staged-upload size cap | `5 GiB` |
| `RB_PG_POOL_MAX` | Per-process Postgres pool ceiling | `10` |
| `CORS_ALLOW_ORIGINS` | Comma-separated extra CORS origins for browser clients | `""` (localhost dev range always allowed) |
| `QUERY_DP_URL` | CP→DP reverse-proxy base URL | `http://localhost:8090` |
| `RB_PROXY_SECRET` | Shared secret CP sends on every CP→DP call | unset → DP relies on network isolation |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | `http://localhost:4318` |
| `OTEL_SDK_DISABLED` | Opt out of telemetry entirely | `false` |

---

## 3. Deploying past localhost

### a. Networking and TLS

The backend does not terminate TLS. Put a real reverse proxy
(Caddy, nginx, Traefik, ALB, Cloudflare, …) in front of the `cp` service on
`:8080`. The CP is the **only** service that should be reachable from outside
your network — every other service (`query_dp`, `validator`, `index_builder`,
`ephemeral_runner`) is private and reached via Redis or via the CP's internal
proxy to the DP.

If you front the backend with a separate-origin dashboard (the canonical Next.js
frontend or your own), set `CORS_ALLOW_ORIGINS` to a comma-separated list of
allowed origins. The localhost dev range is always allowed; production origins
must be enumerated.

Minimal Caddy example:

```caddyfile
api.example.com {
    reverse_proxy cp:8080
}
```

The CP exposes an unauthenticated `GET /healthz` for liveness probes — wire it
up wherever your platform expects one (k8s `livenessProbe`, ALB target group
health check, etc.).

### b. Authentication

Two switches govern the auth surface. Both default OFF so `docker compose up`
works without a signup flow; both must flip to ON for a real deploy.

```bash
export JWT_SECRET=$(openssl rand -hex 32)
export RB_REQUIRE_AUTH=true
```

When `RB_REQUIRE_AUTH=true`:

- Every request must carry `Authorization: Bearer <token>` (a JWT issued by
  signup/login, or an `rb_live_...` API key).
- `/auth/signup`, `/auth/login`, `/auth/me`, `/auth/keys*` become live.
- The bootstrap `default` tenant is NOT loginable (its `password_hash` is the
  sentinel `'!disabled!'`, which no bcrypt comparison can ever match). Sign up
  a new tenant instead.

The first-key flow on a fresh deploy:

```bash
# 1. Sign up. Returns a tenant record, a JWT, AND a first API key (the only
#    time it is ever surfaced — there is no "show key again" path).
curl -X POST https://api.example.com/auth/signup \
  -H 'content-type: application/json' \
  -d '{"email": "you@example.com", "password": "..."}'

# Response body (HTTP 201):
# {
#   "token": "eyJ...",
#   "tenant": {
#     "id": "ten_<hex>",
#     "email": "you@example.com",
#     "plan": "free",
#     "created_at": "2025-01-01T00:00:00Z"
#   },
#   "first_api_key": {
#     "id": "key_<hex>",
#     "name": "Default",
#     "key": "rb_live_<32 chars>",
#     "created_at": "2025-01-01T00:00:00Z",
#     "last_used_at": null,
#     "revoked_at": null
#   }
# }

# 2. Use either the JWT (short-lived, dashboard) or the API key (long-lived,
#    server-to-server). Both go in the same Bearer header.
curl https://api.example.com/v1/datasets \
  -H 'authorization: Bearer rb_live_...'
```

`POST /auth/keys` mints additional keys; it accepts the JWT only (you cannot
bootstrap more keys from an existing key — keys are issued by a logged-in
human via a UI client).

### c. Object storage

Anything S3-compatible works. RosalindDB never writes outside the
`{LANDING,INDEXES}_PREFIX` URIs.

- **Bundled MinIO** (the docker-compose default). Fine for development and
  single-node production if you're comfortable operating MinIO yourself.
  Persistent on the `./data/minio` volume.
- **Self-hosted MinIO cluster.** Same env vars; point `S3_ENDPOINT_URL` at the
  cluster's S3 API endpoint.
- **AWS S3.** Leave `S3_ENDPOINT_URL` empty; set `S3_REGION` to the real
  region; supply IAM access keys via `S3_ACCESS_KEY` / `S3_SECRET_KEY` (or use
  IRSA / IAM Role for Service Accounts on EKS and leave the keys unset).
- **Cloudflare R2.** `S3_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com`,
  `S3_REGION=auto`. Zero egress fees — material for a query-heavy workload.
- **Backblaze B2, Wasabi, DigitalOcean Spaces, Tigris, …** Any S3-compatible
  store. Endpoint + region per the provider's docs.

Buckets are not auto-created in production (the compose `createbuckets`
one-shot only handles dev MinIO). Pre-create the bucket referenced by
`LANDING_PREFIX` / `INDEXES_PREFIX` before the first deploy.

### d. Postgres

Any Postgres 14+. **One sharp gotcha:**

> The index-build coordinator uses session-level `pg_advisory_lock` to
> serialise concurrent builds of the same dataset (see
> `adapters/state/state.py::dataset_build_lock`). Session-level advisory locks
> must be acquired and released on the **same connection**.
>
> **A transaction-pooled connection string will break this.** Common ways to
> hit this:
>
> - **PgBouncer in `transaction` pool mode** — connections are reused per
>   transaction, so the session that took the lock is gone before the unlock.
> - **Supabase's pooled URL** with the `-pooler` suffix — that endpoint is
>   PgBouncer in transaction mode.
> - **Any other pooler in transaction or statement mode.**
>
> Use the **direct** Postgres connection string (or PgBouncer in `session`
> mode). The pooled URL is fine for the rest of the app but will leak locks
> for the builder.

Other notes:

- `python -m scripts.migrate` is idempotent (guarded by
  `pg_advisory_xact_lock` + a `schema_migrations` ledger). Re-running it is
  cheap.
- A small per-process pool is maintained automatically (`RB_PG_POOL_MAX=10`
  default). If you run many CP / DP processes against a small managed Postgres
  (a small managed instance such as RDS micro), tune this down or put a session-mode pooler in
  front.
- Managed Postgres options that work out of the box: Supabase (use the
  direct connection, not the `-pooler` URL), AWS RDS, Google Cloud SQL,
  self-hosted vanilla Postgres.

### e. Redis

Any Redis 6.2+. The queue uses `LMOVE` for the reliable-processing-list
pattern (atomically move a message from the work list to a per-topic
processing list — `BRPOP` would lose in-flight messages on a worker crash).

**Durability.** Without persistence, a Redis restart drops every queued and
in-flight message. Enable AOF:

```
redis-server --appendonly yes --dir /data --maxmemory-policy noeviction
```

`noeviction` is deliberate — a full Redis fails writes loudly rather than
silently evicting a queued message. Mount `/data` on a persistent volume.

Managed Redis (Upstash, ElastiCache, Memorystore, Aiven, Redis Cloud) is fine
as long as it speaks Redis 6.2+ and you're comfortable with the provider's
durability model.

### f. Migrations

`python -m scripts.migrate` runs the schema migration to completion. It is
idempotent and self-locking — running it twice in parallel is safe.

Patterns by platform:

- **docker-compose** — already wired. The `migrator` one-shot service runs
  the migration before any long-running container boots, gated by
  `depends_on: { migrator: { condition: service_completed_successfully } }`.
- **k8s** — an init container on each Deployment, or a `Job` that runs once
  per release with the long-running Deployments rolling out only after it
  succeeds. Helm `--wait` or Argo Sync waves work fine.
- **systemd** — a one-shot unit (`Type=oneshot`, `RemainAfterExit=yes`) that
  every app unit `Requires=` and `After=`.
- **Platform release hooks** — any PaaS pre-deploy/release step (Heroku's
  `release` process, Render's `preDeployCommand`, and equivalents).

If you'd rather skip the explicit migration step, set `RB_SKIP_MIGRATE=0`
(unset / not `1`) and the long-running services will migrate in-process on
boot. This is fine for single-process deploys; in multi-replica setups two
replicas booting simultaneously race the migration locks (correctness is
preserved, but you'll see noisy logs). The out-of-band migration pattern is
recommended for production.

### g. Workers

`validator`, `index_builder`, and `ephemeral_runner` are stateless background
processes consuming from Redis queues. They scale horizontally — run N of each
behind a process supervisor (systemd, Kubernetes Deployment, Nomad, …).

One caveat: per-dataset advisory locks mean only one `index_builder` makes
progress on a given dataset at a time. That's a correctness guarantee
(double-indexing would otherwise be possible under message redelivery), not a
throughput ceiling — concurrent builds of *different* datasets parallelise
fully.

Worker restarts are safe. The Redis queue uses an at-least-once
reliable-processing-list pattern (`LMOVE` + a per-message `attempts` counter +
a DLQ at `<topic>:dlq`), and the workers themselves are idempotent:

- Re-validation re-writes the same Parquet output harmlessly.
- A duplicate `DATASET_READY` is a no-op via the shard's
  `indexed_landing_uris` manifest.
- Ephemeral query re-execution returns the same result.

### h. CP/DP split (advanced)

The OSS docker-compose already splits CP and DP into separate processes
(`cp` and `query_dp`). A single combined process also works — the legacy
`services.query_api.main:app` exists for that — but the split lets you:

- Scale query latency independently of ingest traffic.
- Run different VM sizes per tier (CP is small; DP holds the shard cache and
  wants memory).

The CP→DP wire contract:

- Set `QUERY_DP_URL` on the CP to the DP's base URL (in compose:
  `http://query_dp:8080`).
- Set `RB_PROXY_SECRET` on **both** the CP and the DP to the same value. The
  CP sends it as the `X-RB-Proxy-Secret` header; the DP verifies. Without it
  the DP relies on network isolation alone — that's fine on a private
  VPC / k8s network, but never expose the DP publicly without the secret set.

The DP has **no** `Authorization` parsing and **no** quota check — it trusts
the verified `X-RB-Tenant-Id` header the CP sends. Do not expose the DP
directly to users; route everything via the CP.

---

## 4. Observability

OpenTelemetry is baked in. Every request emits OTLP-compatible traces,
metrics, and logs.

Point it at any OTLP backend:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.your-collector.example.com
```

Grafana Cloud, Honeycomb, Datadog, New Relic, Tempo + Prom + Loki self-hosted —
they all accept OTLP. Or opt out entirely:

```bash
export OTEL_SDK_DISABLED=true
```

Key metrics include `rb_query_latency_seconds`, `rb_ingest_latency_seconds`,
`rb_shard_cache_evictions_total`, and the standard `http.server.duration` /
`http.server.active_requests` HTTP instruments. Per-service `service.name`
attributes (`rosalinddb-cp`, `rosalinddb-query-dp`, etc.) let you slice the
two-service query trace.

---

## 5. Operational notes and known gotchas

- **Worker restarts are safe.** The reliable-queue pattern reclaims in-flight
  messages via the reaper after a timeout, and workers are idempotent on
  redelivery.
- **Builds run alongside queries.** Incremental indexing means the active
  index is never deleted mid-flight; a new shard is published atomically, then
  the old one is retired.
- **Per-tenant data lives under `tenants.<id>/` prefixes** in your object
  store. Deleting a tenant from the catalog does **not** currently sweep
  their bucket data — that's a manual cleanup if you need it (or write a
  small `mc rm --recursive` job).
- **CPU-only FAISS.** The image ships Python 3.11 + FAISS-CPU. No GPU support
  today.
- **The bootstrap `default` tenant cannot be logged into.** Its
  `password_hash` is `'!disabled!'`, a non-bcrypt sentinel that no comparison
  can match. Don't try to flip `RB_REQUIRE_AUTH=true` and reuse it — sign up
  a new tenant.
- **JWTs are HS256 with a 24h TTL.** No refresh-token flow (yet). The
  dashboard re-logs in on expiry. Server-to-server callers should use API
  keys, not JWTs.
- **API keys are SHA-256 hashed at rest** (`api_keys.key_hash`). The raw
  `rb_live_...` value is surfaced exactly once at creation and never again.
  If a user loses it, mint a new one.
- **API-key auth has a 30-second in-process cache.** Revoking a key takes
  effect instantly on the worker that handled the DELETE; other workers see
  the revocation within 30s.

---

## 6. Quick reference

### Fresh local dev

```bash
docker compose up
# → http://localhost:8080
```

### Real deploy (point at your own services)

```bash
export DATABASE_URL=postgresql://user:pw@db.example.com:5432/rosalinddb
export REDIS_URL=redis://redis.example.com:6379/0
export S3_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
export S3_ACCESS_KEY=...
export S3_SECRET_KEY=...
export S3_REGION=auto
export LANDING_PREFIX=s3://yourbucket/landing
export INDEXES_PREFIX=s3://yourbucket/indexes
export JWT_SECRET=$(openssl rand -hex 32)
export RB_REQUIRE_AUTH=true
# Optional but recommended:
export RB_PROXY_SECRET=$(openssl rand -hex 32)
export CORS_ALLOW_ORIGINS=https://dashboard.example.com

# Run the migration once, then bring the stack up.
python -m scripts.migrate
docker compose up -d
```

For non-docker-compose deploys (k8s, Nomad, systemd, …), the same env vars
apply per-process. The repo's `Dockerfile` is the canonical image — use it
unchanged.

Customising compose for production typically means writing your own
`docker-compose.prod.yml` override that drops the bundled MinIO / Postgres /
Redis services and overrides the env on `cp`, `query_dp`, and the workers to
point at your real backends. There is no `docker-compose.prod.yml` in the
repo by design — it's deployment-shape-specific. Use Docker's standard
override file pattern:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## 7. See also

- [`docs/architecture/architecture.md`](../architecture/architecture.md) —
  system design.
- [`docs/api/v1.md`](../api/v1.md) — REST API reference.
- [`docs/indexing.md`](../indexing.md) — incremental-append builder behaviour.
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — dev setup.
- [`SECURITY.md`](../../SECURITY.md) — vulnerability reporting.
