# RosalindDB architecture

> **Status**: source of truth for how RosalindDB is put together today. It
> describes the current architecture — the CP/DP split — and the headline
> self-host topology.

This is an engineering doc, not a marketing one. The codebase is the
underlying truth; this document is the map.

## What RosalindDB is

A small, object-storage-first vector database. The hot query path is a FAISS
**IVFFlat** index loaded into an in-process byte-budgeted cache; the cold path
(indexing, validation, large imports) is queue-driven and runs on cheap
horizontally-scalable workers. State lives in Postgres; bytes (landing data,
index shards) live in S3-compatible object storage (MinIO in self-host,
S3 / R2 / any S3-compatible store elsewhere, an in-memory adapter in unit
tests).

Two things shape every design decision:

- **Self-host is the headline install path.** `git clone && docker compose up`
  on any Docker host stands the whole stack up against a single public origin.
  Auth and per-tenant quotas are **off by default** here.
- **The same image can run a private multi-user deployment by enabling auth and quotas.** The same build, with auth and quotas
  flipped **on** by two env vars. There is no separate "enterprise" build.

## The five service roles

Every process in the stack is one of five roles. The image is the same; the
entrypoint differs.

| Role | Module | Public? | Job |
|---|---|---|---|
| **Control Plane (CP)** | `services.control_plane.cp_app:app` | yes | Auth, dataset CRUD, the public vector-upload + bulk-import surface, the management API, the reverse proxy for `/v1/query`. Stateless. Internally composes the legacy `services.source_registry` FastAPI app (the dataset / ingest router) plus the `query_proxy` router — "Source Registry" survives as an **internal module name**, not a public front door. |
| **Query Data Plane (Query-DP)** | `services.query_api.dp_app:app` | no | Serves `POST /v1/query` and `GET /v1/query/status/{job_id}`. Runs the FAISS search and the in-process shard cache. Trusts the CP's verified `X-RB-Tenant-Id` header — does no auth and no per-query quota of its own. |
| **`validator_worker`** | `services.validator_worker.run` | no | Queue consumer. Reads `VALIDATE_DATASET`, streams uploaded NDJSON / Parquet, runs per-record validation, writes Parquet to landing, publishes `DATASET_READY`. |
| **`index_builder`** | `services.index_builder.run` | no | Queue consumer. Reads `DATASET_READY`, builds (or incrementally appends to) the dataset's FAISS shard, writes the shard + sidecar to object storage, inserts a row into `shard_catalog`. Also hosts the queue reaper thread. |
| **`ephemeral_runner`** | `services.ephemeral_runner.run` | no | Queue consumer. Handles the cold-query fallback (`RUN_EPHEMERAL_QUERY`) — downloads the latest shard, runs FAISS, publishes a `RESULT_READY` message that the DP's status endpoint can drain. |

The CP, Query-DP, `validator_worker`, and `index_builder` are the four
mandatory services; `ephemeral_runner` exists for the async-fallback path and
is optional in a minimal deploy.

## Two topologies, one codebase

### Self-host (default): single-process group on one network

`docker compose up --build` from the repo root brings up the headline
self-host stack:

- `cp` — the only container with a host port (`:8080`). Single public origin.
- `query_dp` — private to the compose network.
- `validator`, `index_builder`, `ephemeral_runner` — private workers.
- `postgres`, `redis`, `minio` — the catalog, the queue bus, and the object
  store. A one-shot `migrator` container applies the schema migrations
  before any app container starts. A one-shot `createbuckets` container
  initialises the MinIO bucket.

The CP/DP split exists *inside* the compose stack — the CP still proxies
`/v1/query` to the DP — but **everything runs on one Docker network on one
host**. There is no public/private network boundary on this topology; adding
one is an orchestrator concern, not a compose property. For self-host that is
fine: the only externally reachable port is the CP's `:8080`.

The compose file is [`docker-compose.yml`](../../docker-compose.yml) at the
repo root, heavily commented.

### Production self-host: keep the DP off the public internet

A production self-host run scales the same five service roles horizontally
behind whatever orchestrator you prefer (Kubernetes, Nomad, Fly, ECS, plain
systemd, ...). The shape is the same as compose; the only addition is a
network boundary so that **only the CP terminates a public TLS port**.
`query_dp`, `validator`, `index_builder` and `ephemeral_runner` should live
on a private network reachable only by the CP and each other.

Defence-in-depth: the CP also sends a shared-secret header
`X-RB-Proxy-Secret`; the DP rejects a mismatch with `403 proxy_unauthorized`
when the secret is set (it isn't in self-host or unit tests, where there is
no separate private network and the check is skipped).

See [`docs/deploy/self-host.md`](../deploy/self-host.md) for the canonical
self-host runbook, including the CP/DP envs and the production-hardening
checklist (`RB_REQUIRE_AUTH`, `JWT_SECRET`, managed Postgres / Redis / S3).

## The Redis reliable-queue contract

The cross-process bus is Redis, used as a **reliable queue** — not pub/sub.
The contract is in [`adapters/queue/queue.py`](../../adapters/queue/queue.py)
and matters because every worker depends on it.

- **Publish**: `LPUSH <topic> <message>`. Messages are JSON with a
  `traceparent` field so OTel trace context survives the queue hop.
- **Consume**: `LMOVE <topic> <topic>:processing LEFT RIGHT` (or blocking
  `BLMOVE`) — atomic move onto a per-topic **processing** list, so a message
  is never absent from Redis mid-handler and a worker crash cannot drop it.
- **Ack**: `LREM <topic>:processing 1 <message>` after the handler succeeds.
- **Nack with retry**: re-`LPUSH` to the live list, bumping a per-message
  attempt counter.
- **Dead-letter**: a message past `QUEUE_MAX_ATTEMPTS` (default 5) goes to
  `<topic>:dlq` and off the processing list. No automatic redrive — DLQ
  entries need operator inspection.
- **Reaper**: a single-leader thread (`adapters/queue/reaper.py`, in the
  `index_builder` process) re-`LPUSH`es any processing-list message older
  than `QUEUE_RECLAIM_TIMEOUT` (default 5 min) — recovering a message whose
  consumer died after `LMOVE` but before `LREM`. Leadership is gated by a
  short-lived Redis `SETNX` lock so only one reaper runs across replicas.

Consequences every handler is built around:

- **At-least-once delivery.** A timed-out attempt can be redelivered, so
  handlers are idempotent (a duplicate `DATASET_READY` for the same landing
  parts must not double-index).
- **No ordering guarantee.** The builder serialises same-dataset messages via
  a Postgres advisory lock (next section).
- **No fan-out.** A down subscriber polls when it returns; there are no
  consumer groups.

Topics in use today:

| Topic | Published by | Consumed by |
|---|---|---|
| `VALIDATE_DATASET` | CP (ingest endpoints + import-complete) | `validator_worker` |
| `DATASET_READY` | `validator_worker` | `index_builder` |
| `RUN_EPHEMERAL_QUERY` | Query-DP (cold-path fallback) | `ephemeral_runner` |
| `RESULT_READY` | `ephemeral_runner` | Query-DP (status endpoint) |

## The Postgres state model

The catalog is Postgres — the local `postgres` container in self-host, any
managed Postgres 14+ elsewhere. Schema lives in
[`adapters/state/migrations/`](../../adapters/state/migrations); the runtime
entry point is `adapters.state.state`.

At a high level the model is:

```
tenants ── datasets ── shards
                  │
                  └── imports         (async bulk-import jobs)
api_keys ── tenants                   (auth)
```

- **`tenants`** — one row per customer. Holds `plan`, the per-tenant
  `dp_pool` column the CP proxy reads when routing `/v1/query`
  (`'shared'` → the default Query-DP pool, `'dedicated-<tenant>'` → a stamped
  per-tenant DP pool), and the usage counters the quota subsystem updates.
  In OSS-mode (`RB_REQUIRE_AUTH` off) a single default tenant exists; every
  request is attributed to it.
- **`api_keys`** — one row per issued `rb_live_…` key, stored as a SHA-256
  digest of the raw key (never the raw key itself; bcrypt is used only for
  human passwords in `tenants.password_hash`). Cross-tenant lookups return
  `404 not_found`, never `403` — we do not leak the existence of another
  tenant's keys.
- **`dataset_catalog`** — one row per `(tenant_id, name)`. Carries
  `dimension`, `status` (`empty`/`validating`/`indexing`/`indexed`/`error`),
  `row_count`, soft-delete `deleted_at`, and the `status_updated_at`
  timestamp the reconciliation reaper reads to detect stuck datasets.
- **`shard_catalog`** — one row per built FAISS shard. Carries the shard's
  `s3://` URI, checksum, vector count, index type, the `supersedes[]` array
  (the previous shards this one replaces — used by the post-build sweeper),
  and a `sealed` flag.
- **`imports`** — one row per async bulk-import job, tracking the staged
  upload URI, the rejected-records sidecar URI, accept / reject counts, and
  job status.

### Advisory locks for per-dataset builds

Two operations would race destructively under multi-replica workers; both use
Postgres advisory locks.

- **Schema migration**. On boot every process briefly takes
  `pg_advisory_xact_lock(_MIGRATE_LOCK_KEY)`. Whichever wins applies any
  pending migrations under the lock; the losers see the post-migration
  schema. An out-of-band release step (`python -m scripts.migrate`) can do
  the same once per deploy with `RB_SKIP_MIGRATE=0` — the in-process check
  is the fallback.
- **Per-dataset builds**. `pg_try_advisory_lock(_BUILD_LOCK_CLASS, hash(tenant, dataset))`
  is taken **non-blockingly** before the `index_builder` starts a build for
  a dataset. A second replica that consumed an overlapping `DATASET_READY`
  for the same dataset and loses the lock NACKs the message back onto the
  queue — the holder finishes and the loser gets redelivered against the
  updated catalog state. This is what makes the builder safe to run with
  more than one replica.

## Object storage layout

Bytes live in S3-compatible storage — MinIO in self-host, any S3-compatible
store (S3, R2, ...) elsewhere. The schemes are `s3://` and the in-memory
unit-test `memory://`; there is deliberately no `file://` adapter.

Landing data:

```
landing/<tenant>/<dataset>/upload-<id>/part-NNNN.parquet
landing/<tenant>/<dataset>/imports/<import_id>/raw.<fmt>     (staged bulk uploads)
landing/<tenant>/<dataset>/imports/<import_id>/rejected.jsonl
```

Index shards have two key shapes, selected by `RB_SHARD_VERSIONED_URIS`
(`adapters/storage/shard_uri.py`, `services/index_builder/run.py`):

```
# Default (flag off): date-partitioned, mutable-key shape
{INDEXES_PREFIX}/{tenant}/{dataset}/indexes/{YYYY-MM-DD}/shard-{epoch_ms}-{uuid8}.bin
                                                         shard-{epoch_ms}-{uuid8}.bin.meta.json

# Versioned (flag on): flat, content-addressed shape under the bucket root
s3://{bucket}/{tenant}/{dataset}/{shard_id}-{content_hash}.bin
```

`content_hash` is the first 16 hex chars of `sha256(serialised_index)`, so the
key is a verifiable receipt for the bytes underneath it. Two prefixes are
configurable via env: `LANDING_PREFIX` (default `s3://<bucket>/landing`) and
`INDEXES_PREFIX` (default `s3://<bucket>/indexes`).

### The sidecar

FAISS stores integer ids, not strings. Every shard `*.bin` has a companion
`*.bin.meta.json` written next to it: a JSON object mapping the int64 FAISS
id of each vector to its caller-supplied string `id` plus the optional
`metadata` object. Query-DP loads both files when caching a shard; the FAISS
search returns int64 ids, the sidecar translates back to the API-visible
`id` and `metadata`.

Incremental ingest merges the new batch's ids and metadata into the existing
sidecar before re-serialising — the builder never loses previously indexed
rows. The full incremental-append behaviour is in
[`docs/indexing.md`](../indexing.md).

## The hot query path

The Query-DP `POST /v1/query` handler is short. Its work is dominated by the
FAISS call, not the surrounding plumbing.

1. Trust `X-RB-Tenant-Id` from the request headers (the CP already
   authenticated the caller and resolved the tenant; the DP does not
   re-authenticate).
2. Resolve the **newest** shard for `(tenant, dataset)` from `shard_catalog`.
3. Look up that shard's id in the in-process **shard cache** — an
   `OrderedDict` LRU keyed by `shard_id`, holding the deserialised FAISS
   index plus the parsed sidecar. On hit: skip steps 4-5.
4. On miss: download the shard `.bin` and `.bin.meta.json` from object
   storage, `faiss.read_index` the bytes, parse the sidecar.
5. Insert into the cache with a measured byte footprint. Evict LRU entries
   until the running total fits `RB_SHARD_CACHE_BYTES` (default 512 MB).
   The byte budget — not a count cap — is what bounds DP memory: shard
   footprints span ~100× across datasets (a 1k-vector shard vs a 1M-vector
   ~430 MB one), so a count cap cannot pin memory usage. See
   [`mmap.md`](mmap.md) for the `RB_FAISS_MMAP` flag that lets a single shard
   exceed the cache budget by paging-in only the touched IVF cells.
6. Configure `nprobe` (how many IVF cells to scan): per-request, or
   `RB_QUERY_NPROBE` (default 64), clamped to a ceiling.
7. Run `index.search(vector, top_k)`.
8. Translate the int64 hits to `(id, score, metadata)` via the sidecar.
   Apply the optional flat AND-of-equals `filter` (post-search — FAISS
   cannot filter by metadata).
9. Return.

The index is FAISS **IVFFlat** — raw float32 vectors partitioned into `nlist`
cells by an `IndexFlatL2` coarse quantizer. Search ranks on **exact** L2
distance within the probed cells. RosalindDB used to build IVF+PQ; recall@10
on a 100k-vector SIFT benchmark was 0.22 (default `nprobe=1`) and ceilinged
at ~0.65 even at `nprobe=nlist`. IVFFlat with `nprobe=64` reaches ~0.99. The
operational detail is in [`docs/indexing.md`](../indexing.md).

When the cache or the network is cold and the synchronous path would breach
the request budget, Query-DP falls back to the **ephemeral path**: it
`LPUSH`es a `RUN_EPHEMERAL_QUERY` message with a correlation id and returns
`202` plus a `job_id`. The `ephemeral_runner` consumes the message, runs the
search, and `LPUSH`es a `RESULT_READY` message that Query-DP drains via a
background thread into an in-memory result store. The client polls
`GET /v1/query/status/{job_id}` until `ready: true`.

## The async ingest path

There is **no synchronous CP→DP hop on ingest**. The CP does the customer-
facing work; everything heavy is queued.

### Small uploads — `POST /v1/datasets/{name}/vectors`

1. CP authenticates the caller, applies the rate limit and the
   `try_consume_vectors` admission quota (all-or-nothing) when
   `RB_ENABLE_QUOTAS` is on, parses the NDJSON body under a hard 10 MiB
   byte cap, validates each record's dimension and shape.
2. CP writes the accepted records as a JSONL part under
   `landing/<tenant>/<dataset>/upload-<id>/`.
3. CP `LPUSH`es a `VALIDATE_DATASET` message and returns
   `{accepted, rejected, errors[], job_id}` to the client. The CP's
   involvement ends at the `LPUSH`.
4. `validator_worker` consumes `VALIDATE_DATASET`, streams the staged JSONL,
   re-validates, writes Parquet to `landing/.../part-*.parquet`, updates
   `dataset_catalog.status` to `validating` → ready, `LPUSH`es
   `DATASET_READY`.
5. `index_builder` consumes `DATASET_READY`, takes the per-dataset advisory
   lock, builds **incrementally** (loads the current shard + sidecar, reads
   only landing parts not already indexed, `add()`s the new vectors onto
   the trained index, merges the sidecar), writes the new shard, inserts a
   `shard_catalog` row that `supersedes[]` the old one, releases the lock.
6. The client polls `GET /v1/datasets/{name}` for `status: indexed`.

### Large uploads — `POST /v1/datasets/{name}/imports*`

The same shape, but the bytes never traverse the application. The CP returns
a **presigned PUT URL** to the client, which uploads directly to object
storage. A subsequent `POST .../complete` call `LPUSH`es
`VALIDATE_DATASET` against the staged object. The validator `head`s the
staged file to enforce the size cap server-side (a presigned PUT cannot
enforce that itself), then proceeds identically.

The bulk-import contract is [`docs/api/imports.md`](../api/imports.md).

## OSS vs SaaS — two env switches

The same image runs both worlds. Two env vars decide which.

### `RB_REQUIRE_AUTH`

Off by default. When **off** (the OSS default), the auth surface — signup,
login, `/auth/me`, `/auth/keys*` — is hidden (those routes return 404), every
request is attributed to a single default tenant, and any caller who can
reach the CP can use the API. A loud startup banner warns about exposing
self-host to the public internet without setting this. When **on**, the full
auth stack lights up: signup, JWT issuance, `rb_live_…` API keys, per-tenant
isolation. See [`services/auth/auth.py`](../../services/auth/auth.py).

### `RB_ENABLE_QUOTAS`

Off by default. When **off** (the OSS default), the rate-limit FastAPI
dependency is a no-op, the ingest and query handlers skip
`try_consume_vectors` / `try_consume_query`, and `GET /auth/usage` returns
`{"enabled": false}`. A self-hoster's own queries are never throttled by the
default 10k-queries/day cap — that would be a footgun on a self-host
install. When **on**, the per-tenant token-bucket rate limiter, the daily
query quota (enforced by `state.try_consume_query` via an atomic
`UPDATE … RETURNING` on `tenants.queries_today` in Postgres), and the
all-or-nothing ingest admission quota are enforced. The schema (the `tenants.vectors_used`,
`queries_today` columns) exists either way — only the runtime checks are
gated. See [`services/auth/quota.py`](../../services/auth/quota.py).

Neither switch is a build flag. Flipping either env var at runtime and
restarting the CP is sufficient.

## Configuration surface

The notable env vars, beyond `RB_REQUIRE_AUTH` and `RB_ENABLE_QUOTAS`:

| Var | Default | What it does |
|---|---|---|
| `DATABASE_URL` | — | Postgres connection string. Required. |
| `REDIS_URL` | — | Redis URL for the queue + ephemeral result store. Required. |
| `S3_ENDPOINT_URL` / `S3_*` | — | Object-storage credentials. Required. |
| `LANDING_PREFIX` | `s3://<bucket>/landing` | Object-store prefix for landing data. |
| `INDEXES_PREFIX` | `s3://<bucket>/indexes` | Object-store prefix for FAISS shards. |
| `QUERY_DP_URL` | (compose: `http://query_dp:8080`) | CP's base URL for the shared Query-DP pool. |
| `RB_PROXY_SECRET` | unset | If set on both `cp` and `query-dp`, the DP rejects CP requests without a matching `X-RB-Proxy-Secret`. |
| `RB_QUERY_NPROBE` | `64` | IVF cells to scan per query. |
| `RB_SHARD_CACHE_BYTES` | `536870912` (512 MB) | Query-DP in-process shard-cache budget. |
| `QUEUE_MAX_ATTEMPTS` | `5` | Deliveries before a message is dead-lettered. |
| `QUEUE_RECLAIM_TIMEOUT` | `300` | Seconds before the reaper considers a processing-list message stale. |
| `IMPORT_MAX_BYTES` | `5368709120` (5 GiB) | Bulk-import staged-upload size cap. |
| `RB_PG_POOL_MAX` | `10` | Postgres pool max per process. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | OTLP destination. |
| `OTEL_SERVICE_NAME` | per process | Trace / metric service name. |

## Observability

OpenTelemetry throughout. Every process exports OTLP/HTTP to a collector you
point it at via `OTEL_EXPORTER_OTLP_ENDPOINT` (any OTLP-compatible backend
works — Grafana Cloud, Honeycomb, Tempo + Loki, Datadog, your own
collector). The query path is a **two-service trace**: a CP server span →
a CP `httpx` client span (`HTTPXClientInstrumentor` injects `traceparent`) →
a Query-DP server span. The ingest path's trace context rides the Redis
message body, so the validator and builder re-establish trace context from
the dequeued message.

## Scaling and resilience

- **CP** — stateless; scale by machine count. The only public surface, so
  keep warm capacity.
- **Query-DP** — latency-critical; scale by machine count, multi-worker per
  machine is fine (`--workers N` ≈ vCPUs). The shard cache is per-process, so
  a new worker is cold until it fills.
- **Ingest workers** (`validator`, `index_builder`) — scale horizontally; the
  per-dataset advisory lock keeps the builder safe under parallel replicas.
- **Postgres** — shared; one pooled request-scoped connection per HTTP
  request.
- **Redis** — shared, and the failure-domain center: queue topics, the
  ephemeral result store, and (with quotas on) the rate-limit and
  daily-query counters all live here. Worker crashes are safe (reliable-queue
  contract above); a *Redis* outage stalls all async work.

## Related documents

- [`docs/api/v1.md`](../api/v1.md) — the public v1 API contract.
- [`docs/api/imports.md`](../api/imports.md) — the bulk-import flow.
- [`docs/indexing.md`](../indexing.md) — incremental-append builder behaviour.
- [`docs/deploy/self-host.md`](../deploy/self-host.md) — the canonical self-host runbook.
- [`docker-compose.yml`](../../docker-compose.yml) — the headline self-host topology.
