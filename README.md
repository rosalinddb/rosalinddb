<div align="center">

<img src="logo.png" alt="RosalindDB logo" width="160" height="160">

# RosalindDB

**Object-storage-first vector database for cold and bursty workloads.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](#quickstart)
[![Self-hostable](https://img.shields.io/badge/self--hostable-yes-green.svg)](docs/deploy/self-host.md)

</div>

---

RosalindDB stores FAISS IVFFlat shards on any S3-compatible object store and
serves nearest-neighbour search out of a byte-budgeted in-process cache. The
always-on footprint is small; heavy work (validate, build, cold queries) runs
on queue-driven workers that can scale to zero.

**Built for:** agent long-term memory, indie or early-stage RAG over
slowly-changing corpora, batch retrieval, internal-tool search, cost-sensitive
similarity lookups where always-on cluster pricing is the wrong shape.

**Not built for:** sub-10ms p50 interactive search at scale, billion-vector
multi-tenant production on a single node, or a drop-in replacement for a
tuned hot-tier in-memory cluster.

## Quickstart

Requirements: Docker and `curl`. No signup.

```bash
git clone https://github.com/rosalinddb/rosalinddb.git
cd rosalinddb
docker compose up
```

The Control Plane comes up on `http://localhost:8080` as the single public
origin. MinIO, Postgres, Redis, and the validator / builder / ephemeral
workers all run privately on the compose network.

Once the stack is healthy:

```bash
# Create a dataset (dimension fixed at create time).
curl -X POST http://localhost:8080/v1/datasets \
  -H 'Content-Type: application/json' \
  -d '{"name": "products", "dimension": 4}'

# Ingest a couple of vectors as NDJSON.
curl -X POST http://localhost:8080/v1/datasets/products/vectors \
  -H 'Content-Type: application/x-ndjson' \
  --data-binary $'{"id":"a","values":[0.1,0.2,0.3,0.4],"metadata":{"category":"books"}}\n{"id":"b","values":[0.5,0.5,0.5,0.5],"metadata":{"category":"movies"}}\n'

# Query (top-k nearest, optional AND-of-equals metadata filter).
curl -X POST http://localhost:8080/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"products","vector":[0.1,0.2,0.3,0.4],"top_k":2,"filter":{"category":"books"}}'
```

For uploads above the 10 MiB request cap, use the async bulk-import flow in
[`docs/api/imports.md`](docs/api/imports.md). The full REST contract — every
endpoint, every error code — is in [`docs/api/v1.md`](docs/api/v1.md).

## Features

- Vector search — FAISS IVFFlat shards on S3-compatible object storage.
- Metadata filtering — flat AND-of-equals, strict type-and-value match
  (no coercion, no ranges, no OR in v1).
- Incremental indexing — subsequent ingests `add()` to the trained shard;
  no full rebuild per batch.
- Async bulk import — NDJSON and Parquet via presigned PUT, with a
  rejected-records report and `continue`/`abort` modes.
- Multi-tenancy and API keys — opt-in via `RB_REQUIRE_AUTH=true`.
- Per-tenant quotas — opt-in via `RB_ENABLE_QUOTAS=true`.
- CP / DP split — public Control Plane, private Data Plane; the query
  path is isolated from auth and ingest admission.
- Reliable queue — Redis-backed, at-least-once delivery, DLQ, reaper.
- OpenTelemetry observability — metrics, traces, structured logs over OTLP
  to any backend you point it at.
- One image, many roles — every service runs from the same `Dockerfile`;
  per-process commands live in `docker-compose.yml`.

## Run modes

| Mode | Auth | Quotas | Use case |
|---|---|---|---|
| OSS default | off | off | Local dev. Single-tenant self-host on a private network. |
| Production self-host | on | optional | Multi-tenant self-host behind a public URL. |

**OSS default.** `docker compose up`. No auth, single implicit `default`
tenant, quotas off, one public port (`:8080`). The CP logs a loud warning on
startup if it detects a likely-public bind. **Do not expose this to the
public internet without flipping `RB_REQUIRE_AUTH=true` first.**

**Production self-host.** Set `RB_REQUIRE_AUTH=true` to turn on the full
signup + API-key + multi-tenant stack. Set `RB_ENABLE_QUOTAS=true` to
enforce per-tenant vector and query caps. Set a real `JWT_SECRET` (e.g.
`openssl rand -hex 32`) — the bundled `dev-secret` is a dev-only default
and a non-starter for anything real. Point `DATABASE_URL`, `REDIS_URL`, and
the `S3_*` envs at your own managed services. Walkthrough in
[`docs/deploy/self-host.md`](docs/deploy/self-host.md).

## Architecture

Five service roles, one image: a public **Control Plane** (auth, dataset
CRUD, ingest admission, `/v1/query` reverse proxy) plus a private
**Query Data Plane** and three async workers (`validator_worker`,
`index_builder`, `ephemeral_runner`) that consume from a Redis reliable
queue. Each dataset owns one or more FAISS IVFFlat shards on object
storage; subsequent ingests `add()` to the existing trained shard instead
of rebuilding from scratch.

Full design — process roles, trust model, queue topics, catalog schema,
shard cache budgeting — in
[`docs/architecture/architecture.md`](docs/architecture/architecture.md).

## Production notes

A short, opinionated list — the things that have bitten real self-hosters.

- **PgBouncer in transaction-pooling mode breaks advisory locks.** The
  migrator and the per-dataset builder lock both rely on PostgreSQL
  session-level advisory locks. PgBouncer's transaction mode releases the
  lock between statements, which deadlocks the migration and lets two
  builders race on the same dataset. Use session-pooling, or skip
  PgBouncer for the catalog connection.
- **The FAISS shard cache is ephemeral by design.** It lives inside the
  container filesystem, warms on first cold query, and is gone on
  `docker compose down`. There is no volume to persist — re-warm takes
  seconds and avoids stale-cache hazards on shard rebuilds.
- **JWTs are HS256 with a 24h TTL** and there is no refresh-token flow
  yet. Server-to-server callers should mint `rb_live_…` API keys instead.
- **API keys are SHA-256 hashed at rest.** The raw value is surfaced
  once on creation and never again — store it where you'd store a
  password.

More gotchas in [`docs/deploy/self-host.md`](docs/deploy/self-host.md) §5.

## MCP server

The companion MCP server for Claude / Cursor users lives at
[rosalinddb/rosalinddb-mcp](https://github.com/rosalinddb/rosalinddb-mcp).
It exposes the full RosalindDB management surface (datasets, ingest,
query) as MCP tools.

## Development

```bash
make test               # unit + integration; integration needs Docker
make test-unit          # fast, hermetic — memory:// storage, no Docker
make test-integration   # real MinIO + Postgres + Redis via testcontainers
make smoke              # post-deploy gate against a running instance
make lint               # ruff check
```

Integration tests run against real MinIO and real Postgres because the
storage and state adapters are the parts that break in production —
fakes get fakey. Patch flow is in [`CONTRIBUTING.md`](CONTRIBUTING.md);
vulnerability reports in [`SECURITY.md`](SECURITY.md).

## License

Apache 2.0. See [`LICENSE`](LICENSE).
