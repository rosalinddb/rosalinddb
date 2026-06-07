<div align="center">

<img src="logo.png" alt="RosalindDB logo" width="160" height="160">

# RosalindDB

**Object-storage-native vector database with read-your-writes.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](#quickstart)
[![Self-hostable](https://img.shields.io/badge/self--hostable-yes-green.svg)](docs/deploy/self-host.md)

</div>

---

## What it is

The index lives where your data already lives — on object storage — and search is served from a small, byte-budgeted in-process cache. No always-on cluster; cost tracks what you query, not corpus size.

The differentiator is **strong read-your-writes**: an optional **recall tier** (pgvector) takes synchronous, immediately-queryable upserts, while the cold **consolidate tier** holds FAISS shards on S3. Queries union the two LSM-style — recall is the memtable, the S3 shards are the SSTables.

- **Built for** — agent memory, early-stage RAG, batch retrieval, cost-sensitive search.
- **Not for** — sub-10ms p50 at scale, billion-vector single-node, a drop-in hot-tier cluster.

## Features

- **Object-storage-native** — immutable FAISS IVFFlat shards on any S3 store; no always-on cluster.
- **Strong read-your-writes** — optional pgvector recall tier; upserts are durable and instantly queryable.
- **LSM tiering** — hot recall ∪ cold S3 shards on a `consolidated_lsn` watermark; optional delta tier for incremental compaction.
- **One image, five roles** — control plane, query DP, validator, index builder, ephemeral runner.
- **Scale-to-zero workers** — Redis reliable queue + DLQ + reaper.
- **Auth & quotas, opt-in** — off by default; two env switches make it multi-tenant (JWT + API keys).
- **OpenTelemetry** — OTLP traces/metrics/logs, fully no-op-able.

## Architecture

Two tiers, partitioned by a write-freshness watermark:

| Tier | Storage | Role | Flag |
|---|---|---|---|
| **Consolidated** (cold) | FAISS IVFFlat shards on S3 | always on | — |
| **Recall** (hot) | separate pgvector instance | read-your-writes | `RB_RECALL_DSN` (off) |

```
read query
  ├─► recall scan (pgvector, exact L2²)     lsn >  consolidated_lsn
  └─► consolidated FAISS search (IVFFlat)    lsn <= consolidated_lsn
         ▼
   merge → recall wins above the watermark, tombstone/filter suppression,
           sort by L2², truncate top_k        (wall-time ≈ max, not sum)
```

- **Watermark seam** — `consolidated_lsn` (on `shard_catalog`) puts every vector in exactly one tier, so the union is complete and non-overlapping.
- **Consolidation** folds recall → cold shards and advances the watermark — on cap (`RB_RECALL_MAX_ROWS`, 2000) or idle (`RB_RECALL_IDLE_S`, 60s).
- **Delta-tier LSM** (`RB_DELTA_TIER`, off) — cold shards become base + ≤8 deltas; minor fold is `O(new rows)`, major compaction at the cap. Validated flat to 1M vectors.

→ [`recall-consolidate.md`](docs/architecture/recall-consolidate.md) · diagrams in [`docs/architecture/diagrams/`](docs/architecture/diagrams/)

### Five roles, one image

| Role | Command | Public |
|---|---|---|
| Control Plane | `services.control_plane.cp_app:app` | **yes — `:8080`** |
| Query Data Plane | `services.query_api.dp_app:app` | no |
| validator_worker | `python -m services.validator_worker.run` | no |
| index_builder | `python -m services.index_builder.run` | no |
| ephemeral_runner | `python -m services.ephemeral_runner.run` | no |

The CP is the only public origin; workers consume a Redis queue. Infra: `postgres` (catalog), `pgvector` (recall, `:5433`, idle until enabled), `redis`, `minio` (S3). → [`architecture.md`](docs/architecture/architecture.md)

## Quickstart

Needs **Docker**.

```bash
git clone https://github.com/rosalinddb/rosalinddb.git && cd rosalinddb
make run-local                         # build + docker compose up -d
curl http://localhost:8080/healthz     # {"status":"ok","service":"control_plane"}
make smoke                             # full happy-path check (health→ingest→query)
```

Only the CP publishes a port (`:8080`); everything else stays private to the compose network. Dev defaults (`postgres/postgres`, `minio/minio123`, auth **off**) are localhost-only.

Minimal flow — auth is off by default, so no header is needed:

```bash
BASE=http://localhost:8080

curl -s -X POST "$BASE/v1/datasets" -H 'Content-Type: application/json' \
  -d '{"name":"demo","dimension":4}'

printf '%s\n' '{"id":"v0","values":[0,1,2,3]}' '{"id":"v1","values":[1,2,3,4]}' \
| curl -s -X POST "$BASE/v1/datasets/demo/vectors" \
  -H 'Content-Type: application/x-ndjson' --data-binary @-

curl -s "$BASE/v1/datasets/demo"       # poll until "status":"indexed"
curl -s -X POST "$BASE/v1/query" -H 'Content-Type: application/json' \
  -d '{"dataset":"demo","vector":[0,1,2,3],"top_k":5}'
```

`score` = raw FAISS L2 (lower is closer). `mode` ∈ `hot | cold | recall | ephemeral`. Auth-on: `POST /auth/signup`, then send `Authorization: Bearer rb_live_…`. Full contract → [`docs/api/v1.md`](docs/api/v1.md).

## Configuration

**Runs on defaults — set only what points at your infra.** Every flag is off by default; the full, typed surface is [`src/adapters/config.py`](src/adapters/config.py).

| Var | Default | Purpose |
|---|---|---|
| `S3_ENDPOINT_URL` · `S3_ACCESS_KEY` · `S3_SECRET_KEY` | AWS / unset | Object store (MinIO, R2, GCS…) |
| `INDEXES_PREFIX` · `LANDING_PREFIX` | `s3://rosalinddb/…` | Where shards / ingests live |
| `DATABASE_URL` | `memory://local` | Catalog DSN — **point at Postgres for any real deploy** |
| `RB_RECALL_DSN` | off | Separate pgvector instance → enables read-your-writes |
| `RB_DELTA_TIER` | off | Delta-tier (LSM) query path |
| `RB_REQUIRE_AUTH` | `false` | **Set `true` for any public deploy** |
| `JWT_SECRET` | `dev-secret` | Required when auth on (`validate()` fails at boot if unset). `openssl rand -hex 32` |

Tuning knobs — pools, `nprobe`, recall lifecycle, SSD cache, quotas, `OTEL_*` — all have safe defaults. → [`.env.example`](.env.example)

## Production notes

What bites real self-hosters:

- **PgBouncer txn-pooling breaks advisory locks** — the migrator + builder lock need **session** pooling (the recall pgvector instance is fine on txn pooling).
- **The shard cache is ephemeral** — warms on first cold query, gone on `compose down`; no volume to persist.
- **JWTs are HS256 / 24h, no refresh** — server-to-server callers should use `rb_live_…` API keys.
- **API keys are SHA-256 at rest** — the raw value is shown once; store it like a password.

More → [`docs/deploy/self-host.md`](docs/deploy/self-host.md)

## API

Served from the CP on `:8080`. Full reference → [`docs/api/`](docs/api/).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness (unauthenticated) |
| `POST` | `/auth/signup` · `/auth/login` | Tenant / JWT (`404` when auth off) |
| `POST` `GET` `DELETE` | `/v1/datasets[/{name}]` | Create / list / get / soft-delete a dataset |
| `POST` `GET` | `/v1/datasets/{name}/vectors` | Ingest (NDJSON, 10 MiB) / list |
| `GET` `DELETE` | `/v1/datasets/{name}/vectors/{id}` | Fetch / delete one vector |
| `POST` | `/v1/datasets/{name}/imports` | Bulk import via presigned upload (5 GiB) |
| `POST` | `/v1/query` | Nearest-neighbour query → `{matches, latency_ms, mode}` |
| `GET` | `/v1/query/status/{job_id}` | Poll an async (`ephemeral`) result |

## MCP server

Operate RosalindDB from any MCP client (Claude Desktop, Cursor, Claude Code) → [rosalinddb/rosalinddb-mcp](https://github.com/rosalinddb/rosalinddb-mcp).

## Development

Real installable package, **src-layout**, `pip install -e .` — imports are `services.*` / `adapters.*`, no path hacks. Python 3.11.

| Target | Does |
|---|---|
| `make venv` | `.venv` + `pip install -e .` |
| `make test-unit` | `pytest -m unit` — hermetic, no Docker |
| `make test-integration` | real MinIO via testcontainers (Docker) |
| `make run-local` | build + `compose up -d` |
| `make fmt` · `make lint` | ruff |

Tests are marked by directory (`tests/unit` → unit, `tests/integration` → integration). Integration runs against real MinIO + Postgres on purpose — fakes get the seams (multipart, presigned URLs, advisory locks) wrong. Patch workflow → [`CONTRIBUTING.md`](CONTRIBUTING.md).

```
src/
  services/   # 5 roles + auth, source_registry, _common
  adapters/   # storage, state, recall, queue, cache, observability, config.py
  schemas/    # request/response models
tests/{unit,integration}/ · docs/ · bench/ (private)
```

## License

Apache 2.0 — [`LICENSE`](LICENSE).
Deploy → [`self-host.md`](docs/deploy/self-host.md) · Architecture → [`docs/architecture/`](docs/architecture/) · Security → [`SECURITY.md`](SECURITY.md)
