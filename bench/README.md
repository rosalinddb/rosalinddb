# bench

Local query-stress harness. Drives the running backend from k6 inside the
same docker network — request RTT is loopback, not WAN — and captures
per-container CPU/memory plus OTLP traces for each cell of the matrix.

## What's in here

| File | Purpose |
| --- | --- |
| `seed_corpus.py` | Seeds N tenants x M vectors at a given dim, writes `cache/dim-<N>.json`. |
| `load_test_queries.js` | k6 pure-query loop reading a corpus file via `open()`. |
| `docker-compose.bench.yml` | Compose override on the base stack: per-service CPU/mem caps, auth + quotas on, OTLP wired, k6 container on the network. |
| `run_matrix.sh` | Drives the 6-cell matrix (128 + 1536 dim x 10/20/50 VU). |
| `run_mmap_comparison.sh` | Runs a 3-cell matrix twice against the same corpus, with `RB_FAISS_MMAP=false` then `=true`. |
| `analyze.py` | Aggregates raw cells into `RESULTS.md` and `summary.json`. |
| `docker-compose.recall-bench.yml` | Self-contained **recall-enabled** bench stack (`RB_RECALL=true`, separate project + remapped ports) for the multi-agent memory bench. |
| `load_test_agents.js` | k6 multi-agent loop: N VUs = N agents writing + searching + a read-your-writes probe on the recall path. |
| `run_agents_bench.sh` | Sweeps `MODE` x `AGENTS` against the recall-bench stack; `--smoke` for harness validation. |
| `analyze_agents.py` | Aggregates agent cells into `RESULTS.md` + `summary.json` with a per-agent-vs-shared comparison. |

## Prerequisites

- Docker, with VM headroom for the capped containers (~5 CPU, ~6 GB)
- Python 3.11+ with `requests` installed
- The `observability/` stack running (OTEL collector + Tempo + Prometheus + Grafana)

## Run it

```bash
# 1. Observability stack
docker compose -p rosalinddb-observability \
  -f observability/docker-compose.yml up -d

# 2. Backend stack with bench overrides
RB_REQUIRE_AUTH=true RB_ENABLE_QUOTAS=true \
  docker compose -p rosalinddb-bench \
  -f docker-compose.yml -f bench/docker-compose.bench.yml up -d

# 3. Drive the matrix (~40 minutes)
bash bench/run_matrix.sh

# 4. Aggregate raw cells into a digest
python bench/analyze.py bench/results/<timestamp>
```

The matrix produces `bench/results/<timestamp>/dim-<N>/vus-<V>/` directories,
each with `k6_summary.json`, `docker_stats.jsonl`, and the cell start/end
timestamps for drill-down in Tempo or Prometheus.

## Tuning

- `DURATION`, `TENANTS`, `VECTORS_PER`, `DIMS`, `VUS_LIST` — env vars on `run_matrix.sh`
- CPU and memory caps — `docker-compose.bench.yml`
- Latency thresholds and percentiles — `load_test_queries.js`

## Mmap comparison bench

`run_mmap_comparison.sh` measures the per-cell QPS and latency tax (or
benefit) of the `RB_FAISS_MMAP` flag on a single seeded corpus. It runs
the same 3-cell matrix (10/20/50 VU x 1 min) twice — once with the flag
off, once on — restarting
only the FAISS-loading services (`query_dp`, `ephemeral_runner`) between
runs. The corpus is seeded once with `--single-tenant` and a high
`--vectors-per`, so the two runs hit identical shards.

### Prereqs

- The observability + backend bench stacks are up (same bring-up as the
  matrix bench above).
- ~45 minutes of wall time for a full 1M-vector, 1536-dim, 3-cell run on
  each side. Use `--smoke` first to verify the harness end-to-end.

### Run

```bash
# full run: 1M x 1536 single tenant, 3 cells (10/20/50 VU x 1 min) x 2
bash bench/run_mmap_comparison.sh

# smoke run: 10-vector seed, 30s cells; ~5 minutes; for harness validation
bash bench/run_mmap_comparison.sh --smoke
```

Override knobs via env vars: `DIM`, `VECTORS_PER`, `VUS_LIST`, `DURATION`,
`INDEX_TIMEOUT`.

### Results

```
bench/results/<timestamp>-mmap/
  seed.log
  off/
    dim-<N>/vus-<V>/{k6_summary.json,k6_stdout.log,docker_stats.jsonl,...}
    RESULTS.md
    summary.json
  on/
    dim-<N>/vus-<V>/...
    RESULTS.md
    summary.json
```

`analyze.py` runs on each subdir at the end of the script.

### Interpretation

- **Same-cell QPS delta** (off vs on) is the steady-state tax or benefit of
  mmap: how much throughput the kernel's page cache costs (or saves) once
  pages are hot.
- **p99 divergence at low VU** is the cold-page-fault story — under thin
  load the on-disk shard is not fully resident, so the long tail picks up
  major faults on a percentage of queries. Watch p99 first, p95 second.
- **`query_dp` mean memory in `docker_stats.jsonl`** drops with mmap on:
  RSS no longer covers the full index, so the per-shard footprint is
  bounded by what the kernel chooses to keep resident.

## Multi-agent memory load

A concurrent-agent stress of the **recall (read-your-writes) path** — the
`RB_RECALL` tier that accepts synchronous writes and unions them into queries.
N k6 VUs model N agents; each agent loops: **write** a batch of memory vectors
(sync `200`), **search** its own memories, and run a **read-your-writes probe**
(write a sentinel id, immediately query with its exact vector, record whether
the id comes back and the round-trip lag).

### What it measures

- **Write throughput** (`rb_writes` counter -> ops/s) and write latency
  (p50/p95/p99) on the synchronous recall write path.
- **Search latency** (p50/p95/p99) of the union query.
- **Read-your-writes**: `rb_ryw_hit` (Rate — was the just-written sentinel
  returned?) and `rb_ryw_lag_ms` (Trend — write→visible round-trip).
- **Error rate** (`rb_errors`) across writes, searches, and probes.
- A **per-agent vs shared** comparison across agent counts.

### Two modes

- **`per-agent`** (default, recommended) — each agent owns its OWN dataset. The
  recall brute-force scan per query is scoped to that small partition; this is
  the model the recall tier is designed for.
- **`shared`** — ALL agents write to ONE dataset, tagging each record with
  `{agent_id: <vu>}`, and searches filter `{agent_id: <vu>}` (exhaustive
  server-side). As the single partition grows, every query brute-force-scans
  the whole thing — this is the **scaling cliff** the comparison exposes.

### Prereqs

- Docker with headroom for the capped recall stack (~6 CPU, ~7 GB incl. the
  extra `pgvector` recall instance).
- The image must contain the recall feature code — `run_agents_bench.sh`
  rebuilds `rosalinddb-backend:latest` from this worktree by default (pass
  `--no-build` to reuse an existing image).
- Nothing else needs to be running; the harness brings the stack up and tears
  it down. It uses a **separate compose project** (`rosalinddb-recall-bench`)
  and **remapped host ports** (CP `18080`, not `8080`) so it never disturbs a
  base stack already on `:8080`.

### Run it

```bash
# Smoke (3 agents, 30s, both modes) — validates the harness end-to-end:
bash bench/run_agents_bench.sh --smoke

# Full sweep: MODE in {per-agent, shared} x AGENTS in {10,50,100}, 2m cells:
bash bench/run_agents_bench.sh

# Aggregate into RESULTS.md + summary.json:
python3 bench/analyze_agents.py bench/results/agents-<timestamp>
```

The runner brings up the recall-bench stack itself:

```bash
docker compose -p rosalinddb-recall-bench \
  -f bench/docker-compose.recall-bench.yml up -d --build
```

It is a **self-contained** compose file (not a `-f` overlay on the root
compose): the pinned Compose v2.6.0 merges/appends `ports` across `-f` files
and has no `!reset`, so a layered overlay could not drop the base file's
hardcoded `8080`/`5432`/`5433` host bindings — it would clash with a base stack
on `:8080`. A standalone file controls the host ports on any Compose version
with zero risk to a running stack.

Results land in `bench/results/agents-<timestamp>/<mode>/agents-<N>/` —
`k6_summary.json`, `k6_stdout.log`, `docker_stats.jsonl`, and the cell
start/end timestamps. The stack is torn down on exit (`--keep-up` to leave it).

### Knobs

- `AGENTS_LIST`, `MODES`, `DURATION`, `DIM`, `MEMORIES_PER`, `TOP_K`,
  `SETTLE_S` — env vars on `run_agents_bench.sh`.
- `RB_RECALL_MAX_ROWS` / `RB_RECALL_IDLE_S` — lifted high by the overlay so
  consolidation does not churn mid-run; override at compose-up time.
- Host ports — `RB_BENCH_CP_PORT` (default 18080), `RB_BENCH_PGVECTOR_PORT`
  (15433), `RB_BENCH_PG_PORT` (15432), `RB_BENCH_REDIS_PORT` (16379),
  `RB_BENCH_MINIO_PORT` (19000) — in `docker-compose.recall-bench.yml`.
- Latency thresholds (incl. `rb_ryw_hit > 0.99`) — `load_test_agents.js`.
