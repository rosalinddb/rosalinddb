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
