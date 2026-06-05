# Query API (`/v1/query`)

The customer-facing vector-search surface. This page documents the
implementation details that the [v1 contract](./v1.md) deliberately keeps
out of the contract proper.

## Endpoints

- `POST /v1/query` — vector similarity search
- `GET /v1/query/status/{job_id}` — poll an ephemeral result

See the **Query** section of [`v1.md`](./v1.md) for the request/response
schemas and error codes. The response shape is exactly:

```json
{ "matches": [{ "id": "<string>", "score": <float>, "metadata": {} }],
  "latency_ms": <int>, "mode": "hot|cold|recall|ephemeral", "job_id": "<string?>" }
```

(`recall` appears only when the recall tier is enabled — see "Recall-tier
union" below; with `RB_RECALL` off the modes are exactly `hot|cold|ephemeral`.)

## Where it is served

The Control Plane (CP) is the single public origin at `:8080`. The CP
authenticates, rate-limits, and consumes daily quota at the edge, then
reverse-proxies the request to a private **Query Data Plane (DP)** node
which runs the FAISS search. The customer never sees the DP — its base URL
is read from `QUERY_DP_URL` (or a per-tenant `QUERY_DP_URL_<TENANT>` for
dedicated pools).

Implementation:

- `services/query_api/query_proxy.py` — the CP reverse-proxy router
  (`POST /v1/query`, `GET /v1/query/status/{job_id}`).
- `services/query_api/dp_query.py` — the DP-side router; trusts the CP's
  verified `X-RB-Tenant-Id` header, re-validates the body, and runs the
  search via `execute_v1_query`.
- `services/query_api/v1_query.py` — the validate-quota-search core
  (`validate_query_body`, `run_query`, `execute_v1_query`, `_hot_search`),
  plus the in-memory shard cache and result-store consumer.

The CP→DP hop is internal to the deployment — externally `/v1/query` looks
like a single request against the Control Plane. The CP injects a verified
`X-RB-Tenant-Id` header so the DP can trust the tenant identity without
re-parsing the `Authorization` header.

## The id / metadata bridge — shard sidecars

FAISS `IndexIDMap2` cannot store strings. The index builder SHA1-hashes each
customer-supplied string `id` into a 63-bit int64 (`_id_to_int64`) before
`add_with_ids`. A FAISS search therefore returns **int64 hashes**, which are
*not* invertible back to the original id.

To bridge that gap, the index builder writes a **sidecar** next to every
shard:

- **Location:** `{shard_uri}.meta.json` (e.g.
  `s3://rosalinddb/indexes/<tenant>/<dataset>/indexes/<date>/shard-<ts>.bin.meta.json`)
- **Format:** a flat JSON object keyed by the *stringified* int64 hash:

  ```json
  {
    "8123456789012345678": { "id": "product-1", "metadata": { "title": "x" } },
    "1029384756102938475": { "id": "product-2", "metadata": {} }
  }
  ```

The query path (`v1_query._hot_search`) and the ephemeral runner
(`ephemeral_runner.handle`) load this sidecar via
`adapters.landing.parquet_reader.read_shard_sidecar` and translate every
FAISS hit back to `{id, score, metadata}`.

**Missing sidecar entry** (should never happen — defensive only): the hit
is returned with `id` set to the stringified int64 hash and `metadata` `{}`.
**Missing sidecar file:** `read_shard_sidecar` returns `{}` and all hits
degrade the same way. FAISS pad ids of `-1` (fewer results than `top_k`)
are skipped.

## `score` semantics

`score` is the **raw FAISS L2 (squared Euclidean) distance** between the
query vector and the matched vector. It is **not normalised** — **lower
means closer**, `0.0` is an exact match. We do not convert it to a cosine
similarity. If a future change normalises vectors at ingest time, L2
distance becomes monotonic with cosine similarity, but the field stays a
distance.

## `mode`

- `hot` — served from FAISS, and the shard was already faulted into this
  process's local cache.
- `cold` — served from FAISS, but the shard was loaded into the cache for
  the first time on this request.
- `recall` — **only with `RB_RECALL` on** (see "Recall-tier union" below): the
  dataset has **no consolidated shard yet**, but the recall tier had matching
  data, so the result was served **synchronously from recall**. There is **no**
  `job_id` — this is the read-your-writes case, not the async ephemeral one. The
  `mode` field always reflects the **consolidated-shard cache state**; `recall` means
  "the consolidated tier contributed nothing, recall answered."
- `ephemeral` — the dataset has no shard yet (not `indexed`) **and** (with
  `RB_RECALL` on) the recall tier had no matching data either. The query is
  enqueued on `RUN_EPHEMERAL_QUERY`; the immediate response carries
  `matches: []` plus a `job_id`. Poll `GET /v1/query/status/{job_id}` until
  `{"ready": true, ...}`.

When a consolidated shard **does** exist, `mode` is `hot` or `cold` exactly as today —
even when recall also contributed to (or overrode entries in) the result. The
recall contribution is invisible in the `mode` label by design; it is the same
top-K answer shape regardless of which tier each match came from.

## Recall-tier union (`RB_RECALL`)

> Default **off**. With `RB_RECALL` unset (or no `RB_RECALL_DSN`) `POST /v1/query`
> is **byte-identical** to the pure-consolidated path documented above — no recall
> connection is ever opened.

When the recall tier is on, `POST /v1/query` searches **both** tiers and merges
them (see [`recall-consolidate.md`](../architecture/recall-consolidate.md),
"Read path — the union"):

1. **Consolidated** — the existing FAISS search over the newest shard
   (via the SSD/RAM shard cache). Returns matches with **L2-squared** distances
   and the `hot`/`cold` cache `mode`. Unchanged.
2. **Recall** — a brute-force **exact** L2 scan over `recall_vectors` in the
   separate recall pgvector instance (`RB_RECALL_DSN`), scoped to
   `tenant_id = ? AND dataset = ? AND lsn > :watermark`, applying the **same**
   AND-of-equals metadata filter as the consolidated path.

**Metric alignment (correctness-critical).** The consolidated tier returns FAISS
**L2-squared** distances; pgvector's `<->` returns **plain** Euclidean L2. The
recall scan **squares** pgvector's distance (`power(embedding <-> q, 2)`) over
the **identical un-normalised** vectors so both tiers' `score`s are directly
comparable. Squaring is monotonic, so the union sorts correctly; without it the
ranking is silently wrong (it has a dedicated test).

**The watermark (`:watermark`).** It is the `consolidated_lsn` of the shard the
consolidated search **actually resolved** — never a value read independently (invariant
I3). The recall tier owns `lsn > consolidated_lsn`; the consolidated shard owns `<=`, so
the union is **complete and non-double-counting** (invariant I1). When no shard
exists yet, the watermark is `0` and **all** recall rows qualify.

**Merge — dedup recall-wins.** The two result sets are unioned and deduped by
`id`:

- A recall **live** row for an id **overrides** a consolidated match for that id (recall
  is newer — its LSN sits above the watermark).
- A recall **tombstone** (`deleted=true`) for an id **suppresses** the consolidated
  match for that id and contributes no match (a deleted vector never appears).
- The surviving matches are sorted **ascending by L2²** and truncated to
  `top_k`.

## `filter`

`filter` is **live and enforced**. A request with a `filter` object applies
an **AND-of-equals** predicate to each candidate record's metadata; only
records that match every key/value pair are returned. The full semantics
are documented in [`v1.md`](./v1.md) under "Query"; the implementation is
`services/query_api/v1_query.py:metadata_matches_filter` +
`_run_faiss_search`.

### Exact match, no coercion

`metadata_matches_filter` requires *identical type and value*:

- `string` compares to `string`, `number` to `number`. A type mismatch is
  never a match — `filter` value `"2024"` does not match metadata `2024`.
- A record missing a filtered key is excluded.
- A `null` filter value is accepted by request validation but never
  matches any record.
- A nested object or array as a filter value is rejected with
  `400 invalid_request` — v1 has no ranges, OR, or nesting.

### Search strategy: exhaustive when filtered

FAISS cannot filter by metadata, so filtering happens after the nearest-
neighbour search. Two strategies, picked at request time:

- **Unfiltered (the hot path)** — `top_k` results are fetched directly
  from FAISS with the configured `nprobe` (server default
  `RB_QUERY_NPROBE=64`, or the per-request `nprobe`). Cheap and the common
  case.
- **Filtered** — the search is run **exhaustively**: `fetch_k = ntotal`
  (every vector) and, on an IVF index, `nprobe = nlist` (every cell).
  Every candidate is then matched against the predicate and the survivors
  — already in ascending-distance order — are truncated to `top_k`. This
  is necessary for correctness: a filter-match in an unprobed IVF cell
  would otherwise be invisible no matter how large a fixed over-fetch is.
  IDSelector-based pre-filtering (so FAISS skips distance math on
  non-matching vectors) is the intended optimisation for large shards and
  is deliberately deferred — the exhaustive scan is correct and fast at
  current dataset sizes.

A per-request `nprobe` override is intentionally **ignored** for filtered
queries — they are exhaustive by construction.

### Partial results

A highly selective filter can legitimately return **fewer than `top_k`**
matches; that is an exact answer, not an approximation or an error. A
filter matching nothing returns `"matches": []` with HTTP 200. The same is
true when the dataset itself has fewer than `top_k` records.

## Error propagation: a successful 200 always means a real result

**A 200 from `POST /v1/query` (or `GET /v1/query/status/{job_id}` with
`ready: true`) always implies the `matches` array is the actual top-K
answer.** A search that *could not run* — shard cache filesystem broken,
object-store outage, FAISS load failure, or any other unrecoverable error
inside the ephemeral runner — surfaces as **HTTP 503** with a v1 error
envelope, never as 200 with an empty list.

| HTTP | code | meaning |
|---|---|---|
| 503 | `cache_unavailable` | local shard cache fs unreadable / unwritable (e.g. a bind-mount permission problem; check `CACHE_DIR`) |
| 503 | `storage_unavailable` | object-store fetch failed (S3 outage, missing shard, network partition) |
| 503 | `recall_unavailable` | the recall (pgvector) tier was unreachable for this query (connection drop / TLS reset / sustained recall-pool exhaustion); the query is safe to retry once recall recovers. Only reachable when the recall union is on (`RB_RECALL`). The query does **not** silently degrade to consolidated-only results — that would drop recent unconsolidated writes (read-your-writes) without signal |
| 503 | `ephemeral_error` | unclassified failure inside the ephemeral runner |
| 503 | `cache_capacity_exceeded` | SSD shard tier rejected the load (admission floor); only reachable when `RB_SHARD_TIER_BYTES` is set — raise the cap |

These codes apply to BOTH paths:

- **Hot path** — `_hot_search` in `services/query_api/v1_query.py:run_query`
  catches the exception, classifies it, and returns the envelope directly.
  It does **not** silently fall through to the ephemeral runner — the
  ephemeral fallback is reserved for its actual semantics (the dataset has
  no shard yet, so `list_shards` returned empty).
- **Ephemeral path** — the worker publishes the same envelope shape on the
  `RESULT_READY` queue (`{ok: false, error: {code, message}}`); the status
  poll surfaces it as 503. The queue message is still NACKed so the
  reliable-queue retry + DLQ semantics are preserved (`QUEUE_MAX_ATTEMPTS`
  deliveries, then dead-letter). The envelope is republished on every
  retry attempt so the caller is unblocked immediately on the first
  failure — they do not have to wait for the retry budget to drain.

The `safe_message` field carries the exception class name only; raw
exception text (which may include paths, bucket names, or signed-URL
parameters) is deliberately omitted so internal details never leak to the
customer. Inspect `code` for the failure shape and self-hosters should
check the worker logs for the full exception.

`{matches: []}` with HTTP 200 still has a legitimate meaning: the search
ran successfully and found no records that satisfy the filter (or the
dataset is empty). It is distinct from the 503 cases above.

## Legacy `/query`

The pre-v1 `POST /query` route is **kept** for back-compat with the
dashboard and existing tests. It now delegates ephemeral result storage to
the shared `v1_query` result store, so a single `RESULT_READY` consumer
serves both routes. New integrations should use `/v1/query`.
