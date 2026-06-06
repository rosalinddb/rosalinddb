# SSD cache: design and rationale

> **Status: default-off.** The SSD tier (`shard_tier.fetch`) is wired into the
> query path behind `RB_SHARD_TIER_BYTES`. The rendezvous-routing primitive is
> shipped but **not yet wired into the proxy** — the CP still routes via the
> static `resolve_dp_base_url`. Residency-override routing is **not yet
> implemented**. The catalog-listener, prewarm consumer, residency writer, and
> versioned-URI modules all exist and are individually flag-gated. Eviction is
> **LRU today**; W-TinyLFU is the long-term target, not the shipped policy.

Design doc for the SSD cache tier in `query_dp`. Pairs with
[`mmap.md`](mmap.md) (the load path this tier sits beneath) and
[`architecture.md`](architecture.md) (the CP/DP split). Every flag below
defaults to off / current-behaviour-preserving: a self-hoster who upgrades and
changes nothing sees identical behaviour to today.

## The problem

The mmap bench showed `RB_FAISS_MMAP=true` decouples on-disk shard size from RSS, but it
surfaced the next ceiling: a 6 GB shard's *first* GET from MinIO over the local
Docker network takes ~100 s, while `RB_QUERY_DP_TIMEOUT_S=5` rejects it ~95 s
earlier. Per-URI single-flight stops the stampede; it does not change the
latency of the one download that runs.

The DP cache before this design was a single in-process tier: an `OrderedDict`
LRU over deserialised FAISS indexes plus parsed sidecars, byte-budgeted against
`RB_SHARD_CACHE_BYTES` (default 1 GiB). The local `CACHE_DIR` on disk held the
downloaded file but was unbudgeted, unmonitored, and never invalidated; catalog
updates mutated `shard_catalog` but never notified the DPs.

This design treats the local SSD as a first-class tier — versioned,
admission-controlled, evictable, and (eventually) observable from the CP —
while keeping the object store as the durable source of truth and the mmap'd
RAM working set on top.

## Three tiers

See [`diagrams/ssd-cache-tiers.puml`](diagrams/ssd-cache-tiers.puml).
Durability and latency split cleanly:

| Tier | Holds | Eviction | Survives DP restart? |
|---|---|---|---|
| Object store (S3-compatible) | Every shard ever published, immutable versioned keys | Periodic GC of orphans (24h) | Yes |
| SSD (`CACHE_DIR`) | Recently-fetched shards | LRU under byte budget (W-TinyLFU planned) | Yes (files persist) |
| RAM (`_SHARD_CACHE` + page cache) | Deserialised FAISS handle + sidecar + touched mmap pages | LRU on the Python object; kernel page-cache LRU on mmap pages | No |

## Flag reference

| Flag | Default | Effect when on |
|---|---|---|
| `RB_SHARD_VERSIONED_URIS` | `false` | Builder writes content-addressed keys instead of date-partitioned ones |
| `RB_SHARD_TIER_BYTES` | unset (tier inactive) | Activates the SSD tier with that byte budget (default sizing 2.5 GiB) |
| `RB_SHARD_TIER_DIR` | `${CACHE_DIR}/tier-managed/` | Where the SSD tier stores its managed files |
| `RB_SHARD_TIER_MIN_RESIDENT_S` | `30` | Admission floor: a `prewarm()` cannot evict an entry younger than this; `fetch()` ignores it |
| `RB_CATALOG_LISTEN` | `false` | DP subscribes to Postgres `LISTEN catalog_updates` |
| `RB_CATALOG_FRESHNESS_S` | `5` | Per-dataset `list_shards` cache TTL (pull fallback); active only when `RB_SHARD_TIER_BYTES` is also set; `0` disables it |
| `RB_PREWARM_ON_BUILD` | `false` | Builder publishes `PREWARM_SHARD` on catalog commit |
| `RB_PREWARM_CONSUMER` | `false` | DP runs the `PREWARM_SHARD` consumer that calls `shard_tier.prewarm` |
| `RB_ADMIN_ENDPOINTS` | `false` | DP mounts `POST /admin/prewarm` and the rest of the admin surface |
| `RB_ROUTING_RENDEZVOUS` | `false` | CP picks DPs via rendezvous hash (primitive shipped, **not yet wired into the proxy**) |
| `RB_DP_ID` | unset | Explicit DP identity; else `HOSTNAME`, else a UUID persisted to `${CACHE_DIR}/.dp_id` |

## Design decisions

### 1. Object store + SSD tier (not SSD-only)

Keep both. The object store is the only tier that survives a DP losing its disk
and the only tier every DP can read from — the same compute/storage split
Snowflake uses for per-warehouse SSD caches and Pinecone's serverless uses for
immutable slabs. SSD-only with replication was rejected: replicating every
shard to N≥2 DPs would let us drop the object store, but a new tenant's shard
would then need placement on N DPs at publish time, and there is no admission
control for "DP-A is full, route elsewhere." The object store side-steps it —
*every* DP can fetch *any* shard, so admission is a per-DP cache decision, not a
per-shard placement decision. No code change here; the existing `s3://` scheme
is already the contract.

### 2. Shard URIs are versioned and immutable (`RB_SHARD_VERSIONED_URIS`)

The default key —
`{INDEXES_PREFIX}/{tenant}/{dataset}/indexes/{YYYY-MM-DD}/shard-{epoch_ms}-{uuid8}.bin`
— *looks* immutable but is not enforced, and nothing in the string identifies
the bytes underneath it. The versioned shape makes the version visible in the
key itself: `s3://{bucket}/{tenant}/{dataset}/{shard_id}-{content_hash}.bin`,
where `content_hash` is the first 16 hex chars of `sha256(serialised_index)`
(64 bits — comfortably above birthday-collision risk for any shard population
this system will hold). Two builds of the same bytes converge on the same key
(cheap dedup); two builds of different bytes can never collide. The layout is
deliberately flatter than the dated legacy shape — a date partition would
defeat dedup across midnight, and a single flat prefix per `{tenant, dataset}`
is what listing tools and the orphan GC want. Bucket-level S3 versioning was
declined: every read would need a `versionId` to be deterministic, and the
catalog row already names the version we want.

`adapters/storage/shard_uri.py` is a pure module (no I/O, no env reads) shared
by the builder, the SSD tier, and the orphan GC: `build()` hashes the bytes (a
verifiable receipt for what was PUT), `parse()` returns a
`NamedTuple(bucket, tenant, dataset, shard_id, content_hash)` and raises on a
legacy URI, and `is_legacy()` is a non-raising classifier the tier uses to
decide migrate-in-place vs accept-as-is. The builder
(`services/index_builder/run.py:_compute_shard_uri`) selects the shape per the
flag; with it off the URI is bit-identical to before — that is the rollback
contract. Catalog `add_shard` is the atomic publish; everything before it is
retriable.

**Operator note.** With the flag on, deployments whose `INDEXES_PREFIX` carries
a non-bucket path segment (e.g. `s3://rosalinddb/indexes`) write flag-on shards
*outside* that segment, directly under the bucket root — the leading path is
dropped so the orphan GC can list one flat prefix per `{tenant, dataset}`. If a
bucket policy or lifecycle rule keys off `indexes/`, update it before flipping
the flag.

### 3. Eviction: LRU today, W-TinyLFU as the target

The SSD tier (`adapters/storage/shard_tier.py`) ships **LRU under a byte
budget** — correct, two lines of `OrderedDict` bookkeeping, and the right first
step. W-TinyLFU (Caffeine's default) is the long-term target: plain LRU
misbehaves under a scan, where a backfill sweep touching every shard once
evicts the frequency-leader and the next real query goes cold; a small
admission filter scored against a count-min frequency sketch fixes that and
reports hit-rates within 99% of Belady's optimal on real traces. It is not yet
implemented — the SSD budget holds only 4–10 shards in the reference
deployment, so the LRU's exposure to the scan pathology is bounded until the
working set grows. Pure LFU was rejected (a one-time frequency burst pins
forever); the RAM tier (`_SHARD_CACHE`, ≤1 GiB) stays plain LRU regardless —
its working set is too small for a frequency sketch to earn its ~1 MB overhead.
Per-tenant pinning to bound a noisy neighbour is an operator-policy concern and is
[out of scope](#out-of-scope).

Eviction `os.unlink`s the SSD file *and* notifies the in-process RAM cache to
`evict_shard(shard_id)`. POSIX guarantees an open mmap on the unlinked file
stays valid until the last fd closes (see the eviction-safety guarantee below).

### 4. Admission: fetch is unconditional, prewarm honours the floor

N prewarms arriving with capacity N−1 has three plausible answers — queue,
reject, evict-the-coldest — and the right one depends on *why* the caller
arrived. A `fetch()` is a real query that already missed RAM and the SSD tier:
there is a client on the far end of that timeout, so refusing to admit turns a
slow query into a wrong one. A `prewarm()` is speculative; if it cannot land,
the worst case is the first query becomes a normal `fetch()`. So the two paths
get two contracts:

| Path | Admission | Cost of being wrong |
|---|---|---|
| `fetch(uri)` | Unconditional — evicts the LRU end until the shard fits, regardless of age | Thrashes a hot resident only when capacity is genuinely too small (telemetry signals it) |
| `prewarm(uri)` | Honours `RB_SHARD_TIER_MIN_RESIDENT_S` (default 30 s): cannot evict an entry younger than the floor; raises `CacheCapacityExceeded` if every candidate is too young | A rejected prewarm — the first query for that shard fetches cold, same as today |

The age floor stops a thrash: a prewarm landing during a write storm cannot
evict a shard that just arrived and has not served a query yet. `fetch()` is
deliberately *not* floored — a fetch reaching the tier has already missed every
faster layer, so an admission rejection there would be a user-visible error for
a query the system can serve. The asymmetry lets prewarm be a free option (it
never makes the cache worse) while keeping query latency under the operator's
direct control via `RB_SHARD_TIER_BYTES`. A queue was rejected: prewarm loses
its value if it cannot land in seconds.

`shard_tier.fetch()` admits unconditionally; `shard_tier.prewarm()` shares the
single-flight and download path but gates admission via
`_check_admission_capacity_locked`, which walks the LRU end accumulating bytes
it *would* evict and raises the moment it hits an entry whose `last_admit_at`
(a `time.monotonic()` stamp, immune to wall-clock steps) is under the floor, or
if the whole table still cannot reclaim enough. The check runs *after* the
download lands (`nbytes` comes from the payload, not a HEAD), and a rejected
prewarm's file is unlinked in `finally`. A pre-flight size probe is the obvious
upgrade once the storage adapter exposes one; today the path trades one wasted
GET per rejection for fewer round-trips.

### 5. Readiness is a hint, upgradable to a lease

A DP that reports "shard X is warm" can evict X before the next query lands, so
readiness is advisory only. The hint is a row in
`dp_shard_residency(dp_id, shard_id, warm_since, last_query_at)`
(migration `007`, written by `services/_common/residency_writer.py`): the DP
writes on cache put and refreshes `last_query_at` on hit; a CP reader would
apply a 60 s freshness filter, and a CP-side prune reaps stale rows. The hint
never *prevents* eviction — a hint pointing at an evicted shard just means the
next query is a normal miss that refetches. A synchronous per-query lease was
declined for v1: it costs a Postgres round-trip on every read and is brittle
under DP crash, where the hint contract recovers automatically. The lease is
the documented upgrade path, gated on a measured need (residency hint racing
eviction in production).

### 6. Invalidation: LISTEN/NOTIFY push + TTL pull fallback

Two channels, because Postgres `LISTEN/NOTIFY` is the cheapest push available
(one statement in the `add_shard` transaction, delivered on commit) but is
best-effort: a service down when the event fires misses it forever, and a
LISTEN session can drop silently on a blip or restart. The standard mitigation
is a TTL fallback — every DP re-reads `list_shards` for a hot dataset whenever
its last lookup is older than `RB_CATALOG_FRESHNESS_S` (default 5 s), which
bounds worst-case staleness even if NOTIFY is silent for a day. Push is the
latency optimisation; pull is the correctness mechanism. The catalog row is the
source of truth; a failed `pg_notify` is caught and logged, and the TTL covers
it.

**Mid-query consistency.** A query mid-search on v1 when v2's catalog update
lands holds an open fd on the mmap'd v1 file. Eviction `unlink`s it, but POSIX
keeps the inode valid until the last fd closes, so the search completes against
v1's bytes; the next query resolves v2 and opens it fresh. *Within* a query the
answer is consistent; *between* queries it can change — the same guarantee
FAISS itself relies on (see [`mmap.md`](mmap.md)). Stale-while-revalidate was
declined: returning v1 while refetching v2 is a silent wrong answer when v2 may
contain the vector being searched for.

`add_shard` runs `pg_notify('catalog_updates', '{tenant, dataset, shard_uri}')`
in the same transaction as the catalog `INSERT` (payload kept minimal —
`pg_notify` caps at 8000 bytes; `shard_uri` rides along only for operator
diagnostics). On the DP side, the cache wrapper around `list_shards` is keyed
by `(tenant, dataset)` and active only when **both** `RB_SHARD_TIER_BYTES` is
set **and** `RB_CATALOG_FRESHNESS_S > 0` — otherwise every call passes straight
through. With `RB_CATALOG_LISTEN=true` the DP runs
`services/_common/catalog_listener.py`: a singleton daemon thread on a
dedicated `AUTOCOMMIT` connection (the LISTEN session cannot share the
request-time pool), reconnecting on `psycopg2.OperationalError` with bounded
backoff (0.5 s → 30 s cap) and logging-and-skipping malformed payloads or
raising callbacks so neither can take down the listener. On a notify it evicts
the affected `(tenant, dataset)` entry. With `RB_CATALOG_LISTEN` unset, the TTL
is the only invalidation channel.

### 7. CP↔DP routing: rendezvous hashing — primitive shipped, NOT yet wired

Today's CP routing (`services/query_api/query_proxy.py:resolve_dp_base_url`) is
a static env-based map — `QUERY_DP_URL` for the shared pool,
`QUERY_DP_URL_<TENANT>` for per-tenant overrides — and **every pool resolves to
exactly one DP URL**. Rendezvous hashing only earns its keep when ≥2 DPs serve
the same pool, so it is purely additive: single-URL deployments (everything
today) see no change regardless of the flag.

Rendezvous (Highest Random Weight) was chosen over consistent hashing: both
give "same key → same DP, mostly," but HRW needs no virtual-node ring —
`pick_dp = argmax_dp hash(routing_key, dp_url)` is the whole algorithm, and for
our DP cardinality (≤16) the per-query hash cost is negligible. Maglev
(table-based, optimised for per-packet routing) and Ringpop (SWIM membership +
consistent hashing) are both overkill for once-per-request routing across a
handful of DPs. The routing key is `(tenant, dataset)`, **not** `shard_uri`, so
a new shard for the same dataset does not reshuffle which DP serves it and the
DP's local cache stays warm across rebuilds. An unhealthy elected DP falls
through HRW rank order; when *all* DPs are unhealthy the router falls back to
the first configured URL (operator-predictable). A multi-URL config with the
flag unset is treated as a misconfiguration: one process-local WARNING naming
`RB_ROUTING_RENDEZVOUS`, then fall back to the first URL so the deployment
still serves.

**Wiring status.** `pick_dp_url(pool, routing_key)` and its pure helpers
(`_hrw_pick`, `_hrw_rank`, `_parse_dp_urls`, `_resolve_pool_urls`,
`_is_dp_healthy`) are shipped and tested, but **`_proxy` does not yet call
them** — it forwards request bytes verbatim and would need a `dataset` argument
or an `X-RB-Dataset` header to derive the routing key. `_is_dp_healthy`
currently defaults to `True` (no active poller). `resolve_dp_base_url` is
unchanged, so the rollback contract holds. The residency-hint override (CP
prefers the warm DP over the HRW-elected one) is **not yet implemented** — it
wants a careful request-path cache for the residency lookup, and the
`dp_shard_residency` table is where residency-aware routing will read once it is wired.

### 8. Prewarm trigger: on-build-completion AND on-first-reference

Pure on-build prewarm wastes capacity on shards that may never be queried; pure
on-first-reference makes every first query cold. The hybrid: the builder
publishes a `PREWARM_SHARD` message on completion, and a query that misses
triggers a normal fetch (covering a dropped, dead-lettered, or
wrong-DP-consumed prewarm). In today's single-DP-per-pool topology the queue
topic has one consumer per pool, so each `PREWARM_SHARD` is dispatched to a
single DP — fan-out is bounded by consumer count, not DP count. This is safe
under contention because the admission floor makes a prewarm a free option: a
prewarm that lands while the tier is full of just-arrived shards is rejected,
not allowed to thrash, and the worst case is a normal cold fetch. Predictive
(ML-based) prewarm was declined; reactive coverage is high at this scale.

With `RB_PREWARM_ON_BUILD=true`, `services/index_builder/run.py` publishes
`PREWARM_SHARD {tenant, dataset, shard_uri}` *after* `add_shard` returns, wrapped
best-effort so a queue blip cannot reference an un-cataloged shard and a builder
crash between commit and publish just leaves the cold-fetch path as the safety
net. With `RB_PREWARM_CONSUMER=true`, the DP runs
`services/_common/prewarm_consumer.py` — a daemon thread that pulls the messages
and calls `shard_tier.prewarm(shard_uri)`, `nack`-ing on `CacheCapacityExceeded`
(bounded retry, then DLQ) or on a malformed message, isolated so one bad message
cannot kill the worker. With `RB_ADMIN_ENDPOINTS=true`, `POST /admin/prewarm`
exposes the same `shard_tier.prewarm` call for manual and smoke-test use: 200 +
`{shard_uri, local_path}` on success, 503 `cache_capacity_exceeded` when the
floor blocks, 404 `shard_not_found`, 400 `invalid_request`. All three flags
default off, so an untouched upgrade sees no new traffic.

### 9. Local-tier capacity sizing

`RB_SHARD_TIER_BYTES` default is 2.5 GiB — roughly 2.5× `RB_SHARD_CACHE_BYTES`,
and ~10% of the reference deployment's ~25 GB hot set (4 shards × ~6 GB +
sidecars), matching the classic working-set cache-sizing heuristic. It holds
~4 reference shards with sidecars. The default is intentionally conservative;
size production from `rosalinddb.shard.tier_miss{tier=ssd}` rates. The headroom
for one inbound prewarm is one shard's worth, and the
`RB_SHARD_TIER_MIN_RESIDENT_S` floor keeps a capacity-tight tier from
admission-thrash. `RB_SHARD_TIER_DIR` defaults to `${CACHE_DIR}/tier-managed/`
to keep managed files distinct from legacy `_ensure_cached` files. Per-tenant
SSD quotas are [out of scope](#out-of-scope) — the budget is shared.

## Edge cases proven handled

| # | Case | Mechanism |
|---|---|---|
| 1 | Builder writes v2, updates catalog, crashes before deleting v1. v1 leaks. | Versioned URIs + periodic GC sweeping S3 keys with no catalog row older than `RB_SHARD_ORPHAN_MAX_AGE_H` (default 24h). |
| 2 | Builder writes v2 but the catalog update fails. v2 leaks. | Same GC — a key with no catalog row ages out regardless of cause. |
| 3 | DP A and DP B start downloading v1 simultaneously. | Within a DP: per-URI `threading.Event` in `shard_tier.fetch`. Across DPs: each populates its own SSD; the object store handles parallel GETs. |
| 4 | DP starts downloading v1; catalog moves to v2 mid-download. | Download completes against v1's immutable content-addressed URL; the post-NOTIFY re-read sees v2; v1 ages out normally. |
| 5 | DP has v1 cached; catalog moves to v2; query arrives. v1 or v2? | The query resolves the newest shard via `list_shards`; v2's `shard_id` is new, so the cache returns `None` and fetches. v1 stays until evicted but is never reachable. |
| 6 | Query mid-search on v1 when v2 lands. | POSIX unlink-with-open-fd. The mmap stays valid for the fd's lifetime; the query completes on v1. |
| 7 | Prewarm reports "v1 ready"; query arrives 5 min later; v1 was evicted. | Readiness is a hint. Normal miss → refetch; ~100 ms on a warm SSD-tier `fetch` (which checks the filesystem first). |
| 8 | DP crashes mid-download (partial tmp file). | `fetch` writes `{path}.{pid}.{uuid8}.tmp` and `os.replace`-renames on success; a crashed write leaves a tmp file that a startup sweep unlinks (any `*.tmp` >1h). Readers never see partial bytes. |
| 9 | Two tenants prewarm shards that together exceed capacity by 10%. | Admission control. One succeeds; the other evicts an old-enough LRU entry or is rejected with `cache_capacity_exceeded`. No thrash. |
| 10 | DP restart — does it rebuild its resident set? | Yes. SSD files persist; on startup `shard_tier.recover()` walks `RB_SHARD_TIER_DIR` and re-indexes present files. RAM cache is cold; first query per shard pays the `read_index` cost, not the GET cost. |
| 11 | Network partition: DP can't reach the object store mid-download. | `fetch` raises; `_classify_hot_path_error` (`services/query_api/v1_query.py`) maps it to `storage_unavailable` → HTTP 503; client retries. The single-flight registry clears in `finally`. |
| 12 | Catalog update succeeded but the NOTIFY was lost. | TTL pull fallback: a DP that missed the NOTIFY discovers v2 within `RB_CATALOG_FRESHNESS_S` (5 s). |
| 13 | Clock skew makes DPs disagree on "newest". | "Newest" is `shard_catalog.created_at DESC` from one Postgres source of truth; no DP-local clock is consulted. |
| 14 | Tenant deletes a dataset while a DP is mid-query. | `delete_shards` runs in a transaction (FK CASCADE on `dataset_catalog`). The in-flight query holds an open fd (POSIX, as case 6) and completes; the next query returns `dataset_not_found`. The SSD file is unlinked by the next eviction or the orphan GC. |

## Telemetry

Metric names this design defines (emission lags the wiring in several cases):

| Metric | Reads as |
|---|---|
| `rosalinddb.shard.tier_miss{tier=ssd\|ram}` | High `ssd` → SSD budget too small; high `ram` with low `ssd` → mmap working |
| `rosalinddb.shard.write{outcome=new\|duplicate}` | High duplicate → bit-identical rebuilds (dedup opportunity) |
| `rosalinddb.shard.orphan_count` | Non-zero → builders leaking S3 keys |
| `rosalinddb.shard.cache_hit_rate{tier=ssd}` | Eviction effectiveness vs the hit-rate curves we bench against |
| `rosalinddb.shard.admission{outcome=admitted\|evicted_old\|rejected_young}` | Sustained `rejected_young` → raise `RB_SHARD_TIER_BYTES`; any `rejected_young` on `fetch` would be a bug |
| `rosalinddb.routing.residency_match` | Below 80% → the residency-lease upgrade becomes worth its cost |
| `rosalinddb.catalog.invalidation{source=notify\|ttl}` | Healthy: NOTIFY > 95%, TTL > 0 (the safety net firing occasionally) |
| `rosalinddb.routing.outcome{path=static\|hrw\|health_fallback\|all_unhealthy}` | Sustained `all_unhealthy` → page the operator (not yet emitted; needs the routing wiring) |
| `rosalinddb.shard.first_query_cold_rate` | Target < 5% under steady builder→query traffic |
| `rosalinddb.shard.tier_bytes_used / RB_SHARD_TIER_BYTES` | Sustained 100% + non-zero miss → upsize |

## Out of scope

Sensible follow-ups, not on the critical path:

- **Replication (shard on N≥2 DPs).** v1 is single-DP residency; HA is a later layer.
- **Distributed consensus for cache state.** The residency registry is plain Postgres rows; the hint contract makes coordination unnecessary.
- **ML-based prewarm prediction.** Reactive prewarm ships; predictive is a follow-up only if cold-fetch tails persist.
- **Per-tenant SLA tiers.** All tenants share the SSD budget; pinning and reserved capacity are out of scope for the engine.
- **Cross-region SSD replication.** The SSD tier is per-DP.

## References

- [TinyLFU: A Highly Efficient Cache Admission Policy](https://arxiv.org/abs/1512.00727) — the W-TinyLFU eviction target.
- [Caffeine — Efficiency wiki](https://github.com/ben-manes/caffeine/wiki/Efficiency) — W-TinyLFU comparison curves.
- [Pinecone slab architecture](https://www.pinecone.io/learn/slab-architecture/) — immutable slabs, prewarm-to-executor.
- [Snowflake architecture overview](https://opsiocloud.com/blogs/snowflake-data-warehouse-architecture-layers/) — compute/storage split, per-warehouse SSD cache.
- [Cloudflare — How Workers KV works](https://developers.cloudflare.com/kv/concepts/how-kv-works/) — tiered cache invalidation model.
- [Rendezvous hashing — Wikipedia](https://en.wikipedia.org/wiki/Rendezvous_hashing) — the HRW algorithm.
- [Postgres LISTEN/NOTIFY reliability](https://tapoueh.org/blog/2018/07/postgresql-listen-notify/) — best-effort delivery caveats.
- [Amazon S3 Versioning](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html) — the bucket-versioning model we declined.
- [POSIX `rename(2)` semantics](https://pubs.opengroup.org/onlinepubs/9699919799/functions/rename.html) — atomic temp+rename publish.
- [`docs/architecture/mmap.md`](mmap.md) — the load path beneath this tier; cgroup page-cache accounting and FUSE caveats apply here unchanged.
