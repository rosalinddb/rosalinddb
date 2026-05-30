# Indexing — incremental append

How the `index_builder` service turns landing data into queryable FAISS shards,
and how it avoids re-indexing data it has already indexed.

## The pipeline

1. A customer uploads vectors. `validator_worker` validates them and writes
   each upload into its own landing sub-prefix:
   `…/landing/<tenant>/<dataset>/upload-<id>/part-0001.parquet`.
2. The validator publishes a `DATASET_READY` message.
3. `index_builder` consumes `DATASET_READY` and builds/updates the dataset's
   FAISS shard, then records it in `shard_catalog`.
4. `query_api` loads the **newest** shard for the dataset and searches it.

## Incremental append — what changed

Before incremental indexing, every `DATASET_READY` re-read **all** landing parquet under the
dataset prefix and rebuilt the whole FAISS shard from scratch ("rebuild
amplification"): a 1M-vector dataset that received 100 new vectors re-read and
re-indexed all 1,000,100. Cost scaled with total dataset size on every ingest.

Now the builder is incremental:

- **First ingest** for a dataset (no shard exists): train an IVFFlat index
  (or a flat index for tiny datasets — see *Index-type gate* below), `add()`
  all vectors, write the shard. Unchanged from before.
- **Subsequent ingest** (a shard already exists): load the current shard's
  FAISS index and its `{shard_uri}.meta.json` sidecar, read **only** the
  landing parts not already indexed, `index.add()` the new vectors onto the
  already-trained index, merge the new id/metadata into the sidecar, and write
  an updated shard. Previously-indexed uploads are never re-read.

A FAISS IVF index, once **trained**, supports `add()` of new vectors without
retraining — the quantizer/centroids are fixed. The incremental path relies on
exactly that: it deserialises the trained index and only calls `add_with_ids`.

## Index type — IVFFlat (the recall touch-up)

RosalindDB builds a FAISS **IVFFlat** index for any non-tiny dataset.

IVFFlat is an IVF index — an `IndexFlatL2` coarse quantizer partitions the
space into `nlist` cells — that stores **raw, uncompressed float32 vectors**
in each cell. It is *not* the same as an exact `IndexFlatL2` ("flat" below):
IVFFlat still does the IVF cell-pruning that makes search sub-linear, it just
does not compress the vectors.

**Why not IVF+PQ.** RosalindDB originally built **IVF+PQ** — IVF plus Product
Quantization, which compresses each vector to a handful of bytes (~8x). PQ is
lossy *by construction*: search ranks candidates on PQ-approximate distances,
not exact ones, so recall@10 ceilinged at **~0.65** on the SIFT benchmark even
at `nprobe = nlist` (exhaustive IVF). IVFFlat ranks on **exact L2 distances**
and reaches **~0.95** on the same data. The cost is shard size — raw float32
vectors are ~8x larger than 8-bit PQ codes — which is the right tradeoff for
an object-storage-first product: object storage is cheap, and RosalindDB's
cost pitch is about *not paying for idle compute*, not about squeezing index
bytes.

### Sizing `nlist`

`nlist` (the coarse-quantizer cell count) is sized per dataset by the FAISS
rule of thumb `nlist ≈ 4*sqrt(N)`, clamped so k-means can always train
(`nlist <= N//8`) and capped by the optional `IVF_NLIST` env ceiling. This was
informed by **OpenData Vector**'s flat-IVF sizing, which targets roughly
**~100 vectors per cluster** (its bench config and RFC-0005 — "it is optimal to
maintain one centroid for ~100 vectors"). For SIFT-scale data `4*sqrt(N)`
lands close to that target (100k vectors → ~1264 cells → ~80 per cell).

### Index-type gate — tiny → flat, otherwise → IVFFlat

A *first* ingest picks one of two index types based on the batch size:

- **IVFFlat** (`ivfflat`) — chosen when the batch clears IVF's single training
  floor: `>= 64` rows and `nlist >= 4`. IVFFlat has just one training step —
  k-means on the `nlist` centroids.
- **flat** (`IndexFlatL2`, exact) — the fallback for a *tiny* first ingest
  below that floor, where IVF cell partitioning buys nothing.

The gate is deliberately simple: there is **no second, larger PQ-codebook
training floor**. IVF+PQ needed `>= 2^PQ_NBITS` rows (256 for an 8-bit
codebook) to train its PQ codebook *in addition* to IVF's `>= 64`; IVFFlat
stores raw vectors and trains no codebook, so that floor — and the `PQ_NBITS`
/ `PQ_M` knobs — are gone.

This gate applies only to the *first* ingest. A subsequent ingest always
`add()`s onto whatever index the first build produced — IVFFlat stays IVFFlat,
flat stays flat.

### No migration of old shards

There is no production data, so old `ivfpq` shards are simply **superseded**
by the next IVFFlat build (the superseded-shard sweep then reclaims them — see
*Storage reclamation* below). No migration step is needed. The incremental
`add()` path still loads a legacy `ivfpq` shard correctly if one is the newest
shard for a dataset, since `add_with_ids` works on any trained IVF index.

## Querying — IVF `nprobe`

An IVF index partitions the vector space into `nlist` cells; a query searches
only `nprobe` of them. FAISS defaults `nprobe` to **1** — a query then inspects
a single cell out of (typically) thousands and misses any true neighbour that
landed in an adjacent cell. The query path now sets `nprobe` explicitly for
every search, delivered as a **per-search `faiss.SearchParametersIVF`** object
passed to `index.search(...)` — the shared cached index is *never* mutated, so
concurrent queries cannot race on each other's `nprobe`:

- **Server default** — `RB_QUERY_NPROBE` (default **64**), read by `query_api`.
- **Per-query override** — the `POST /v1/query` body may carry an optional
  positive-integer `nprobe`, which overrides the server default for that query
  only.
- **Upper bound** — the effective `nprobe` is clamped to `MAX_NPROBE` (**1024**);
  a per-query override above it is rejected with a `400 nprobe_out_of_range`
  (mirrors the `top_k` ceiling). This prevents an unbounded override from
  turning a cheap ANN search into a full scan.

`nprobe` is a **query-time** parameter: changing it tunes the recall/latency
tradeoff with **no index rebuild**. It is an IVF knob and applies unchanged to
**IVFFlat** — IVFFlat is an IVF index, so `SearchParametersIVF(nprobe=…)` works
exactly as it did for IVF+PQ. The default of 64 was chosen from an `nprobe`
sweep against the SIFT 1M benchmark — recall climbs steeply with `nprobe`
then flattens, and query latency is effectively flat across the whole sweep
(the FAISS IVF search is sub-millisecond), so a generous default costs
nothing measurable. On a flat (non-IVF) index `nprobe` has no meaning and
the setting is a harmless no-op.

> **Recall is no longer index-ceilinged.** Under IVF+PQ, `nprobe` removed the
> dominant recall loss but PQ's lossy ~8x compression still ceilinged recall@10
> at ~0.65 even at `nprobe = nlist`. **IVFFlat** stores raw vectors and ranks
> on exact L2 distances, so there is no PQ ceiling: with a sensible `nprobe`,
> recall@10 reaches ~0.99 on the SIFT 1M benchmark.

## In-memory shard cache (query path)

The query hot path used to `faiss.read_index()` (deserialise a ~13 MB index)
and parse the multi-MB JSON sidecar on **every** query — the `mode: "hot"`
label was cosmetic, nothing was actually held in memory.

`query_api` now keeps a real in-memory cache, keyed by **shard id**, holding
the *deserialised* FAISS index and the *parsed* sidecar:

- The **first** query for a shard does a genuine cold load (`mode: "cold"`);
  every subsequent query reuses the in-memory objects (`mode: "hot"`) and
  collapses to just the FAISS search.
- The cache is **byte-budgeted LRU**. An entry's footprint varies ~100x across
  datasets (a 1k-vector shard vs a 1M-vector shard's ~430 MB index), so a
  count cap cannot bound memory. Each entry's approximate footprint —
  `faiss.serialize_index(index).nbytes` plus the serialised sidecar size — is
  measured once at insert time and a running total is kept; on insert the LRU
  end is evicted until the total fits `RB_SHARD_CACHE_BYTES` (default
  **512 MB**). A single entry larger than the whole budget is admitted then
  immediately evicted — usable for that one query, never retained.
  `RB_SHARD_CACHE_SIZE` remains an optional secondary count cap (0 = disabled).
- It is **evicted in step with the superseded-shard sweep**: when the builder
  sweeps a superseded shard it calls `evict_shard()` so the stale (now-deleted)
  index is dropped and can never be served. Because every query resolves the
  *newest* shard, a freshly-built shard is a natural cache miss → cold load.
- A cache lookup emits the `rosalinddb.shard_cache` counter (`result =
  hit | miss`).

## Tracking what has been indexed

Each shard row in `shard_catalog` carries an `indexed_landing_uris` manifest:
the list of landing parquet part URIs already folded into that shard
(migration `003_shard_incremental_indexing.sql`). The **newest** shard's
manifest is the authoritative record of "what has been indexed".

On each build the builder:

1. lists every landing part under the dataset prefix;
2. subtracts the newest shard's `indexed_landing_uris`;
3. indexes only the remainder.

This also makes a **sequential duplicate `DATASET_READY`** for an already-indexed
batch a clean no-op (no new parts → nothing to do → no new shard, no
double-counting). Concurrent duplicates are a separate matter — see
*Operational constraints* below.

`build_type` on the shard row is `full` (first ingest) or `incremental`.

## Edge cases

- **Empty new batch** — no new landing parts, or new parts with no rows: the
  build is a clean no-op; the dataset status is left untouched.
- **Dimension mismatch** in a later batch — if a new batch's vectors do not
  match the existing index dimension, the build fails and the dataset status
  is set to `error` with an explanatory `error_message`. The bad batch never
  produces a shard.
- **Duplicate `DATASET_READY`** — see above; the manifest is authoritative, so
  a batch is never indexed twice.

## Retraining policy (v1 limitation)

For v1 the builder **always** does an incremental `add()` once a shard exists.
It does **not** auto-retrain on vector-distribution drift.

If a dataset's vector distribution shifts far from the sample the IVF
quantizer was originally trained on, IVF recall can degrade (new vectors fall
into poorly-fitted cells). The v1 remedy is operational: delete and re-create
the dataset to force a fresh full build.

A future loop can add a drift threshold (e.g. retrain when added vectors
exceed N× the original training set, or when measured recall drops) — kept out
of v1 deliberately, in favour of a simple, predictable, documented behaviour.

## Operational constraints & known gaps

- **Single-replica builder.** The build sequence — read the newest shard's
  manifest, compute the new parts, write a new shard — is **not atomic**. It is
  safe only because `index_builder` processes `DATASET_READY` messages serially
  in a single process. Running more than one builder replica for the same
  dataset would let two concurrent builds read the same manifest and index
  overlapping parts (double-counted vectors). The builder MUST stay
  single-replica until a future loop adds shard-level locking.

- **Sidecar is rewritten in full each build.** The FAISS index *append* is
  incremental, but the `{shard_uri}.meta.json` sidecar is read, merged, and
  re-written whole on every incremental build. "No rebuild amplification"
  therefore applies to the index, not the sidecar — sidecar write cost still
  scales with total dataset size. Acceptable at MVP scale; a future loop can
  make the sidecar append-structured.

## Storage reclamation (sweepers)

Each successful build ends by reclaiming storage the dataset no longer needs.
Both sweepers are **best-effort**: a storage error on one object is logged and
the sweep moves on; the build itself has already succeeded and is never failed
by a sweep error. The next build retries.

### Superseded-shard sweep

Each build writes a *new* FAISS shard and inserts a new `shard_catalog` row;
the previous shard is now superseded. The sweeper deletes superseded shards —
the `.bin`, its `.meta.json` sidecar, and the catalog row.

It retains the newest **two** shards (configurable via `SHARD_KEEP`, minimum
2): the newest is the one queries load, and the one before it is a **grace
buffer**. This is what makes the sweep race-safe without locking or
timestamps:

- Queries only ever resolve `get_latest_shard`. A query that resolved shard
  *N* as the latest, then a build writes shard *N+1* and sweeps — the query is
  still faulting *N*'s `.bin`/`.meta.json` into its local cache.
- Keeping *N+1* **and** *N* guarantees that in-flight query's objects are still
  present. Only shards `≤ N-1` are deleted, and no query that started *after*
  *N-1* was superseded could ever have picked it as the latest shard.

**Safety assumption — the grace buffer is count-based, not time-based.** The
"keep newest 2" retention gives an in-flight query exactly *one* extra build
of slack. It is therefore race-safe **only on the assumption that no two
builds complete within a single query's cold-load window** (the time it takes
that query to fault the shard `.bin`/`.meta.json` into its local cache). If
two builds were to complete back-to-back inside that window — writing *N+1*
and *N+2* — shard *N* would fall to `≤ N-1` and could be swept while a query
that resolved *N* is still cold-loading it. The builder is single-replica and
builds are far slower than a cold load, so in practice this never happens, but
the guarantee is bounded by build cadence, not an absolute time-based hold.
Raising `SHARD_KEEP` widens the buffer if a deployment ever needs more slack.

The catalog row is deleted *after* the objects, so a crash mid-sweep leaves a
harmless orphan object (retried next build) rather than a catalog row pointing
at a missing object.

### Indexed-landing sweep

A landing `.parquet` part folded into a shard is recorded in the newest shard's
`indexed_landing_uris` manifest. Once recorded the part's bytes are dead
weight, so the sweeper deletes every object named in that manifest.

This does **not** break incremental indexing: the manifest still records the
URI (so a duplicate `DATASET_READY` is still a no-op — the part simply no
longer appears in the landing listing, which is exactly what the incremental
builder wants). It is safe because the builder is single-replica, so no
concurrent build is mid-read of the same part, and queries never read landing.

Bulk-import staging is pruned by the same sweep — see `docs/api/imports.md`
("Storage retention").

## Observability

- `rosalinddb.index_builds` — counter, attribute `build_type` = `full` |
  `incremental`.
- `rosalinddb.index_build.vectors_added` — histogram (count), attribute
  `build_type`. Vectors folded in by a single build.
- `rosalinddb.index_build.duration` — histogram (ms), attribute `index_type`.
- `rosalinddb.storage.swept` — counter, attribute `kind` = `shard` |
  `landing`. Objects reclaimed by the post-build sweepers.

No tenant/dataset labels on any of these — keeping cardinality bounded.
