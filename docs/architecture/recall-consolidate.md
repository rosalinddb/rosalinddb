# Recall & Consolidate: read-your-writes on object storage

> **Status: DRAFT / not yet implemented.** Design under active iteration.
> The recall tier ships behind `RB_RECALL` (default **off**); a self-hoster who
> upgrades and changes nothing sees identical behaviour to today (same convention as
> [`ssd-cache.md`](ssd-cache.md)). Pairs with [`indexing.md`](indexing.md) (the build
> path consolidation reuses), [`ssd-cache.md`](ssd-cache.md) (the read-cache hierarchy this
> sits *beside*, not inside), and [`architecture.md`](architecture.md).

## The problem

RosalindDB today is **eventually consistent on writes**: `POST /v1/datasets/{name}/vectors`
returns `202`, lands NDJSON, and a `DATASET_READY` build folds it into a FAISS shard
asynchronously. A query issued between the write and the build completing does **not**
see the new vector — it returns `ephemeral` (empty + `job_id`) or hits an older shard.

For batch RAG over slowly-changing corpora that is fine. For **agent memory it is not**:
an agent that stores "the user is allergic to peanuts" and asks "what should I avoid?"
on the next turn must get the fact back *now*. That property — **read-your-writes** — is
the table stakes the async pipeline cannot provide.

The recall tier adds a small, synchronously-writable, immediately-queryable **recall tier**
in front of the immutable shards, and unions the two at query time. It is the classic
**LSM-tree** split (memtable + SSTables) applied to an object-storage vector index — the
same shape Turbopuffer (WAL + object storage) and Pinecone serverless (a "freshness
layer") use.

## Two axes — do not conflate them

RosalindDB will have two things that both get called "tiers." They are orthogonal:

| Axis | Members | Purpose | Authority | Affects correctness? |
|---|---|---|---|---|
| **Storage distance** (today, see `ssd-cache.md`) | Object store → SSD → RAM | make *reaching cold data* fast | disposable copies; S3 is truth | **No** — a miss is only slower |
| **Write freshness** (this doc) | **recall** ↔ **consolidated (shards)** | make *just-written data* visible | recall is authoritative until consolidated | **Yes** — the union must be complete |

The SSD/RAM cache (the storage-distance axis) holds **copies of consolidated shards**, and its
members are still called **hot/cold** there — that is the cache hit/miss distance, a *separate*
axis from this one. It can never lose data — a miss just re-fetches from S3. The recall tier
holds **data that is in no shard yet**. The only place completeness can break is the
**recall↔consolidated seam** (§The watermark, §Invariants) — never the cache.

## Diagrams

Dedicated, richer PlantUML diagrams focused on **how the Recall tier snaps onto
the Consolidate tier and how they work in tandem** (the inline ```plantuml blocks
below are kept as quick in-context sketches; these files are the detailed
companions). All live in [`diagrams/`](diagrams/):

| File | Kind | Shows |
|---|---|---|
| [`recall-consolidate-tandem.puml`](diagrams/recall-consolidate-tandem.puml) | component / overview | Both tiers either side of the `consolidated_lsn` seam (Recall owns `lsn > W`, Consolidate owns `lsn <= W`), the `index_builder`, and all three flows: synchronous write → Recall; consolidation drains Recall → shard + advances the watermark; query reads BOTH and merges. |
| [`recall-consolidate-watermark.puml`](diagrams/recall-consolidate-watermark.puml) | the "snap" picture | An LSN number line with `consolidated_lsn` as the divider, BEFORE vs AFTER a consolidation: the watermark snaps `W → N`, and the grace-bounded trim (I4) reclaims Recall only to the 2nd-newest shard's watermark. |
| [`recall-consolidate-lifecycle.puml`](diagrams/recall-consolidate-lifecycle.puml) | sequence | One vector's journey: write → Recall (queryable now) → union serves it from Recall → consolidation commits `consolidated_lsn=N` → union serves it from Consolidate → its Recall row is eventually trimmed. |
| [`recall-consolidate-consolidation.puml`](diagrams/recall-consolidate-consolidation.puml) | handoff sequence (I2/I4) | snapshot up to N → build + commit shard → grace-bounded idempotent trim, annotating commit-then-trim ordering (I2), the grace buffer (I4), and the crash-safe windows across the two-DB seam. |
| [`recall-consolidate-query-union.puml`](diagrams/recall-consolidate-query-union.puml) | query union sequence | resolve latest shard + its watermark (I3) → search Recall (`lsn > W`, brute-force) + Consolidate (FAISS via cache) → merge: square pgvector L2 → L2² (metric alignment), recall-wins dedup, tombstone suppression, sort, top_k. |

Validated with PlantUML 1.2024.7 (`-checkonly` + PNG render, both clean) — see the PR for details.

### Recall tier internals

A companion set focused on **how the Recall tier works BY ITSELF** — its own
data model and flows, independent of the Consolidate tandem above. These are
accurate to the implementation (`adapters.state.state.recall_*` +
`recall/001_recall_vectors.sql`); CONSOLIDATE appears only as the outflow.
All live in [`diagrams/`](diagrams/):

| File | Kind | Shows |
|---|---|---|
| [`recall-internal-model.puml`](diagrams/recall-internal-model.puml) | structure / data-model | The Recall tier as a SEPARATE data-plane pgvector instance (`RB_RECALL_DSN`, never the control-plane PG): the `recall_vectors` table (columns + `(tenant_id, dataset, id)` PK), the `recall_lsn_seq` table (per-`(tenant,dataset)` `last_lsn`), the partition b-tree on `(tenant_id, dataset, lsn)`; brute-force exact by default (HNSW is a flagged escape hatch needing a fixed dim). |
| [`recall-internal-write.puml`](diagrams/recall-internal-write.puml) | sequence | Synchronous write (`recall_upsert_vectors`): allocate an LSN BLOCK via the atomic `recall_lsn_seq` upsert-increment (seq-row lock → commit-order == LSN-order → strictly monotonic), then a SINGLE multi-row UPSERT into `recall_vectors` (last-write-wins, `deleted=false`); one transaction, committed before returning → immediately queryable (read-your-writes). |
| [`recall-internal-search.puml`](diagrams/recall-internal-search.puml) | sequence | Brute-force search internals (`recall_search`): the TWO scans — MATCH (`NOT deleted`, `lsn > watermark`, `ORDER BY embedding <-> q`, `score = power(<->,2) = L2²`) and SUPPRESS-id (all ids above the watermark, live + tombstoned) — both scoped to the `(tenant,dataset)` partition; why an exact full-partition scan is fine (the set is bounded small by the cap + idle drain, not by data size). |
| [`recall-internal-delete.puml`](diagrams/recall-internal-delete.puml) | sequence | Recall-delete (`recall_delete_vector`): UPSERT a tombstone (`deleted=true`) stamped with a FRESH `lsn` from `recall_lsn_seq` (strictly above the watermark — never an in-place flip), last-write-wins; committed before returning → synchronous read-your-deletes. |
| [`recall-internal-bounding.puml`](diagrams/recall-internal-bounding.puml) | flow | What keeps Recall SMALL (so brute-force stays cheap) from Recall's own POV: the per-`(tenant,dataset)` cap `RB_RECALL_MAX_ROWS` → enqueue `CONSOLIDATE`; the idle window `RB_RECALL_IDLE_S` → drain to zero rows. CONSOLIDATE is the outflow only. |

## Architecture

> Diagram: the component/overview is [`diagrams/recall-consolidate-tandem.puml`](diagrams/recall-consolidate-tandem.puml).

```plantuml
@startuml
title Recall tier — components (RB_RECALL on)
skinparam componentStyle rectangle

actor Client
component "Control Plane\n(HTTP origin)" as CP
database "Recall tier\npgvector\n(tenant, dataset, id,\nembedding, metadata, lsn, deleted)" as RECALL
queue "consolidate signal" as Q
component "index_builder\n(consolidate = compaction)" as IB
database "shard_catalog\n(+ consolidated_lsn)" as CAT
cloud "Object store\nFAISS shards + sidecars" as S3
component "query_dp\n(union + merge)" as DP

Client --> CP : POST /vectors (sync)
CP --> RECALL : 1. assign lsn, UPSERT
CP --> Q  : 2. enqueue consolidate (async)
Q --> IB
IB --> RECALL : drain lsn <= N
IB --> S3  : write shard + sidecar
IB --> CAT : commit consolidated_lsn = N
Client --> CP : POST /query
CP --> DP
DP --> RECALL : search lsn > consolidated_lsn
DP --> S3  : search newest shard (via SSD/RAM cache)
DP --> CAT : resolve latest shard + its consolidated_lsn
@enduml
```

The recall tier is **pgvector, deployed as a separate data-plane instance** (not the
control-plane Postgres — see §Blast radius). Consolidation is the **existing `index_builder`**
fed from pgvector rows instead of landing parquet. The watermark (`consolidated_lsn`) is the seam.

## Blast radius & control/data-plane isolation

The recall tier is **data-plane** work and MUST NOT share fate with the control-plane Postgres,
which is on the critical path of *every* query (the DP resolves tenant → dataset → latest
shard → `consolidated_lsn` from it). Today the vector *data* never touches the control-plane PG
(ingest goes S3 landing → builder → S3 shards); the recall tier must preserve that property.

Where the recall tier lives sets the blast radius of a write storm:

| Recall tier placement | A tenant write-storms → | Blast radius |
|---|---|---|
| Co-located on control-plane PG | metadata reads starve → no query can resolve its shard | **Total multi-tenant outage** |
| Separate shared recall instance (**default**) | co-tenants on that instance degrade; truth DB survives | degraded co-tenants, system up |
| Sharded / per-tenant recall (future) | only the noisy tenant degrades | noisy tenant only |

**Decision:** the recall tier defaults to a **separate pgvector instance** (data-plane),
addressed via `RB_RECALL_DSN`. This mirrors RosalindDB's existing control-plane/data-plane split
(the CP already proxies `/v1/query` to a private Query DP). Per-tenant load is bounded by
quotas + the recall-row cap + consolidation, and the design stays shardable so blast radius can
later shrink to per-tenant.

**The CP is protected by construction:** the LSN sequence lives in the recall store, so the
per-write path never touches the control-plane PG. The control-plane PG sees the recall tier only
as a low-frequency `consolidated_lsn` update at consolidation time — never per write.

## Running recall behind pgbouncer

In production the recall tier is run the way Postgres is run elsewhere: **behind pgbouncer in
TRANSACTION pooling mode**. The full connection topology is:

```
   app process  ──►  app-side pool (#15, RB_RECALL_POOL_MAX)  ──►  RB_RECALL_DSN
                                                                       │
                                                            pgbouncer (transaction mode)
                                                                       │
                                                            recall pgvector (RB_RECALL_DSN target db)
```

`RB_RECALL_DSN` points at **pgbouncer**, which targets the recall pgvector database. Each app
process keeps its own small app-side pool (the `ThreadedConnectionPool` added in #15); those
pools fan in to pgbouncer, which multiplexes them onto a small set of real pgvector backends.
A wired reference is `bench/docker-compose.recall-bench.yml` (the `pgbouncer` service +
`RB_RECALL_DSN: …@pgbouncer:6432/recall` on every recall-touching service and the migrator).

### Transaction-mode safety (the contract the recall code keeps)

In transaction pooling mode a server (pgvector) connection is held only for the **duration of one
transaction**, then returned to pgbouncer's pool and potentially handed to a *different* client.
So **no session-level state may span transactions**. The recall code path is safe by construction
(audited in `adapters/state/state.py`, `_recall_connect`):

| Hazard in transaction mode | Recall path |
|---|---|
| Server-side **named prepared statements** persisting across a checkout | None. psycopg2 (the driver) never emits named server-side prepared statements; recall connections are additionally minted with `prepare_threshold=None` as an explicit, tested, psycopg3-forward marker. |
| `SET` / session GUCs | None. Isolation is the server default (READ COMMITTED); each single-statement read (`recall_search`, `recall_list_rows`, the consolidation snapshot) relies only on a **statement-level** snapshot, never session state. |
| `LISTEN`/`NOTIFY` | None on the recall tier (catalog NOTIFY is control-plane only). |
| Advisory locks held **across** transactions | None. The recall migrator uses `pg_advisory_xact_lock` (transaction-scoped) on a dedicated connection; the only session-level advisory lock (`dataset_build_lock`) is control-plane, not recall. |
| Server-side **named cursors** | None. Plain client cursors, fetched within the same transaction. |
| Two statements assuming a shared session beyond one txn | None. Every recall op is **one self-contained transaction on one app-side checkout** — including the multi-statement write ops (LSN-block alloc + UPSERT; LSN alloc + tombstone), which are the *same* transaction. |

### Honest scope: when pgbouncer actually helps

pgbouncer is the **horizontal / prod-scale** option, not a single-host speedup. On a single host
the app-side pool (#15) already captures essentially all of the latency win: the recall-bench
finding was that the recall cost is **query work + RTT**, not connection setup, and the app-side
pool already amortises connection setup across requests. Adding pgbouncer on the same host neither
removes RTT nor speeds up the pgvector scan.

Where pgbouncer pays off is **connection multiplexing across many app replicas**. Each app
process holds up to `RB_RECALL_POOL_MAX` recall connections; with R replicas the naive ceiling is
`R × RB_RECALL_POOL_MAX` real backends against one recall instance, which quickly exceeds
pgvector's `max_connections`. pgbouncer in transaction mode lets a large number of *client*
connections share a small, bounded set of *server* backends, so you can scale app replicas
without scaling (or exhausting) the recall instance's backend cap. Size the app-side pools and
pgbouncer's `default_pool_size` together so the server-side pool stays under the recall instance's
connection limit.

## The watermark (the seam)

> Diagram: the "snap" before/after picture is [`diagrams/recall-consolidate-watermark.puml`](diagrams/recall-consolidate-watermark.puml).

Every write is stamped with a monotonic **`lsn`** (log sequence number) from a per-dataset
sequence. Each `shard_catalog` row gains **`consolidated_lsn`** = the highest LSN folded into
that shard. This single number partitions the universe of vectors:

```
   lsn <= consolidated_lsn   ->  lives in CONSOLIDATED  (the shard)
   lsn >  consolidated_lsn   ->  lives in RECALL        (pgvector)
```

Every vector has exactly one LSN, so it is in **exactly one** set. Union = complete.
This is the whole correctness story; everything below protects this invariant.

The LSN sequence lives in the **recall store** (so the per-write path never touches the
control-plane PG); `consolidated_lsn` is written to the control-plane `shard_catalog` only at
consolidation. The two live in different databases by design (§Blast radius) — consolidation's
commit-then-trim ordering (I2) plus an **idempotent trim** make that split safe without a
distributed transaction.

> **New to LSN / LSM / SSTable?** See the reading list in the design journal
> (kept out of this repo). Short version: an LSN is a monotonic version stamp on each
> write (like a Postgres WAL LSN or a RocksDB sequence number); an LSM-tree buffers writes
> in an in-memory *memtable*, then flushes them to immutable on-disk *SSTables*, merged by
> *compaction*. Here: pgvector = memtable, S3 shards = SSTables, `index_builder` = compaction.

## Write path

```plantuml
@startuml
title Write (RB_RECALL on)
Client -> CP : POST /vectors {id, values, metadata}
CP -> CP : validate (dim, id, metadata)
CP -> RECALL : SELECT nextval(lsn_seq); UPSERT (… , lsn, deleted=false)
note right : durable in Postgres -> immediately queryable
CP -> Queue : enqueue consolidate signal (best-effort)
CP -> Client : 200 OK {accepted, rejected, errors}   // 202 preserved when flag OFF
@enduml
```

**Consolidation-cadence change.** Today every ingest batch produces a new shard. With the
recall tier, **writes no longer create shards** — they accumulate in pgvector and are baked
into a shard on **consolidation**, which coalesces many writes into one build. Net effect vs
today:

| | Today | With recall tier |
|---|---|---|
| Shard created per | ingest batch | **consolidation** (batches many ingests) |
| Queryable when | after build | **immediately** (from recall) |
| Sidecar rewrites | per addition | per consolidation |

This *reduces* the write amplification `indexing.md` flags, and decouples shard-creation
rate from write rate.

**Delete / update.** Delete = `UPDATE … SET deleted=true` in recall (immediate tombstone).
Update = UPSERT (last-write-wins, new LSN). Tombstones are applied to the consolidated tier at
consolidation via the existing `_remove_ids`.

**Bulk imports bypass recall.** The async import path (`POST …/imports`) lands directly to
the consolidated tier (landing → builder → shard). Large dumps never enter the recall tier —
this is what keeps the recall set small enough for brute-force search (see §Recall search).

## Read path — the union

> Diagrams: the full query union (metric alignment, dedup, tombstone suppression, mode labels) is [`diagrams/recall-consolidate-query-union.puml`](diagrams/recall-consolidate-query-union.puml); one vector's end-to-end journey across the seam is [`diagrams/recall-consolidate-lifecycle.puml`](diagrams/recall-consolidate-lifecycle.puml).

```plantuml
@startuml
title Query (RB_RECALL on)
Client -> DP : POST /query {vector, top_k, filter}
DP -> CAT : resolve latest shard S, read consolidated_lsn(S)
DP -> S3 : FAISS search S  (via SSD/RAM cache)   -> consolidated hits (L2^2)
DP -> RECALL : exact search WHERE tenant,dataset, NOT deleted, lsn > consolidated_lsn(S), filter
note right : brute-force over the small recall set, scoped to this (tenant,dataset)
DP -> DP : align metric (square pgvector L2 -> L2^2)
DP -> DP : dedup by id (RECALL wins), drop recall-tombstoned ids, sort asc, take top_k
DP -> Client : {matches, mode, ...}
@enduml
```

**Recall search = brute-force exact** (no ANN index), scoped to the `(tenant, dataset)`
partition via a b-tree filter, then exact L2 over those rows. Correct *by construction*
because consolidation keeps the partition small (§Recall search). HNSW is a flagged escape
hatch (`RB_RECALL_INDEX=hnsw`, default off), expected never to be needed.

**Metric alignment (correctness-critical):** the consolidated tier returns FAISS **L2-squared**;
pgvector `<->` returns plain L2. Square pgvector's distance before merging, over **identical
un-normalised** vectors, or the union ranks wrong. This is the most likely silent bug — it
gets a dedicated test. *Implemented* as `power(embedding <-> q, 2)` in the recall scan
(`adapters.state.state.recall_search`), so the recall `score` is L2² and merges directly with
the consolidated shard's FAISS L2² distances.

**Dedup:** a re-upserted id can be in both tiers during the consolidation grace window;
**recall is authoritative** for any id above the watermark (its version is newer). The merge
(`services.query_api.v1_query._merge_recall_and_consolidated`) keys suppression on the FULL set of recall
ids above the watermark: a recall row for id X — **live, tombstoned, filtered-out, or
ranked-past-`top_k`** — always drops the stale consolidated copy of X. Only a **filter-passing live**
recall row contributes an actual match; a tombstone, or a live row that fails the query filter,
suppresses its consolidated twin but adds nothing (so an authoritative re-upsert that no longer matches
the filter cannot let a stale, filter-matching consolidated copy leak). The survivors are sorted ascending
by L2² and truncated to `top_k`.

**No consolidated shard + recall has data → synchronous recall answer.** When the dataset has
no shard yet (`list_shards` → empty) the watermark is `0`, so every recall row qualifies and
the query is answered **synchronously from recall** — it does **not** fall through to the
`ephemeral` empty+`job_id` path (that path is reserved for "nothing can answer": no shard AND no
recall row). This is the read-your-writes property for a brand-new dataset.

**`mode` semantics under the union.** The response `mode` always reflects the **consolidated-shard cache
state**, never the recall contribution:

| Consolidated shard | Recall contributed | `mode` | `job_id`? |
|---|---|---|---|
| resolved (warm cache) | maybe | `hot` | no |
| resolved (cold load) | maybe | `cold` | no |
| none | yes | `recall` | no |
| none | no | `ephemeral` | yes |

So a `hot`/`cold` mode does **not** imply the recall tier was idle — recall may have overridden
or added matches; the label only describes the consolidated-shard cache state. `recall` is the dedicated value for
"no consolidated shard, recall answered." This is documented for callers in
[`docs/api/query.md`](../api/query.md).

**Watermark/shard pairing (I3) in code.** Shard resolution (`_resolve_shard`) reports the exact
shard row it resolved back to `run_query` (via an out-dict), the consolidated FAISS search
(`_search_consolidated_shard`) reads *that same* shard, and the recall scan is filtered with
*that* shard's `consolidated_lsn` — never a watermark resolved by an independent
`list_shards` lookup — so a stale cached shard version can never open a partition gap.

**Overlap (#31).** Resolution and the FAISS search are split so `run_query` resolves the shard
once, then runs the consolidated FAISS search and the recall scan **concurrently** (recall needs
only the watermark, not the FAISS result): the recall scan is submitted to a bounded worker pool
while the FAISS search runs inline, so the union latency is ~`max(consolidated, recall)` instead of
their sum. The worker re-attaches the request's OpenTelemetry context so the `recall.search` span
stays a child of the request span. If both branches fail, the consolidated error takes precedence
(preserving the pre-overlap serial ordering); a recall failure is never masked and a consolidated
failure is never reported as a recall error.

## Consolidation / flush

> Diagram: the build → commit (I2) → grace-bounded trim (I4) handoff, with the crash-safe windows, is [`diagrams/recall-consolidate-consolidation.puml`](diagrams/recall-consolidate-consolidation.puml).

The recall→consolidated flush *(implemented in `services.index_builder.run` —
`run_consolidate_once` / `_run_consolidate_locked` / `_build_consolidated_shard`)* is the
LSM **compaction** that folds a recall partition into the immutable shards and advances the
watermark. It runs in the **single-replica `index_builder`**, gated by the existing
per-dataset advisory lock (serialised against any concurrent build/delete), and is triggered
by a new `CONSOLIDATE` queue topic. **The order is load-bearing (I2): build → commit catalog
→ grace-bounded trim. NEVER trim before commit.**

1. **Snapshot** the `(tenant, dataset)` recall partition up to the current `max(lsn) = N`
   (live rows **and** tombstones) in one transaction from `RB_RECALL_DSN`
   (`recall_snapshot_for_consolidation`). A write that lands *after* the snapshot (higher LSN)
   is left in recall — it stays queryable via the union and the next consolidation folds it
   (this is what makes read-your-writes hold **through** a consolidation).
2. **Fold** the LIVE rows into a new Consolidated shard via the existing build tail
   (`_add_to_index` onto the current shard's index, or a fresh build; `_write_shard` + sidecar).
   **Apply tombstones**: `deleted=true` ids are `_remove_ids`'d from the index and dropped from
   the sidecar; they are never added.
3. **Commit** the `shard_catalog` row with `consolidated_lsn = N`, `build_type='consolidate'`;
   run the supersede sweep (keep newest 2) and evict superseded shards from the query cache.
4. **Then** (strictly after the commit — I2) **trim** `recall_vectors` idempotently and
   **grace-bounded (I4)**: `recall_trim` hard-deletes only rows whose `lsn <=` the
   **2nd-newest** shard's `consolidated_lsn` (NOT the newest just committed) — so an in-flight
   query that resolved an older shard still finds its recall rows. Re-runnable
   (`DELETE WHERE lsn <= grace_watermark`); the first consolidation, whose new shard is the
   only one, trims nothing.

**Triggers** (both `recall_enabled()`-gated, so flag-off enqueues nothing):
- **Per-tenant cap** `RB_RECALL_MAX_ROWS` (default 2000): after a recall write, if the
  partition row count exceeds the cap, `POST /vectors` enqueues `CONSOLIDATE` (best-effort,
  after the durable write — a cap-check failure never fails the committed write). This also
  bounds the union's brute-force recall scan.
- **Consolidate-on-idle** `RB_RECALL_IDLE_S` (default 60): a rate-limited sweep on the builder
  loop's idle tick (`recall_idle_partitions`) enqueues `CONSOLIDATE` for every partition whose
  newest write (`max(created_at)`) is older than the idle window → drains it to ZERO recall
  rows → idle queries skip pgvector entirely (scale-to-zero).

**Cross-DB crash safety (no distributed transaction).** Recall rows live in `RB_RECALL_DSN`;
`consolidated_lsn` lives in the control-plane catalog. Commit-then-trim (I2) + the idempotent
grace-bounded trim make the two-database seam safe **without** a distributed transaction: a
crash between commit and trim leaves recall rows with `lsn <= consolidated_lsn` that the union
**harmlessly excludes** (`lsn > consolidated_lsn` is false) and the next consolidation GCs.
No loss (the shard has them), no duplicate (the union excludes the recall copy).

## Invariants

These are named so tests and reviews can reference them.

- **I1 — Partition.** Every vector has exactly one `lsn`; the consolidated tier owns
  `lsn <= consolidated_lsn`, recall owns `lsn > consolidated_lsn`. ⇒ no vector is in *neither*
  tier.
- **I2 — Consolidation ordering.** Consolidation MUST: build shard → **commit**
  `consolidated_lsn=N` → **then** trim recall (`lsn <= N`). Never trim before commit. ⇒ no
  window where a row is in neither.
- **I3 — Watermark/shard pairing.** A query filters recall with
  `lsn > consolidated_lsn(**the shard it actually resolved/read**)`, not the catalog's claimed
  latest. ⇒ a stale cached shard version can never open a gap.
- **I4 — Grace buffer.** A consolidated recall row is physically deleted only once its covering
  shard is ≥ 2 generations old (symmetric to the `SHARD_KEEP=2` sweep). ⇒ an in-flight query
  that resolved an older shard still finds its rows in recall.

## Failure-mode table

| Scenario | Without protection | With invariants |
|---|---|---|
| Crash between shard commit and recall trim | rows in neither → lost reads | I2: rows still in recall (not yet trimmed) → served; trimmed next consolidation |
| Crash before shard commit | shard half-written | shard not committed; recall still authoritative; retried |
| Query reads stale cached shard V while V+1 exists | rows in (lsn_V, lsn_V+1] vanish | I3+I4: query uses V's watermark; those rows still in recall (grace buffer) |
| Re-upsert of an id present in the consolidated tier | duplicate in union | dedup recall-wins |
| Delete then immediate query | stale hit from consolidated tier | recall tombstone suppresses consolidated id |
| Chatty tenant outpaces consolidation | recall set grows unbounded → slow brute-force | per-tenant recall cap forces consolidation |

## Scale-to-zero preservation

An always-on pgvector in the recall path would quietly defeat scale-to-zero (idle tenants must
cost ~0). Mitigations, all **v1 requirements, not nice-to-haves**:

- **Consolidate-on-idle.** A `(tenant, dataset)` with no writes for `RB_RECALL_IDLE_S`
  is consolidated to completion → its recall row count → 0 → idle queries skip pgvector entirely
  (pure consolidated path / on-demand shard load). Postgres holds only the **active working set**.
  *Implemented* as a rate-limited sweep on the `index_builder` loop's idle tick
  (`adapters.state.state.recall_idle_partitions` → enqueue `CONSOLIDATE`).
- **Per-tenant recall cap** (`RB_RECALL_MAX_ROWS` per tenant/dataset) → forces a consolidation;
  bounds memory and keeps brute-force fast; stops one tenant evicting another's working set.
- **Bulk imports bypass recall** (above).

**Honest caveat (don't overclaim).** Read-your-writes requires *something* always-on to accept
synchronous writes, so the recall tier is a small, fixed, always-on **data-plane** cost
(Turbopuffer and Pinecone serverless have the same — their "freshness layers" are always-on
too). The **consolidated** tier scales to zero; the system as a whole does not go to literal
zero. A serverless scale-to-zero Postgres (e.g. Neon) as the recall store could reclaim even
that, at the price of cold-start latency on an idle tenant's first write — a future option, not
a v1 default.

## Recall search — why brute-force

The recall set is bounded by **consolidation cadence, not data size**. Total memory may be
millions of vectors (consolidated); recall holds only the trickle since the last consolidation —
hundreds to a few thousand rows. Exact L2 over that is sub-millisecond and has **zero recall
loss**; HNSW adds index-maintenance churn on a set that is about to be consolidated away. This
mirrors the existing codebase judgment in `query.md` ("the exhaustive scan is correct and fast
at current dataset sizes"), applied one tier up. Note `RB_RECALL_INDEX=hnsw` is not a drop-in
flag: the v1 `recall_vectors.embedding` column is an unparameterised `vector` (mixed per-dataset
dims), and a pgvector HNSW index requires a fixed dimension — so enabling it first needs a
fixed-dimension schema migration (e.g. a table/partition per embedding dimension).

## Config flags (all default off / current-behaviour-preserving)

| Flag | Default | Effect when set |
|---|---|---|
| `RB_RECALL` | `false` | Master switch: sync recall write, query union, consolidation worker |
| `RB_RECALL_MAX_ROWS` | `2000` | Per-(tenant,dataset) recall-row cap that forces a consolidation (after a recall write, if the partition row count exceeds this, a `CONSOLIDATE` is enqueued — also bounds the union's brute-force recall scan) |
| `RB_RECALL_IDLE_S` | `60` | Idle window after which a dataset is consolidated to zero recall rows (the builder's idle-tick sweep enqueues `CONSOLIDATE` for any partition whose newest write is older than this → drains to 0 → idle queries skip pgvector entirely) |
| `RB_RECALL_INDEX` | `bruteforce` | `hnsw` to add a pgvector ANN index (escape hatch — requires a fixed-dimension schema migration first, since the v1 `recall_vectors.embedding` is an unparameterised `vector` for mixed per-dataset dims) |
| `RB_RECALL_DSN` | separate recall pgvector instance | DSN of the recall tier (data-plane), isolated from the control-plane PG by default. A single-tenant self-hoster MAY point it at the control-plane DSN to accept shared-fate. In prod this points at **pgbouncer** (transaction mode), which targets the recall pgvector db — see §Running recall behind pgbouncer. |
| `RB_RECALL_POOL_MAX` | `10` | Per-process app-side recall connection-pool ceiling (the #15 `ThreadedConnectionPool` against `RB_RECALL_DSN`). Independent of the control-plane `RB_PG_POOL_MAX`. Size it with pgbouncer's `default_pool_size` so the server-side pool stays under the recall instance's `max_connections` across all app replicas. |

## TDD test plan

Unit (memory:// + a pgvector test instance; no S3 needed for merge logic):
- `test_lsn_monotonic_per_dataset`
- `test_union_merge_metric_alignment` — pgvector-L2 squared equals FAISS-L2² ordering
- `test_dedup_recall_wins_on_reupsert`
- `test_recall_tombstone_suppresses_consolidated_id`
- `test_recall_search_scoped_to_tenant_dataset`
- `test_query_skips_pgvector_when_recall_empty` (scale-to-zero path)

Integration (real PG/MinIO/Redis):
- `test_read_your_writes` — write → immediate query returns it
- `test_visibility_gap_during_consolidation` — query throughout a consolidation always returns the vector (I1/I2)
- `test_crash_between_commit_and_trim` — no loss, no dupe (I2)
- `test_stale_cache_version_uses_resolved_watermark` (I3)
- `test_grace_buffer_in_flight_older_shard` (I4)
- `test_consolidate_on_idle_drains_to_zero`
- `test_per_tenant_cap_forces_consolidation`
- `test_bulk_import_bypasses_recall`

## Decisions

**Decided**
- Recall tier = **pgvector**; consolidation reuses the existing `index_builder`.
- Recall search = **brute-force exact** (HNSW behind a flag).
- Feature-flagged, default-off, current-behaviour-preserving.
- LSN = **per-dataset** monotonic sequence, generated in the **recall store**; watermark =
  `shard_catalog.consolidated_lsn` (control-plane PG), written only at consolidation.
- **pgvector placement = separate data-plane instance by default** (`RB_RECALL_DSN`-overridable
  to the control-plane PG for single-tenant self-hosters). Rationale: blast-radius isolation
  of the control-plane truth DB (§Blast radius). The cross-DB watermark is made safe by I2
  (commit-then-trim) + an idempotent trim — no distributed transaction.
- **Ingest contract:** when `RB_RECALL` is on, `POST /vectors` returns **`200`** (write is
  synchronous, durable, immediately queryable); when off it keeps **`202`**. Body shape
  unchanged (`{accepted, rejected, errors}`, `job_id` optional). Flag-conditional, documented
  in `docs/api/v1.md`.
- **Sequencing (PR plan):** PR1 consolidated get/list/delete-by-id (flag-off correct) → PR2
  migrations + separate pgvector container → PR3 sync recall write + flag → PR4 query
  union/merge *(implemented — `recall_search` + `_merge_recall_and_consolidated`; see §Read path)* → PR5
  consolidation/compaction + consolidate-on-idle + caps *(implemented — `run_consolidate_once` +
  the `CONSOLIDATE` topic + the cap/idle triggers; see §Consolidation / flush)* → PR6 recall +
  consolidated union for get/list/delete + mem0 adapter + docs. Each flag-gated so `main` stays
  shippable.

**Open**
- (none blocking — ready to split into agents)

## Out of scope (future)
- HNSW recall index by default; W-TinyLFU-style recall eviction.
- Range/OR filters (v1 stays AND-of-equals, matching `query.md`).
- Append-structured sidecar (separate `indexing.md` follow-up).
- Multi-replica builder / shard-level locking (still single-replica).
