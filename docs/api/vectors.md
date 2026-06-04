# Vectors API (get / list / delete by id)

Get, list, and delete individual vectors by their customer-supplied string id.
This page documents the implementation details the [v1 contract](./v1.md)
keeps out of the contract proper (sidecar mechanics, pagination cursor,
consistency model, the recall union).

## Endpoints

- `GET    /v1/datasets/{name}/vectors/{id}` — get one vector's id + metadata
- `GET    /v1/datasets/{name}/vectors` — list vectors (filter + pagination)
- `DELETE /v1/datasets/{name}/vectors/{id}` — delete one vector by id

See the **Vectors (get / list / delete)** section of [`v1.md`](./v1.md) for the
request/response schemas and error codes.

## Two modes: flag-off (consolidated-only) vs flag-on (recall union)

With `RB_RECALL` **off** (the default), this surface serves data purely from
today's asynchronously-built consolidated shards and is **byte-identical** to the
shipping pipeline — no recall connection is ever opened.

With `RB_RECALL` **on** (`RB_RECALL=true` + `RB_RECALL_DSN` set), these endpoints
**union** the consolidated shard with the [recall tier](../architecture/recall-consolidate.md),
with **recall authoritative** for any id above the resolved shard's watermark
(`consolidated_lsn`) — exactly the recall-wins / tombstone-suppress rule the
[query union](./query.md) uses. This makes a just-written-but-not-yet-consolidated
vector visible to get/list, and adds a **synchronous recall-delete** (below).

| Operation | Flag OFF | Flag ON (recall) |
|---|---|---|
| **get** | cold sidecar lookup | live recall row (`lsn > watermark`) **wins**; a recall **tombstone** → `404`; else fall back to the cold sidecar |
| **list** | cold sidecar, filtered + paginated | union cold + recall live rows (recall-wins dedup), **suppress** any id with a recall tombstone above the watermark, then filter + sort + paginate |
| **delete** | publish `DELETE_VECTORS`, `202 {job_id}` (async) | write an **above-watermark tombstone**, `204` (synchronous, read-your-deletes); **no** `DELETE_VECTORS` |

The watermark used for the union is the `consolidated_lsn` of the **same** shard
the cold sidecar was read from (invariant I3 — never an independently-resolved
watermark). When the dataset has **no shard yet**, the watermark is `0`, so every
recall row qualifies — a brand-new dataset's writes are visible to get/list
straight from recall.

## How a vector is resolved (the sidecar)

A FAISS `IndexIDMap2` stores only the SHA1-derived **int64** hash of each
string id (see `adapters/landing/parquet_reader.id_to_int64` — the single
shared hash used by both the builder and this surface). The original id and
its metadata live in the shard's `{shard_uri}.meta.json` **sidecar**:

```json
{ "<int64-hash>": { "id": "<original id>", "metadata": { } } }
```

- **get-by-id** resolves the newest shard (`state.get_latest_shard`), reads
  its sidecar (`read_shard_sidecar`), hashes the requested id, and looks up the
  entry. Missing shard or missing key → `404 not_found`. With the recall union
  on, the recall point-lookup (`recall_get_vector`) runs first against the same
  resolved shard's watermark and wins (live) / 404s (tombstone) / defers (absent)
  before this cold lookup.
- **list** reads the whole sidecar, applies the optional `filter`
  (`metadata_matches_filter`, the same AND-of-equals predicate as
  `POST /v1/query`), stably sorts by original id, and paginates. With the recall
  union on, the cold sidecar is first deduped against the recall partition
  (`recall_list_rows`): ids recall is authoritative for are dropped from the cold
  set, recall live rows are appended (recall-wins), and the filter is applied to
  the **merged** records so it sees each id's authoritative metadata.

The raw vector values are not returned in v1. Returning them requires a FAISS
`reconstruct` against the index (the sidecar holds only id + metadata); that is
a noted follow-up exposed later as `?include_values`.

## Pagination cursor

`GET /v1/datasets/{name}/vectors` returns an opaque `next_cursor`. It is a
base64-encoded JSON offset (`{"o": N}`) into the id-sorted result — opaque on
purpose so the scheme can change later (e.g. to a keyset cursor) without
breaking clients. `limit` defaults to 100 and is capped at 1000 (a larger value
is silently clamped to 1000; a non-integer or `< 1` value is rejected with
`400 invalid_limit`). A malformed cursor — including one whose decoded offset is
negative — is rejected with `400 invalid_cursor` rather than silently restarting
from the beginning.

### Continuation contract (resend the same `filter` and `limit`)

The cursor encodes **only the offset** — it does **not** capture the active
`filter` or `limit`. A continuation request **MUST resend the same `filter` and
`limit`** it used for the first page. Changing either mid-pagination applies the
old offset to a *different* (re-filtered / re-sorted) result set, silently
**skipping or duplicating** rows. Treat `(filter, limit)` as fixed for the
lifetime of a pagination run; to change them, start a new run without a cursor.

### Eventual-consistency caveat

The offset is resolved against the **newest shard at request time**, not a
snapshot taken when pagination began. A concurrent rebuild — an ingest
(`DATASET_READY`) or a delete (`DELETE_VECTORS`) — that produces a new shard
generation between pages can add, remove, or re-order rows under a stable
offset, so a long pagination run can miss or repeat a row that was inserted or
deleted while it was in progress. This is the expected behaviour of a simple
offset cursor over an eventually-consistent consolidated tier. v1 keeps the offset
cursor deliberately (a keyset cursor that is stable across rebuilds is a
possible later change, enabled by the cursor being opaque); callers that need a
strictly consistent full scan should page quickly and tolerate the small
windows, or re-run the scan.

## Delete (flag OFF): asynchronous and eventually consistent

`DELETE /v1/datasets/{name}/vectors/{id}` mirrors the `POST .../vectors`
contract: it publishes a `DELETE_VECTORS` job, flips the dataset to `indexing`,
and returns `202 {job_id}`. The **index builder** consumes `DELETE_VECTORS`
alongside `DATASET_READY`:

1. load the newest shard's FAISS index + sidecar;
2. `_remove_ids([hash])` (a no-op if the id is absent — deleting an unknown id
   still returns `202`);
3. drop the id from the sidecar;
4. write a new superseded shard via the shared build/catalog/sweep tail
   (`_write_shard` — the exact path an incremental ingest uses), then sweep the
   old shard and evict it from the query cache;
5. flip the dataset back to `indexed`.

Poll `GET /v1/datasets/{name}` to observe the `indexing` → `indexed`
transition and the decremented `row_count`. Until the build lands, a query may
still return the id (the old shard is authoritative); after it lands, get,
list, and query all agree the id is gone.

```plantuml
@startuml
title DELETE /v1/datasets/{name}/vectors/{id}
actor Client
participant "Control Plane\n(source_registry)" as CP
queue "DELETE_VECTORS" as Q
participant "index_builder" as IB
database "shard_catalog" as CAT
cloud "Object store\nshard + sidecar" as S3

Client -> CP : DELETE .../vectors/{id}
CP -> CP : resolve dataset (tenant-scoped)
CP -> Q  : publish {tenant, dataset, id, job_id}
CP -> CAT : status = indexing
CP -> Client : 202 {job_id}

Q -> IB
IB -> S3  : load newest shard + sidecar
IB -> IB  : _remove_ids([hash]); drop id from sidecar
IB -> S3  : write superseded shard + sidecar
IB -> CAT : add_shard; reconcile row_count; status = indexed
IB -> IB  : sweep old shard; evict query cache
@enduml
```

## Delete (flag ON, recall): synchronous, read-your-deletes

With the recall tier on, `DELETE /v1/datasets/{name}/vectors/{id}` is
**synchronous**. It writes a **tombstone** (`deleted = true`) into the recall tier
and returns **`204 No Content`** — there is no async job. The union immediately
hides the id: an immediate `GET` → `404`, a `POST /query` no longer returns it,
and `list` omits it. It does **not** publish `DELETE_VECTORS`; the next
**consolidation** applies the tombstone to the consolidated tier (`_remove_ids`)
and the recall row is then grace-trimmed. Deleting an absent id is still a
clean no-op (it writes a tombstone that consolidation discards) → still `204`.

### The above-watermark lsn contract (hard requirement)

The recall-delete tombstone is **always allocated a fresh `lsn` ABOVE the current
watermark**, from the same `recall_lsn_seq` atomic upsert-increment a recall
write uses (`recall_delete_vector`). The delete is a last-write-wins UPSERT on
`(tenant, dataset, id)` stamped with that fresh lsn — it **never** flips
`deleted=true` in place at the row's old lsn, and **never** reuses an old lsn.
This is load-bearing for correctness:

- A tombstone at or below the watermark (`lsn <= consolidated_lsn`) is **excluded
  from every union** (`lsn > consolidated_lsn` is false) — the id would never
  delete.
- It is also **trim-eligible but unapplied**: the grace-bounded trim could GC the
  recall row before a consolidation folds the delete into cold — the id would
  **resurrect** from the consolidated shard.

Allocating fresh, strictly above `max(lsn) >= consolidated_lsn`, guarantees the
tombstone is inside the union's `lsn > watermark` scan window and is applied by
the next consolidation. See
[recall & consolidate](../architecture/recall-consolidate.md), "Write path"
(Delete) + invariants I1/I2, and the regression test
`tests/.../test_*crud_union*::*above_watermark*`.

A cold-only delete (an id with no prior recall row) writes a brand-new tombstone
with a zero-vector placeholder embedding of the dataset's dimension — never read
(tombstones are not search candidates) but dimension-matched so the recall
search scan's distance computation over the partition does not hit a
dimension-mismatch.

```plantuml
@startuml
title DELETE /v1/datasets/{name}/vectors/{id} (RB_RECALL on)
actor Client
participant "Control Plane\n(source_registry)" as CP
database "Recall tier\npgvector" as R
participant "index_builder\n(consolidate)" as IB
cloud "Object store\nshard + sidecar" as S3

Client -> CP : DELETE .../vectors/{id}
CP -> CP : resolve dataset (tenant-scoped)
CP -> R  : SELECT nextval(lsn_seq); UPSERT tombstone (deleted=true, lsn > watermark)
note right : durable -> union hides the id immediately
CP -> Client : 204 (read-your-deletes)
... later (consolidation) ...
R -> IB  : snapshot recall partition (incl. tombstones)
IB -> S3 : fold live rows; _remove_ids(tombstoned ids); write shard
IB -> R  : grace-bounded trim of consolidated rows
@enduml
```

## Tenant scoping

Every state/storage lookup is keyed by the caller's `tenant_id` (resolved by
the auth dependency). A missing or cross-tenant dataset returns
`404 dataset_not_found` and never leaks another tenant's vectors.

## Implementation

- Handlers: `services/source_registry/main.py`
  (`get_vector_endpoint`, `list_vectors_endpoint`, `delete_vector_endpoint`;
  shard/watermark pairing via `_resolve_shard_sidecar_and_watermark`).
- Recall union helpers: `adapters/state/state.py`
  (`recall_get_vector`, `recall_list_rows`, `recall_delete_vector` — the
  above-watermark tombstone write).
- Builder consumers (flag-off delete + consolidation): `services/index_builder/run.py`
  (`run_delete_once`, `_handle_delete_vectors`; `run_consolidate_once` applies
  recall tombstones to cold).
- Shared id hash + sidecar reader: `adapters/landing/parquet_reader.py`.
