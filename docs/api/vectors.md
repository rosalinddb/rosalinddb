# Vectors API (cold-tier CRUD)

Get, list, and delete individual vectors by their customer-supplied string id.
This page documents the implementation details the [v1 contract](./v1.md)
keeps out of the contract proper (sidecar mechanics, pagination cursor,
consistency model).

## Endpoints

- `GET    /v1/datasets/{name}/vectors/{id}` — get one vector's id + metadata
- `GET    /v1/datasets/{name}/vectors` — list vectors (filter + pagination)
- `DELETE /v1/datasets/{name}/vectors/{id}` — delete one vector by id

See the **Vectors (cold-tier CRUD)** section of [`v1.md`](./v1.md) for the
request/response schemas and error codes.

## Flag-independent

This surface is **not** gated by `RB_DELTA_TIER`. It serves data from today's
asynchronously-built cold shards and works with the shipping pipeline. The
hot↔cold union for these operations (so a just-written-but-not-yet-flushed
vector is visible to get/list/delete) is a later step in the
[delta-tier plan](../architecture/delta-tier.md) (PR6) and does not change
this contract — it only widens what these endpoints can see.

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
  entry. Missing shard or missing key → `404 not_found`.
- **list** reads the whole sidecar, applies the optional `filter`
  (`metadata_matches_filter`, the same AND-of-equals predicate as
  `POST /v1/query`), stably sorts by original id, and paginates.

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
offset cursor over an eventually-consistent cold tier. v1 keeps the offset
cursor deliberately (a keyset cursor that is stable across rebuilds is a
possible later change, enabled by the cursor being opaque); callers that need a
strictly consistent full scan should page quickly and tolerate the small
windows, or re-run the scan.

## Delete: asynchronous and eventually consistent

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

## Tenant scoping

Every state/storage lookup is keyed by the caller's `tenant_id` (resolved by
the auth dependency). A missing or cross-tenant dataset returns
`404 dataset_not_found` and never leaks another tenant's vectors.

## Implementation

- Handlers: `services/source_registry/main.py`
  (`get_vector_endpoint`, `list_vectors_endpoint`, `delete_vector_endpoint`).
- Builder consumer: `services/index_builder/run.py`
  (`run_delete_once`, `_handle_delete_vectors`, wired into `main_loop`).
- Shared id hash + sidecar reader: `adapters/landing/parquet_reader.py`.
