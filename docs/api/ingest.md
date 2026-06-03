# Ingest API

Customer-facing dataset + vector ingest endpoints. Served by the Control
Plane (which internally composes the `source_registry` FastAPI app — see
`services/source_registry/main.py`). All endpoints are tenant-scoped via
`Authorization: Bearer <token>` (JWT or `rb_live_...` API key) when
`RB_REQUIRE_AUTH=true`; in the OSS default mode every call resolves to the
built-in `default` tenant. The full contract is in [`v1.md`](./v1.md); this
page is a quickstart with curl examples.

## Create a dataset

```bash
curl -s -X POST http://localhost:8080/v1/datasets \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"name":"products","dimension":768}'
```

`name`: 1-64 chars, `[a-z0-9_-]+`, unique per tenant.
`dimension`: positive int; every uploaded vector must match.

Errors: `400 invalid_name`, `400 invalid_dimension`, `409 dataset_exists`.

## Upload vectors (NDJSON)

```bash
printf '{"id":"a","values":[0.1,0.2,0.3,0.4]}\n{"id":"b","values":[0.5,0.6,0.7,0.8]}\n' | \
  curl -s -X POST http://localhost:8080/v1/datasets/products/vectors \
    -H "Authorization: Bearer $API_KEY" \
    -H 'Content-Type: application/x-ndjson' \
    --data-binary @-
```

Each line is a JSON object with `id` (non-empty string, max 256 chars),
`values` (list of floats, length == dataset.dimension), and optional
`metadata` (object). This endpoint is an upsert: re-sending a record with
an `id` that already exists in the dataset overwrites it (last write wins)
rather than creating a duplicate. Returns `202` with
`{accepted, rejected, errors[], job_id}`. Records that fail validation are
reported in `errors[].reason`; accepted records are processed
asynchronously — poll `GET /v1/datasets/{name}` for status. The `accepted`
count reflects validated NDJSON lines counted before upsert dedup, so a
request that re-sends the same `id` twice counts both lines even though
last-write-wins leaves only one stored row.

Body cap: 10 MiB per request. Larger payloads return `413 payload_too_large`.
For larger embedding dumps, use the async bulk-import flow instead — see
[`imports.md`](./imports.md).

## Poll status

```bash
curl -s http://localhost:8080/v1/datasets/products \
  -H "Authorization: Bearer $TOKEN"
```

The `status` field walks `empty -> validating -> indexing -> indexed` as
the pipeline processes uploads. On failure it lands on `error` with
`error_message` populated.

## List / delete

```bash
curl -s http://localhost:8080/v1/datasets -H "Authorization: Bearer $TOKEN"
curl -s -X DELETE http://localhost:8080/v1/datasets/products -H "Authorization: Bearer $TOKEN"
```

`DELETE` is a soft-delete: the row is marked `deleted_at = now()` and
subsequent `GET` returns `404 dataset_not_found`. The shard catalog rows
are purged synchronously in the same transaction; the underlying
object-storage bytes are reclaimed by a background sweep.

## Get / list / delete individual vectors

Operate on a single vector by its customer-supplied string id, served from
the cold shards. Full reference: [`vectors.md`](./vectors.md).

```bash
# get one vector's id + metadata
curl -s http://localhost:8080/v1/datasets/products/vectors/doc-42 \
  -H "Authorization: Bearer $TOKEN"

# list (paginated; optional metadata filter as URL-encoded JSON)
curl -s "http://localhost:8080/v1/datasets/products/vectors?limit=100" \
  -H "Authorization: Bearer $TOKEN"

# delete by id (async: returns 202 {job_id}; poll dataset status to confirm)
curl -s -X DELETE http://localhost:8080/v1/datasets/products/vectors/doc-42 \
  -H "Authorization: Bearer $TOKEN"
```

`list` returns `{vectors: [{id, metadata}], next_cursor}` — follow
`next_cursor` to page. `delete` is asynchronous and eventually consistent
(same contract as upload): the vector disappears from get/list/query once the
builder rewrites the shard. The raw vector values are not returned in v1.

## Tenant isolation

All endpoints filter by `current_tenant_id` resolved from the bearer token
(or the bootstrap `default` tenant when `RB_REQUIRE_AUTH` is off).
Cross-tenant requests return `404 dataset_not_found`, never `403` — the
contract intentionally hides existence.
