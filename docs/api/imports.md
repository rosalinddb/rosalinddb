# RosalindDB bulk import — the async import-job API

The async bulk-ingest flow: stage a large embedding dump directly into object
storage via a presigned upload, then a background job validates and indexes
it. Implementation: `services/source_registry/main.py` (the HTTP endpoints) +
`services/validator_worker/run.py` (the validation/landing worker).

## Why a separate flow

`POST /v1/datasets/{name}/vectors` buffers the whole NDJSON body in the
application and caps it at 10 MiB — fine for tiny interactive upserts, far too
small for a real embedding dump. The **import-job** flow, modelled on Pinecone
import / Milvus bulkinsert / BigQuery load jobs, lets a client stage a large
file **directly into object storage** via a presigned upload; a job then
validates and indexes it asynchronously. The bytes never flow through the API.

The small `POST .../vectors` endpoint is unchanged and remains the right tool
for small, low-latency upserts.

## Lifecycle

```
awaiting_upload ──(complete, object present)──▶ validating ──▶ indexing ──▶ completed
       │                                            │             │
       └──────────────── any stage ─────────────────┴─────────────┴──▶ failed
```

- `awaiting_upload` — job created, presigned upload target issued, no file yet.
- `validating` — `complete` was called, the validator is reading the upload.
- `indexing` — validation succeeded, the index builder is folding the batch in.
- `completed` — terminal; the shard is built and queryable.
- `failed` — terminal; `error_message` explains why (bad file, quota, etc.).

## Endpoints

### `POST /v1/datasets/{name}/imports` — create an import job

Auth required. Creates a job in `awaiting_upload` and returns a presigned
upload target.

**Request:**
```json
{ "format": "ndjson", "error_mode": "continue", "max_bad_records": 100 }
```

- `format` (required): `"ndjson"` or `"parquet"`.
- `error_mode` (optional, default `"continue"`): `"continue"` drops bad records
  and records them in a rejected-records file; `"abort"` fails the job on the
  first bad record.
- `max_bad_records` (optional, default `null` = unlimited): with
  `error_mode=continue`, if `records_rejected` exceeds this the job `failed`s.

**Response 201:**
```json
{
  "import_id": "imp_a1b2c3...",
  "dataset": "products",
  "status": "awaiting_upload",
  "format": "ndjson",
  "error_mode": "continue",
  "max_bad_records": 100,
  "upload": {
    "method": "PUT",
    "url": "https://minio.example/rosalinddb/staging/...?X-Amz-Signature=...",
    "content_type": "application/octet-stream",
    "max_bytes": 5368709120,
    "expires_at": "2026-05-15T13:34:56Z"
  },
  "created_at": "2026-05-15T12:34:56Z"
}
```

**Uploading.** The client does a single HTTP `PUT upload.url` with the staged
file as the **raw request body** — no multipart form, no fields — and a
`Content-Type` header set to exactly `upload.content_type`. A presigned PUT is
used (not POST) because Cloudflare R2 (one supported backend) does not
implement presigned POST. The presigned URL is signed for that exact
`Content-Type`; sending any other value (or omitting it) is rejected
`403 SignatureDoesNotMatch`.

A presigned PUT URL carries no upload policy, so it cannot cap the upload size
server-side the way a presigned-POST `content-length-range` condition did.
Instead, `max_bytes` is enforced after the fact: the import worker `head`s the
staged object when validation starts and fails the job (`status: failed`,
with an `error_message`) if the object is larger than `max_bytes`. Clients
should still check the file size against `max_bytes` before uploading to fail
fast.

**Errors:**
- `404 dataset_not_found` — dataset missing or not owned by the tenant.
- `400 invalid_format` / `400 invalid_error_mode` / `400 invalid_request`.
- `429 vector_quota_exceeded` — admission check: the tenant is already at/over
  its vector quota (see *Quota* below).

### `POST /v1/datasets/{name}/imports/{import_id}/complete` — signal upload done

Verifies the staged object is present in object storage, transitions the job
`awaiting_upload` → `validating`, and enqueues validation.

**Response 202:** the job object with `status: "validating"`.

**Errors:**
- `404 import_not_found` — unknown import (or another tenant's).
- `409 import_not_pending` — status is not `awaiting_upload`.
- `400 upload_missing` — the expected object is not present in storage.

### `GET /v1/datasets/{name}/imports/{import_id}` — job status

**Response 200:**
```json
{
  "import_id": "imp_a1b2c3...",
  "dataset": "products",
  "format": "ndjson",
  "status": "completed",
  "error_mode": "continue",
  "max_bad_records": 100,
  "records_processed": 10000,
  "records_accepted": 9998,
  "records_rejected": 2,
  "percent_complete": 100,
  "rejected_records_url": "https://minio.example/...&X-Amz-Signature=...",
  "error_message": null,
  "created_at": "2026-05-15T12:34:56Z",
  "completed_at": "2026-05-15T12:36:10Z"
}
```

- `percent_complete` — 0 (`awaiting_upload`), 25 (`validating`), 90
  (`indexing`), 100 (`completed`/`failed`).
- `rejected_records_url` — a presigned GET for the rejected-records file;
  present only when `records_rejected > 0`, else `null`.
- `error_message` — populated only when `status == "failed"`.
- `completed_at` — `null` until the job reaches a terminal state.

### `GET /v1/datasets/{name}/imports` — list a dataset's import jobs

**Response 200:** `{ "imports": [<job objects>] }`, newest first.

## Validation semantics

The validator reads the staged upload from landing storage.

### NDJSON

Each line is one JSON record `{"id", "values", "metadata"?}`. A record is valid
when `id` is a non-empty string, `values` has length == `dataset.dimension`,
and `metadata` (if present) is a JSON object. Valid records are written as
internal landing **Parquet**, which feeds the incremental indexer.

### Parquet

The uploaded file IS Parquet and must conform to RosalindDB's **internal
landing schema**:

| column     | type                                              | notes |
|------------|---------------------------------------------------|-------|
| `id`       | `string`                                          | non-empty |
| `values`   | `list<float>` of length == `dataset.dimension`    | `list`, `large_list`, or `fixed_size_list` of `float32`/`float64` |
| `metadata` | per-row JSON object (`struct`)                    | optional column |

A conforming Parquet file **skips the NDJSON→Parquet conversion** — it is used
directly, byte-for-byte, as a landing Parquet part. A non-conforming file
fails the whole job (Parquet cannot be partially conforming).

### Rejected records (`error_mode=continue`)

Bad records are dropped and appended to a rejected-records file in landing
storage at `imports/{import_id}/rejected.jsonl` — one JSON object per line:

```json
{ "line": 47, "reason": "dimension mismatch: got 3 expected 4", "record": "{...}" }
```

The `record` field is the offending record, truncated if larger than ~2 KiB.
If `max_bad_records` is set and `records_rejected` exceeds it, the job
`failed`s. With `error_mode=abort` the first bad record fails the job
immediately and nothing is indexed.

## Quota — two-stage

1. **Admission** (at create): if the tenant is already at/over its vector
   quota, create returns `429 vector_quota_exceeded`, before any object is
   staged. The staged-upload byte cap (`max_bytes`) is enforced separately by
   the import worker, which `head`s the staged object before validation.
2. **Settlement** (after validation): `records_accepted` is charged via
   `try_consume_vectors`. If the accepted count would cross the tenant's
   remaining quota, the job is aborted → `failed` with a quota `error_message`,
   and nothing is indexed.

> **Note:** admission and settlement are not atomic, so two or more
> imports running concurrently can each pass admission and transiently
> overshoot the quota before settlement fails the losers. This bounded,
> self-correcting overshoot is accepted.

## Storage layout

The *raw* client upload is staged in a dedicated `staging/` root — a sibling
of the landing root that the index builder never scans — so a raw
`upload.parquet` is never picked up as a landing part. Only the validator's
*produced* landing part lives under the landing prefix and is indexed.

For tenant `T`, dataset `D`, import `imp_X`:

```
staging/T/D/imports/imp_X/upload.ndjson      ← staged client upload (or .parquet)
landing/T/D/imports/imp_X/landing/part-<uuid>.parquet  ← validated landing part
landing/T/D/imports/imp_X/rejected.jsonl     ← rejected records (continue mode)
```

## Storage retention

Once an import is captured in a queryable shard, the index builder reclaims the
objects it produced (see `docs/indexing.md`, "Storage reclamation"):

- **Validated landing part** — deleted by the indexed-landing sweep once it is
  folded into a shard (recorded in the newest shard's `indexed_landing_uris`
  manifest). The manifest still records the URI, so re-running the import is
  still a clean no-op.
- **Staged raw client upload** (`staging/.../upload.*`) — deleted once the
  import job reaches a **terminal** status (`completed` or `failed`). Until
  then it is retained so the validator can read it (and a retry can re-read
  it). A `failed` job's raw upload is *not* kept for inspection — the
  `error_message` on the job is the record of what went wrong.
- **`rejected.jsonl`** — **retained.** Customers download it via the
  `rejected_records_url` presigned link *after* the job finishes, so it is left
  in place. It is not deleted by the post-build sweep; a separate time-based
  job prunes rejected sidecars after the retention window. Treat it as
  available for at least 30 days post-completion.

## Presigned upload — S3/MinIO/R2 vs the `memory://` test fake

The storage adapter's `presign_put(uri, expires)` returns `{"url", "method"}`
(`method` is always `"PUT"`); there is no `fields` dict because a presigned
PUT carries no upload policy:

- **S3 / MinIO / R2**: a real `generate_presigned_url("put_object", ...)`. The
  upload is a single cross-origin HTTP `PUT` of the raw file body straight to
  the storage host. Presigned PUT — not POST — is used because Cloudflare R2
  (one supported backend) does not implement presigned POST (it returns
  `501 NotImplemented`). Presigned PUT is supported on S3, MinIO and R2 alike.
- **`memory://`** (unit tests): a faithful fake — there is no HTTP server, so
  `url` is the `memory://...` object key itself. A test "uploads" by writing
  through the storage adapter, mirroring what a browser PUT to MinIO/R2 lands.

Because a presigned PUT URL cannot enforce a `content-length-range` size cap
server-side, the `max_bytes` cap is re-homed: the import worker `head`s the
staged object before validation and fails an oversized job.

## CORS

Because a browser `PUT`s the staged file **directly** to the storage host, the
bucket must allow the cross-origin preflight. In dev, the MinIO container sets
`MINIO_API_CORS_ALLOW_ORIGIN` to the localhost dashboard origins (see
`docker-compose.yml`). In production, set it (or the bucket CORS config) to
the real dashboard origin(s), allowing `PUT` and `OPTIONS`.
