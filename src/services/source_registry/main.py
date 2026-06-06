from __future__ import annotations

"""Source Registry service.

Hosts the customer-facing `/v1/datasets*` surface alongside the auth router.
Datasets are tenant-scoped (the auth dependency resolves `current_tenant_id`);
the `/how-to-connect` helper endpoint is an unauthenticated docs pointer.

Endpoints:
  - POST   /v1/datasets                      create an empty dataset
  - POST   /v1/datasets/{name}/vectors       stream NDJSON records into the dataset
  - GET    /v1/datasets                       list the tenant's datasets
  - GET    /v1/datasets/{name}               get a single dataset
  - DELETE /v1/datasets/{name}               soft-delete
  - GET    /v1/datasets/{name}/vectors/{id}  get one consolidated-tier vector by id
  - GET    /v1/datasets/{name}/vectors       list consolidated-tier vectors (filter + pagination)
  - DELETE /v1/datasets/{name}/vectors/{id}  delete one consolidated-tier vector by id
"""

import base64
import json
import logging
import math
import re
import uuid
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from adapters import config
from adapters.errors import error_envelope
from adapters.landing.parquet_reader import id_to_int64, read_shard_sidecar
from adapters.observability import init_observability
from adapters.observability.otel import instrument_fastapi
from adapters.observability import metrics as obs_metrics
from adapters.queue.queue import publish
from adapters.state import state as state_mod
from adapters.state.conn_middleware import RequestScopedConnectionMiddleware
from adapters.storage.storage import exists as storage_exists, presign_put, write_bytes
from services.auth.auth import (
    install_exception_handlers,
    install_pool_exhaustion_handler,
    router as auth_router,
)
from services.auth.jwt_utils import auth_required, current_tenant_id
from services.auth.quota import (
    install_rate_limit_handler,
    quotas_enabled,
    rate_limit,
    vector_quota_429,
)

# Recall-tier flag + sync write path. Default OFF — when off, nothing below is
# reached and `post_vectors` behaves byte-identically to today (202, landing
# write, VALIDATE_DATASET). See docs/architecture/recall-consolidate.md.
from adapters.state.state import (
    recall_delete_vector,
    recall_enabled,
    recall_get_vector,
    recall_get_vector_with_embedding,
    recall_list_rows,
    recall_partition_count,
    recall_upsert_vectors,
)


logger = logging.getLogger(__name__)

# Per-(tenant, dataset) recall-row cap. After a recall write, if the partition's
# row count exceeds this, a `CONSOLIDATE` is enqueued to flush the partition into
# a Consolidated shard — bounding the recall set the union brute-force-scans and
# keeping one tenant from evicting another's working set. Read live (per call)
# so a test can retune it without a reload; a missing/malformed value falls back
# to the default. Only consulted under `recall_enabled()`. See
# docs/architecture/recall-consolidate.md, "Scale-to-zero preservation".
_DEFAULT_RECALL_MAX_ROWS = 2000


def _recall_max_rows() -> int:
    """Return the per-(tenant, dataset) recall-row cap (`RB_RECALL_MAX_ROWS`)."""
    return config.recall_max_rows()


# Observability bootstrap. Runs at import so it works both when this module is
# the standalone uvicorn entrypoint AND when a single-process dev/test harness
# imports `app` (first caller wins; `init_observability` is idempotent).
# Default service name is overridden by `OTEL_SERVICE_NAME`.
init_observability("rosalinddb-source-registry")

app = FastAPI(title="Source Registry")
# FastAPI HTTP server traces + metrics (request count/duration by route+status).
instrument_fastapi(app)

# Bind ONE pooled Postgres connection per HTTP request so a request that calls
# N state functions costs one pool checkout, not N. A pure no-op in
# `memory://` mode (no pool). Added before the routes are exercised;
# `add_middleware` wraps the app, so it runs outermost-but-one (CORS stays
# outermost). `cp_app.py` reuses this `app`, so the CP inherits it for free.
app.add_middleware(RequestScopedConnectionMiddleware)

# CORS: a browser client calls this service cross-origin. In dev the
# Next dev server picks a free port (3000/3001/3002…) so we allow the whole
# localhost range. Prod origins are passed via the comma-separated
# CORS_ALLOW_ORIGINS env var. Auth is via Bearer token (Authorization header,
# not cookies), so credentials=False is correct.
_dev_origin_regex = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
_extra_origins = config.cors_allow_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_extra_origins,
    allow_origin_regex=_dev_origin_regex,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
    max_age=600,
)

# Mount the auth surface and rewrite HTTPException payloads into the v1
# `{"error": {"code", "message"}}` envelope.
app.include_router(auth_router, prefix="/auth")
install_exception_handlers(app)
# Map a `PoolCheckoutTimeout` escaping a handler to a v1 503 envelope
# (genuine sustained overload), never the bare 500 a raw fail-fast
# `PoolError` would become.
install_pool_exhaustion_handler(app)
# Turn a `RateLimited` raised by the `rate_limit` dependency into the v1
# `rate_limited` 429. The `/auth/*` surface is deliberately not rate-limited.
install_rate_limit_handler(app)


# --- OSS auth-disabled startup warning ------------------------------------
#
# When `RB_REQUIRE_AUTH` is unset/false (the headline self-host default) the
# auth/tenancy stack is bypassed: every request resolves to the bootstrap
# "default" tenant regardless of the `Authorization` header. A self-hoster
# running `docker compose up` on localhost is exactly the principal this is
# for — but if that deployment ever fronts a public URL without flipping
# `RB_REQUIRE_AUTH=true`, the entire dataset surface is open to the
# internet. Log it loudly, once, at process startup so an accidental public
# deploy is the loudest line in `docker compose logs` and any centralized
# log search can alert on it.
_AUTH_DISABLED_WARNING = (
    "Auth disabled (RB_REQUIRE_AUTH=false). API is open to anyone who can "
    "reach this process. Do NOT expose this deployment to the public "
    "internet without setting RB_REQUIRE_AUTH=true AND a stable JWT_SECRET "
    "(e.g. `openssl rand -hex 32`). Without a stable secret, every restart "
    "invalidates all existing tokens. See docs/deploy/self-host.md."
)


@app.on_event("startup")
def _oss_startup() -> None:
    """Per-process startup: bootstrap the default tenant + log the auth banner.

    The bootstrap call is idempotent (memory mode is a dict upsert, Postgres
    mode runs entirely inside `migrate()` / `scripts/migrate.py` so this is a
    no-op there) so it is safe to fire on every worker boot. The banner runs
    once per process at the WARNING level when auth is disabled — quiet when
    `RB_REQUIRE_AUTH=true`.
    """
    # Fail-fast on required-at-boot config (e.g. JWT_SECRET when auth is on)
    # before this process starts serving. Reads env fresh. This also covers the
    # Control Plane, which reuses this exact `app` (see control_plane.cp_app).
    config.validate()
    # Seed the "default" tenant row in memory mode. Postgres mode seeds it in
    # the migration runner (`scripts/migrate.py` -> `_apply_migrations`); this
    # call is a guarded no-op there.
    state_mod._bootstrap_default_tenant_memory()
    if not auth_required():
        logger.warning(_AUTH_DISABLED_WARNING)


@app.get("/healthz", include_in_schema=False)
def healthz():
    """Unauthenticated liveness probe.

    Returns 200 with a tiny JSON body and does NO DB/storage round-trip — it
    only proves the process is up and routing. `make smoke` and any post-deploy
    health gate hit this first. The `{"status": "ok", "service": ...}` shape is
    shared across every RosalindDB HTTP service for consistency. The service
    name is `control_plane` because this app is the Control Plane (the
    `source_registry` module name is an internal implementation detail; the
    CP reuses this app wholesale — see `services/control_plane/cp_app.py`).
    """
    return {"status": "ok", "service": "control_plane"}


@app.get("/how-to-connect")
def how_to_connect():
    """Return basic pointers for configuring access to object storage.

    Only schemes the storage adapter actually accepts are advertised here:
    ``s3://`` (S3, MinIO, R2, or any S3-compatible store) for read/write,
    and ``http(s)://`` for read-only public datasets. Other vendor schemes
    (``gs://``, ``az://``) are NOT supported — see ``adapters/storage/storage.py``.
    """
    return {
        "s3": (
            "Provide s3:// URIs. Works with AWS S3, MinIO, Cloudflare R2, and "
            "any S3-compatible store. Configure via the S3_ENDPOINT_URL, "
            "S3_ACCESS_KEY, S3_SECRET_KEY, and S3_REGION env vars; grant the "
            "credentials GetObject (and PutObject for writes) on your prefix."
        ),
        "http": (
            "Public HTTP(S) URLs are accepted read-only for external datasets "
            "(e.g. a public bucket served over https)."
        ),
    }


# --- v1 datasets surface --------------------------------------------------


_DATASET_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
_INGEST_MAX_BYTES = config.ingest_max_bytes()  # 10 MiB
# Largest finite float4 (single-precision). Embeddings are stored as pgvector
# `vector` (float4) in the recall tier and cast to float32 on the consolidated
# path, so a magnitude beyond this overflows to Infinity on storage. Validation
# rejects any value above it (or non-finite) per-line, identically in both flag
# modes.
_FLOAT4_MAX = 3.4028235e38
_LANDING_PREFIX = config.landing_prefix()
# Raw bulk-import uploads are staged OUTSIDE the dataset landing prefix so
# the index builder (which scans `landing/{tenant}/{dataset}/` recursively for
# `.parquet`) never sees a raw `upload.parquet` and double-indexes it. The
# staging root is a sibling of the landing root — same bucket, different
# top-level prefix the builder is never pointed at. Defaults to the landing
# root with its last path segment swapped to `staging`.
def _default_staging_prefix() -> str:
    base = config.landing_prefix().rstrip("/")
    head, _, _ = base.rpartition("/")
    return f"{head}/staging" if head else f"{base}-staging"


# Honor an explicitly-set empty STAGING_PREFIX as "" (original used
# os.getenv(..., default), so only an UNSET value derives the default).
_sp = config.staging_prefix()
_STAGING_PREFIX = _sp if _sp is not None else _default_staging_prefix()
_TENANT_PREFIX = config.tenant_prefix()

# --- bulk import limits ---------------------------------------------------
# The staged upload cap for the async import flow — far larger than the small
# `POST .../vectors` in-app cap because the bytes never touch the application;
# they go straight to object storage. A presigned-PUT URL cannot enforce this
# server-side (only a presigned-POST policy could, and presigned POST is not
# universally supported across S3-compatible backends — so the bulk-import
# flow uses presigned PUT). The import worker `head`s the staged object and
# fails the job if it exceeds this cap. Default 5 GiB; overridable per-deployment.
_IMPORT_MAX_BYTES = config.import_max_bytes()
# Presigned upload URL lifetime, seconds (default 1 hour).
_IMPORT_UPLOAD_TTL_S = config.import_upload_ttl_s()
_IMPORT_FORMATS = ("ndjson", "parquet")
_IMPORT_ERROR_MODES = ("continue", "abort")
_IMPORT_EXT = {"ndjson": "ndjson", "parquet": "parquet"}


def _err(status_code: int, code: str, message: str, details: Optional[dict] = None) -> JSONResponse:
    """Build a v1 error envelope response. Delegates to the canonical
    `adapters.errors.error_envelope` (same byte-for-byte body)."""
    return error_envelope(status_code, code, message, details)


def _dataset_response(row: dict) -> dict:
    """Project an internal dataset row down to the v1 `Dataset` shape.

    `last_indexed_at` is normalised to an ISO 8601 string (or None) so the
    JSON response matches the contract regardless of whether the value
    came from Postgres (`datetime`) or the in-memory adapter (`str`).
    """
    return {
        "name": row["dataset_name"],
        "dimension": int(row["dimension"]),
        "status": row.get("status", "empty"),
        "row_count": int(row.get("row_count", 0)),
        "created_at": _stringify_ts(row.get("created_at")),
        "last_indexed_at": _stringify_ts(row.get("last_indexed_at")) if row.get("last_indexed_at") else None,
        "error_message": row.get("error_message"),
    }


def _stringify_ts(value) -> str:
    """Coerce a timestamp value (datetime or str) to ISO 8601 UTC."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(value)


def _landing_prefix_for(tenant: str, dataset: str) -> str:
    """Compute landing prefix for a tenant/dataset (mirrors validator/builder)."""
    base = _LANDING_PREFIX
    if not base.endswith("/"):
        base += "/"
    if _TENANT_PREFIX:
        return f"{base}{tenant}/{dataset}"
    return f"{base}{dataset}"


def _staging_prefix_for(tenant: str, dataset: str) -> str:
    """Compute the staging prefix for a tenant/dataset's raw import uploads.

    A sibling of `_landing_prefix_for` rooted at `_STAGING_PREFIX` rather than
    the landing prefix — the index builder never scans this root, so a raw
    `upload.parquet` staged here is not picked up as a landing part.
    """
    base = _STAGING_PREFIX
    if not base.endswith("/"):
        base += "/"
    if _TENANT_PREFIX:
        return f"{base}{tenant}/{dataset}"
    return f"{base}{dataset}"


class _CreateDatasetRequest(BaseModel):
    """Pydantic shape for POST /v1/datasets.

    Both `name` and `dimension` are validated manually in the handler so
    failures map to the contract-spec error codes (`invalid_name`,
    `invalid_dimension`) rather than pydantic's generic envelope.
    """
    name: Optional[str] = None
    dimension: Optional[int] = None


@app.post("/v1/datasets", status_code=201)
async def create_dataset(
    request: Request,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """Create an empty dataset bound to the caller's tenant."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _err(400, "invalid_name", "Request body must be JSON")
    if not isinstance(body, dict):
        return _err(400, "invalid_name", "Request body must be a JSON object")

    name = body.get("name")
    if not isinstance(name, str) or not _DATASET_NAME_RE.match(name):
        return _err(400, "invalid_name", "name must be 1-64 chars matching [a-z0-9_-]+")

    dimension = body.get("dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        return _err(400, "invalid_dimension", "dimension must be a positive integer")

    # `state_mod.create_dataset` is a SYNC blocking Postgres write. Running it
    # inline in this `async def` would block the CP worker's event loop for the
    # INSERT round-trip, so under a burst the loop serialises and every other
    # request — including its own request-scoped connection still checked out —
    # stalls. Offload it to a worker thread. The request-scoped `_REQUEST_CONN`
    # contextvar is copied into the thread, so `pooled_conn()` inside still
    # reuses this request's bound connection (same discipline the conn
    # middleware and query proxy rely on).
    try:
        row = await run_in_threadpool(
            state_mod.create_dataset, tenant_id, name, dimension
        )
    except ValueError as exc:
        if str(exc) == "dataset_exists":
            return _err(409, "dataset_exists", f"Dataset '{name}' already exists for this tenant")
        raise
    # rosalinddb.datasets.created — no attributes (cardinality rule).
    obs_metrics.record_dataset_created()
    return JSONResponse(status_code=201, content=_dataset_response(row))


@app.get("/v1/datasets")
def list_datasets_endpoint(
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """List the caller's datasets (excludes soft-deleted)."""
    rows = state_mod.list_datasets(tenant_id)
    return {"datasets": [_dataset_response(r) for r in rows]}


@app.get("/v1/datasets/{name}")
def get_dataset_endpoint(
    name: str,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """Get a single dataset. Returns 404 for missing OR cross-tenant lookups."""
    row = state_mod.get_dataset(tenant_id, name)
    if row is None:
        return _err(404, "dataset_not_found", f"Dataset '{name}' not found")
    return _dataset_response(row)


@app.delete("/v1/datasets/{name}", status_code=204)
def delete_dataset_endpoint(
    name: str,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """Soft-delete a dataset. Subsequent GET → 404."""
    ok = state_mod.delete_dataset(tenant_id, name)
    if not ok:
        return _err(404, "dataset_not_found", f"Dataset '{name}' not found")
    return Response(status_code=204)


@app.post("/v1/datasets/{name}/vectors", status_code=202)
async def post_vectors(
    name: str,
    request: Request,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """Accept an NDJSON stream of vectors for `name`.

    Each line: `{"id": str, "values": [float], "metadata": object?}`. Records
    are validated (id non-empty, values length matches dataset.dimension,
    metadata absent or object). Accepted records are persisted to the
    landing area as a JSONL file and a `VALIDATE_DATASET` message is
    published — the validator does the canonical validation and writes
    parquet that the index_builder will read.

    Returns 202 with `{accepted, rejected, errors, job_id}`. The dataset's
    `status` flips through `validating` -> `indexing` -> `indexed` as the
    pipeline progresses; the caller polls via `GET /v1/datasets/{name}`.

    Recall tier (`RB_RECALL`, default OFF): when on, the write is instead
    SYNCHRONOUS — each accepted record is UPSERTed into the recall pgvector store
    (durable + immediately queryable) and the endpoint returns **200** with
    `{accepted, rejected, errors}` (no `job_id`, no landing write, no
    `VALIDATE_DATASET`). Validation is identical in both modes. The status code
    is the only flag-conditional difference; see docs/api/v1.md and
    docs/architecture/recall-consolidate.md.
    """
    dataset = state_mod.get_dataset(tenant_id, name)
    if dataset is None:
        return _err(404, "dataset_not_found", f"Dataset '{name}' not found")

    expected_dim = int(dataset["dimension"])

    # Read the body with a hard byte cap so we reject oversized payloads
    # before they consume memory. Using stream() lets us short-circuit
    # well before reading the whole body — important once payloads are
    # in the multi-MB range.
    body_chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > _INGEST_MAX_BYTES:
            return _err(
                413,
                "payload_too_large",
                f"Request body exceeds {_INGEST_MAX_BYTES} bytes",
            )
        body_chunks.append(chunk)
    body = b"".join(body_chunks)

    if not body.strip():
        return _err(400, "invalid_ndjson", "Empty NDJSON body")

    # Parse + per-line-validate + normalise the NDJSON body. For an OpenAI-
    # embedding-size batch (1536-dim, ~9 MB, ~300 records) this is hundreds of
    # ms of pure-Python CPU: a `utf-8` decode, then per-line `json.loads` /
    # `json.dumps` of large float arrays. Running it inline in this `async def`
    # would block the single CP worker's event loop for that whole window,
    # freezing every other request (a k6 1536-dim load sweep showed concurrent
    # small `POST /v1/datasets` creates timing out the 2.5s pool checkout into
    # 503s while ingests held the loop). Offload the CPU work to a worker
    # thread so the loop stays responsive.
    bad_utf8, accepted_lines, accepted_count, errors = await run_in_threadpool(
        _parse_ndjson_body, body, expected_dim
    )
    if bad_utf8:
        return _err(400, "invalid_ndjson", "body must be UTF-8")

    rejected_count = len(errors)
    job_id = "job_" + uuid4().hex

    if accepted_count == 0 and rejected_count == 0:
        return _err(400, "invalid_ndjson", "No valid NDJSON records found")

    # Enforce the per-tenant vector quota *before* anything is persisted or
    # published. `vectors_used` is incremented by `accepted_count` — the count
    # of records that passed this service's per-line validation. The canonical
    # validator may reject a few more downstream, so this can slightly overcount;
    # that is an accepted tradeoff for a default cap (documented in
    # docs/api/quotas.md). Quota enforcement is all-or-nothing: if the upload
    # would cross the cap the WHOLE upload is rejected (no partial acceptance)
    # and nothing lands or is published.
    # OSS opt-in: skipped entirely when `RB_ENABLE_QUOTAS` is unset/false (the
    # self-host default). The counter row stays — only enforcement is gated.
    if accepted_count > 0 and quotas_enabled():
        ok, usage = state_mod.try_consume_vectors(tenant_id, accepted_count)
        if not ok:
            # rosalinddb.ingest.uploads{outcome=rejected} + quota.rejections{kind=vector}.
            obs_metrics.record_upload("rejected")
            obs_metrics.record_quota_rejection("vector")
            return vector_quota_429(usage)

    # --- recall tier (RB_RECALL): synchronous recall-tier write path ------
    #
    # When the recall tier is on, the write is SYNCHRONOUS: each accepted record
    # is assigned a per-(tenant, dataset) LSN and UPSERTed into the recall
    # pgvector store (last-write-wins), making it durable and immediately
    # queryable. We then return 200 — NOT 202 — because there is nothing async to
    # wait for.
    #
    # In this mode we deliberately do NOT write a landing object and do NOT
    # publish `VALIDATE_DATASET`: the recall→consolidated consolidation is a later
    # PR, so until then flag-on data lives in the recall tier only (acceptable —
    # the flag defaults off and the full recall tier is not user-complete until
    # consolidation ships). The per-line validation above is identical to the
    # flag-off path, so dimension / id / metadata rules are unchanged. Body shape
    # is unchanged; `job_id` is omitted (no async job exists). See
    # docs/architecture/recall-consolidate.md, "Write path".
    if recall_enabled():
        if accepted_count > 0:
            # `accepted_lines` are canonical, already-validated NDJSON strings
            # (id/values/metadata). Parse them back to records for the UPSERT —
            # the recall write needs the structured embedding, not the wire line.
            records = [json.loads(line) for line in accepted_lines]
            # The recall UPSERT is a SYNC psycopg2 batch round-trip against the
            # separate recall instance; offload it so the CP event loop stays
            # responsive under a burst (same discipline as the landing write).
            #
            # The recall store is a SEPARATE data-plane instance (RB_RECALL_DSN):
            # a connection drop, statement timeout, or constraint failure
            # surfaces as a psycopg2 error here. Map any such failure into the v1
            # error envelope (503 recall_write_failed) so the contract holds
            # instead of a raw 500 leaking outside `{error:{code,message}}`. The
            # whole batch is one transaction, so a failure leaves the recall tier
            # unchanged.
            #
            # KNOWN FOLLOW-UP (quota leak): `try_consume_vectors` above already
            # COMMITTED the quota increment before this write. If the recall write
            # fails, that quota stays consumed (no refund). Refund/compensation
            # is deliberately out of scope for this PR — tracked as a follow-up.
            try:
                await run_in_threadpool(
                    recall_upsert_vectors, tenant_id, name, records
                )
            except Exception:  # noqa: BLE001 - any recall-store failure -> 503
                logger.exception(
                    "recall-tier write failed for tenant=%s dataset=%s",
                    tenant_id,
                    name,
                )
                obs_metrics.record_upload("rejected")
                return _err(
                    503,
                    "recall_write_failed",
                    "Recall-tier write failed; the batch was not persisted",
                )

            # Per-tenant recall cap: if this partition now exceeds
            # `RB_RECALL_MAX_ROWS`, enqueue a `CONSOLIDATE` so the builder flushes
            # it into a Consolidated shard. This bounds the recall set the query
            # union brute-force-scans (the union PR's otherwise-unbounded scan)
            # and stops one tenant evicting another's working set. Best-effort
            # AFTER the durable write: a count/enqueue failure must NOT fail the
            # already-committed write (the next write — or the idle sweep — re-
            # checks and re-enqueues), so it never turns a 200 into an error.
            try:
                if recall_partition_count(tenant_id, name) > _recall_max_rows():
                    await run_in_threadpool(
                        publish, "CONSOLIDATE", {"tenant": tenant_id, "dataset": name}
                    )
            except Exception:  # noqa: BLE001 - cap check is best-effort
                logger.warning(
                    "recall cap check/enqueue failed for tenant=%s dataset=%s "
                    "(write already committed; will retry next write/idle sweep)",
                    tenant_id,
                    name,
                    exc_info=True,
                )
        obs_metrics.record_upload("accepted" if accepted_count > 0 else "rejected")
        obs_metrics.record_vectors_ingested(accepted_count)
        return JSONResponse(
            status_code=200,
            content={
                "accepted": accepted_count,
                "rejected": rejected_count,
                "errors": errors,
            },
        )

    if accepted_count > 0:
        # Persist as a uniquely-named JSONL so successive uploads coexist
        # in the landing area (the validator reads JSONL via the storage
        # adapter and writes parquet per-upload into a sub-prefix).
        upload_id = uuid.uuid4().hex[:12]
        landing_uri = f"{_landing_prefix_for(tenant_id, name)}/uploads/upload-{upload_id}.jsonl"
        landing_bytes = ("\n".join(accepted_lines) + "\n").encode("utf-8")

        # `write_bytes` is a SYNC boto3 `put_object` of a multi-MB landing
        # object and `publish` is a SYNC Redis call. Running them directly in
        # this `async def` would block the event loop for the whole — possibly
        # multi-second — object-storage write, stalling every other request the
        # CP worker is serving (the CP runs a single uvicorn worker / one event
        # loop). Offload the blocking I/O to a worker thread so a slow ~9 MB
        # ingest does not freeze the loop. (Identified by a k6 load sweep at
        # 1536-dim ingest sizes — see conn_middleware's vector-upload note.)
        await run_in_threadpool(write_bytes, landing_uri, landing_bytes)

        await run_in_threadpool(
            publish,
            "VALIDATE_DATASET",
            {
                "dataset": name,
                "tenant": tenant_id,
                "uri": landing_uri,
                "file_type": "jsonl",
                "job_id": job_id,
            },
        )

    # rosalinddb.ingest.uploads + rosalinddb.vectors.ingested. `outcome` is
    # `accepted` when at least one record landed, else `rejected`. Only the
    # low-cardinality `outcome` is attached — no tenant/dataset labels.
    obs_metrics.record_upload("accepted" if accepted_count > 0 else "rejected")
    obs_metrics.record_vectors_ingested(accepted_count)

    return JSONResponse(
        status_code=202,
        content={
            "accepted": accepted_count,
            "rejected": rejected_count,
            "errors": errors,
            "job_id": job_id,
        },
    )


# --- consolidated-tier vector CRUD (+ recall union, RB_RECALL) -------------
#
# Get / list / delete a single vector by its customer-supplied string id,
# served from the immutable consolidated shards (the FAISS index + its
# `.meta.json` sidecar).
#
# When the recall tier is ON (`RB_RECALL`, default OFF), these endpoints UNION
# the consolidated sidecar with the recall tier, with RECALL AUTHORITATIVE for any id
# above the resolved shard's watermark — the same recall-wins / tombstone-
# suppress rule the QUERY union uses (`recall_search` + `_merge_recall_and_consolidated`):
#   - get: a live recall row (lsn > watermark) wins (recall metadata); a recall
#     tombstone → 404 (deleted); otherwise fall back to the consolidated sidecar.
#   - list: union recall live rows with the consolidated sidecar (recall-wins dedup),
#     suppress any id with a recall tombstone above the watermark, then filter +
#     sort + paginate as before.
#   - delete: write an ABOVE-watermark TOMBSTONE into recall (fresh lsn) and
#     return 200/204 synchronously (read-your-deletes); consolidation applies it
#     to the consolidated tier later. Does NOT publish DELETE_VECTORS in flag-on mode.
# With the flag OFF every endpoint is byte-identical to the consolidated-only path and
# NEVER opens a recall connection. See docs/architecture/recall-consolidate.md
# (PR6) + docs/api/vectors.md.
#
# The sidecar maps each SHA1->int64 hash (the same `id_to_int64` the builder
# stamps onto every FAISS vector) back to `{id, metadata}`. Get/delete hash
# the incoming string id to that int64 and look it up; list reads the whole
# sidecar. `?include_values` (FAISS reconstruct to return the raw vector) is a
# noted follow-up — v1 returns metadata only.

# Default and max page size for the list endpoint. The cap mirrors the
# `MAX_TOP_K`-style ceiling pattern: an unbounded `limit` would let one
# request materialise an entire shard's sidecar into the response body.
_VECTORS_LIST_DEFAULT_LIMIT = 100
_VECTORS_LIST_MAX_LIMIT = 1000


def _encode_cursor(offset: int) -> str:
    """Encode a list offset into an opaque base64 cursor.

    The cursor is deliberately opaque (a base64-encoded JSON `{"o": N}`) so
    the offset scheme is not part of the public contract and can change later
    (e.g. to a keyset cursor) without breaking clients. The list is stably
    sorted by original id, so a plain offset is a correct, simple paginator at
    MVP scale.
    """
    return base64.urlsafe_b64encode(json.dumps({"o": int(offset)}).encode("utf-8")).decode("ascii")


def _decode_cursor(cursor: str) -> Optional[int]:
    """Decode an opaque cursor back to its offset, or None if malformed.

    Returns None (the caller maps it to `400 invalid_cursor`) on any decode /
    shape error so a hand-edited or truncated cursor fails loudly rather than
    silently restarting pagination from zero.
    """
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
        offset = int(data["o"])
        if offset < 0:
            return None
        return offset
    except Exception:  # noqa: BLE001 - any malformed cursor is a 400
        return None


def _newest_sidecar(tenant_id: str, dataset: str) -> Optional[dict]:
    """Return the newest shard's parsed `.meta.json` sidecar, or None.

    None means there is no shard yet for `(tenant_id, dataset)` — the caller
    treats that as an empty consolidated tier (404 for get/delete-by-id, empty
    list for list). Tenant-scoped via `get_latest_shard`.
    """
    shard = state_mod.get_latest_shard(tenant_id, dataset)
    if shard is None:
        return None
    return read_shard_sidecar(shard["shard_uri"])


def _resolve_shard_sidecar_and_watermark(
    tenant_id: str, dataset: str
) -> tuple[Optional[dict], int]:
    """Resolve the newest shard ONCE; return `(sidecar_or_None, watermark)`.

    The recall-union CRUD paths need BOTH the consolidated sidecar and the recall
    watermark, and invariant I3 (watermark/shard pairing) requires they come from
    the SAME resolved shard — never a sidecar from one shard and a watermark read
    from an independent lookup. This resolves `get_latest_shard` once and derives
    both: the sidecar (None when no shard exists yet) and the watermark (the
    shard's `consolidated_lsn`, or `0` when no shard exists so EVERY recall row
    qualifies — a brand-new dataset's writes live only in recall).
    """
    shard = state_mod.get_latest_shard(tenant_id, dataset)
    if shard is None:
        return None, 0
    return read_shard_sidecar(shard["shard_uri"]), _watermark_of(shard)


def _watermark_of(shard: Optional[dict]) -> int:
    """Recall watermark = the resolved shard's `consolidated_lsn` (I3), else 0.

    Mirrors `services.query_api.v1_query._watermark_for_shard`: `0` when no shard
    (all recall rows qualify), and a missing/None `consolidated_lsn` (a pre-008
    or memory-mode shard row) defaults to `0` — backward-compatible with the
    `NOT NULL DEFAULT 0` column.
    """
    if not shard:
        return 0
    raw = shard.get("consolidated_lsn", 0)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _is_truthy_query_flag(value: Optional[str]) -> bool:
    """Parse a query-string boolean flag (`?include_values=true`).

    Treats `true`/`1`/`yes`/`on` (case-insensitive), and a bare valueless flag
    (`?include_values`, which FastAPI surfaces as the empty string), as True.
    Everything else (absent → `None`, `false`, `0`, ...) is False.
    """
    if value is None:
        return False
    return value.strip().lower() in ("", "true", "1", "yes", "on")


@app.get("/v1/datasets/{name}/vectors/{vector_id}")
def get_vector_endpoint(
    name: str,
    vector_id: str,
    include_values: Optional[str] = None,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """Get one vector by id. Returns `{id, metadata}` (`{id, metadata, embedding}`
    when `?include_values=true` and the vector is recall-resident).

    Resolves the newest shard, reads its sidecar, hashes `vector_id` with the
    builder's SHA1->int64, and looks it up. `404 not_found` when there is no
    shard yet or the id is absent. A missing/cross-tenant dataset is
    `404 dataset_not_found` (tenant scoping is enforced by `get_dataset` /
    `get_latest_shard`).

    `?include_values=true` (the cheap recall path): a LIVE recall row's stored
    `embedding` is a plain `vector` COLUMN on `recall_vectors`, so it is returned
    via a single SELECT (no FAISS). This backs the mem0 adapter's metadata-only
    `update`, which must re-upsert WITHOUT clobbering the real embedding. For a
    CONSOLIDATED id, returning values would require a FAISS
    `reconstruct` against the shard — deferred — so `embedding` is OMITTED from
    the response (the caller treats the absence as "not recall-resident"). The
    metadata-only response shape is unchanged when `include_values` is unset.

    Recall union (`RB_RECALL`, default OFF): when on, the recall tier is
    AUTHORITATIVE for any id above the resolved shard's watermark. A LIVE recall
    row (`lsn > watermark`) wins → its `{id, metadata}` is returned (recall-wins);
    a recall TOMBSTONE → `404 not_found` (the id was deleted, never fall back to
    the stale consolidated copy); otherwise fall back to the consolidated sidecar lookup below.
    Flag OFF → consolidated-only (byte-identical). See
    docs/architecture/recall-consolidate.md (PR6).
    """
    if state_mod.get_dataset(tenant_id, name) is None:
        return _err(404, "dataset_not_found", f"Dataset '{name}' not found")

    want_values = _is_truthy_query_flag(include_values)

    # Recall union. Resolve the shard ONCE so the recall watermark and the consolidated
    # sidecar come from the SAME shard (I3 pairing). With the flag off this whole
    # branch is skipped and no recall connection is ever opened.
    if recall_enabled():
        sidecar, watermark = _resolve_shard_sidecar_and_watermark(tenant_id, name)
        if want_values:
            # `include_values` recall path: also read the stored embedding column.
            status, metadata, embedding = recall_get_vector_with_embedding(
                tenant_id, name, vector_id, watermark
            )
            if status == "live":
                resp = {"id": vector_id, "metadata": metadata or {}}
                if embedding is not None:
                    resp["embedding"] = embedding
                return resp
            if status == "tombstone":
                return _err(404, "not_found", f"Vector '{vector_id}' not found")
            # status is None: not recall-resident — fall through to the consolidated lookup.
        else:
            status, metadata = recall_get_vector(tenant_id, name, vector_id, watermark)
            if status == "live":
                # Recall-wins: a live row above the watermark is the authoritative
                # version, newer than any consolidated copy.
                return {"id": vector_id, "metadata": metadata or {}}
            if status == "tombstone":
                # Deleted in recall — authoritative; do NOT fall back to the consolidated copy.
                return _err(404, "not_found", f"Vector '{vector_id}' not found")
            # status is None: no recall row above the watermark — fall through to
            # the consolidated sidecar (the id, if any, is consolidated below the watermark).
    else:
        sidecar = _newest_sidecar(tenant_id, name)

    if not sidecar:
        return _err(404, "not_found", f"Vector '{vector_id}' not found")
    entry = sidecar.get(str(id_to_int64(vector_id)))
    if entry is None:
        return _err(404, "not_found", f"Vector '{vector_id}' not found")
    # Consolidated-only hit. `?include_values` cannot reconstruct the embedding here (a
    # FAISS `reconstruct` is deferred), so the response stays metadata-only — the
    # absent `embedding` signals "not recall-resident" to the caller.
    return {"id": entry.get("id", vector_id), "metadata": entry.get("metadata") or {}}


@app.get("/v1/datasets/{name}/vectors")
def list_vectors_endpoint(
    name: str,
    limit: Optional[str] = None,
    cursor: Optional[str] = None,
    filter: Optional[str] = None,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """List vectors with an optional `filter` and pagination.

    Reads the newest shard's sidecar, applies an optional AND-of-equals
    `filter` (a JSON object, reusing the query path's `metadata_matches_filter`
    so the semantics match `POST /v1/query`), stably sorts by original id, and
    paginates with `limit` (default 100, capped at 1000) and an opaque
    `cursor`. Returns `{vectors: [{id, metadata}], next_cursor}`; an empty list
    when no shard exists. Tenant-scoped via `get_dataset` / `get_latest_shard`.

    Recall union (`RB_RECALL`, default OFF): when on, the consolidated sidecar entries are
    unioned with the recall tier's LIVE rows above the resolved shard's watermark,
    RECALL-WINS on a shared id (recall metadata overrides the consolidated copy), and any
    id with a recall TOMBSTONE above the watermark is SUPPRESSED from the list —
    the same recall-authoritative dedup the QUERY union uses. The union happens
    BEFORE the metadata `filter`, so the filter sees each id's authoritative
    (recall-or-consolidated) metadata. Flag OFF → consolidated-only (byte-identical). See
    docs/architecture/recall-consolidate.md (PR6) + docs/api/vectors.md.

    Pagination contract (offset cursor — read this before paging):
    `next_cursor` encodes ONLY the offset into the id-sorted result; it does
    NOT capture the active `filter` or `limit`. A continuation request MUST
    resend the SAME `filter` and `limit` it used for the first page. Changing
    either mid-pagination re-sorts/re-filters a different result set under the
    old offset and silently skips or duplicates rows. The offset is also
    resolved against the NEWEST shard at request time, so a concurrent rebuild
    (ingest/delete) that produces a new shard generation between pages can shift
    rows under a stable offset — pagination is eventually-consistent, not a
    snapshot. v1 deliberately keeps the simple offset cursor (a keyset cursor is
    a later option) and documents this contract instead. See docs/api/vectors.md.

    `limit` is validated manually (not via FastAPI's typed query param) so a
    non-integer value returns the v1 `{error:{code,message}}` envelope
    (`400 invalid_limit`) rather than FastAPI's generic 422, mirroring how
    `POST /v1/query` validates `top_k`/`nprobe`.
    """
    if state_mod.get_dataset(tenant_id, name) is None:
        return _err(404, "dataset_not_found", f"Dataset '{name}' not found")

    # Manual `limit` coercion → v1 envelope. Absent/blank → default; a
    # non-integer or out-of-range value → 400 invalid_limit (NOT FastAPI 422).
    if limit is None or limit == "":
        limit_val = _VECTORS_LIST_DEFAULT_LIMIT
    else:
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return _err(400, "invalid_limit", "limit must be a positive integer")
    if limit_val < 1:
        return _err(400, "invalid_limit", "limit must be a positive integer")
    limit = min(limit_val, _VECTORS_LIST_MAX_LIMIT)

    offset = 0
    if cursor is not None:
        decoded = _decode_cursor(cursor)
        if decoded is None:
            return _err(400, "invalid_cursor", "cursor is malformed")
        offset = decoded

    flt: Optional[dict] = None
    if filter is not None:
        try:
            flt = json.loads(filter)
        except (ValueError, TypeError):
            return _err(400, "invalid_filter", "filter must be a JSON object")
        if not isinstance(flt, dict):
            return _err(400, "invalid_filter", "filter must be a JSON object")

    # Resolve the consolidated sidecar and (under the union) the recall partition above
    # the SAME resolved shard's watermark (I3 pairing). With the flag off this is
    # a plain sidecar read and no recall connection is ever opened.
    if recall_enabled():
        sidecar, watermark = _resolve_shard_sidecar_and_watermark(tenant_id, name)
        sidecar = sidecar or {}
        recall_live, recall_suppress_ids = recall_list_rows(tenant_id, name, watermark)
    else:
        sidecar = _newest_sidecar(tenant_id, name) or {}
        recall_live, recall_suppress_ids = [], set()

    # Project the consolidated sidecar to {id, metadata}, DROPPING any id recall is
    # authoritative for (`recall_suppress_ids`) — a recall tombstone hides the
    # consolidated id, and a recall live row replaces the consolidated copy with the recall
    # version appended below (recall-wins dedup, mirroring the query union).
    records = [
        {"id": e.get("id"), "metadata": e.get("metadata") or {}}
        for e in sidecar.values()
        if e.get("id") is not None and e.get("id") not in recall_suppress_ids
    ]
    # Append the recall LIVE rows (their ids were suppressed from the consolidated set
    # above, so this is the only copy of each — no duplicate). Empty when the
    # flag is off.
    records.extend(recall_live)
    if flt:
        # Reuse the query path's AND-of-equals predicate so list/query agree.
        from services.query_api.v1_query import metadata_matches_filter

        records = [r for r in records if metadata_matches_filter(r["metadata"], flt)]
    records.sort(key=lambda r: r["id"])

    page = records[offset : offset + limit]
    next_cursor = (
        _encode_cursor(offset + limit) if offset + limit < len(records) else None
    )
    return {"vectors": page, "next_cursor": next_cursor}


@app.delete(
    "/v1/datasets/{name}/vectors/{vector_id}",
    status_code=202,
    # The success code is FLAG-CONDITIONAL: 204 (No Content, synchronous
    # read-your-deletes) on the recall path (`RB_RECALL` on), 202 (`{job_id}`,
    # eventually-consistent builder delete) on the default consolidated-only path. The
    # declared `status_code=202` is the default; the recall path returns an
    # explicit `Response(204)` at runtime. Document BOTH in the OpenAPI schema so
    # the generated contract reflects the real flag-conditional behaviour rather
    # than advertising only 202.
    responses={
        204: {"description": "Recall path (RB_RECALL on): tombstone written "
                             "synchronously; no body (read-your-deletes)."},
        202: {"description": "Default path: DELETE_VECTORS enqueued; returns "
                             "`{job_id}` (eventually consistent)."},
    },
)
def delete_vector_endpoint(
    name: str,
    vector_id: str,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """Delete one vector by id. Returns `202 {job_id}` (flag off) or `204` (recall).

    Flag OFF (default — eventually consistent via the builder): publishes a
    `DELETE_VECTORS` message; the index builder's consumer loads the newest shard,
    removes the hashed id, rewrites the sidecar without it, and writes a superseded
    shard. Tenant-scoped — a missing/cross-tenant dataset is `404 dataset_not_found`.

    Status handling is shard-aware so a delete never masks a dataset's true
    state. The status is flipped to `indexing` ONLY when a shard actually
    exists for the dataset (the in-flight delete will then produce the next
    shard generation and the builder flips it back to `indexed`). When there is
    NO shard the delete is a guaranteed no-op — there is nothing to reindex —
    so the status is left untouched: deleting on a never-ingested (`empty`) or
    failed (`error`) dataset must keep reporting `empty`/`error`, not be
    rewritten to `indexing`/`indexed` with `row_count=0`. The per-dataset
    advisory lock the builder holds serializes this delete against any
    concurrent build, so the shard-existence check is not racy in a way that
    can clobber a real status.

    The delete is accepted optimistically (202) without first proving the id
    exists in the shard: the builder treats an absent id as a clean no-op, and
    this keeps the endpoint a single cheap publish (plus, when a shard exists,
    one status flip) rather than a synchronous shard read on the request path.
    This mirrors the eventual-consistency contract of `POST .../vectors`.

    Recall (`RB_RECALL`, default OFF — SYNCHRONOUS, read-your-deletes): writes a
    TOMBSTONE into the recall tier with a FRESH lsn allocated ABOVE the watermark
    (`recall_delete_vector`), so the union immediately hides the id (an immediate
    GET → 404, a `POST /query` no longer returns it) and the next consolidation
    removes it from the consolidated tier. Returns `204` — there is no async job. It does NOT
    publish `DELETE_VECTORS`: consolidation applies the tombstone to the consolidated tier
    later (that machinery already exists), so a builder delete-rebuild here would
    be redundant. THE ABOVE-WATERMARK LSN IS A HARD CONTRACT (a below-watermark
    tombstone would be excluded from the union AND trim-eligible-unapplied — the
    id would resurrect); see docs/architecture/recall-consolidate.md, invariants
    I1/I2. A recall-store failure maps to the v1 `503 recall_delete_failed`
    envelope, never a raw 500.
    """
    dataset = state_mod.get_dataset(tenant_id, name)
    if dataset is None:
        return _err(404, "dataset_not_found", f"Dataset '{name}' not found")

    # --- recall-delete write path (RB_RECALL): synchronous tombstone ----------
    if recall_enabled():
        try:
            recall_delete_vector(
                tenant_id, name, vector_id, int(dataset["dimension"])
            )
        except Exception:  # noqa: BLE001 - any recall-store failure -> 503
            logger.exception(
                "recall-tier delete failed for tenant=%s dataset=%s id=%s",
                tenant_id,
                name,
                vector_id,
            )
            return _err(
                503,
                "recall_delete_failed",
                "Recall-tier delete failed; the vector was not tombstoned",
            )
        # Per-tenant recall cap (liveness): a delete WRITES a tombstone row, which
        # counts against the partition like a live row, so a delete-heavy workload
        # would otherwise accumulate tombstones unbounded between idle sweeps and
        # bloat the brute-force recall scan. Mirror the write path: if this
        # partition now exceeds `RB_RECALL_MAX_ROWS`, enqueue a `CONSOLIDATE` so
        # the builder flushes it (folding live rows + APPLYING tombstones) into a
        # Consolidated shard. Best-effort AFTER the durable tombstone write: a
        # count/enqueue failure must NEVER turn the already-committed delete into
        # an error (the next write/delete — or the idle sweep — re-checks and
        # re-enqueues), so it never turns the 204 into a 5xx.
        try:
            if recall_partition_count(tenant_id, name) > _recall_max_rows():
                publish("CONSOLIDATE", {"tenant": tenant_id, "dataset": name})
        except Exception:  # noqa: BLE001 - cap check is best-effort
            logger.warning(
                "recall cap check/enqueue failed on delete for tenant=%s "
                "dataset=%s (tombstone already committed; will retry next "
                "write/delete/idle sweep)",
                tenant_id,
                name,
                exc_info=True,
            )
        # Read-your-deletes: the tombstone is committed; the union hides the id
        # immediately. No async job → 204 No Content (no body).
        return Response(status_code=204)

    job_id = "job_" + uuid4().hex
    publish(
        "DELETE_VECTORS",
        {"dataset": name, "tenant": tenant_id, "id": vector_id, "job_id": job_id},
    )
    # Only flip to `indexing` when a shard exists. With no shard the delete is
    # a no-op against the consolidated tier — flipping status here would falsely report
    # `indexing`/`indexed` on an `empty`/`error` dataset and mask its real state.
    if state_mod.get_latest_shard(tenant_id, name) is not None:
        state_mod.update_dataset_status(tenant_id, name, "indexing")
    return JSONResponse(status_code=202, content={"job_id": job_id})


# --- bulk import surface --------------------------------------------------
#
# Async import-job flow modelled on Pinecone import / Milvus bulkinsert /
# BigQuery load jobs: the client stages a large NDJSON/Parquet file directly
# into object storage via a presigned upload, then a job validates + indexes
# it asynchronously. The small `POST .../vectors` endpoint above is unchanged
# and stays the right tool for tiny interactive upserts.


def _import_upload_uri(tenant: str, dataset: str, import_id: str, fmt: str) -> str:
    """Deterministic key for an import's *raw* staged upload object.

    This lives under the dedicated **staging** root
    (`staging/{tenant}/{dataset}/imports/{import_id}/upload.<ext>`), NOT the
    dataset landing prefix. The index builder scans the landing prefix
    recursively for `.parquet` parts; staging a raw `upload.parquet` inside the
    landing prefix would cause every Parquet import to be indexed twice (once
    for the raw upload, once for the validator's produced landing part). Keeping
    the raw upload in a sibling root the builder never scans avoids that. The
    validator's *produced* landing part still goes under the landing prefix.
    """
    ext = _IMPORT_EXT.get(fmt, "bin")
    return f"{_staging_prefix_for(tenant, dataset)}/imports/{import_id}/upload.{ext}"


def _import_response(row: dict, include_upload: bool = False) -> dict:
    """Project an `import_jobs` row to the v1 import-job response shape.

    `include_upload` is True only on the create (201) response — that is the
    one moment the presigned upload target is handed back. Subsequent GETs do
    not re-mint it (the object is staged once).
    """
    rejected = int(row.get("records_rejected", 0))
    status = row.get("status", "awaiting_upload")
    body = {
        "import_id": row["import_id"],
        "dataset": row["dataset"],
        "format": row["format"],
        "status": status,
        "error_mode": row.get("error_mode", "continue"),
        "max_bad_records": row.get("max_bad_records"),
        "records_processed": int(row.get("records_processed", 0)),
        "records_accepted": int(row.get("records_accepted", 0)),
        "records_rejected": rejected,
        "percent_complete": _import_percent(row),
        "rejected_records_url": (
            _presign_rejected(row) if rejected > 0 and row.get("rejected_uri") else None
        ),
        "error_message": row.get("error_message") if status == "failed" else None,
        "created_at": _stringify_ts(row.get("created_at")),
        "completed_at": (
            _stringify_ts(row.get("completed_at")) if row.get("completed_at") else None
        ),
    }
    if include_upload:
        target = presign_put(row["upload_uri"], _IMPORT_UPLOAD_TTL_S)
        body["upload"] = {
            "method": "PUT",
            "url": target["url"],
            "content_type": target["content_type"],
            "max_bytes": _IMPORT_MAX_BYTES,
            "expires_at": _expires_at(_IMPORT_UPLOAD_TTL_S),
        }
    return body


def _import_percent(row: dict) -> int:
    """Map a job's lifecycle state to a 0-100 integer progress value.

    `awaiting_upload`/`validating` are pre-completion stages; `completed` and
    `failed` are terminal at 100. `indexing` is reported as 90 — validation
    (the bulk of the work) is done, the shard build is the final step.
    """
    status = row.get("status", "awaiting_upload")
    return {
        "awaiting_upload": 0,
        "validating": 25,
        "indexing": 90,
        "completed": 100,
        "failed": 100,
    }.get(status, 0)


def _presign_rejected(row: dict) -> Optional[str]:
    """Presigned GET URL for the import's rejected-records file."""
    from adapters.storage.storage import presign_get

    return presign_get(row["rejected_uri"], _IMPORT_UPLOAD_TTL_S)


def _expires_at(ttl_s: int) -> str:
    """ISO 8601 UTC timestamp `ttl_s` seconds in the future."""
    import datetime as _dt

    return (_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=ttl_s)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


@app.post("/v1/datasets/{name}/imports", status_code=201)
async def create_import(
    name: str,
    request: Request,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """Create an async bulk-import job and return a presigned upload target.

    The job starts in `awaiting_upload`. The response carries an `upload`
    object `{"method": "PUT", "url", "content_type", "max_bytes",
    "expires_at"}`. The client stages its NDJSON/Parquet file directly into
    object storage by doing a single `PUT upload.url` with the file as the raw
    request body (no multipart form, no fields) and a `Content-Type` header
    set to `upload.content_type` — the presigned URL is signed for that exact
    Content-Type, so any other value is rejected `403`. It then calls
    `.../complete` to kick off validation + indexing.

    Presigned PUT — not POST — is used because presigned PUT is universally
    supported across S3-compatible backends (S3, MinIO, R2, …), whereas
    presigned POST is not. A PUT URL carries no upload policy, so it cannot
    cap the upload size server-side; the import worker enforces `max_bytes`
    instead by `head`ing the staged object.

    Two-stage quota: this is the admission check — a tenant already at/over
    its vector quota is rejected 429 here, before any object is staged. Final
    settlement (`try_consume_vectors`) happens after validation.
    """
    dataset = state_mod.get_dataset(tenant_id, name)
    if dataset is None:
        return _err(404, "dataset_not_found", f"Dataset '{name}' not found")

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _err(400, "invalid_request", "Request body must be JSON")
    if not isinstance(body, dict):
        return _err(400, "invalid_request", "Request body must be a JSON object")

    fmt = body.get("format")
    if fmt not in _IMPORT_FORMATS:
        return _err(400, "invalid_format", f"format must be one of {list(_IMPORT_FORMATS)}")

    error_mode = body.get("error_mode", "continue")
    if error_mode not in _IMPORT_ERROR_MODES:
        return _err(
            400, "invalid_error_mode",
            f"error_mode must be one of {list(_IMPORT_ERROR_MODES)}",
        )

    max_bad = body.get("max_bad_records", None)
    if max_bad is not None:
        if not isinstance(max_bad, int) or isinstance(max_bad, bool) or max_bad < 0:
            return _err(
                400, "invalid_request",
                "max_bad_records must be a non-negative integer or null",
            )

    # Admission check: reject if the tenant is already at/over the vector cap.
    # We consume 0 — `try_consume_vectors(0)` succeeds iff used <= quota, but a
    # tenant exactly at the cap has no room for an import, so check explicitly.
    #
    # OSS opt-in: skipped entirely when `RB_ENABLE_QUOTAS` is unset/false.
    if quotas_enabled():
        try:
            usage = state_mod.get_usage(tenant_id)
        except ValueError:
            usage = {"vectors_used": 0, "vector_quota": 0}
        if int(usage.get("vectors_used", 0)) >= int(usage.get("vector_quota", 0)):
            obs_metrics.record_quota_rejection("vector")
            return vector_quota_429(usage)

    import_id = "imp_" + uuid4().hex
    upload_uri = _import_upload_uri(tenant_id, name, import_id, fmt)
    row = state_mod.create_import_job(
        import_id=import_id,
        tenant_id=tenant_id,
        dataset=name,
        fmt=fmt,
        error_mode=error_mode,
        max_bad_records=max_bad,
        upload_uri=upload_uri,
    )
    return JSONResponse(status_code=201, content=_import_response(row, include_upload=True))


@app.post("/v1/datasets/{name}/imports/{import_id}/complete", status_code=202)
def complete_import(
    name: str,
    import_id: str,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """Signal that the staged upload is done; enqueue validation.

    Verifies the expected object is actually present in object storage, then
    transitions the job `awaiting_upload` → `validating` and publishes a
    `VALIDATE_DATASET` message carrying the `import_id`. The validator worker's
    `process_import` path picks it up.
    """
    job = state_mod.get_import_job(tenant_id, import_id)
    if job is None or job["dataset"] != name:
        return _err(404, "import_not_found", f"Import '{import_id}' not found")
    if job["status"] != "awaiting_upload":
        return _err(
            409, "import_not_pending",
            f"Import is '{job['status']}', expected 'awaiting_upload'",
        )
    if not storage_exists(job["upload_uri"]):
        return _err(
            400, "upload_missing",
            "No uploaded object found; PUT the file to the presigned upload URL first",
        )

    state_mod.update_import_job(import_id, status="validating")
    publish(
        "VALIDATE_DATASET",
        {
            "dataset": name,
            "tenant": tenant_id,
            "uri": job["upload_uri"],
            "file_type": job["format"],
            "import_id": import_id,
        },
    )
    job = state_mod.get_import_job(tenant_id, import_id)
    return JSONResponse(status_code=202, content=_import_response(job))


@app.get("/v1/datasets/{name}/imports/{import_id}")
def get_import(
    name: str,
    import_id: str,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """Return a single import job's status. Cross-tenant lookups → 404."""
    job = state_mod.get_import_job(tenant_id, import_id)
    if job is None or job["dataset"] != name:
        return _err(404, "import_not_found", f"Import '{import_id}' not found")
    return _import_response(job)


@app.get("/v1/datasets/{name}/imports")
def list_imports(
    name: str,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """List this dataset's import jobs, newest first."""
    dataset = state_mod.get_dataset(tenant_id, name)
    if dataset is None:
        return _err(404, "dataset_not_found", f"Dataset '{name}' not found")
    jobs = state_mod.list_import_jobs(tenant_id, name)
    return {"imports": [_import_response(j) for j in jobs]}


def _parse_ndjson_body(
    body: bytes, expected_dim: int
) -> tuple[bool, list[str], int, list[dict]]:
    """Decode + per-line-validate + normalise an NDJSON ingest body.

    Pure, CPU-bound, and synchronous so it can run off the event loop via
    `run_in_threadpool` — see the caller in `post_vectors`. For a ~9 MB
    1536-dim batch this is the bulk of the request's CPU cost.

    Per-line validation failures are reported but do not abort the upload —
    accepted lines still go to landing. This matches Pinecone/Weaviate
    semantics where customers expect partial successes on bulk inserts.

    Returns `(bad_utf8, accepted_lines, accepted_count, errors)`. `bad_utf8` is
    True when the body is not valid UTF-8 (the caller turns that into a 400);
    in that case the other fields are empty.
    """
    # Decoding as utf-8 is required by the NDJSON convention. Bad bytes
    # short-circuit the request — we cannot tell which line was bad.
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return True, [], 0, []

    accepted_lines: list[str] = []
    accepted_count = 0
    errors: list[dict] = []
    line_no = 0
    for line in text.splitlines():
        line_no += 1
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            errors.append({"line": line_no, "reason": "invalid json"})
            continue
        reason = _validate_ndjson_record(obj, expected_dim)
        if reason:
            errors.append({"line": line_no, "reason": reason})
            continue
        # Re-emit the normalised record so downstream gets a canonical shape.
        accepted_lines.append(json.dumps({
            "id": obj["id"],
            "values": [float(v) for v in obj["values"]],
            "metadata": obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {},
        }))
        accepted_count += 1

    return False, accepted_lines, accepted_count, errors


def _validate_ndjson_record(obj, expected_dim: int) -> Optional[str]:
    """Return None if `obj` is a valid record, else a human-readable reason.

    The reason strings are surfaced verbatim under `errors[].reason` in the
    response so the customer can find the offending row.
    """
    if not isinstance(obj, dict):
        return "record must be a JSON object"
    rid = obj.get("id")
    if not isinstance(rid, str) or not rid:
        return "id must be a non-empty string"
    if len(rid) > 256:
        return "id too long (max 256 chars)"
    values = obj.get("values")
    if not isinstance(values, list):
        return "values must be list[float]"
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in values):
        return "values must be list[float]"
    if len(values) != expected_dim:
        return f"dimension mismatch: got {len(values)} expected {expected_dim}"
    # Reject NaN / +-Infinity and magnitudes that overflow float4 BEFORE the
    # write splits on the flag. The recall tier stores embeddings as pgvector
    # `vector` (float4) which rejects non-finite and out-of-range values at
    # insert time: without this guard a flag-ON batch would fail the all-or-
    # nothing transaction and roll back EVERY valid record as a bare 500, while
    # the flag-OFF path silently cast the same value to float32 and returned
    # 202 (Inf/NaN landing as garbage in a consolidated shard). Rejecting per-line
    # here makes both modes behave identically and closes the consolidated-path hole.
    if not all(math.isfinite(x) and abs(x) <= _FLOAT4_MAX for x in values):
        return "values must be finite and within float4 range"
    metadata = obj.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return "metadata must be object"
    return None
