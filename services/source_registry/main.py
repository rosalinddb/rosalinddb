from __future__ import annotations

"""Source Registry service.

Hosts the customer-facing `/v1/datasets*` surface alongside the auth router.
Datasets are tenant-scoped (the auth dependency resolves `current_tenant_id`);
the `/how-to-connect` helper endpoint is an unauthenticated docs pointer.

Endpoints:
  - POST   /v1/datasets               create an empty dataset
  - POST   /v1/datasets/{name}/vectors  stream NDJSON records into the dataset
  - GET    /v1/datasets               list the tenant's datasets
  - GET    /v1/datasets/{name}        get a single dataset
  - DELETE /v1/datasets/{name}        soft-delete
"""

import json
import logging
import os
import re
import uuid
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

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


logger = logging.getLogger(__name__)


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
_extra_origins = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()
]
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
_INGEST_MAX_BYTES = int(os.getenv("INGEST_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MiB
_LANDING_PREFIX = os.getenv("LANDING_PREFIX", "s3://rosalinddb/landing")
# Raw bulk-import uploads are staged OUTSIDE the dataset landing prefix so
# the index builder (which scans `landing/{tenant}/{dataset}/` recursively for
# `.parquet`) never sees a raw `upload.parquet` and double-indexes it. The
# staging root is a sibling of the landing root — same bucket, different
# top-level prefix the builder is never pointed at. Defaults to the landing
# root with its last path segment swapped to `staging`.
def _default_staging_prefix() -> str:
    base = os.getenv("LANDING_PREFIX", "s3://rosalinddb/landing").rstrip("/")
    head, _, _ = base.rpartition("/")
    return f"{head}/staging" if head else f"{base}-staging"


_STAGING_PREFIX = os.getenv("STAGING_PREFIX", _default_staging_prefix())
_TENANT_PREFIX = os.getenv("TENANT_PREFIX", "true").lower() == "true"

# --- bulk import limits ---------------------------------------------------
# The staged upload cap for the async import flow — far larger than the small
# `POST .../vectors` in-app cap because the bytes never touch the application;
# they go straight to object storage. A presigned-PUT URL cannot enforce this
# server-side (only a presigned-POST policy could, and presigned POST is not
# universally supported across S3-compatible backends — so the bulk-import
# flow uses presigned PUT). The import worker `head`s the staged object and
# fails the job if it exceeds this cap. Default 5 GiB; overridable per-deployment.
_IMPORT_MAX_BYTES = int(os.getenv("IMPORT_MAX_BYTES", str(5 * 1024 * 1024 * 1024)))
# Presigned upload URL lifetime, seconds (default 1 hour).
_IMPORT_UPLOAD_TTL_S = int(os.getenv("IMPORT_UPLOAD_TTL_S", "3600"))
_IMPORT_FORMATS = ("ndjson", "parquet")
_IMPORT_ERROR_MODES = ("continue", "abort")
_IMPORT_EXT = {"ndjson": "ndjson", "parquet": "parquet"}


def _err(status_code: int, code: str, message: str, details: Optional[dict] = None) -> JSONResponse:
    """Build a v1 error envelope response."""
    body: dict = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


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
    metadata = obj.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return "metadata must be object"
    return None
