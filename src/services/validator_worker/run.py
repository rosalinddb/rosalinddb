from __future__ import annotations

"""Validator Worker service.

Validates raw input records, writes them to the dataset landing area, updates
catalog state, and signals downstream index building via events.

Queue message handling:
  - The queue message's `tenant` is passed through to state writes so
    `dataset_catalog` and `shard_catalog` stay tenant-isolated end-to-end.
  - The dataset's `status` transitions through `validating` -> `indexing`
    (on success) or `error` (on failure) so the customer-facing
    `GET /v1/datasets/{name}` reflects real pipeline progress.
  - Validated records land in a per-upload subdirectory so the index
    builder, which walks the dataset's landing prefix recursively, sees
    every upload rather than just the most recent one.
"""

import io
import json
import os
import time
import uuid
from typing import Dict, Any, Iterable

from adapters.observability import init_observability
from adapters.observability import metrics as obs_metrics
from adapters.observability.tracing import validate_dataset_span, landing_write_span
from adapters.queue.queue import consume, publish, ack, nack
from adapters.queue.shutdown import install_signal_handlers, should_stop
from adapters.storage.storage import (
    delete as storage_delete,
    object_size,
    open_reader,
    read_bytes,
    write_bytes,
)
from adapters.landing.parquet_writer import write_parquet
from adapters.state.state import (
    migrate,
    upsert_dataset,
    increment_row_count,
    update_dataset_status,
    get_dataset,
    get_import_job_by_id,
    update_import_job,
    try_consume_vectors,
)
from adapters.metrics.metrics import counter
from adapters.metrics.server import (
    make_metrics_handler,
    start_metrics_server as _start_metrics_server,
)
from services.auth.quota import quotas_enabled

# Observability bootstrap at import so it works whether this module runs as a
# standalone worker process OR is imported by a single-process dev/test harness
# (idempotent; the first caller wins, and OTEL_SERVICE_NAME overrides).
init_observability("rosalinddb-validator")


DIMENSION = int(os.getenv("VECTOR_DIM", os.getenv("DIMENSION", "1536")))
# Landing parts are written exclusively as parquet — `bson_writer.py` and the
# `LANDING_FORMAT` env switch were removed on this branch.
LANDING_FORMAT = "parquet"
LANDING_PREFIX = os.getenv("LANDING_PREFIX", "s3://rosalinddb/landing")
TENANT_PREFIX = os.getenv("TENANT_PREFIX", "true").lower() == "true"
METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))
# Bulk-import staged-upload size cap. The presigned-PUT URL handed to the
# client cannot enforce this server-side (only a presigned-POST policy could,
# and presigned POST is not universally supported across S3-compatible
# backends — so the bulk-import flow uses presigned PUT). The cap is enforced
# here: `process_import` `head`s the staged object and fails the job if it is
# larger. Must mirror `_IMPORT_MAX_BYTES` in `source_registry/main.py`.
IMPORT_MAX_BYTES = int(os.getenv("IMPORT_MAX_BYTES", str(5 * 1024 * 1024 * 1024)))


def _dataset_dimension(tenant: str, dataset: str, fallback: int) -> int:
    """Resolve the per-dataset dimension, falling back to env default.

    Datasets carry their own `dimension` so vectors get validated against the
    customer's declared shape rather than the global env var. Internal callers
    that publish a `VALIDATE_DATASET` message directly (tests, scripts) may not
    have created a catalog row first; in that case we fall back to the env
    default so the existing integration test keeps passing.
    """
    row = get_dataset(tenant, dataset)
    if row is None:
        return fallback
    return int(row.get("dimension", fallback))


def _validate_record(obj: Dict[str, Any], dim: int | None = None) -> Dict[str, Any]:
    """Validate and normalize a record against the expected `dim`.

    If `dim` is omitted, falls back to the env-default `DIMENSION` (set from
    `VECTOR_DIM`/`DIMENSION`). The env-default path keeps the older unit
    tests (`tests/unit/test_validator.py`) working without modification.
    """
    if dim is None:
        dim = int(os.getenv("VECTOR_DIM", os.getenv("DIMENSION", str(DIMENSION))))
    if "id" not in obj or not isinstance(obj["id"], str) or not obj["id"]:
        raise ValueError("missing id")
    values = obj.get("values")
    if not isinstance(values, list) or not all(isinstance(x, (float, int)) for x in values):
        raise ValueError("values must be list[float]")
    if len(values) != dim:
        raise ValueError(f"dimension mismatch: got {len(values)} expected {dim}")
    metadata = obj.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be object")
    # `metadata` is stored verbatim. `parquet_writer` JSON-encodes it into a
    # string column, so an empty `{}` (or any nested object) round-trips with
    # a stable schema — no sentinel field needed.
    return {"id": obj["id"], "values": [float(v) for v in values], "metadata": metadata}


def process_uri(dataset: str, tenant: str, uri: str, file_type: str | None) -> int:
    """Validate rows from `uri`, write to landing, update catalog state.

    The catalog `status` is flipped to `validating` on entry and then to
    `indexing` (downstream `DATASET_READY` event will trigger the builder)
    or `error` on failure, with `error_message` populated.
    """
    # `validate_dataset` span — child of the originating upload trace (the
    # queue adapter attached the producer context on consume()). Tenant /
    # dataset are high-cardinality but correct on a span.
    with validate_dataset_span(tenant=tenant, dataset=dataset):
        update_dataset_status(tenant, dataset, "validating")

        dim = _dataset_dimension(tenant, dataset, DIMENSION)
        good: list[Dict[str, Any]] = []
        bad: list[Dict[str, Any]] = []
        total = 0
        try:
            for rec in _stream_records(uri, file_type):
                total += 1
                try:
                    good.append(_validate_record(rec, dim))
                except Exception as e:  # noqa: BLE001
                    bad.append({"payload": rec, "reason": str(e)})
        except Exception as exc:  # noqa: BLE001
            update_dataset_status(tenant, dataset, "error", error_message=f"validator: {exc}")
            counter("ingest_errors", 1)
            raise

        if good:
            # Write each upload into its own `upload-<id>/` sub-prefix. The
            # parquet writer already gives every part a unique name, so this is
            # not needed for correctness — it groups one upload's parts under a
            # single prefix so the landing sweeper can prune a whole upload's
            # directory once it is captured in a shard.
            upload_id = uuid.uuid4().hex[:12]
            landing_prefix = f"{_landing_prefix(dataset, tenant)}/upload-{upload_id}"
            try:
                with landing_write_span(uri=landing_prefix):
                    write_parquet(landing_prefix, good)
            except Exception as exc:  # noqa: BLE001
                update_dataset_status(tenant, dataset, "error", error_message=f"landing write: {exc}")
                raise
            increment_row_count(tenant, dataset, len(good))
            upsert_dataset(tenant, dataset, dim, uri, LANDING_FORMAT)

        counter("rows_ingested", len(good))
        counter("ingest_errors", len(bad))

        # Transition to indexing so clients see pipeline progress
        # even if zero records were accepted (builder skips empty landing).
        update_dataset_status(tenant, dataset, "indexing")
        return len(good)


# --- bulk import ----------------------------------------------------------
#
# Internal landing Parquet schema. A bulk-import Parquet upload must conform
# to *exactly* this so it can be used directly as a landing part (skipping the
# NDJSON->Parquet conversion). This is the same schema `parquet_writer` emits:
#   - id        : string
#   - values    : list<float> (any list/large_list/fixed_size_list of float32
#                 or float64) of length == dataset.dimension
#   - metadata  : a struct/map (JSON object per row); optional column
# Documented in `docs/api/imports.md`.

# How big an offending record may get in the rejected.jsonl `record` field
# before it is truncated (keeps the sidecar bounded for pathological input).
_REJECTED_RECORD_MAX_CHARS = 2000


def _truncate_record(value: Any) -> Any:
    """Return `value` for the rejected-records sidecar, truncated if large."""
    try:
        text = json.dumps(value, default=str)
    except Exception:  # noqa: BLE001
        text = repr(value)
    if len(text) > _REJECTED_RECORD_MAX_CHARS:
        return text[:_REJECTED_RECORD_MAX_CHARS] + "...[truncated]"
    return value if len(text) <= _REJECTED_RECORD_MAX_CHARS else text


def _validate_parquet_schema(schema) -> None:
    """Strictly validate an uploaded Parquet *schema* against the landing schema.

    The internal landing schema is exactly:
      - id        : string-like
      - values    : list/large_list/fixed_size_list of float32/float64
      - metadata  : struct/map (optional)

    Raises ValueError on any deviation — a missing required column, an
    unexpected/unknown extra column, a non-list `values` column, or a `values`
    list whose element type is not a float (a `list<string>`/`list<int>`
    `values` column must be rejected, not passed through).
    """
    import pyarrow as pa

    names = list(schema.names)
    name_set = set(names)
    if len(names) != len(name_set):
        raise ValueError("Parquet has duplicate column names")

    required = {"id", "values"}
    allowed = {"id", "values", "metadata"}
    missing = required - name_set
    if missing:
        raise ValueError(
            f"Parquet missing required column(s) {sorted(missing)} "
            "(RosalindDB landing schema: id, values, optional metadata)"
        )
    extra = name_set - allowed
    if extra:
        raise ValueError(
            f"Parquet has unexpected column(s) {sorted(extra)}; "
            "landing schema allows only: id, values, metadata"
        )

    id_type = schema.field("id").type
    if not (pa.types.is_string(id_type) or pa.types.is_large_string(id_type)):
        raise ValueError(f"'id' column must be a string, got {id_type}")

    values_type = schema.field("values").type
    if not (
        pa.types.is_list(values_type)
        or pa.types.is_large_list(values_type)
        or pa.types.is_fixed_size_list(values_type)
    ):
        raise ValueError(
            f"'values' column must be a list of floats, got {values_type}"
        )
    elem_type = values_type.value_type
    if not (pa.types.is_float32(elem_type) or pa.types.is_float64(elem_type)):
        raise ValueError(
            f"'values' must be a list of float32/float64 elements, "
            f"got list<{elem_type}>"
        )

    if "metadata" in name_set:
        meta_type = schema.field("metadata").type
        if not (pa.types.is_struct(meta_type) or pa.types.is_map(meta_type)):
            raise ValueError(
                f"'metadata' column must be a struct/map (per-row JSON object), "
                f"got {meta_type}"
            )


# Row count per `iter_batches` step in `_validate_parquet_landing`. Kept small
# on purpose: each batch's columns are `to_pylist()`-materialised into Python
# objects (~4 KB per 128-dim vector — Python float/list overhead, ~8x the
# columnar Arrow footprint), so a large batch OOMs a small worker. 4096 rows is
# ~16 MB transient, freed per batch. Do NOT raise this for "throughput" — the
# real fix is columnar validation with no to_pylist() (see issue #19).
_VALIDATE_BATCH_ROWS = 4096


def _validate_parquet_landing(uri: str, dim: int) -> tuple[int, bytes]:
    """Strictly validate an uploaded Parquet, returning `(row_count, landing_bytes)`.

    Previously this only checked that `id`/`values` columns existed and
    that vectors matched the dimension, then the caller wrote the raw bytes
    through byte-for-byte — so a schema-passing-but-wrong file (e.g. a
    `list<string>` `values` column, or an unknown extra column) reached the
    index builder. It also `read_table`-d the whole (potentially multi-GiB)
    file into memory *on top of* the already-buffered raw bytes — twice.

    This now (a) validates the schema *strictly* — exactly the expected columns
    with no extras, a `values` column that is a list of float32/float64 — and
    (b) streams the file by row-group via `pyarrow.parquet.ParquetFile`,
    validating per-row dimension/metadata shape and re-emitting each row-group
    into the returned landing bytes with a bounded working set instead of
    materialising the whole table a second time.

    The returned `landing_bytes` is the validated, re-emitted landing Parquet —
    never a raw passthrough, so a malformed file can never reach the index
    builder. The caller writes it to the landing part only after the quota
    settlement check passes. Raises ValueError on any non-conformance.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        pf = pq.ParquetFile(io.BytesIO(read_bytes(uri)))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"not a readable Parquet file: {exc}") from exc

    _validate_parquet_schema(pf.schema_arrow)

    total = 0
    writer: pq.ParquetWriter | None = None
    out_buf = io.BytesIO()
    row_offset = 0
    try:
        # Stream in small fixed-size batches across row-groups. The batch is
        # deliberately small (`_VALIDATE_BATCH_ROWS`) — each batch is
        # `to_pylist()`-materialised below, so a large batch OOMs a small
        # worker (see the constant's comment and issue #19).
        for batch in pf.iter_batches(batch_size=_VALIDATE_BATCH_ROWS):
            table = pa.Table.from_batches([batch])
            ids = table.column("id").to_pylist()
            values = table.column("values").to_pylist()
            for i, (rid, vec) in enumerate(zip(ids, values)):
                if rid is None or str(rid) == "":
                    raise ValueError(f"row {row_offset + i}: empty id")
                if vec is None or not isinstance(vec, list):
                    raise ValueError(
                        f"row {row_offset + i}: 'values' must be a list of floats"
                    )
                if len(vec) != dim:
                    raise ValueError(
                        f"row {row_offset + i}: dimension mismatch: "
                        f"got {len(vec)} expected {dim}"
                    )
                if any(x is None for x in vec):
                    raise ValueError(
                        f"row {row_offset + i}: 'values' contains a null element"
                    )
            if "metadata" in table.column_names:
                for i, meta in enumerate(table.column("metadata").to_pylist()):
                    if meta is not None and not isinstance(meta, dict):
                        raise ValueError(
                            f"row {row_offset + i}: 'metadata' must be a JSON object"
                        )
            if writer is None:
                writer = pq.ParquetWriter(out_buf, table.schema)
            writer.write_table(table)
            total += table.num_rows
            row_offset += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if total == 0:
        raise ValueError("Parquet upload contains no rows")

    return total, out_buf.getvalue()


def _import_landing_prefix(tenant: str, dataset: str, import_id: str) -> str:
    """Per-import sub-prefix under the dataset *landing* prefix.

    The raw upload is staged outside the landing prefix (in a `staging/`
    sibling root) so the index builder does not double-index it. The
    validator's *produced* landing Parquet — which SHOULD be indexed — and the
    rejected-records sidecar live here, at
    `landing/{tenant}/{dataset}/imports/{import_id}/`, so the builder's
    recursive landing scan picks up exactly the produced part and nothing else.
    """
    return f"{_landing_prefix(dataset, tenant)}/imports/{import_id}"


def process_import(import_id: str) -> int:
    """Validate + land an async bulk-import job, advancing its lifecycle.

    Reads the staged NDJSON/Parquet upload, validates each record against the
    dataset's dimension, and writes the accepted rows as internal landing
    Parquet (which feeds the incremental indexer). Then:

      - `error_mode=continue`: bad records are dropped and appended to a
        rejected-records file at `imports/{import_id}/rejected.jsonl`. If
        `max_bad_records` is set and exceeded, the job fails.
      - `error_mode=abort`: the first bad record fails the job; nothing lands.

    Two-stage quota — settlement: `records_accepted` is charged via
    `try_consume_vectors`. If the accepted count would cross the tenant's
    remaining quota the job fails with a quota error and nothing is indexed.

    Size cap: the staged upload arrived via a presigned-PUT URL (presigned
    POST is not universally supported across S3-compatible backends), which
    cannot enforce a size limit server-side. This path `head`s the staged
    object first and fails the job if it exceeds `IMPORT_MAX_BYTES` — the
    re-homed `content-length-range` enforcement.

    Returns the number of accepted records (0 on failure). Status moves
    `validating` → `indexing` on success; on failure it is set to `failed`
    with an `error_message`.
    """
    job = get_import_job_by_id(import_id)
    if job is None:
        return 0
    dataset = job["dataset"]
    tenant = job["tenant_id"]
    fmt = job["format"]
    error_mode = job["error_mode"]
    max_bad = job.get("max_bad_records")

    def _fail(message: str) -> int:
        update_import_job(
            import_id, status="failed", error_message=message,
            completed_at=_now_iso(),
        )
        update_dataset_status(tenant, dataset, "error", error_message=f"import: {message}")
        # Delete the staged raw upload directly. A failed import's staged file
        # is never read again, so this is safe for ALL failure paths. Without
        # it, an oversized (or otherwise failed) import would orphan its staged
        # object in object storage indefinitely: `index_builder`'s
        # `_sweep_indexed_landing` only prunes after a *successful* build for
        # the dataset, which may never happen for a tenant whose first/only
        # import fails. Best-effort — a delete failure must not mask the
        # original failure, and the deferred sweep deleting an already-gone
        # object is a harmless idempotent no-op.
        try:
            storage_delete(job["upload_uri"])
        except Exception as exc:  # noqa: BLE001
            print(f"validator: import={import_id} staged-object cleanup failed: {exc}")
        obs_metrics.record_import_terminal("failed", fmt)
        return 0

    # Re-homed size cap. The staged file arrived via a presigned-PUT URL, which
    # — unlike the old presigned-POST policy — cannot reject an oversized
    # upload server-side. Enforce `IMPORT_MAX_BYTES` here: `head` the staged
    # object and fail the job before reading a single byte of an oversized
    # file into memory.
    staged_size = object_size(job["upload_uri"])
    if staged_size is not None and staged_size > IMPORT_MAX_BYTES:
        return _fail(
            f"uploaded file is {staged_size} bytes, which exceeds the "
            f"{IMPORT_MAX_BYTES}-byte import size limit"
        )

    # The import's landing sub-prefix lives under the dataset *landing* prefix
    # — the produced landing part there SHOULD be indexed. The raw upload
    # itself sits in the staging root and is never scanned by the builder.
    import_prefix = _import_landing_prefix(tenant, dataset, import_id)

    with validate_dataset_span(tenant=tenant, dataset=dataset):
        update_dataset_status(tenant, dataset, "validating")
        dim = _dataset_dimension(tenant, dataset, DIMENSION)

        good: list[Dict[str, Any]] = []
        rejected: list[Dict[str, Any]] = []
        processed = 0
        # When the upload is conforming Parquet, `parquet_passthrough` holds the
        # validated row count and `parquet_landing_bytes` the validated,
        # row-group-re-emitted landing Parquet (never the raw passthrough).
        parquet_passthrough = -1
        parquet_landing_bytes: bytes | None = None

        if fmt == "parquet":
            # A conforming Parquet file IS the landing schema — validate it
            # strictly by streaming row-groups so a multi-GiB file is not
            # loaded whole into memory twice. A malformed-but-columns-present
            # file is rejected here, not passed through to the index builder.
            try:
                parquet_passthrough, parquet_landing_bytes = _validate_parquet_landing(
                    job["upload_uri"], dim
                )
            except ValueError as exc:
                return _fail(str(exc))
            processed = parquet_passthrough
        else:
            try:
                raw = read_bytes(job["upload_uri"])
            except Exception as exc:  # noqa: BLE001
                return _fail(f"could not read uploaded file: {exc}")
            # NDJSON: per-record validation, honouring continue/abort.
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return _fail("uploaded file is not valid UTF-8")
            line_no = 0
            for line in text.splitlines():
                line_no += 1
                stripped = line.strip()
                if not stripped:
                    continue
                processed += 1
                reason = None
                rec = None
                try:
                    obj = json.loads(stripped)
                    rec = _validate_record(obj, dim)
                except Exception as exc:  # noqa: BLE001
                    reason = str(exc)
                if reason is not None:
                    if error_mode == "abort":
                        return _fail(f"line {line_no}: {reason} (error_mode=abort)")
                    rejected.append({
                        "line": line_no,
                        "reason": reason,
                        "record": _truncate_record(stripped),
                    })
                    if max_bad is not None and len(rejected) > int(max_bad):
                        _write_rejected(import_prefix, rejected)
                        return _fail(
                            f"max_bad_records ({max_bad}) exceeded: "
                            f"{len(rejected)} bad records"
                        )
                else:
                    good.append(rec)

        records_accepted = len(good) if parquet_passthrough < 0 else parquet_passthrough
        records_rejected = len(rejected)

        # Two-stage quota — settlement. Charge the accepted count. If the
        # tenant lacks the headroom, fail the job and index nothing.
        #
        # OSS opt-in: skipped entirely when `RB_ENABLE_QUOTAS` is unset/false
        # (the self-host default). The admission check in the CP is also off
        # in that mode, so a self-hoster's import never hits a quota wall.
        if records_accepted > 0 and quotas_enabled():
            ok, usage = try_consume_vectors(tenant, records_accepted)
            if not ok:
                if rejected:
                    _write_rejected(import_prefix, rejected)
                obs_metrics.record_quota_rejection("vector")
                return _fail(
                    f"vector quota exceeded: {records_accepted} accepted records "
                    f"would exceed quota {usage.get('vector_quota')} "
                    f"(used {usage.get('vectors_used')})"
                )

        rejected_uri = None
        if rejected:
            rejected_uri = _write_rejected(import_prefix, rejected)

        if records_accepted > 0:
            # Write a landing Parquet part under the import's landing sub-prefix
            # so the incremental indexer folds exactly this batch in. This lives
            # under the dataset landing prefix (and SHOULD be indexed); the raw
            # upload sits in the staging root and is never scanned. A conforming
            # Parquet upload writes the validated, row-group-re-emitted bytes; an
            # NDJSON upload's accepted rows are serialised via `write_parquet`.
            landing_prefix = f"{import_prefix}/landing"
            try:
                with landing_write_span(uri=landing_prefix):
                    if parquet_landing_bytes is not None:
                        write_bytes(
                            f"{landing_prefix}/part-0001.parquet",
                            parquet_landing_bytes,
                        )
                    else:
                        write_parquet(landing_prefix, good)
            except Exception as exc:  # noqa: BLE001
                return _fail(f"landing write failed: {exc}")
            increment_row_count(tenant, dataset, records_accepted)
            upsert_dataset(tenant, dataset, dim, job["upload_uri"], fmt)

        update_import_job(
            import_id,
            status="indexing",
            records_processed=processed,
            records_accepted=records_accepted,
            records_rejected=records_rejected,
            rejected_uri=rejected_uri,
        )
        obs_metrics.record_import_records(records_accepted, "accepted")
        obs_metrics.record_import_records(records_rejected, "rejected")

        update_dataset_status(tenant, dataset, "indexing")
        counter("rows_ingested", len(good))
        counter("ingest_errors", records_rejected)
        return records_accepted


def finalize_import(import_id: str) -> None:
    """Mark an import job `completed` once its index build has finished.

    Called after the index builder runs for a job. Idempotent: a job already
    in a terminal state is left alone.
    """
    job = get_import_job_by_id(import_id)
    if job is None or job["status"] in ("completed", "failed"):
        return
    update_import_job(import_id, status="completed", completed_at=_now_iso())
    obs_metrics.record_import_terminal("completed", job["format"])


def fail_import(import_id: str, message: str) -> None:
    """Mark an import job `failed` with `message`. Catch-all terminal sweep.

    A guarantee that ANY unhandled exception while processing an import —
    whether the validator crashes mid-`process_import` or the index builder
    crashes before `finalize_import` — ends the job in a terminal state rather
    than leaving it stuck in `validating`/`indexing` forever. Idempotent: a job
    already terminal (or absent) is left alone.
    """
    job = get_import_job_by_id(import_id)
    if job is None or job["status"] in ("completed", "failed"):
        return
    update_import_job(
        import_id, status="failed", error_message=message, completed_at=_now_iso(),
    )
    try:
        update_dataset_status(
            job["tenant_id"], job["dataset"], "error",
            error_message=f"import: {message}",
        )
    except Exception:  # noqa: BLE001
        pass
    obs_metrics.record_import_terminal("failed", job["format"])


# Backwards-friendly alias used by `main_loop`'s catch-all.
_fail_import = fail_import


def _write_rejected(import_prefix: str, rejected: list[Dict[str, Any]]) -> str:
    """Write the rejected-records JSONL sidecar and return its URI.

    Lands at `imports/{import_id}/rejected.jsonl` — one JSON object per line:
    `{"line": int, "reason": str, "record": <offending record, truncated>}`.
    `import_prefix` is the import's landing sub-prefix (`_import_landing_prefix`).
    """
    rejected_uri = f"{import_prefix}/rejected.jsonl"
    body = ("\n".join(json.dumps(r) for r in rejected) + "\n").encode("utf-8")
    write_bytes(rejected_uri, body)
    return rejected_uri


def _now_iso() -> str:
    """Current UTC time, ISO 8601 with trailing Z (matches the v1 contract)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stream_records(uri: str, file_type: str | None) -> Iterable[Dict[str, Any]]:
    """Yield decoded records from a URI according to file type."""
    ftype = file_type or ("jsonl" if uri.endswith(".jsonl") else "json" if uri.endswith(".json") else "parquet")
    for chunk in open_reader(uri, ftype):
        if ftype == "jsonl":
            yield json.loads(chunk)
        elif ftype == "json":
            data = json.loads(chunk)
            if isinstance(data, list):
                for row in data:
                    yield row
            else:
                yield data
        else:
            # Parquet is validated on the import path via `_validate_parquet_landing`;
            # `_stream_records` only handles JSON/JSONL. This branch is therefore
            # unreachable for parquet — the import path never routes parquet through here.
            raise NotImplementedError(
                "parquet records are validated via _validate_parquet_landing; "
                "_stream_records only handles JSON/JSONL"
            )


def _landing_prefix(dataset: str, tenant: str) -> str:
    """Compute landing prefix respecting tenancy setting."""
    base = LANDING_PREFIX
    if not base.endswith("/"):
        base += "/"
    if TENANT_PREFIX:
        return f"{base}{tenant}/{dataset}"
    return f"{base}{dataset}"


# The metrics HTTP handler + server are the canonical implementation in
# `adapters.metrics.server`. `MetricsHandler` is re-exported (a configured
# subclass with this service's `/healthz` service name + Prometheus prefix) so
# the name stays importable from this module; `start_metrics_server()` keeps its
# no-arg signature and forwards this service's two strings + `METRICS_PORT`.
MetricsHandler = make_metrics_handler("validator_worker", "validator_")


def start_metrics_server():
    """Start the metrics HTTP server in a background thread."""
    return _start_metrics_server("validator_worker", "validator_", METRICS_PORT)


def main_loop():
    """Blocking loop that consumes VALIDATE_DATASET and performs validation.

    Reliable-queue contract: a message is `ack`-ed only after the job finishes
    successfully (or terminally — a validation that flips the dataset to
    `error` is *handled*, so it is acked, not redelivered). An UNHANDLED crash
    `nack`s the message so it is redelivered (and dead-lettered after
    `QUEUE_MAX_ATTEMPTS`). On `SIGTERM` the loop stops pulling new messages and
    exits cleanly so the in-flight message is acked/nacked before shutdown.
    """
    migrate()
    install_signal_handlers()
    start_metrics_server()
    while not should_stop():
        msg = consume("VALIDATE_DATASET", block=True, timeout=1.0)
        if not msg:
            continue
        try:
            _handle_validate(msg)
        except Exception as exc:  # noqa: BLE001
            # An UNHANDLED crash here means the job did not reach a terminal
            # state — nack so the message is redelivered (or dead-lettered
            # past the retry cap) rather than silently lost.
            print(f"validator: unhandled error, nacking message: {exc}")
            nack(msg, requeue=True)
            continue
        ack(msg)
    print("validator: shutdown signal received — exiting consume loop")


def _handle_validate(msg) -> None:
    """Process one VALIDATE_DATASET message to a terminal outcome.

    Returns normally once the job has reached a terminal state (success OR a
    handled failure that flipped the dataset/import to `error`/`failed`); the
    caller then acks. Raising propagates to the caller's nack path.
    """
    dataset = msg["dataset"]
    tenant = msg.get("tenant", "default")
    uri = msg.get("uri")
    file_type = msg.get("file_type")
    import_id = msg.get("import_id")
    if import_id:
        # Async bulk-import path: validates the staged upload and advances
        # the import job. A failed import has already been flipped to
        # `failed`; only a successful one publishes onward.
        try:
            count = process_import(import_id)
        except Exception as exc:  # noqa: BLE001
            # Errors handled *inside* `process_import` already flip the job to
            # `failed` via `_fail`. An UNHANDLED exception here (a bug, an OOM,
            # a storage outage mid-stream) is flipped terminal too so the job is
            # never stuck — the message is then acked (the job IS terminal;
            # redelivery would not change the outcome).
            print(f"validator: import={import_id} failed: {exc}")
            _fail_import(import_id, f"validator crashed: {exc}")
            return
        job = get_import_job_by_id(import_id)
        if job is not None and job["status"] == "indexing":
            publish(
                "DATASET_READY",
                {"dataset": dataset, "tenant": tenant, "rows": count,
                 "import_id": import_id},
            )
        return
    try:
        count = process_uri(dataset, tenant, uri, file_type)
    except Exception as exc:  # noqa: BLE001
        # Status already moved to `error` by `process_uri`; nothing left to
        # publish — the builder must not run on a failed validation. The job
        # IS terminal, so this is acked, not redelivered.
        print(f"validator: dataset={dataset} tenant={tenant} failed: {exc}")
        return
    publish("DATASET_READY", {"dataset": dataset, "tenant": tenant, "rows": count})


if __name__ == "__main__":
    main_loop()
