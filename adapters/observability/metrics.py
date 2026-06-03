from __future__ import annotations

"""Custom business metrics — the contract-pinned RosalindDB instruments.

Names, types, and attributes must match the infra side (Prometheus / Grafana).
This module owns the OTel
instrument objects and exposes one terse `record_*` helper per metric so call
sites stay readable.

**Hard cardinality rule.** Metric attributes are low-cardinality ONLY:
`outcome`, `mode`, `kind`, `index_type`. NEVER attach `tenant_id`,
`dataset_name`, `api_key`, or `email` — that explodes Prometheus series count.
High-cardinality detail belongs on spans/logs, not here. The `record_*`
helpers below deliberately take no per-entity id arguments, so a caller
*cannot* leak one into a metric.

When the SDK is disabled / not yet initialised, `get_meter` returns a no-op
meter and every instrument silently records nothing — call sites need no
guards.
"""

import threading

from opentelemetry import metrics as _metrics_api

_LOCK = threading.Lock()
_instruments: dict | None = None


def _get_instruments() -> dict:
    """Lazily build the instrument set against the current MeterProvider.

    Built lazily (not at import) so the instruments bind to the real
    `MeterProvider` installed by `init_observability`, not the no-op one that
    exists at import time.
    """
    global _instruments
    if _instruments is not None:
        return _instruments
    with _LOCK:
        if _instruments is not None:
            return _instruments
        meter = _metrics_api.get_meter("rosalinddb.business")
        _instruments = {
            "auth.signups": meter.create_counter(
                "rosalinddb.auth.signups",
                unit="1",
                description="Tenant signups.",
            ),
            "auth.logins": meter.create_counter(
                "rosalinddb.auth.logins",
                unit="1",
                description="Login attempts, by outcome.",
            ),
            "datasets.created": meter.create_counter(
                "rosalinddb.datasets.created",
                unit="1",
                description="Datasets created.",
            ),
            "ingest.uploads": meter.create_counter(
                "rosalinddb.ingest.uploads",
                unit="1",
                description="Vector upload requests, by outcome.",
            ),
            "vectors.ingested": meter.create_counter(
                "rosalinddb.vectors.ingested",
                unit="1",
                description="Individual vectors accepted into landing.",
            ),
            "queries": meter.create_counter(
                "rosalinddb.queries",
                unit="1",
                description="Vector search queries, by serving mode.",
            ),
            "shard_cache": meter.create_counter(
                "rosalinddb.shard_cache",
                unit="1",
                description="In-memory shard index cache lookups, by result (hit|miss).",
            ),
            "shard_page_faults": meter.create_counter(
                "rosalinddb.shard.page_faults",
                unit="1",
                description="Major page faults charged during a shard's FAISS search call (cold/warm mmap signal).",
            ),
            "queries.filtered": meter.create_counter(
                "rosalinddb.queries.filtered",
                unit="1",
                description="Vector search queries that included a non-empty metadata filter.",
            ),
            "query.filtered_results": meter.create_histogram(
                "rosalinddb.query.filtered_results",
                unit="1",
                description="Post-filter result count for queries with a non-empty metadata filter.",
            ),
            "query.duration": meter.create_histogram(
                "rosalinddb.query.duration",
                unit="ms",
                description="Vector query latency in milliseconds.",
            ),
            "index_build.duration": meter.create_histogram(
                "rosalinddb.index_build.duration",
                unit="ms",
                description="FAISS index build duration in milliseconds.",
            ),
            "index_builds": meter.create_counter(
                "rosalinddb.index_builds",
                unit="1",
                description="Index builds, by build type (full vs incremental).",
            ),
            "index_build.vectors_added": meter.create_histogram(
                "rosalinddb.index_build.vectors_added",
                unit="1",
                description="Vectors added per index build, by build type.",
            ),
            "quota.rejections": meter.create_counter(
                "rosalinddb.quota.rejections",
                unit="1",
                description="Requests rejected by a quota/rate limit, by kind.",
            ),
            "imports": meter.create_counter(
                "rosalinddb.imports",
                unit="1",
                description="Bulk import jobs reaching a terminal status, by status and format.",
            ),
            "import.records": meter.create_counter(
                "rosalinddb.import.records",
                unit="1",
                description="Records processed by bulk import jobs, by outcome.",
            ),
            "storage.swept": meter.create_counter(
                "rosalinddb.storage.swept",
                unit="1",
                description="Objects reclaimed by background sweepers, by kind.",
            ),
        }
        return _instruments


# --- record helpers -------------------------------------------------------
#
# None of these accept a tenant/dataset/key/email argument — the cardinality
# rule is enforced structurally, not by convention.


def record_signup() -> None:
    """`rosalinddb.auth.signups` +1 — no attributes."""
    _get_instruments()["auth.signups"].add(1)


def record_login(outcome: str) -> None:
    """`rosalinddb.auth.logins` +1. `outcome` is `success` or `failure`."""
    _get_instruments()["auth.logins"].add(1, {"outcome": outcome})


def record_dataset_created() -> None:
    """`rosalinddb.datasets.created` +1 — no attributes."""
    _get_instruments()["datasets.created"].add(1)


def record_upload(outcome: str) -> None:
    """`rosalinddb.ingest.uploads` +1. `outcome` is `accepted` or `rejected`."""
    _get_instruments()["ingest.uploads"].add(1, {"outcome": outcome})


def record_vectors_ingested(count: int) -> None:
    """`rosalinddb.vectors.ingested` += count — no attributes."""
    if count > 0:
        _get_instruments()["vectors.ingested"].add(count)


def record_query(mode: str) -> None:
    """`rosalinddb.queries` +1. `mode` is `hot`, `cold`, or `ephemeral`."""
    _get_instruments()["queries"].add(1, {"mode": mode})


def record_query_duration(duration_ms: float, mode: str) -> None:
    """`rosalinddb.query.duration` histogram sample (ms), tagged by `mode`."""
    _get_instruments()["query.duration"].record(float(duration_ms), {"mode": mode})


def record_shard_cache(result: str) -> None:
    """`rosalinddb.shard_cache` +1. `result` is `hit` or `miss`.

    Counts in-memory shard index cache outcomes. A `hit` means the query
    reused an already-deserialised FAISS index + parsed sidecar; a `miss`
    means a genuine cold load. `result` is the only (low-cardinality)
    attribute — no tenant/dataset/shard label.
    """
    _get_instruments()["shard_cache"].add(1, {"result": result})


def record_shard_page_faults(count: int) -> None:
    """`rosalinddb.shard.page_faults` += count — no attributes.

    Sourced from the delta of `/proc/self/stat` field 12 (`majflt`, per
    `man 5 proc`) around a `faiss.search` call. The counter is process-wide
    on purpose: per-tenant cardinality is owned by spans, not metrics, so
    no tenant/dataset/shard label is attached here. Zero/negative deltas
    are dropped so a monotonic counter never absorbs sampler noise.

    A non-trivial page-fault rate per query is the operator-facing signal that
    the shard's pages are NOT warm in the page cache — i.e. the working set
    has cooled and the next query will incur synchronous disk reads. With the
    legacy non-mmap path this metric stays flat (the whole index is already in
    RSS), so a rise is a clean "mmap is on AND the cache is cold" indicator.
    """
    if count <= 0:
        return
    _get_instruments()["shard_page_faults"].add(count)


def record_filtered_query() -> None:
    """`rosalinddb.queries.filtered` +1 — a query with a non-empty filter.

    No attributes: filter keys/values are customer data (high cardinality)
    and belong on spans, not metrics.
    """
    _get_instruments()["queries.filtered"].add(1)


def record_filtered_result_count(count: int) -> None:
    """`rosalinddb.query.filtered_results` histogram sample — post-filter
    result count for a query that carried a non-empty metadata filter."""
    _get_instruments()["query.filtered_results"].record(float(count))


def record_index_build_duration(duration_ms: float, index_type: str) -> None:
    """`rosalinddb.index_build.duration` histogram sample (ms).

    `index_type` is `ivfflat` or `flat`.
    """
    _get_instruments()["index_build.duration"].record(
        float(duration_ms), {"index_type": index_type}
    )


def record_index_build(build_type: str) -> None:
    """`rosalinddb.index_builds` +1. `build_type` is `full`, `incremental`, or `delete`.

    Lets the incremental-vs-full-rebuild ratio be observed without any
    per-tenant/dataset label — `build_type` is the only (low-cardinality)
    attribute, so operators can alert on a rising full rebuild rate.
    """
    _get_instruments()["index_builds"].add(1, {"build_type": build_type})


def record_vectors_added(count: int, build_type: str) -> None:
    """`rosalinddb.index_build.vectors_added` histogram sample.

    The number of vectors folded into the index by a single build, tagged by
    `build_type` (`full` or `incremental`). For a `full` build this is the
    whole dataset; for an `incremental` build it is just the new batch.
    """
    _get_instruments()["index_build.vectors_added"].record(
        float(count), {"build_type": build_type}
    )


def record_quota_rejection(kind: str) -> None:
    """`rosalinddb.quota.rejections` +1.

    `kind` is `vector`, `query`, or `rate_limit`.
    """
    _get_instruments()["quota.rejections"].add(1, {"kind": kind})


def record_import_terminal(status: str, fmt: str) -> None:
    """`rosalinddb.imports` +1 — a bulk import job reaching a terminal state.

    `status` is `completed` or `failed`; `fmt` is `ndjson` or `parquet`. Both
    are low-cardinality (2 values each) — no tenant/dataset/import_id label,
    so the import success rate by format is observable without blowing the
    cardinality budget.
    """
    _get_instruments()["imports"].add(1, {"status": status, "format": fmt})


def record_import_records(count: int, outcome: str) -> None:
    """`rosalinddb.import.records` += count.

    `outcome` is `accepted` or `rejected`. No per-entity label.
    """
    if count > 0:
        _get_instruments()["import.records"].add(count, {"outcome": outcome})


def record_storage_swept(count: int, kind: str) -> None:
    """`rosalinddb.storage.swept` += count.

    `kind` is `shard` (a superseded FAISS shard `.bin`/`.meta.json` pair) or
    `landing` (an indexed landing object — part, staged upload, or rejected
    sidecar). Low-cardinality: no tenant/dataset label.
    """
    if count > 0:
        _get_instruments()["storage.swept"].add(count, {"kind": kind})
