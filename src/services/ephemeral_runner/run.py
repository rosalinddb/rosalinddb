from __future__ import annotations

"""Ephemeral Runner service.

Consumes `RUN_EPHEMERAL_QUERY` messages, downloads the latest index shard for a
dataset, and performs a FAISS top-K search. In the minimal implementation,
results are computed and discarded; production would return via callback.
"""

import time
from typing import Any, Dict, Optional, Tuple

import faiss  # type: ignore
import numpy as np

from adapters import config
from adapters.cache import CatalogCache, ensure_cached as _ensure_cached_shared
from adapters.errors import classify_query_error
from adapters.observability import init_observability
from adapters.observability import metrics as obs_metrics
from adapters.observability.tracing import (
    ephemeral_query_span,
    faiss_load_index_span,
    faiss_search_span,
    shard_download_span,
)
from adapters.queue.queue import consume, publish, ack, nack
from adapters.queue.shutdown import install_signal_handlers, should_stop
from adapters.state.state import list_shards, migrate
from adapters.landing.parquet_reader import read_shard_sidecar
from adapters.metrics.server import (
    make_metrics_handler,
    start_metrics_server as _start_metrics_server,
)
# SSD-tier import. The import-time `.tmp` orphan sweep is bounded and
# log-and-continue, so importing unconditionally is cheap; the activation
# gate inside `_ensure_cached` makes sure the tier's `fetch` only runs when
# the deployment has opted in via `RB_SHARD_TIER_BYTES`.
from adapters.storage import shard_tier  # noqa: F401  (imported for its import-time .tmp orphan sweep; see comment above)

# Reuse the hot-path AND-of-equals predicate so the two query paths cannot
# drift. The ephemeral runner does a brute-force search and already has every
# candidate's metadata in memory (from the shard sidecar), so it simply
# excludes non-matching hits — no over-fetch is needed.
from services.query_api.v1_query import (
    _MMAP_ENABLED,
    _ivf_search_params,
    _read_major_faults,
    metadata_matches_filter,
)

# This module's `_ensure_cached` (below) and catalog cache call the shared
# `adapters.cache` helpers. The ephemeral path deliberately runs WITHOUT
# download coalescing (`coalescing=False`): it runs at much lower concurrency
# than the hot path, where the download stampede was observed and fixed. If
# ephemeral concurrency grows, flip `coalescing=True` and supply the
# single-flight state.

# Observability bootstrap at import — idempotent; see validator_worker/run.py.
init_observability("rosalinddb-ephemeral-runner")

CACHE_DIR = config.cache_dir()
METRICS_PORT = config.ephemeral_metrics_port()


# --- Per-`(tenant, dataset)` catalog cache -----------------------------------
#
# The cache logic lives once in `adapters.cache.CatalogCache`; this runner
# owns its own instance (state is per-service — independent of the hot path's
# cache). The thin wrappers below resolve `list_shards` / `_now` from THIS
# module's namespace at call time so a test monkeypatching either retunes the
# cache. Public shape is unchanged:
#   - `_cached_list_shards(tenant, dataset) -> list`
#   - `_invalidate_catalog_cache(tenant, dataset) -> bool`
#   - `_on_catalog_notify(payload: dict) -> None`
#   - `_now()` indirected for testability
#   - `_catalog_cache_clear()` test helper
_CATALOG_CACHE = CatalogCache()


def _now() -> float:
    """Monotonic clock source for cache TTL (patchable for tests)."""
    return time.monotonic()


def _catalog_freshness_s() -> float:
    """`RB_CATALOG_FRESHNESS_S` in seconds (default 5; 0 disables cache)."""
    return config.catalog_freshness_s()


def _catalog_cache_active() -> bool:
    """Active iff SSD tier is on AND TTL is positive (default-off rollback)."""
    return config.shard_tier_bytes_set() and _catalog_freshness_s() > 0


def _cached_list_shards(tenant: str, dataset: str) -> list:
    """TTL-cached wrapper around `list_shards`.

    Single-flight is NOT a contract: N concurrent callers on a cold miss
    can each call the source. `list_shards` is cheap relative to the
    search itself; the bounded worst case is a startup thundering herd.
    """
    return _CATALOG_CACHE.cached_list_shards(
        tenant,
        dataset,
        list_shards,
        _now,
        _catalog_freshness_s(),
        _catalog_cache_active(),
    )


def _invalidate_catalog_cache(tenant: str, dataset: str) -> bool:
    """Drop the cached `(tenant, dataset)` entry. Idempotent.

    Bumps the per-key generation counter so an in-flight fetch refuses
    to install the now-stale rows.
    """
    return _CATALOG_CACHE.invalidate(tenant, dataset)


def _on_catalog_notify(payload: dict) -> None:
    """Listener subscriber — evict the affected entry on a notify payload."""
    tenant = payload.get("tenant") or ""
    dataset = payload.get("dataset") or ""
    if tenant and dataset:
        _invalidate_catalog_cache(tenant, dataset)


def _catalog_cache_clear() -> None:
    """Test helper — flush every cached entry."""
    _CATALOG_CACHE.clear()


# Match the v1_query opt-in: subscribe the cache invalidator to the
# LISTEN consumer when `RB_CATALOG_LISTEN=true`. Default-off; with the
# env unset, no thread is spawned and the TTL pull is the only channel.
if config.catalog_listen():
    from services._common import catalog_listener as _catalog_listener

    _catalog_listener.subscribe(_on_catalog_notify)


# Classify a worker exception into a v1 error envelope before it is published
# on RESULT_READY. The classification answers two questions for a caller: *what
# kind of failure* (cache fs broken, S3 outage, anything else) and a HUMAN-SAFE
# message that does not leak internal paths, credentials, or the full traceback.
#
# Order matters — `PermissionError` is a subclass of `OSError`, so it must be
# matched first; `FileNotFoundError` is also an `OSError` so it sits ahead of
# the generic OSError branch as well. The unknown bucket is the catch-all
# `ephemeral_error` so an unexpected exception still propagates as a
# structured envelope (never `{matches:[]}`).
def _classify_error(exc: BaseException) -> Tuple[str, str]:
    """Map an exception to `(error_code, safe_message)` for the v1 envelope.

    Codes are drawn from / extended on the v1 error catalog (`docs/api/v1.md`):

      - `recall_unavailable` — the recall (pgvector) tier is unreachable; the
        query is safe to retry once recall recovers. Surfaces as 503.
      - `cache_unavailable` — the local shard cache directory is unreadable
        or unwritable. Self-host hint: check `CACHE_DIR` mount permissions.
      - `storage_unavailable` — the object-store fetch failed. The query is
        safe to retry once storage recovers.
      - `ephemeral_error` — generic; the exact cause is not safe to surface.

    The returned `safe_message` carries only the EXCEPTION CLASS NAME — never
    `str(exc)` — so a botocore `ClientError` cannot leak an S3 endpoint URL,
    a bucket name, or signed-URL parameters into the customer-visible error.

    The classification table itself now lives once in
    `adapters.errors.classify_query_error` (it owns the exception hierarchy and
    the only adapter-layer imports the table needs, so importing it here does NOT
    violate the one-way rule the way importing from `services.query_api` would).
    This wrapper pins the cold-shard catch-all message
    (`"Cold-shard query failed: <Cls>"`); the hot path's
    `_classify_hot_path_error` delegates to the same canonical table with its own
    `"Query failed: <Cls>"` prefix, so the two classifiers can no longer drift.
    """
    return classify_query_error(exc, default_message_prefix="Cold-shard query failed")


def _ensure_cached(shard_uri: str) -> str:
    """Ensure a shard is present in local cache and return its path.

    FAISS's `read_index` needs a filesystem path, so an object-store shard
    (`s3://` or `memory://`) is fetched once into `CACHE_DIR`. There is no
    `file://` branch — RosalindDB is object-storage-first.

    The fetch is written **atomically** — to a unique temp file, then
    `os.replace`-renamed into place. When several RosalindDB processes share
    one `CACHE_DIR` (a multi-worker / multi-service deployment), a plain
    `open(path,"wb").write(...)` races: one process creates the file (so a
    concurrent `os.path.exists` sees it) while another reads it mid-write and
    `faiss.read_index` fails on the truncated bytes. An atomic rename means a
    reader sees either no file or the fully-written file — never a partial one.

    SSD-tier activation gate (same wiring as the hot path in
    `services/query_api/v1_query.py`). When `RB_SHARD_TIER_BYTES` is set,
    delegation runs through `shard_tier.fetch(shard_uri)`. When the env is
    unset the legacy body runs unchanged so a deployment that has not opted
    in sees unchanged behaviour.

    The body lives in `adapters.cache.shard_fetch.ensure_cached`. The
    ephemeral path passes `coalescing=False`: it runs at much lower
    concurrency than the hot path, where the download stampede was observed
    and fixed.
    """
    return _ensure_cached_shared(
        shard_uri,
        cache_dir=CACHE_DIR,
        coalescing=False,
    )


def handle(
    tenant: str,
    dataset: str,
    vector: list[float],
    top_k: int,
    flt: Optional[Dict[str, Any]] = None,
):
    """Execute a top-K FAISS search against the newest shard for a dataset.

    Results carry the customer's original string `id` and `metadata`, not the
    raw FAISS int64 hash. FAISS `IndexIDMap2` only stores the SHA1-derived hash
    of each id, so we load the `{shard_uri}.meta.json` sidecar written by the
    index builder and map every hit back. This keeps the ephemeral
    `RESULT_READY` payload identical in shape to the hot path. A hit missing
    from the sidecar (defensive) falls back to the stringified hash with empty
    metadata. `score` is the raw FAISS L2 distance.

    Metadata filtering: when `flt` is a non-empty dict, the AND-of-equals
    predicate (`metadata_matches_filter`, shared with the hot path) is applied
    to each candidate. The filtered path is *exhaustive*: FAISS is asked for
    the full shard (`fetch_k = ntotal`) and, on an IVF index, every cell is
    scanned (`nprobe = nlist`) — a filter-match in an unprobed cell would
    otherwise be invisible. The predicate then filters the hits and the
    survivors — already nearest-first — are truncated to `top_k`. The search
    is exhaustive (every cell, every vector) and therefore exact for IVFFlat
    and flat indexes; a legacy IVF+PQ shard's per-vector distances stay
    PQ-approximate (full cell coverage fixes which cells are scanned, not
    PQ's lossy quantization). A filtered query returns exactly `min(top_k,
    total_matching)`. A query can still legitimately return fewer than `top_k`
    results (or zero) when the dataset genuinely contains fewer than `top_k`
    matching records — an exact answer, not an approximation or an error
    (identical to the hot path).
    """
    # Catalog cache wrapper. With the SSD tier off OR `RB_CATALOG_FRESHNESS_S=0`
    # this is a passthrough to `list_shards`; with both active, repeated
    # lookups for the same dataset within the TTL skip the Postgres round-trip.
    # Invalidated on notify when `RB_CATALOG_LISTEN=true` (the LISTEN consumer
    # is the latency optimisation; the TTL is the correctness mechanism).
    shards = _cached_list_shards(tenant, dataset)
    if not shards:
        return []
    latest = shards[0]
    has_filter = bool(flt)
    # Cold load decomposed into separate spans — `shard.download` and
    # `faiss.load_index` — so the ephemeral path is attributable the same way
    # the query hot path is; `faiss.search` then times the vector search
    # alone. All nest under the `ephemeral_query` span (and the upload/query
    # trace, via queue propagation).
    with shard_download_span(uri=latest["shard_uri"]):
        # `_ensure_cached` internally checks `RB_SHARD_TIER_BYTES` to
        # decide between the SSD tier and the legacy in-process body.
        # The URI is the cache key in either path.
        local = _ensure_cached(latest["shard_uri"])
    with faiss_load_index_span(uri=latest["shard_uri"], mmap=_MMAP_ENABLED):
        # Mmap parity with the hot path: when `RB_FAISS_MMAP=true` the cold
        # query in this runner must also avoid slurping a multi-GB shard into
        # RSS. The flag is captured at v1_query import time and re-exported so
        # the two services cannot drift apart on a deploy that flips the env.
        if _MMAP_ENABLED:
            index = faiss.read_index(
                local, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            )
        else:
            index = faiss.read_index(local)
    sidecar = read_shard_sidecar(latest["shard_uri"])
    x = np.array([vector], dtype=np.float32)
    # `faiss.search` span — high-cardinality tenant/dataset attrs are correct
    # on a span. Now wraps the vector search alone.
    #
    # Page-fault sampler mirrors the hot path's accounting (see
    # `services/query_api/v1_query.py` around the matching block): read
    # `/proc/self/stat`'s `majflt` before and after the search and record the
    # delta to `rosalinddb.shard.page_faults`. On the mmap path this is the
    # operator-facing signal that an ephemeral cold-search had to fault pages
    # in from disk; with mmap off the delta stays at zero. Without this the
    # counter undercounted whenever an ephemeral cell ran.
    maj_before = _read_major_faults()
    with faiss_search_span(tenant=tenant, dataset=dataset, top_k=top_k):
        # With a filter, search the entire shard so no matching record can be
        # missed; without one, top_k is sufficient.
        if has_filter:
            # Exhaustive-when-filtered: fetch every vector AND, on an IVF
            # index, scan every cell (`nprobe = nlist`). Without full cell
            # coverage a filter-match in an unprobed cell stays invisible no
            # matter how large `fetch_k` is. A flat index is already exact, so
            # `_ivf_search_params` returns no params for it.
            #
            # IDSelector-based pre-filtering is the intended optimization for
            # large shards and is deliberately deferred: the full scan is
            # correct and fast at MVP scale (datasets capped at 100k vectors).
            fetch_k = int(getattr(index, "ntotal", 0)) or top_k
            search_params, _nprobe = _ivf_search_params(index, full_coverage=True)
            search_kwargs = (
                {"params": search_params} if search_params is not None else {}
            )
        else:
            fetch_k = top_k
            search_kwargs = {}
        distances, ids = index.search(x, fetch_k, **search_kwargs)
    maj_after = _read_major_faults()
    if maj_before is not None and maj_after is not None:
        obs_metrics.record_shard_page_faults(max(0, maj_after - maj_before))
    matches: list[dict] = []
    for i in range(min(fetch_k, len(ids[0]))):
        raw_id = int(ids[0][i])
        if raw_id == -1:
            continue
        entry = sidecar.get(str(raw_id))
        if entry is not None:
            match = {
                "id": entry.get("id", str(raw_id)),
                "score": float(distances[0][i]),
                "metadata": entry.get("metadata") or {},
            }
        else:
            match = {"id": str(raw_id), "score": float(distances[0][i]), "metadata": {}}
        # AND-of-equals filtering: drop candidates that violate the filter.
        if has_filter and not metadata_matches_filter(match["metadata"], flt):
            continue
        matches.append(match)
        # Hits arrive nearest-first; once we have top_k survivors, stop.
        if len(matches) >= top_k:
            break
    if has_filter:
        # Count the filtered query and its post-filter result size, mirroring
        # the hot path (`v1_query.v1_query`).
        obs_metrics.record_filtered_query()
        obs_metrics.record_filtered_result_count(len(matches))
    return matches


# The metrics HTTP handler + server are the canonical implementation in
# `adapters.metrics.server`. `MetricsHandler` is re-exported (a configured
# subclass with this service's `/healthz` service name + Prometheus prefix) so
# the name stays importable from this module; `start_metrics_server()` keeps its
# no-arg signature and forwards this service's two strings + `METRICS_PORT`.
MetricsHandler = make_metrics_handler("ephemeral_runner", "ephemeral_")


def start_metrics_server():
    """Start the metrics HTTP server in a background thread."""
    return _start_metrics_server("ephemeral_runner", "ephemeral_", METRICS_PORT)


def main_loop():
    """Blocking loop that consumes and handles ephemeral query requests.

    Reliable-queue contract: a RUN_EPHEMERAL_QUERY message is `ack`-ed once the
    search completes and the RESULT_READY reply (if any) has been published; an
    UNHANDLED crash `nack`s it for redelivery (then dead-lettering past
    `QUEUE_MAX_ATTEMPTS`). On `SIGTERM` the loop stops pulling new messages and
    exits cleanly.

    A search that raises (storage outage, cache fs unwritable, FAISS load
    failure, …) now publishes a STRUCTURED ERROR ENVELOPE to `reply_to` before
    re-raising so the status poll surfaces a real failure (HTTP 503) rather than
    an empty 200. The NACK + DLQ behaviour is preserved: every delivery attempt
    still publishes its own envelope, and a message exhausting
    `QUEUE_MAX_ATTEMPTS` is dead-lettered exactly as before.
    """
    # Fail-fast on required-at-boot config (reads env fresh) before consuming.
    config.validate()
    migrate()
    install_signal_handlers()
    start_metrics_server()
    while not should_stop():
        msg = consume("RUN_EPHEMERAL_QUERY", block=True, timeout=1.0)
        if not msg:
            continue
        try:
            _handle_ephemeral(msg)
        except Exception as exc:  # noqa: BLE001
            # The error envelope was already published inside `_handle_ephemeral`
            # (so the status poll surfaces a 503, not a silent empty 200);
            # NACK lets the reliable queue retry up to `QUEUE_MAX_ATTEMPTS`
            # before dead-lettering. Logging carries the exception class + a
            # safe message so an operator can correlate the DLQ growth with
            # the failure shape without leaking paths/credentials.
            print(
                "ephemeral: handler error, nacking message: "
                f"exc_class={type(exc).__name__} msg={exc!s}"
            )
            nack(msg, requeue=True)
            continue
        ack(msg)
    print("ephemeral: shutdown signal received — exiting consume loop")


def _publish_error_envelope(
    reply_to: str,
    correlation_id: Optional[str],
    dataset: str,
    code: str,
    message: str,
    latency_ms: int,
) -> None:
    """Publish a structured failure envelope on the reply queue.

    The `RESULT_READY` consumer stashes the envelope under `correlation_id` in
    the shared result store, so `GET /v1/query/status/{job_id}` finds an
    `ok: false` payload instead of a missing key and surfaces it as HTTP 503.

    An unaddressable message (`correlation_id` absent) cannot be routed back
    to any client — we still record the failure in the trace/log but skip the
    publish (a message on `RESULT_READY` with no correlation id is unreachable
    by any status poll).
    """
    if not correlation_id:
        return
    publish(
        reply_to,
        {
            "correlation_id": correlation_id,
            "dataset": dataset,
            "ok": False,
            "error": {"code": code, "message": message},
            "latency_ms": latency_ms,
        },
    )


def _handle_ephemeral(msg) -> None:
    """Run one RUN_EPHEMERAL_QUERY search and publish its RESULT_READY reply.

    On a handler exception the worker publishes an EXPLICIT ERROR ENVELOPE
    (`{ok: false, error: {code, message}}`) to `reply_to` before re-raising,
    instead of silently NACKing and leaving the caller's status poll to see
    `{matches: []}` forever. The exception is then re-raised so `main_loop`
    still NACKs the queue message — both behaviours are needed: the envelope
    unblocks the caller, the NACK lets the reliable queue retry up to
    `QUEUE_MAX_ATTEMPTS` before dead-lettering.
    """
    dataset = msg["dataset"]
    tenant = msg.get("tenant", "default")
    vector = msg["vector"]
    top_k = int(msg.get("top_k", 10))
    # The filter rides on the RUN_EPHEMERAL_QUERY message; absent / null
    # collapses to no filtering.
    flt = msg.get("filter") or None
    correlation_id = msg.get("correlation_id")
    reply_to = msg.get("reply_to", "RESULT_READY")
    start = time.time()
    # `ephemeral_query` span — child of the originating `POST /v1/query`
    # trace (the queue adapter attached the producer context on consume()).
    # The `span(...)` helper records the exception and sets ERROR status on
    # the span automatically, so the trace already shows the failure.
    try:
        with ephemeral_query_span(tenant=tenant, dataset=dataset):
            matches = handle(tenant, dataset, vector, top_k, flt)
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.time() - start) * 1000.0)
        code, safe_message = _classify_error(exc)
        # Structured log: dataset, tenant, exception class, raw message.
        # `repr(exc)` instead of `str(exc)` so a bare exception (e.g.
        # `PermissionError()` with no message) is still distinguishable in
        # logs from one with a path attached.
        print(
            "ephemeral: search failed: "
            f"tenant={tenant} dataset={dataset} "
            f"exc_class={type(exc).__name__} exc={exc!r} code={code}"
        )
        _publish_error_envelope(
            reply_to, correlation_id, dataset, code, safe_message, latency_ms
        )
        # Re-raise so `main_loop` runs its NACK + retry logic. The envelope
        # is already on the wire so the caller is unblocked regardless.
        raise
    latency_ms = int((time.time() - start) * 1000.0)
    if correlation_id:
        publish(
            reply_to,
            {
                "correlation_id": correlation_id,
                "dataset": dataset,
                "ok": True,
                "matches": matches,
                "latency_ms": latency_ms,
            },
        )


if __name__ == "__main__":
    main_loop()

